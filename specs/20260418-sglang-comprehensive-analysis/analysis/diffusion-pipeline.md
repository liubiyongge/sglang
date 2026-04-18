# Diffusion Pipeline Architecture: Stage-Based Pipeline System

## Overview

The diffusion runtime uses a **stage-based pipeline architecture** where each generation request flows through a sequence of composable stages. This design enables model-agnostic orchestration — the same executor can run Flux, WAN Video, HunyuanVideo, or QwenImage pipelines by composing different stage sequences.

**File Locations**:
- Base: `python/sglang/multimodal_gen/runtime/pipelines_core/`
- Concrete Pipelines: `python/sglang/multimodal_gen/runtime/pipelines/`

---

## Architecture Layers

```
GPUWorker.execute_forward(req)
    └── ComposedPipelineBase.forward(req, server_args)
            └── PipelineExecutor.execute_with_profiling(stages, batch, server_args)
                    └── ParallelExecutor._execute(stages, batch, server_args)
                            ├── Stage 1: InputValidationStage
                            ├── Stage 2: TextEncodingStage
                            ├── Stage 3: ConditioningStage
                            ├── Stage 4: LatentPreparationStage
                            ├── Stage 5: TimestepPreparationStage
                            ├── Stage 6: DenoisingStage
                            └── Stage 7: DecodingStage → OutputBatch
```

---

## Core Classes

### `ComposedPipelineBase` (ABC)

**File**: `pipelines_core/composed_pipeline_base.py`

The base class for all diffusion pipelines:

```python
class ComposedPipelineBase(ABC):
    _required_config_modules: list[str] = []  # e.g. ["vae", "text_encoder", "transformer", "scheduler"]
    modules: dict[str, Any] = {}               # loaded model components
    executor: PipelineExecutor | None = None    # stage execution engine
    
    def forward(self, batch: Req, server_args: ServerArgs) -> OutputBatch:
        return self.executor.execute_with_profiling(self.stages, batch, server_args)
```

**Key Methods**:
- `load_modules()`: Loads all required model components from `model_index.json` using `PipelineComponentLoader` (supports custom VAE paths, boundary_ratio for MoE DiTs)
- `create_pipeline_stages()`: Abstract — subclasses define their stage sequence
- `build_executor()`: Creates a `ParallelExecutor` (default)

### `PipelineStage` (ABC)

**File**: `pipelines_core/stages/base.py`

Each stage is a discrete, composable processing step:

```python
class PipelineStage(ABC):
    def __call__(self, batch: Req, server_args: ServerArgs) -> Req:
        # 1. Input verification
        input_result = self.verify_input(batch, server_args)
        self._run_verification(input_result, stage_name, "input")
        
        # 2. Execute with profiling
        with StageProfiler(stage_name, ...):
            result = self.forward(batch, server_args)
        
        # 3. Output verification
        output_result = self.verify_output(result, server_args)
        self._run_verification(output_result, stage_name, "output")
        
        return result
    
    @abstractmethod
    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        raise NotImplementedError
```

**Stage Parallelism Types**:

| Type | Behavior |
|------|----------|
| `REPLICATED` | Execute on all GPUs (default) |
| `MAIN_RANK_ONLY` | Only rank 0 executes; others barrier |
| `CFG_PARALLEL` | Broadcast batch from rank 0, all execute, barrier |

### `PipelineExecutor` / `ParallelExecutor`

**File**: `pipelines_core/executors/parallel_executor.py`

The executor orchestrates stage execution with parallelism awareness:

```python
class ParallelExecutor(PipelineExecutor):
    def _execute(self, stages, batch, server_args) -> OutputBatch:
        for stage in stages:
            paradigm = stage.parallelism_type
            if paradigm == StageParallelismType.MAIN_RANK_ONLY:
                if rank == 0: batch = stage(batch, server_args)
                torch.distributed.barrier()
            elif paradigm == StageParallelismType.CFG_PARALLEL:
                batch = broadcast_pyobj(...) if rank != 0
                batch = stage(batch, server_args)
                torch.distributed.barrier()
            elif paradigm == StageParallelismType.REPLICATED:
                batch = stage(batch, server_args)
```

---

## Standard Pipeline Stages

### Stage 1: Input Validation

**File**: `stages/input_validation.py`

Validates request parameters (dimensions, prompts, model compatibility).

### Stage 2: Text Encoding (`TextEncodingStage`)

**File**: `stages/text_encoding.py`

- Encodes positive and negative prompts using configured text encoders
- Supports multi-encoder pipelines (e.g., CLIP + T5 for Flux)
- Handles tokenization, model forward, and post-processing per encoder config
- Outputs: `batch.prompt_embeds`, `batch.negative_prompt_embeds`, `batch.pooled_embeds`

### Stage 3: Conditioning (`ConditioningStage`)

**File**: `stages/conditioning.py`

- Prepares conditioning signals (text embeddings, image embeddings, CLIP features)
- Model-specific conditioning preparation via pipeline config

### Stage 4: Latent Preparation (`LatentPreparationStage`)

**File**: `stages/latent_preparation.py`

- Generates initial noise latents: `torch.randn(shape, generator=...)`
- Handles I2V: encodes condition images to latent space
- Applies VAE scaling factor
- Shape: `[B, C, T, H, W]` for video, `[B, C, H, W]` for images

### Stage 5: Timestep Preparation (`TimestepPreparationStage`)

**File**: `stages/timestep_preparation.py`

- Computes denoising schedule using the configured noise scheduler
- Sets `batch.timesteps`, `batch.num_inference_steps`

### Stage 6: Denoising (`DenoisingStage`)

**File**: `stages/denoising.py` — **The core computation stage**

This is the iterative denoising loop:

```python
for i, t in enumerate(timesteps):
    # 1. Select model (high-noise vs low-noise expert for MoE/Wan2.2)
    current_model = self._select_and_manage_model(t, boundary_timestep)
    
    # 2. Prepare input
    latent_model_input = scheduler.scale_model_input(latents, t)
    
    # 3. Predict noise with CFG
    noise_pred = self._predict_noise_with_cfg(
        current_model, latent_model_input, timestep, ...)
    
    # 4. Scheduler step (Euler, UniPC, etc.)
    latents = scheduler.step(noise_pred, t, latents)
```

**Key Features**:
- **Dual-transformer support** (Wan2.2): Switches between high-noise and low-noise expert at `boundary_timestep`
- **CFG Parallel**: Positive/negative passes distributed across CFG ranks with `all_reduce`
- **Sequence Parallelism**: Latents sharded/gathered before/after denoising
- **TeaCache integration**: Temporal similarity caching for step-skipping
- **Cache-DiT**: Block-level caching for DiT layers
- **torch.compile**: Optional JIT compilation of transformer
- **STA/VSA/VMoBA**: Attention backend metadata construction per-step
- **Memory management**: LayerWise offload, DiT CPU offload between experts

### Stage 7: Decoding (`DecodingStage`)

**File**: `stages/decoding.py`

- Applies inverse VAE scaling + shift
- Runs `vae.decode(latents)` with autocast
- Supports tiled VAE for memory efficiency
- Normalizes output to [0, 1] range
- Handles trajectory decoding for visualization
- CPU offloads VAE after use

---

## Concrete Pipeline Examples

### WAN Video Pipeline

```python
class WanPipeline(ComposedPipelineBase):
    _required_config_modules = ["vae", "text_encoder", "transformer", "scheduler", "tokenizer"]
    
    def create_pipeline_stages(self, server_args):
        self.add_stage("input_validation", InputValidationStage(...))
        self.add_stage("text_encoding", TextEncodingStage([text_encoder], [tokenizer]))
        self.add_stage("conditioning", ConditioningStage(...))
        self.add_stage("latent_preparation", LatentPreparationStage(...))
        self.add_stage("timestep_preparation", TimestepPreparationStage(...))
        self.add_stage("denoising", DenoisingStage(transformer, scheduler))
        self.add_stage("decoding", DecodingStage(vae))
```

### Flux Image Pipeline

Similar structure but with dual text encoders (CLIP + T5) and different conditioning.

---

## Data Flow (Req Object)

The `Req` object (`schedule_batch.py`) carries all state through stages:

```
Input:  req.prompt, req.width, req.height, req.num_frames, req.guidance_scale
                              ↓
TextEncoding:   → req.prompt_embeds, req.negative_prompt_embeds
                              ↓
Conditioning:   → req.image_embeds, req.clip_embedding_pos/neg
                              ↓
LatentPrep:     → req.latents [B, C, T, H, W], req.image_latent
                              ↓
TimestepPrep:   → req.timesteps, req.num_inference_steps
                              ↓
Denoising:      → req.latents (denoised)
                              ↓
Decoding:       → OutputBatch(output=frames [B, C, T, H, W])
```

---

## Supported Models

The diffusion runtime supports multiple model families through pipeline configs:

| Model | Pipeline | Task Types |
|-------|----------|-----------|
| Flux 1/2 | `FluxPipeline` | T2I |
| WAN 2.1/2.2 | `WanPipeline` | T2V, I2V, TI2V |
| HunyuanVideo | `HunyuanPipeline` | T2V |
| GLM Image | `GlmImagePipeline` | T2I |
| QwenImage | `QwenImagePipeline` | T2I, I2I |
| ZImage | `ZImagePipeline` | T2I |
| LTX-2 | `Ltx2Pipeline` | T2V, T2A |
| MoVA | `MoVAPipeline` | T2V, T2A |

---

## Source References

- ComposedPipelineBase: `pipelines_core/composed_pipeline_base.py`
- PipelineStage: `pipelines_core/stages/base.py`
- DenoisingStage: `pipelines_core/stages/denoising.py`
- TextEncodingStage: `pipelines_core/stages/text_encoding.py`
- DecodingStage: `pipelines_core/stages/decoding.py`
- ParallelExecutor: `pipelines_core/executors/parallel_executor.py`
- Pipeline Registry: `python/sglang/multimodal_gen/runtime/pipelines/`
