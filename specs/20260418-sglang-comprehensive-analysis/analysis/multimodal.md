# Multi-Modal Model Support

## Overview

SGLang provides built-in support for Vision-Language Models (VLMs), audio-language models, and other multi-modal architectures. The system handles image/video/audio preprocessing, vision encoder execution, and cross-modal token embedding within the same serving infrastructure as text-only LLMs.

**File**: `python/sglang/srt/multimodal/`

---

## Architecture

```
Client Request (text + images/video/audio)
    ↓
TokenizerManager: BaseMultimodalProcessor.process(text, media)
    ↓ → MultiModalProcessorOutput (input_ids, pixel_values, image_offsets, ...)
Scheduler: Embed visual tokens + prepare KV allocation
    ↓
ModelRunner:
    ├── Vision Encoder forward → image embeddings
    ├── Merge text embeddings + image embeddings at placeholder positions
    └── LLM forward with fused input
```

---

## BaseMultimodalProcessor

**File**: `multimodal/processors/base_processor.py`

The abstract base class (~41KB) that all model-specific processors inherit:

```python
class BaseMultimodalProcessor(ABC):
    def __init__(self, hf_config, server_args, _processor):
        self.hf_config = hf_config
        self._processor = _processor  # HuggingFace processor
    
    def process(self, text: str, media: List[Media]) -> MultiModalProcessorOutput:
        """Convert raw inputs into model-ready format."""
        # 1. Load media (images/videos/audio from URLs, paths, base64)
        # 2. Apply model-specific preprocessing (resize, normalize, patch)
        # 3. Tokenize text with placeholder tokens for media
        # 4. Return structured output
```

### MultiModalProcessorOutput

```python
@dataclass
class MultiModalProcessorOutput:
    input_ids: List[int]                    # Token IDs with media placeholders
    pixel_values: Optional[torch.Tensor]    # [N_images, C, H, W] or patches
    image_sizes: Optional[List[Tuple]]      # Original image dimensions
    image_offsets: Optional[List[int]]      # Positions of image tokens in sequence
    video_inputs: Optional[Dict]            # Video frames/features
    audio_inputs: Optional[Dict]            # Audio features
    modalities: List[str]                   # ["image", "video", "audio"]
```

---

## Supported Models (40+ families)

### Vision-Language Models

| Model Family | Processor File | Key Features |
|---|---|---|
| LLaVA | `llava.py` | Multi-image, dynamic resolution |
| Qwen-VL | `qwen_vl.py` | Multi-modal with interleaved text-image |
| InternVL | `internvl.py` | Dynamic tiling, multi-image |
| Deepseek-VL | `deepseek_vl_v2.py` | OCR-optimized vision |
| Phi-4-MM | `phi4mm.py` | Multi-modal Phi |
| Pixtral | `pixtral.py` | Instruction-following VLM |
| MiniCPM | `minicpm.py` | Efficient multi-modal |
| GLM-4V | `glm4v.py` | ChatGLM vision variant |
| Gemma-3 | `gemma3.py` | Google multi-modal |
| Molmo | `molmo.py` | Pointing/grounding VLM |

### Audio-Language Models

| Model Family | Processor File | Key Features |
|---|---|---|
| Qwen-Audio | `qwen_audio.py` | Audio understanding |
| GLM-ASR | `glmasr.py` | Speech recognition |

### Video-Language Models

| Model Family | Processor File | Key Features |
|---|---|---|
| Qwen-VL (video) | `qwen_vl.py` | Temporal understanding |
| InternVL (video) | `internvl.py` | Long video support |

---

## Media Loading

The base processor handles multiple input formats:

```python
# Image loading supports:
# - HTTP/HTTPS URLs → async download
# - Local file paths
# - Base64 encoded data
# - PIL Image objects
# - Torch tensors

def load_image(self, image_input) -> PIL.Image:
    if isinstance(image_input, str):
        if image_input.startswith("http"):
            return load_from_url(image_input)
        elif image_input.startswith("data:image"):
            return load_from_base64(image_input)
        else:
            return load_from_file(image_input)
```

---

## Vision Encoder Execution

### Standard Path

Vision encoders run as part of the model forward:

```python
# In model forward:
def forward(self, input_ids, pixel_values, image_offsets, ...):
    # 1. Text embedding
    text_embeds = self.embed_tokens(input_ids)
    
    # 2. Vision encoding (if media present)
    if pixel_values is not None:
        image_embeds = self.vision_encoder(pixel_values)
        # Projection to text embedding space
        image_embeds = self.multi_modal_projector(image_embeds)
    
    # 3. Merge at placeholder positions
    for i, offset in enumerate(image_offsets):
        text_embeds[offset:offset+n_image_tokens] = image_embeds[i]
    
    # 4. LLM forward with fused embeddings
    return self.language_model(inputs_embeds=text_embeds)
```

### CUDA Graph Optimization

**File**: `multimodal/vit_cuda_graph_runner.py`

For repeated vision encoder calls with the same resolution:

```python
class VitCudaGraphRunner:
    """Captures vision encoder as CUDA graph for zero-overhead replay."""
    
    def __init__(self, vision_model, max_batch_size):
        # Pre-allocate fixed-size buffers for vision inputs
        self.static_pixel_values = torch.empty(max_bs, 3, H, W)
        # Capture graph
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.static_output = vision_model(self.static_pixel_values)
    
    def forward(self, pixel_values):
        self.static_pixel_values.copy_(pixel_values)
        self.graph.replay()
        return self.static_output.clone()
```

### Data-Parallel Vision Encoding

**File**: `multimodal/mm_utils.py`

For large batches, vision encoding can be distributed:

```python
def get_dp_encoder_lb_assignment(num_images, num_gpus):
    """Load-balance image encoding across data-parallel GPUs."""
    # Assigns images to GPUs for parallel encoding
    # Results gathered back to coordinating rank
```

---

## EVS (Early Exit Vision Sampling)

**File**: `multimodal/evs/`

An optimization that reduces vision encoder computation:

```
Standard: Run all ViT layers (24-48 layers) for every image patch
EVS:      Exit early for "easy" patches, full computation for "hard" patches
```

Components:
- `evs_core.py`: Core sampling logic
- `evs_module.py`: ViT integration
- `evs_processor.py`: Processing pipeline

---

## Disaggregated Vision Encoding

**File**: `srt/disaggregation/encode_server.py`

For very large deployments, vision encoding can run on separate GPU(s):

```
Separate Encode Server:
    - Receives images/video
    - Runs vision encoder
    - Returns embeddings via network

Main LLM Server:
    - Receives pre-computed embeddings
    - Merges with text tokens
    - Runs LLM inference only
```

This enables:
- Independent scaling of vision vs language compute
- Caching of repeated image embeddings
- Cross-request embedding reuse

---

## Token Management

### Placeholder Tokens

Each model family uses specific placeholder tokens for media positions:

```python
# Example: LLaVA uses <image> token
# Qwen-VL uses <|vision_start|>...<|vision_end|>
# InternVL uses <IMG_CONTEXT> tokens

class MultimodalSpecialTokens:
    image_token: str        # e.g., "<image>"
    video_token: str        # e.g., "<video>"
    audio_token: str        # e.g., "<audio>"
    image_token_id: int     # Vocabulary ID for image placeholder
```

### Dynamic Resolution

Many VLMs support variable image sizes through dynamic patching:

```python
# InternVL dynamic resolution:
# 224px image → 1 tile (256 tokens)
# 448px image → 4 tiles (1024 tokens)  
# 896px image → 16 tiles (4096 tokens)

def get_num_image_tokens(self, image_size, model_config):
    """Calculate token count based on image resolution."""
    num_tiles = calculate_tiles(image_size, max_tiles=model_config.max_tiles)
    return num_tiles * self.tokens_per_tile
```

---

## Configuration

```bash
# Launch VLM server
python -m sglang.launch_server \
    --model-path Qwen/Qwen2.5-VL-7B-Instruct \
    --chat-template qwen2-vl

# With disaggregated vision encoding
python -m sglang.launch_server \
    --model-path InternVL2-8B \
    --enable-dp-encoder-lb  # Load-balanced vision encoding
```

---

## Source References

- Base Processor: `multimodal/processors/base_processor.py`
- MM Utils: `multimodal/mm_utils.py`
- ViT CUDA Graph: `multimodal/vit_cuda_graph_runner.py`
- EVS: `multimodal/evs/`
- Disaggregated Encoding: `srt/disaggregation/encode_server.py`
- Model Processors: `multimodal/processors/` (40+ model-specific files)
