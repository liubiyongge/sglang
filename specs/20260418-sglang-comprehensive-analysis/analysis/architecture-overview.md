# Architecture Overview: SGLang Process Model

**Source**: `python/sglang/srt/entrypoints/engine.py`, `python/sglang/srt/managers/`

---

## High-Level Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                              SGLang System Architecture                               │
├─────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                     │
│  ┌──────────────────────────────────────────────────────────────────────────────┐   │
│  │                         USER INTERFACES                                       │   │
│  │  HTTP Server ─── gRPC Server ─── Python Engine API ─── CLI                   │   │
│  │  (FastAPI/uvicorn)  (grpc_server.py)  (engine.py)      (launch_server.py)    │   │
│  └─────────────────────────────────┬────────────────────────────────────────────┘   │
│                                    │                                                 │
│  ┌─────────────────────────────────┴────────────────────────────────────────────┐   │
│  │                    TOKENIZER MANAGER PROCESS                                  │   │
│  │  ┌────────────┐  ┌──────────────┐  ┌────────────────┐  ┌───────────────┐    │   │
│  │  │ Tokenizer  │  │ Chat Template│  │ Multimodal     │  │ Multi-Tokenizer│   │   │
│  │  │ (HF Fast)  │  │ Manager      │  │ Preprocessor   │  │ Router        │    │   │
│  │  └────────────┘  └──────────────┘  └────────────────┘  └───────────────┘    │   │
│  └─────────────────────────────────┬────────────────────────────────────────────┘   │
│                                    │ ZMQ (tokenized requests)                        │
│                                    ▼                                                 │
│  ┌──────────────────────────────────────────────────────────────────────────────┐   │
│  │              DATA PARALLEL CONTROLLER (optional)                              │   │
│  │  Load Balancing: ROUND_ROBIN | TOTAL_REQUESTS | TOTAL_TOKENS                  │   │
│  └───────────┬──────────────────────┬──────────────────────┬────────────────────┘   │
│              │                      │                      │                         │
│              ▼                      ▼                      ▼                         │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐                     │
│  │  SCHEDULER #0   │  │  SCHEDULER #1   │  │  SCHEDULER #N   │  (DP workers)       │
│  │  ┌───────────┐  │  │                 │  │                 │                     │
│  │  │Event Loop │  │  │    (same as     │  │    (same as     │                     │
│  │  │ ├─recv    │  │  │     #0)         │  │     #0)         │                     │
│  │  │ ├─schedule│  │  │                 │  │                 │                     │
│  │  │ ├─forward │  │  │                 │  │                 │                     │
│  │  │ └─process │  │  │                 │  │                 │                     │
│  │  ├───────────┤  │  └─────────────────┘  └─────────────────┘                     │
│  │  │           │  │                                                                │
│  │  │ ┌───────┐ │  │  ┌────────────────────────────────────────────────────────┐   │
│  │  │ │Waiting│ │  │  │              GPU EXECUTION LAYER                        │   │
│  │  │ │Queue  │ │  │  │                                                        │   │
│  │  │ ├───────┤ │  │  │  ┌──────────┐  ┌──────────┐  ┌──────────┐            │   │
│  │  │ │Running│ │  │  │  │TP Worker │  │TP Worker │  │TP Worker │  (TP=4)    │   │
│  │  │ │Batch  │ │  │  │  │  Rank 0  │  │  Rank 1  │  │  Rank 3  │            │   │
│  │  │ ├───────┤ │  │  │  ├──────────┤  ├──────────┤  ├──────────┤            │   │
│  │  │ │Radix  │ │  │  │  │Model     │  │Model     │  │Model     │            │   │
│  │  │ │Cache  │ │  │  │  │Runner    │  │Runner    │  │Runner    │            │   │
│  │  │ ├───────┤ │  │  │  ├──────────┤  ├──────────┤  ├──────────┤            │   │
│  │  │ │Memory │ │  │  │  │CUDA Graph│  │CUDA Graph│  │CUDA Graph│            │   │
│  │  │ │Pool   │ │  │  │  │Runner    │  │Runner    │  │Runner    │            │   │
│  │  │ └───────┘ │  │  │  └──────────┘  └──────────┘  └──────────┘            │   │
│  │  └───────────┘  │  │         │              │              │                │   │
│  └─────────────────┘  │         └──────────────┼──────────────┘                │   │
│                        │                 NCCL AllReduce                          │   │
│                        └────────────────────────────────────────────────────────┘   │
│                                    │                                                 │
│                                    │ ZMQ (output tokens)                             │
│                                    ▼                                                 │
│  ┌──────────────────────────────────────────────────────────────────────────────┐   │
│  │                    DETOKENIZER MANAGER PROCESS                                │   │
│  │  Streaming detokenization → SSE/WebSocket response                            │   │
│  └──────────────────────────────────────────────────────────────────────────────┘   │
│                                                                                     │
└─────────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────────┐
│                        DIFFUSION RUNTIME (Separate)                                   │
├─────────────────────────────────────────────────────────────────────────────────────┤
│  HTTP Server → ZMQ Scheduler → GPU Worker → Pipeline (Encode→Denoise→Decode)        │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

---

## Process Spawning Model

```
Engine.__init__()
    │
    ├── spawn TokenizerManager (Process)
    │     └── Handles: tokenization, chat templates, multimodal data
    │
    ├── spawn DataParallelController (Process, if dp_size > 1)
    │     └── Routes requests to multiple scheduler instances
    │
    ├── spawn Scheduler(s) (Process, one per DP rank)
    │     ├── Manages: waiting queue, running batch, RadixCache
    │     ├── spawn TpModelWorker(s) (Thread per TP rank)
    │     │     └── GPU forward pass, CUDA graphs
    │     └── Optional: Draft worker (for speculative decoding)
    │
    └── spawn DetokenizerManager (Process)
          └── Converts token IDs back to text, streams to client
```

---

## Communication Topology

```
Client ──HTTP──→ [TokenizerManager] ──ZMQ PUSH──→ [Scheduler]
                                                       │
                                                  GPU Forward
                                                       │
Client ←──HTTP── [DetokenizerManager] ←──ZMQ PUSH── [Scheduler]
```

**Protocol**: ZeroMQ with msgpack serialization
**Pattern**: PUSH/PULL for unidirectional message flow
**Addressing**: IPC sockets (unix domain) or TCP for distributed

---

## Key Design Principles

1. **Separation of Concerns**: CPU-bound (tokenization) isolated from GPU-bound (inference)
2. **Zero-Copy Where Possible**: Shared memory for large tensors between processes
3. **Overlap**: CPU scheduling runs while GPU computes (overlap event loop)
4. **Scalability**: DP controller enables horizontal scaling across GPU groups
5. **Fault Isolation**: Process boundaries prevent GPU hangs from blocking tokenization
