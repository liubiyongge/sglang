"""
hiradix_cache.py - 层次化基数树(Hierarchical Radix Tree)缓存实现

================================================================================
HiRadixCache vs RadixCache 核心区别
================================================================================

1. 继承关系：
   HiRadixCache 继承自 RadixCache，是其层次化扩展版本。

2. 存储层级对比：
   ┌─────────────────────────────────────────────────────────────────────────┐
   │  RadixCache (单级存储):                                                 │
   │  ┌─────────────────┐                                                   │
   │  │   GPU Memory    │  ← KV Cache 只存在GPU，驱逐时直接删除              │
   │  └─────────────────┘                                                   │
   ├─────────────────────────────────────────────────────────────────────────┤
   │  HiRadixCache (三级层次存储):                                           │
   │  ┌─────────────────┐                                                   │
   │  │   L1: GPU       │  ← 热数据，快速访问，容量有限                       │
   │  ├─────────────────┤                                                   │
   │  │   L2: CPU Host  │  ← 温数据，中等延迟，容量较大                       │
   │  ├─────────────────┤                                                   │
   │  │   L3: Storage   │  ← 冷数据，持久化存储，跨会话共享                   │
   │  └─────────────────┘                                                   │
   └─────────────────────────────────────────────────────────────────────────┘

3. 关键特性对比：
   ┌──────────────────┬────────────────────┬────────────────────────────────┐
   │      特性         │    RadixCache      │        HiRadixCache            │
   ├──────────────────┼────────────────────┼────────────────────────────────┤
   │ 存储层级          │ 单级 (仅GPU)       │ 三级 (GPU→CPU→Storage)         │
   │ 驱逐目标          │ 直接删除           │ 可驱逐到CPU/存储               │
   │ 预取机制          │ 无                │ 支持从存储预取                 │
   │ 写策略            │ 无                │ write_back/write_through       │
   │ 分布式存储        │ 不支持            │ 支持外部存储后端               │
   │ 长上下文支持      │ 受GPU内存限制      │ 可卸载到CPU                    │
   │ 跨会话共享        │ 不支持            │ 支持存储后端                   │
   └──────────────────┴────────────────────┴────────────────────────────────┘

4. 核心工作流程：
   - 写入流程: GPU写入 → (可选)写入CPU → (可选)写入存储
   - 读取流程: 先查GPU → 未命中查CPU → 未命中从存储预取
   - 驱逐流程: GPU驱逐到CPU → CPU驱逐到存储 → 最终删除

5. 写策略说明：
   - write_through: 写入GPU时同时写入CPU，适合读多写少场景
   - write_back: 驱逐时才写入CPU，适合写多读少场景
   - write_through_selective: 选择性写穿透，根据命中次数决定

6. 适用场景：
   - RadixCache: 短请求、GPU内存充足、无需持久化
   - HiRadixCache: 长上下文、多轮对话、跨会话共享、成本敏感

================================================================================
与 Storage 后端的集成架构
================================================================================

┌─────────────────────────────────────────────────────────────────────────────┐
│                           HiRadixCache                                      │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                    RadixCache (父类)                                 │  │
│  │  - 基数树结构管理 (TreeNode, RadixKey)                               │  │
│  │  - 前缀匹配、插入、驱逐逻辑                                          │  │
│  │  - L1 (GPU) 内存管理                                                 │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                              │                                              │
│                              ▼                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                    HiCacheController                                 │  │
│  │  - 管理GPU↔CPU数据传输                                               │  │
│  │  - 管理CPU↔Storage数据传输                                           │  │
│  │  - 异步操作队列和线程管理                                            │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│           │                              │                                  │
│           ▼                              ▼                                  │
│  ┌─────────────────┐           ┌─────────────────────────────────────┐   │
│  │ L2: CPU Host    │           │ L3: Storage Backend                 │   │
│  │ Memory Pool     │           │ (hicache_storage.py)                │   │
│  │ (host_value)    │           │                                     │   │
│  └─────────────────┘           └─────────────────────────────────────┘   │
│                                              │                              │
└──────────────────────────────────────────────│──────────────────────────────┘
                                               │
                                               ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                     StorageBackendFactory                                   │
│  - 动态创建和加载存储后端                                                   │
│  - 支持内置后端和动态加载后端                                               │
└─────────────────────────────────────────────────────────────────────────────┘
                                               │
        ┌──────────────┬──────────────┬────────┴────────┬──────────────┐
        ▼              ▼              ▼                 ▼              ▼
┌───────────┐  ┌───────────┐  ┌───────────────┐  ┌───────────┐  ┌───────────┐
│ HiCache   │  │ HiCache   │  │ MooncakeStore │  │ HiCache   │  │ Aibrix    │
│ File      │  │ Nixl      │  │               │  │ HF3FS     │  │ KVCache   │
│ (本地文件) │  │ (NVIDIA)  │  │ (分布式存储)   │  │ (3FS)     │  │ (AIBrix)  │
└───────────┘  └───────────┘  └───────────────┘  └───────────┘  └───────────┘
     storage/      storage/      storage/          storage/       storage/
     hicache_      nixl/         mooncake_store/   hf3fs/         aibrix_kvcache/
     storage.py    hicache_      mooncake_store.py storage_hf3fs.py aibrix_kvcache
                   nixl.py                                      _storage.py

================================================================================
数据流转详解
================================================================================

写入路径 (Insert):
─────────────────
1. 新KV缓存写入GPU (L1)
   HiRadixCache.insert() → node.value = GPU_indices

2. 根据写策略决定是否写入CPU (L2)
   - write_through: 立即写入
   - write_through_selective: 命中阈值后写入
   - write_back: 驱逐时写入
   HiRadixCache._inc_hit_count() → write_backup()

3. 可选写入存储后端 (L3)
   HiRadixCache.write_backup_storage() → storage_backend.set()

读取路径 (Match Prefix):
────────────────────────
1. 在基数树中匹配前缀
   HiRadixCache.match_prefix() → _match_prefix_helper()

2. 检查数据位置
   - node.value != None: 数据在GPU，直接返回
   - node.evicted: 数据被驱逐，需要从CPU加载

3. 从CPU加载回GPU
   HiRadixCache.load_back() → cache_controller.load()

4. 可选：从存储预取到CPU
   HiRadixCache.prefetch_from_storage() → storage_backend.get()

驱逐路径 (Evict):
─────────────────
1. 根据驱逐策略选择节点
   HiRadixCache.evict() → 从evictable_leaves中选择

2. 处理被选中的节点
   - 已备份: 只释放GPU内存 (_evict_backuped)
   - 未备份 + write_back: 先写入CPU再驱逐
   - 未备份 + 其他策略: 直接释放 (_evict_regular)

3. 更新树结构
   node.value = None, 保留node.host_value

作者: SGLang Team
版权所有 2023-2024 SGLang Team
许可证: Apache License 2.0
"""

from __future__ import annotations

import atexit
import heapq
import json
import logging
import os
import threading
import time
from queue import Empty
from typing import TYPE_CHECKING, Dict, List, Optional

import torch

# HiCacheController: 管理GPU/CPU/Storage之间的数据移动控制器
from sglang.srt.managers.cache_controller import HiCacheController, PrefetchOperation
from sglang.srt.mem_cache.base_prefix_cache import (
    EvictParams,
    EvictResult,
    InsertParams,
    InsertResult,
    MatchPrefixParams,
    MatchResult,
)
from sglang.srt.mem_cache.memory_pool import (
    MHATokenToKVPool,
    MLATokenToKVPool,
    NSATokenToKVPool,
)
from sglang.srt.mem_cache.memory_pool_host import (
    MHATokenToKVPoolHost,
    MLATokenToKVPoolHost,
    NSATokenToKVPoolHost,
)
from sglang.srt.mem_cache.radix_cache import (
    RadixCache,
    RadixKey,
    TreeNode,
    compute_node_hash_values,
    split_node_hash_value,
)
from sglang.srt.metrics.collector import StorageMetricsCollector
from sglang.srt.utils import bind_to_closest_numa_node_cuda

if TYPE_CHECKING:
    from sglang.srt.mem_cache.cache_init_params import CacheInitParams
    from sglang.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)


class HiRadixCache(RadixCache):
    """
    层次化基数树缓存 (Hierarchical Radix Cache)

    继承自 RadixCache，扩展为三级存储架构: GPU → CPU Host → Storage。

    核心概念：
    ==========
    1. 三级存储层次：
       - L1 (GPU): 热数据，延迟最低 (~微秒级)
       - L2 (CPU Host): 温数据，延迟中等 (~毫秒级)
       - L3 (Storage): 冷数据，持久化存储 (~秒级)

    2. 数据状态：
       - 驻留 (resident): 数据在GPU上
       - 已驱逐 (evicted): 数据被移到CPU，node.value=None
       - 已备份 (backuped): 数据在CPU有备份，node.host_value不为None

    3. 节点生命周期：
       ┌──────────┐     驱逐      ┌──────────┐    驱逐     ┌──────────┐
       │ GPU驻留  │ ─────────────→ │ CPU备份  │ ──────────→ │ Storage  │
       │ (value)  │                │(host_val)│             │(hash_val)│
       └──────────┘                └──────────┘             └──────────┘
            ↑                           ↑                        │
            │        加载回(load_back)  │       预取(prefetch)   │
            └───────────────────────────┴────────────────────────┘

    4. 写策略 (write_policy):
       - write_through: 每次写入GPU时同步写入CPU，保证数据一致性
       - write_back: 延迟写入，仅在驱逐时写入CPU，减少写入开销
       - write_through_selective: 根据节点命中次数选择性写穿透

    5. 预取机制 (prefetch):
       当匹配前缀时，如果发现数据在存储层，可以异步预取到CPU，
       减少后续请求的延迟。

    属性说明：
    ==========
    - token_to_kv_pool_host: CPU内存池，存储被驱逐的KV缓存
    - cache_controller: 缓存控制器，管理GPU/CPU/Storage之间的数据传输
    - ongoing_write_through: 正在进行写穿透操作的节点映射
    - ongoing_load_back: 正在从CPU加载回GPU的节点映射
    - ongoing_prefetch: 正在从存储预取的请求映射
    - ongoing_backup: 正在备份到存储的节点映射
    - evictable_host_leaves: 可以从CPU驱逐的叶子节点集合
    """

    def __init__(self, params: CacheInitParams, server_args: ServerArgs):
        """
        初始化层次化基数树缓存

        Args:
            params: 缓存初始化参数，包含GPU内存池、页面大小等配置
            server_args: 服务器参数，包含hicache相关配置：
                - hicache_io_backend: IO后端类型 (direct/standard)
                - hicache_mem_layout: 内存布局 (page_first/token_first)
                - hicache_ratio: CPU内存与GPU内存的比例
                - hicache_size: 指定的CPU内存大小
                - hicache_write_policy: 写策略
                - hicache_storage_backend: 存储后端类型 (如nixl, mooncake等)
        """
        self._enable_metrics_flag = params.enable_metrics
        if server_args.hicache_io_backend == "direct":
            # FIXME: move this logic into server_args parsing
            if server_args.hicache_mem_layout == "page_first":
                server_args.hicache_mem_layout = "page_first_direct"
                logger.warning(
                    "Page first layout is not supported with direct IO backend, switching to page first direct layout"
                )

        if not server_args.disable_hicache_numa_detect:
            bind_to_closest_numa_node_cuda()

        # 页面大小，决定了KV缓存分配和匹配的基本单位
        self.page_size = params.page_size
        # 获取GPU上的KV缓存池
        self.kv_cache = params.token_to_kv_pool_allocator.get_kvcache()

        # =========================================================================
        # L2层: CPU Host 内存池初始化
        # 根据GPU缓存池的类型创建对应的CPU内存池
        # CPU内存池用于存储被驱逐的KV缓存，实现GPU到CPU的offloading
        # =========================================================================
        if isinstance(self.kv_cache, MHATokenToKVPool):
            # Multi-Head Attention 的 CPU 内存池
            self.token_to_kv_pool_host = MHATokenToKVPoolHost(
                self.kv_cache,
                server_args.hicache_ratio,      # CPU/GPU内存比例
                server_args.hicache_size,       # 指定的CPU内存大小
                self.page_size,
                server_args.hicache_mem_layout, # 内存布局方式
                allocator_type=server_args.hicache_storage_backend,
            )
        elif isinstance(self.kv_cache, NSATokenToKVPool):
            # Native Sparse Attention 的 CPU 内存池
            self.token_to_kv_pool_host = NSATokenToKVPoolHost(
                self.kv_cache,
                server_args.hicache_ratio,
                server_args.hicache_size,
                self.page_size,
                server_args.hicache_mem_layout,
                allocator_type=server_args.hicache_storage_backend,
            )
        elif isinstance(self.kv_cache, MLATokenToKVPool):
            # Multi-Latent Attention 的 CPU 内存池
            self.token_to_kv_pool_host = MLATokenToKVPoolHost(
                self.kv_cache,
                server_args.hicache_ratio,
                server_args.hicache_size,
                self.page_size,
                server_args.hicache_mem_layout,
                allocator_type=server_args.hicache_storage_backend,
            )
        else:
            raise ValueError(f"HiRadixCache only supports MHA and MLA yet")

        # =========================================================================
        # 分布式训练相关配置
        # =========================================================================
        # 张量并行组，用于TP workers之间的同步
        self.tp_group = params.tp_cache_group
        self.tp_world_size = torch.distributed.get_world_size(group=self.tp_group)
        # 流水线并行的rank和size
        self.pp_rank = params.pp_rank
        self.pp_size = params.pp_size

        # =========================================================================
        # L3层: 存储后端配置
        # =========================================================================
        # 是否启用外部存储后端 (如NIXL, Mooncake等)
        self.enable_storage = server_args.hicache_storage_backend is not None
        self.enable_storage_metrics = self.enable_storage and params.enable_metrics
        self.extra_metric_labels = server_args.extra_metric_labels

        (
            extra_config,
            prefetch_threshold,
            prefetch_timeout_base,
            prefetch_timeout_per_ki_token,
            hicache_storage_pass_prefix_keys,
        ) = self._parse_storage_backend_extra_config(
            server_args.hicache_storage_backend_extra_config
        )
        # TODO: support more timeout check functions
        # 预取超时检查函数，使用线性超时策略
        self.is_prefetch_timeout = self._prefetch_timeout_check_linear_func
        # 预取停止策略: best_effort/wait_complete/timeout
        self.prefetch_stop_policy = server_args.hicache_storage_prefetch_policy

        # 用于同步KV缓存加载的事件
        self.load_cache_event = threading.Event()

        # =========================================================================
        # 核心组件: 缓存控制器
        # 管理GPU/CPU/Storage之间的所有数据移动操作
        # =========================================================================
        self.cache_controller = HiCacheController(
            params.token_to_kv_pool_allocator,  # GPU内存分配器
            self.token_to_kv_pool_host,          # CPU内存池
            self.page_size,
            self.tp_group,
            load_cache_event=self.load_cache_event,
            write_policy=server_args.hicache_write_policy,  # 写策略
            io_backend=server_args.hicache_io_backend,      # IO后端
            storage_backend=server_args.hicache_storage_backend,  # 存储后端
            prefetch_threshold=prefetch_threshold,
            model_name=server_args.served_model_name,
            storage_backend_extra_config=extra_config,
            pp_rank=self.pp_rank,
            pp_size=self.pp_size,
        )
        self._apply_storage_runtime_config(
            storage_backend=server_args.hicache_storage_backend,
            prefetch_threshold=prefetch_threshold,
            prefetch_timeout_base=prefetch_timeout_base,
            prefetch_timeout_per_ki_token=prefetch_timeout_per_ki_token,
            hicache_storage_pass_prefix_keys=hicache_storage_pass_prefix_keys,
            enable_storage=self.enable_storage,
            enable_storage_metrics=self.enable_storage_metrics,
            extra_metric_labels=self.extra_metric_labels,
        )

        # =========================================================================
        # 数据传输状态跟踪
        # 这些字典跟踪正在进行中的异步数据传输操作
        # =========================================================================
        # 记录正在进行写穿透(write-through)操作的节点
        # key: node.id, value: TreeNode
        # 用于确保写操作完成后再释放锁
        self.ongoing_write_through = {}

        # 记录正在从CPU加载回GPU的节点段
        # key: node.id, value: TreeNode
        # 用于跟踪异步加载操作，防止重复加载
        self.ongoing_load_back = {}

        # 记录正在进行的预取请求
        # key: request_id, value: (last_host_node, token_ids, host_indices, operation)
        # 预取是从存储层异步加载KV缓存到CPU的过程
        self.ongoing_prefetch = {}

        # 记录正在备份到存储的节点
        # key: operation_id, value: TreeNode
        self.ongoing_backup = {}

        # 跟踪每个请求从存储加载的token数量 (L3命中)
        # key: request_id, value: 实际从存储加载的token数量
        self.prefetch_loaded_tokens_by_reqid: dict[str, int] = {}

        # =========================================================================
        # 阈值配置
        # =========================================================================
        # 写穿透阈值: 节点被访问几次后触发写入CPU
        # write_through策略时阈值为1，write_back策略时阈值为2
        self.write_through_threshold = (
            1 if server_args.hicache_write_policy == "write_through" else 2
        )
        # 加载回阈值: 从CPU加载回GPU的最小token数量
        # 避免小块数据的频繁传输开销
        self.load_back_threshold = 10

        # 进程退出时自动分离存储后端
        atexit.register(self.shutdown)

        # 可从CPU驱逐的叶子节点集合
        # 与evictable_leaves类似，但针对CPU内存
        self.evictable_host_leaves = set()

        # 调用父类RadixCache的初始化
        super().__init__(params=params)

    def shutdown(self):
        """
        进程关闭时自动分离存储后端

        在进程退出时自动调用（通过atexit注册），确保存储后端正确清理。
        这是一个尽力而为的操作，即使失败也不会导致程序崩溃。
        """
        try:
            if self.enable_storage:
                self.detach_storage_backend()
        except Exception:
            logger.exception("Failed to detach storage backend on process shutdown.")

    def _apply_storage_runtime_config(
        self,
        *,
        storage_backend: Optional[str],
        prefetch_threshold: int,
        prefetch_timeout_base: float,
        prefetch_timeout_per_ki_token: float,
        hicache_storage_pass_prefix_keys: bool,
        enable_storage: bool,
        enable_storage_metrics: bool,
        extra_metric_labels: Optional[Dict[str, str]],
    ) -> None:
        prefetch_timeout_per_page = (
            self.page_size / 1024 * prefetch_timeout_per_ki_token
        )

        self.enable_storage = enable_storage
        self.prefetch_threshold = prefetch_threshold
        self.prefetch_timeout_base = prefetch_timeout_base
        self.prefetch_timeout_per_page = prefetch_timeout_per_page
        self.hicache_storage_pass_prefix_keys = hicache_storage_pass_prefix_keys
        self.enable_storage_metrics = enable_storage_metrics

        if self.enable_storage_metrics:
            labels = {
                "storage_backend": storage_backend,
                "tp_rank": self.cache_controller.tp_rank,
                "dp_rank": self.cache_controller.dp_rank,
                "pp_rank": self.cache_controller.pp_rank,
                "pp_size": self.cache_controller.pp_size,
            }
            if extra_metric_labels:
                labels.update(extra_metric_labels)
            existing_collector = getattr(self, "storage_metrics_collector", None)
            if existing_collector is None:
                self.storage_metrics_collector = StorageMetricsCollector(labels=labels)
            elif set(existing_collector.labels.keys()) == set(labels.keys()):
                existing_collector.labels = labels
            else:
                logger.warning(
                    "Storage metrics labels changed (%s -> %s). Keep existing labels to "
                    "avoid duplicate metric registration.",
                    sorted(existing_collector.labels.keys()),
                    sorted(labels.keys()),
                )

    def attach_storage_backend(
        self,
        storage_backend: str,
        storage_backend_extra_config_json: Optional[str] = None,
        served_model_name: Optional[str] = None,
        hicache_storage_prefetch_policy: Optional[str] = None,
        hicache_write_policy: Optional[str] = None,
    ) -> tuple[bool, str]:
        """
        运行时动态附加（启用）存储后端

        这将启动HiCacheController内部的存储线程，并启用预取/备份路径。
        调用者必须确保没有正在运行/排队的请求以避免竞争条件。

        Args:
            storage_backend: 存储后端类型（如 'nixl', 'mooncake' 等）
            storage_backend_extra_config_json: 存储后端额外配置的JSON字符串
            served_model_name: 服务的模型名称
            hicache_storage_prefetch_policy: 预取停止策略
                - 'best_effort': 尽力而为，可随时终止
                - 'wait_complete': 等待完成
                - 'timeout': 超时后终止
            hicache_write_policy: 写策略
                - 'write_back': 驱逐时写入
                - 'write_through': 同步写入
                - 'write_through_selective': 选择性写入

        Returns:
            tuple[bool, str]: (是否成功, 消息)
        """
        # Validate inputs first (no side effects).
        if hicache_storage_prefetch_policy is not None:
            allowed = ["best_effort", "wait_complete", "timeout"]
            if hicache_storage_prefetch_policy not in allowed:
                return (
                    False,
                    f"Invalid hicache_storage_prefetch_policy: {hicache_storage_prefetch_policy!r}. "
                    f"Expected one of {allowed}.",
                )

        if hicache_write_policy is not None:
            allowed = ["write_back", "write_through", "write_through_selective"]
            if hicache_write_policy not in allowed:
                return (
                    False,
                    f"Invalid hicache_write_policy: {hicache_write_policy!r}. "
                    f"Expected one of {allowed}.",
                )

        # If already enabled:
        # - backend unchanged: treat as success, update policies only.
        # - backend changed: treat as failure, do NOT update policies.
        if self.enable_storage:
            current_backend = self.cache_controller.storage_backend_type

            if current_backend == storage_backend:
                if hicache_storage_prefetch_policy is not None:
                    self.prefetch_stop_policy = hicache_storage_prefetch_policy
                    logger.info(
                        f"Set hicache_storage_prefetch_policy to {hicache_storage_prefetch_policy}"
                    )
                if hicache_write_policy is not None:
                    self.cache_controller.write_policy = hicache_write_policy
                    self.write_through_threshold = (
                        1 if hicache_write_policy == "write_through" else 2
                    )
                    logger.info(f"Set hicache_write_policy to {hicache_write_policy}")
                return (
                    True,
                    "HiCache storage backend already enabled with same backend; policies updated.",
                )

            return (
                False,
                f"HiCache storage backend is already enabled with backend '{current_backend}'. "
                f"Cannot attach different backend '{storage_backend}'. Detach first.",
            )

        # Not enabled: update policies before controller attach so storage threads observe new values.
        if hicache_storage_prefetch_policy is not None:
            self.prefetch_stop_policy = hicache_storage_prefetch_policy
            logger.info(
                f"Set hicache_storage_prefetch_policy to {hicache_storage_prefetch_policy}"
            )

        if hicache_write_policy is not None:
            self.cache_controller.write_policy = hicache_write_policy
            self.write_through_threshold = (
                1 if hicache_write_policy == "write_through" else 2
            )
            logger.info(f"Set hicache_write_policy to {hicache_write_policy}")

        logger.info(f"Attaching HiCache storage backend: {storage_backend}")
        try:
            (
                extra_config,
                prefetch_threshold,
                prefetch_timeout_base,
                prefetch_timeout_per_ki_token,
                hicache_storage_pass_prefix_keys,
            ) = self._parse_storage_backend_extra_config(
                storage_backend_extra_config_json
            )
        except Exception as e:
            logger.exception(f"Failed to parse storage_backend_extra_config_json: {e}")
            return (
                False,
                f"Failed to parse storage_backend_extra_config_json '{storage_backend_extra_config_json}': {e}",
            )

        try:
            self.cache_controller.attach_storage_backend(
                storage_backend=storage_backend,
                prefetch_threshold=prefetch_threshold,
                model_name=served_model_name,
                storage_backend_extra_config=extra_config,
            )
        except Exception as e:
            logger.exception(
                f"Failed to attach storage backend '{storage_backend}': {e}"
            )
            return False, f"Failed to attach storage backend '{storage_backend}': {e}"

        self._apply_storage_runtime_config(
            storage_backend=storage_backend,
            prefetch_threshold=prefetch_threshold,
            prefetch_timeout_base=prefetch_timeout_base,
            prefetch_timeout_per_ki_token=prefetch_timeout_per_ki_token,
            hicache_storage_pass_prefix_keys=hicache_storage_pass_prefix_keys,
            enable_storage=True,
            enable_storage_metrics=self._enable_metrics_flag,
            extra_metric_labels=self.extra_metric_labels,
        )
        return True, "Attached HiCache storage backend successfully."

    def detach_storage_backend(self) -> tuple[bool, str]:
        """
        运行时动态分离（禁用）存储后端

        调用者必须确保没有正在运行/排队的请求以避免竞争条件。

        分离过程：
        1. 排空存储控制队列
        2. 停止存储线程
        3. 强制释放所有待处理的操作
        4. 更新状态标志

        Returns:
            tuple[bool, str]: (是否成功, 消息)
        """
        try:
            # Drain any pending control queues before tearing down storage threads/backend.
            # IMPORTANT: this must happen before we clear `ongoing_*`, otherwise acks/releases
            # cannot be matched to nodes and may leak host pages / locks.
            self._drain_storage_control_queues_local()
            # Idempotent detach: always ask controller to best-effort cleanup, even if
            # `self.enable_storage` is already False (may be leftover state from a
            # previous partial detach).
            self.cache_controller.detach_storage_backend()
        except Exception as e:
            logger.exception("Failed to detach storage backend.")
            # Do NOT crash the server for admin operations. Return failure with detail.
            return False, f"Failed to detach HiCache storage backend: {e}"

        # Best-effort cleanup of any leftover bookkeeping.
        self._drain_storage_control_queues_local()
        # After controller threads are fully stopped, it's safe to force-release any
        # leftover pending ops (e.g., async prefetch/backup that didn't get a revoke/ack).
        self._force_release_pending_storage_ops()

        self.enable_storage = False
        self.enable_storage_metrics = False
        return True, "Detached HiCache storage backend successfully."

    def _force_release_pending_storage_ops(self):
        """
        强制释放所有待处理的预取/备份操作

        这是detach/shutdown路径的安全网。假设存储线程已经停止，
        所以不会有对这些结构的并发访问。

        处理流程：
        1. 释放所有待处理预取操作的主机页和锁
        2. 释放所有待处理备份操作的节点保护
        """
        cc = self.cache_controller

        # Force release leftover prefetch ops: free pre-allocated host pages and
        # drop the host protection on the matched prefix node.
        try:
            for req_id, info in list(self.ongoing_prefetch.items()):
                try:
                    last_host_node, token_ids, host_indices, _operation = info
                except Exception:
                    # Unexpected shape; just drop it.
                    self.ongoing_prefetch.pop(req_id, None)
                    continue

                try:
                    if host_indices is not None:
                        cc.mem_pool_host.free(host_indices)
                except Exception:
                    logger.exception(
                        "Failed to free host indices for prefetch %s", req_id
                    )

                try:
                    last_host_node.release_host()
                except Exception:
                    logger.exception(
                        "Failed to release host protection for prefetch %s", req_id
                    )

                try:
                    cc.prefetch_tokens_occupied -= len(token_ids)
                    if cc.prefetch_tokens_occupied < 0:
                        cc.prefetch_tokens_occupied = 0
                except Exception:
                    pass

                self.ongoing_prefetch.pop(req_id, None)
        except Exception:
            logger.exception("Force release pending prefetch ops failed.")

        # Force release leftover backup ops: drop host protection on nodes.
        try:
            for ack_id, node in list(self.ongoing_backup.items()):
                try:
                    node.release_host()
                except Exception:
                    logger.exception(
                        "Failed to release host protection for backup op %s", ack_id
                    )
                self.ongoing_backup.pop(ack_id, None)
        except Exception:
            logger.exception("Force release pending backup ops failed.")

    def _drain_storage_control_queues_local(self):
        """Drain storage control queues without TP synchronization.

        This is intended for shutdown/detach paths where we want to make best-effort
        cleanup even if queue sizes temporarily differ across ranks.
        """
        self._drain_storage_control_queues_impl(
            n_revoke=None,
            n_backup=None,
            n_release=None,
            log_metrics=False,
        )

    def _drain_storage_control_queues_impl(
        self,
        n_revoke: Optional[int],
        n_backup: Optional[int],
        n_release: Optional[int],
        log_metrics: bool,
    ):
        cc = self.cache_controller

        def _drain_queue(q, limit: Optional[int]):
            drained = 0
            while limit is None or drained < limit:
                try:
                    item = q.get_nowait()
                except Empty:
                    break
                drained += 1
                yield item

        def _drain_revoke():
            for req_id in _drain_queue(cc.prefetch_revoke_queue, n_revoke):
                info = self.ongoing_prefetch.pop(req_id, None)
                if info is not None:
                    last_host_node, token_ids, _, _ = info
                    last_host_node.release_host()
                    cc.prefetch_tokens_occupied -= len(token_ids)
                    if cc.prefetch_tokens_occupied < 0:
                        cc.prefetch_tokens_occupied = 0

        def _drain_backup():
            for operation in _drain_queue(cc.ack_backup_queue, n_backup):
                ack_id = operation.id
                entry = self.ongoing_backup.pop(ack_id, None)
                if entry is not None:
                    entry.release_host()
                if log_metrics and self.enable_storage_metrics:
                    self.storage_metrics_collector.log_backuped_tokens(
                        operation.completed_tokens
                    )

        def _drain_release():
            host_indices_list = []
            for host_indices in _drain_queue(cc.host_mem_release_queue, n_release):
                host_indices_list.append(host_indices)
            if host_indices_list:
                host_indices = torch.cat(host_indices_list, dim=0)
                cc.mem_pool_host.free(host_indices)

        _drain_revoke()
        _drain_backup()
        _drain_release()

    def _parse_storage_backend_extra_config(
        self, storage_backend_extra_config: Optional[str]
    ):
        """
        Parse storage backend extra config JSON and extract specific parameters.

        Args:
            storage_backend_extra_config: JSON string containing extra configuration

        Returns:
            tuple: (extra_config_dict, prefetch_threshold, prefetch_timeout_base, prefetch_timeout_per_ki_token, hicache_storage_pass_prefix_keys)
        """
        # Parse extra config if provided. Extra config can be a JSON string or a json/toml/yaml file path prefixed with "@".
        extra_config = {}
        if storage_backend_extra_config:
            try:
                if storage_backend_extra_config.startswith("@"):
                    # Read config from a json/toml/yaml file
                    path = storage_backend_extra_config[1:]
                    ext = os.path.splitext(path)[1].lower()
                    with open(path, "rb" if ext == ".toml" else "r") as f:
                        if ext == ".json":
                            extra_config = json.load(f)
                        elif ext == ".toml":
                            import tomllib

                            extra_config = tomllib.load(f)
                        elif ext in (".yaml", ".yml"):
                            import yaml

                            extra_config = yaml.safe_load(f)
                        else:
                            raise ValueError(
                                f"Unsupported config file {path} (config format: {ext})"
                            )
                else:
                    # read config from JSON string
                    extra_config = json.loads(storage_backend_extra_config)
            except Exception as e:
                logger.error(f"Invalid backend extra config JSON: {e}")
                raise e

        prefetch_threshold = extra_config.pop("prefetch_threshold", 256)  # tokens
        prefetch_timeout_base = extra_config.pop("prefetch_timeout_base", 1)  # seconds
        prefetch_timeout_per_ki_token = extra_config.pop(
            "prefetch_timeout_per_ki_token", 0.25
        )  # seconds per 1024 tokens
        hicache_storage_pass_prefix_keys = extra_config.pop(
            "hicache_storage_pass_prefix_keys", False
        )

        if not isinstance(prefetch_threshold, int):
            raise ValueError(
                f"prefetch_threshold must be int, got {type(prefetch_threshold).__name__}"
            )
        if not isinstance(prefetch_timeout_base, (int, float)):
            raise ValueError(
                f"prefetch_timeout_base must be number, got {type(prefetch_timeout_base).__name__}"
            )
        if not isinstance(prefetch_timeout_per_ki_token, (int, float)):
            raise ValueError(
                f"prefetch_timeout_per_ki_token must be number, got {type(prefetch_timeout_per_ki_token).__name__}"
            )
        if not isinstance(hicache_storage_pass_prefix_keys, bool):
            raise ValueError(
                "hicache_storage_pass_prefix_keys must be bool, got "
                f"{type(hicache_storage_pass_prefix_keys).__name__}"
            )

        return (
            extra_config,
            prefetch_threshold,
            float(prefetch_timeout_base),
            float(prefetch_timeout_per_ki_token),
            hicache_storage_pass_prefix_keys,
        )

    def reset(self):
        TreeNode.counter = 0
        self.cache_controller.reset()
        self.token_to_kv_pool_host.clear()
        # Clear per-request tracking dicts
        self.prefetch_loaded_tokens_by_reqid.clear()
        self.evictable_host_leaves.clear()
        super().reset()

    def get_height(self, node: TreeNode):
        height = 0
        while node != self.root_node:
            node = node.parent
            height += 1
        return height

    def clear_storage_backend(self) -> bool:
        if self.enable_storage:
            try:
                # Check if the storage backend has a clear method (for nixl backends)
                if hasattr(self.cache_controller.storage_backend, "clear"):
                    self.cache_controller.storage_backend.clear()
                    logger.info(
                        "Hierarchical cache storage backend cleared successfully!"
                    )
                    return True
                else:
                    logger.warning(
                        f"Storage backend {type(self.cache_controller.storage_backend).__name__} does not support clear operation."
                    )
                    return False
            except Exception as e:
                logger.error(f"Failed to clear hierarchical cache storage backend: {e}")
                return False
        else:
            logger.warning("Hierarchical cache storage backend is not enabled.")
            return False

    def write_backup(self, node: TreeNode, write_back=False):
        """
        将GPU上的KV缓存备份到CPU内存

        这是L1→L2的数据传输过程。当GPU内存不足或节点命中次数达到阈值时调用。

        工作流程：
        1. 在CPU内存池中分配空间
        2. 如果CPU内存不足，先驱逐CPU中的旧数据
        3. 将GPU数据拷贝到CPU
        4. 记录到ongoing_write_through以跟踪异步操作

        Args:
            node: 要备份的树节点
            write_back: 是否为write_back策略（驱逐时的备份）

        Returns:
            备份的token数量，如果失败返回0
        """
        host_indices = self.cache_controller.write(
            device_indices=node.value,
            node_id=node.id,
        )
        if host_indices is None:
            # CPU内存不足，先驱逐一些旧数据
            self.evict_host(len(node.value))
            host_indices = self.cache_controller.write(
                device_indices=node.value,
                node_id=node.id,
            )
        if host_indices is not None:
            node.host_value = host_indices
            assert len(node.host_value) > 0
            self.ongoing_write_through[node.id] = node
            if not write_back:
                # write_through策略需要增加锁引用计数保护节点
                self.inc_lock_ref(node)
        else:
            return 0

        return len(host_indices)

    def write_backup_storage(self, node: TreeNode):
        """
        将CPU内存中的KV缓存备份到外部存储后端 (L2→L3)

        这是L2→L3的数据传输过程。当启用storage backend时，
        可以将CPU数据持久化到外部存储系统，实现跨会话共享。

        支持的存储后端：
        - file: 本地文件系统存储
        - nixl: NVIDIA NIXL存储
        - mooncake: Mooncake分布式存储
        - hf3fs: 3FS高性能存储
        - aibrix: AIBrix KV Cache存储
        - eic: EIC存储

        Args:
            node: 要备份到存储的树节点，必须有host_value
        """
        prefix_keys = (
            node.get_prefix_hash_values(node.parent)
            if self.hicache_storage_pass_prefix_keys
            else None
        )

        operation_id = self.cache_controller.write_storage(
            node.host_value, node.key, node.hash_value, prefix_keys
        )
        self.ongoing_backup[operation_id] = node
        node.protect_host()

    def _inc_hit_count(self, node: TreeNode, chunked=False):
        """
        增加节点命中计数，并在达到阈值时触发写穿透

        这是write_through_selective策略的核心逻辑：
        - 只有当节点被访问足够多次时才写入CPU
        - 避免冷数据占用CPU内存
        - 写回策略(write_back)跳过此逻辑

        Args:
            node: 被访问的树节点
            chunked: 是否为分块请求（分块请求不更新计数）
        """
        # skip the hit count update for chunked requests
        if self.cache_controller.write_policy == "write_back" or chunked:
            return
        node.hit_count += 1

        if not node.backuped:
            if node.hit_count >= self.write_through_threshold:
                # write to host if the node is not backuped
                self.write_backup(node)

    def writing_check(self, write_back=False):
        """
        检查并处理正在进行的写操作

        这个方法协调GPU→CPU的异步数据传输：
        1. 检查CUDA事件是否完成
        2. 在TP workers之间同步完成状态
        3. 更新节点状态并触发存储备份

        对于write_back策略：
        - 阻塞等待所有写操作完成
        - 确保数据安全后再进行驱逐

        对于write_through策略：
        - 非阻塞检查已完成的事件
        - 释放锁引用计数
        - 触发存储备份

        Args:
            write_back: 是否为write_back策略的阻塞检查
        """
        if write_back:
            # blocking till all write back complete
            while len(self.ongoing_write_through) > 0:
                for _, finish_event, ack_list in self.cache_controller.ack_write_queue:
                    finish_event.synchronize()
                    for ack_id in ack_list:
                        backuped_node = self.ongoing_write_through.pop(ack_id)
                        if self.enable_storage:
                            self.write_backup_storage(backuped_node)
                self.cache_controller.ack_write_queue.clear()
                assert len(self.ongoing_write_through) == 0
            return

        # NOTE: all ranks has the same ongoing_write_through, can skip sync if empty
        if len(self.ongoing_write_through) == 0:
            return

        finish_count = 0
        for _, finish_event, ack_list in self.cache_controller.ack_write_queue:
            if not finish_event.query():
                break
            finish_count += 1
        queue_size = torch.tensor(finish_count, dtype=torch.int, device="cpu")
        if self.tp_world_size > 1:
            # synchronize TP workers to make the same update to radix cache
            torch.distributed.all_reduce(
                queue_size,
                op=torch.distributed.ReduceOp.MIN,
                group=self.tp_group,
            )

        finish_count = int(queue_size.item())
        while finish_count > 0:
            _, finish_event, ack_list = self.cache_controller.ack_write_queue.pop(0)
            finish_event.synchronize()
            for ack_id in ack_list:
                backuped_node = self.ongoing_write_through.pop(ack_id)
                self.dec_lock_ref(backuped_node)
                if self.enable_storage:
                    self.write_backup_storage(backuped_node)
            finish_count -= 1

    def loading_check(self):
        finish_count = 0
        for _, finish_event, ack_list in self.cache_controller.ack_load_queue:
            if not finish_event.query():
                # the KV cache loading is still ongoing
                break
            finish_count += 1
            # no need to sync across TP workers as batch forwarding is synced
            for ack_id in ack_list:
                end_node = self.ongoing_load_back.pop(ack_id)
                self.dec_lock_ref(end_node)

        # ACK until all events are processed
        del self.cache_controller.ack_load_queue[:finish_count]

    def evictable_size(self):
        return self.evictable_size_

    def inc_lock_ref(self, node: TreeNode):
        if self.disable:
            return 0

        delta = 0
        while node != self.root_node:
            if node.lock_ref == 0:
                self.evictable_size_ -= len(node.key)
                self.protected_size_ += len(node.key)
                delta -= len(node.key)
            node.lock_ref += 1
            self._update_leaf_status(node)
            self._update_host_leaf_status(node)
            node = node.parent
        return delta

    def dec_lock_ref(self, node: TreeNode):
        if self.disable:
            return 0

        delta = 0
        while node != self.root_node:
            if node.lock_ref == 1:
                self.evictable_size_ += len(node.key)
                self.protected_size_ -= len(node.key)
                delta += len(node.key)
            node.lock_ref -= 1
            self._update_leaf_status(node)
            self._update_host_leaf_status(node)
            if node.parent is None:
                assert (
                    node is self.root_node
                ), f"This request holds the node from another tree"
            node = node.parent
        return delta

    def _update_host_leaf_status(self, node: TreeNode):
        if not node.evicted or node.lock_ref > 0:
            if node in self.evictable_host_leaves:
                self.evictable_host_leaves.remove(node)
            return

        for child in node.children.values():
            if child.evicted:
                if node in self.evictable_host_leaves:
                    self.evictable_host_leaves.remove(node)
                return

        if node not in self.evictable_host_leaves:
            self.evictable_host_leaves.add(node)

    def evict(self, params: EvictParams) -> EvictResult:
        """
        驱逐GPU上的KV缓存

        这是层次化缓存的核心驱逐逻辑，与RadixCache的主要区别：
        1. 支持将被驱逐数据写入CPU (write_back策略)
        2. 已备份的节点只释放GPU内存，保留CPU备份
        3. 驱逐后检查写入完成状态

        驱逐优先级由eviction_strategy决定：
        - LRU: 最近最少使用
        - LFU: 最不经常使用
        - FIFO: 先进先出
        - MRU: 最近最常使用
        - FILO: 后进先出
        - Priority: 优先级

        Args:
            params: 驱逐参数，包含要驱逐的token数量

        Returns:
            EvictResult: 驱逐结果，包含实际驱逐的token数量
        """
        start_time = time.perf_counter()
        num_tokens = params.num_tokens
        leaves = list(self.evictable_leaves)
        eviction_heap = [
            (self.eviction_strategy.get_priority(node), node) for node in leaves
        ]
        heapq.heapify(eviction_heap)

        num_evicted = 0
        write_back_nodes = []
        while num_evicted < num_tokens and len(eviction_heap):
            _priority, x = heapq.heappop(eviction_heap)

            if x.lock_ref > 0:
                continue

            if not x.backuped:
                if self.cache_controller.write_policy == "write_back":
                    # write to host if the node is not backuped
                    num_evicted += self.write_backup(x, write_back=True)
                    write_back_nodes.append(x)
                else:
                    num_evicted += self._evict_regular(x)
            else:
                num_evicted += self._evict_backuped(x)

            for child in x.parent.children.values():
                if child in write_back_nodes:
                    continue
                if not child.evicted:
                    break
            else:
                # all children are evicted or no children
                new_priority = self.eviction_strategy.get_priority(x.parent)
                heapq.heappush(eviction_heap, (new_priority, x.parent))

        if self.cache_controller.write_policy == "write_back":
            self.writing_check(write_back=True)
            for node in write_back_nodes:
                assert node.backuped
                self._evict_backuped(node)

        self.update_eviction_metrics(num_evicted, start_time)
        return EvictResult(num_tokens_evicted=num_evicted)

    def _evict_backuped(self, node: TreeNode):
        """
        驱逐已备份到CPU的节点 (只释放GPU内存)

        这是L1→L2驱逐的完成阶段。节点已经在CPU有备份，
        只需要释放GPU上的内存，CPU数据保留以备后续加载回。

        处理流程：
        1. 释放GPU内存 (value置None)
        2. 更新evictable_size统计
        3. 更新节点和父节点的叶子状态

        Args:
            node: 已备份的树节点

        Returns:
            释放的token数量
        """
        # evict a node already written to host
        num_evicted = self.cache_controller.evict_device(node.value)
        assert num_evicted > 0
        self.evictable_size_ -= num_evicted
        node.value = None  # GPU值置空，但host_value保留
        self._update_leaf_status(node)
        self._update_host_leaf_status(node)
        # update leaf status for the parent because the node is evicted
        self._update_leaf_status(node.parent)
        return num_evicted

    def _evict_regular(self, node: TreeNode):
        # evict a node not initiated write to host
        self.cache_controller.mem_pool_device_allocator.free(node.value)
        num_evicted = len(node.value)
        self._delete_leaf(node)
        return num_evicted

    def evict_host(self, num_tokens: int):
        leaves = list(self.evictable_host_leaves)
        eviction_heap = [
            (self.eviction_strategy.get_priority(node), node) for node in leaves
        ]
        heapq.heapify(eviction_heap)

        num_evicted = 0
        while num_evicted < num_tokens and len(eviction_heap):
            _priority, x = heapq.heappop(eviction_heap)
            if x == self.root_node:
                break
            # only evict the host value of evicted nodes
            if not x.evicted:
                continue

            # node is protected from eviction as it has ongoing prefetch or backup to storage
            if x.host_ref_counter > 0:
                continue

            num_evicted += self.cache_controller.evict_host(x.host_value)

            key = self.get_child_key_fn(x.key)
            v = x.parent.children.pop(key, None)
            assert v == x, f"parent does not have child key, {key}"
            if x in self.evictable_host_leaves:
                self.evictable_host_leaves.remove(x)
            self._update_host_leaf_status(x.parent)

            if len(x.parent.children) == 0 and x.parent.evicted:
                new_priority = self.eviction_strategy.get_priority(x.parent)
                heapq.heappush(eviction_heap, (new_priority, x.parent))

    def load_back(
        self, node: TreeNode, mem_quota: Optional[int] = None
    ) -> Optional[torch.Tensor]:
        """
        从CPU内存加载KV缓存回GPU (L2→L1)

        当匹配前缀时发现节点已被驱逐到CPU，调用此方法将数据加载回GPU。
        这是CPU到GPU的反向数据传输。

        工作流程：
        1. 从被驱逐节点向上遍历，收集所有需要加载的节点
        2. 检查是否满足加载条件（大小阈值、内存配额）
        3. 分配GPU内存并执行异步传输
        4. 更新节点状态和锁引用计数

        Args:
            node: 起始节点（通常是被驱逐的叶子节点）
            mem_quota: 可用的GPU内存配额

        Returns:
            加载的GPU索引张量，如果加载失败返回None
        """

        start_time = time.perf_counter()
        last_hit_node = node
        nodes_to_load = []
        while node.evicted:
            assert (
                node.backuped
            ), "No backup available on evicted nodes, should not happen"
            nodes_to_load.insert(0, node)
            node = node.parent
        else:
            ancester_node = node

        # protect the ancestor nodes from eviction
        delta = self.inc_lock_ref(ancester_node)

        # load it all or not at all
        host_indices = torch.cat([n.host_value for n in nodes_to_load])
        if len(host_indices) < self.load_back_threshold or (
            len(host_indices) > mem_quota + delta if mem_quota is not None else False
        ):
            # skip loading back if the total size is too small or exceeding the memory quota
            self.dec_lock_ref(ancester_node)
            return None

        device_indices = self.cache_controller.load(
            host_indices=host_indices, node_id=last_hit_node.id
        )
        if device_indices is None:
            self.evict(EvictParams(num_tokens=len(host_indices)))
            device_indices = self.cache_controller.load(
                host_indices=host_indices, node_id=last_hit_node.id
            )
        self.dec_lock_ref(ancester_node)
        if device_indices is None:
            # no sufficient GPU memory to load back KV caches
            return None

        self.ongoing_load_back[last_hit_node.id] = last_hit_node
        offset = 0
        for node in nodes_to_load:
            node.value = device_indices[offset : offset + len(node.host_value)]
            offset += len(node.host_value)
        self.evictable_size_ += len(device_indices)
        self.inc_lock_ref(last_hit_node)

        if self.metrics_collector is not None:
            self.metrics_collector.observe_load_back_duration(
                time.perf_counter() - start_time
            )
            self.metrics_collector.increment_load_back_num_tokens(len(device_indices))

        return device_indices

    def init_load_back(
        self,
        last_node: TreeNode,
        host_hit_length: int,
        mem_quota: Optional[int] = None,
    ):
        _ = host_hit_length  # unused, but kept for compatibility
        if last_node.evicted:
            loading_values = self.load_back(last_node, mem_quota)
            if loading_values is not None:
                logger.debug(
                    f"loading back {len(loading_values)} tokens for node {last_node.id}"
                )
                return loading_values, last_node

            while last_node.evicted:
                last_node = last_node.parent

        return (
            torch.empty((0,), dtype=torch.int64, device=self.device),
            last_node,
        )

    def ready_to_load_host_cache(self) -> int:
        """
        Notify the cache controller to start the KV cache loading.
        Return the consumer index for the schedule batch manager to track.
        """
        return self.cache_controller.start_loading()

    def check_hicache_events(self):
        self.writing_check()
        self.loading_check()
        if self.enable_storage:
            self.drain_storage_control_queues()
        if self.enable_storage_metrics:
            self.storage_metrics_collector.log_storage_metrics(
                self.cache_controller.storage_backend.get_stats()
            )

    def drain_storage_control_queues(self):
        """
        Combine prefetch revoke, backup ack, and host mem release checks
        to minimize TP synchronization and Python overhead.
        """
        cc = self.cache_controller

        qsizes = torch.tensor(
            [
                cc.prefetch_revoke_queue.qsize(),
                cc.ack_backup_queue.qsize(),
                cc.host_mem_release_queue.qsize(),
            ],
            dtype=torch.int,
        )
        if self.tp_world_size > 1:
            torch.distributed.all_reduce(
                qsizes, op=torch.distributed.ReduceOp.MIN, group=self.tp_group
            )

        n_revoke, n_backup, n_release = map(int, qsizes.tolist())
        self._drain_storage_control_queues_impl(
            n_revoke=n_revoke,
            n_backup=n_backup,
            n_release=n_release,
            log_metrics=True,
        )

    # Timeout is linearly increasing with the number of pages
    def _prefetch_timeout_check_linear_func(self, operation: PrefetchOperation):
        # If hash_value has not been computed in timeout_base seconds, terminate it.
        return (
            time.monotonic() - operation.start_time
            > self.prefetch_timeout_base
            + len(operation.hash_value) * self.prefetch_timeout_per_page
        )

    def can_terminate_prefetch(self, operation: PrefetchOperation):
        can_terminate = True

        if self.prefetch_stop_policy == "best_effort":
            return can_terminate

        if len(operation.hash_value) == 0:
            completed = False
        else:
            completed = (
                operation.completed_tokens == len(operation.hash_value) * self.page_size
            )

        if self.prefetch_stop_policy == "wait_complete":
            can_terminate = completed
        elif self.prefetch_stop_policy == "timeout":
            can_terminate = completed or self.is_prefetch_timeout(operation)
        else:
            # unknown prefetch stop policy, just return True
            return True

        operation_terminated = operation.is_terminated()
        if self.tp_world_size > 1:
            states = torch.tensor(
                [1 - int(can_terminate), int(operation_terminated)],
                dtype=torch.int,
            )
            torch.distributed.all_reduce(
                states,
                op=torch.distributed.ReduceOp.MAX,
                group=self.tp_group,
            )
            can_terminate = states[0].item() == 0
            operation_terminated = states[1].item() == 1
        # the operation should be terminated if it is already terminated on any TP worker
        # or it meets the termination condition on all TP workers
        can_terminate = can_terminate or operation_terminated
        return can_terminate

    def check_prefetch_progress(self, req_id: str) -> bool:
        if req_id not in self.ongoing_prefetch:
            # there is no ongoing prefetch for this request or it has been revoked
            return True

        # todo: more policies for prefetch progress such as timeout
        # the current policy is to prefetch with best effort and terminate when queuing is over
        last_host_node, token_ids, host_indices, operation = self.ongoing_prefetch[
            req_id
        ]

        if operation.host_indices is None:
            # prefetch has not been issued due to insufficient host memory
            return True

        if not self.can_terminate_prefetch(operation):
            return False

        completed_tokens, hash_value = self.cache_controller.terminate_prefetch(
            operation
        )
        logger.debug(f"Prefetch {req_id} completed with {completed_tokens} tokens")

        min_completed_tokens = completed_tokens
        if self.tp_world_size > 1:
            # synchrnoize TP workers to make the same update to hiradix cache
            completed_tokens_tensor = torch.tensor(
                min_completed_tokens, dtype=torch.int
            )
            torch.distributed.all_reduce(
                completed_tokens_tensor,
                op=torch.distributed.ReduceOp.MIN,
                group=self.tp_group,
            )
            min_completed_tokens = completed_tokens_tensor.item()
        fetched_token_ids = token_ids[:min_completed_tokens]
        written_indices = host_indices[:min_completed_tokens]
        matched_length = self._insert_helper_host(
            last_host_node,
            RadixKey(
                token_ids=fetched_token_ids, extra_key=last_host_node.key.extra_key
            ),
            written_indices,
            hash_value[: min_completed_tokens // self.page_size],
        )

        self.cache_controller.mem_pool_host.free(host_indices[:matched_length])
        self.cache_controller.append_host_mem_release(
            host_indices[min_completed_tokens:completed_tokens]
        )
        last_host_node.release_host()
        del self.ongoing_prefetch[req_id]
        self.cache_controller.prefetch_tokens_occupied -= len(token_ids)

        # Track tokens actually loaded from storage for this request (L3 hits)
        loaded_from_storage = min_completed_tokens - matched_length
        self.prefetch_loaded_tokens_by_reqid[req_id] = loaded_from_storage

        if self.enable_storage_metrics:
            self.storage_metrics_collector.log_prefetched_tokens(loaded_from_storage)

        return True

    def terminate_prefetch(self, req_id: str):
        if req_id not in self.ongoing_prefetch:
            return

        _, _, _, operation = self.ongoing_prefetch[req_id]
        if operation.host_indices is None:
            return
        operation.mark_terminate()

    def pop_prefetch_loaded_tokens(self, req_id: str) -> int:
        """
        Pop and return the number of tokens loaded from storage for a request.
        Returns 0 if no prefetch was done or was revoked.
        This should be called after check_prefetch_progress() returns True.
        """
        return self.prefetch_loaded_tokens_by_reqid.pop(req_id, 0)

    def match_prefix(self, params: MatchPrefixParams):
        key = params.key
        empty_value = torch.empty((0,), dtype=torch.int64, device=self.device)
        key, _ = self.maybe_bigram_convert(key)
        if self.disable or len(key) == 0:
            return MatchResult(
                device_indices=empty_value,
                last_device_node=self.root_node,
                last_host_node=self.root_node,
                host_hit_length=0,
            )

        if self.page_size != 1:
            page_aligned_len = len(key) // self.page_size * self.page_size
            key = key[:page_aligned_len]

        value, last_node = self._match_prefix_helper(self.root_node, key)
        if value:
            value = torch.cat(value)
        else:
            value = empty_value

        host_hit_length = 0
        last_host_node = last_node
        while last_node.evicted:
            host_hit_length += len(last_node.host_value)
            last_node = last_node.parent
        while not last_host_node.backuped:
            last_host_node = last_host_node.parent

        return MatchResult(
            device_indices=value,
            last_device_node=last_node,
            last_host_node=last_host_node,
            host_hit_length=host_hit_length,
        )

    def prefetch_from_storage(
        self,
        req_id: str,
        last_host_node: TreeNode,
        new_input_tokens: List[int],
        last_hash: Optional[str] = None,
        prefix_keys: Optional[List[str]] = None,
    ):
        # align the number of fetching tokens to the page size
        prefetch_length = len(new_input_tokens) - (
            len(new_input_tokens) % self.page_size
        )
        new_input_tokens = new_input_tokens[:prefetch_length]
        if (
            not self.enable_storage
            or prefetch_length < self.prefetch_threshold
            or self.cache_controller.prefetch_rate_limited()
        ):
            return

        last_host_node.protect_host()
        host_indices = self.cache_controller.mem_pool_host.alloc(prefetch_length)
        if host_indices is None:
            self.evict_host(prefetch_length)
            host_indices = self.cache_controller.mem_pool_host.alloc(prefetch_length)
        if host_indices is None:
            last_host_node.release_host()
            # no sufficient host memory for prefetch
            return
        operation = self.cache_controller.prefetch(
            req_id, host_indices, new_input_tokens, last_hash, prefix_keys
        )
        self.ongoing_prefetch[req_id] = (
            last_host_node,
            new_input_tokens,
            host_indices,
            operation,
        )
        self.cache_controller.prefetch_tokens_occupied += len(new_input_tokens)

    def _insert_helper_host(
        self, node: TreeNode, key: RadixKey, host_value, hash_value
    ):
        node.last_access_time = time.monotonic()
        if len(key) == 0:
            return 0

        child_key = self.get_child_key_fn(key)

        matched_length = 0
        while len(key) > 0 and child_key in node.children.keys():
            node = node.children[child_key]
            node.last_access_time = time.monotonic()
            prefix_len = self.key_match_fn(node.key, key)
            key = key[prefix_len:]
            host_value = host_value[prefix_len:]
            hash_value = hash_value[prefix_len // self.page_size :]
            matched_length += prefix_len

            if prefix_len < len(node.key):
                new_node = self._split_node(node.key, node, prefix_len)
                node = new_node

            if len(key):
                child_key = self.get_child_key_fn(key)

        if len(key):
            new_node = TreeNode(priority=node.priority)
            new_node.parent = node
            new_node.key = key
            new_node.value = None
            new_node.host_value = host_value.clone()
            new_node.hash_value = hash_value
            node.children[child_key] = new_node
            self._update_host_leaf_status(new_node)
            self._update_leaf_status(node)
            self._update_host_leaf_status(node)

        return matched_length

    def _match_prefix_helper(self, node: TreeNode, key: RadixKey):
        node.last_access_time = time.monotonic()
        child_key = self.get_child_key_fn(key)
        value = []

        while len(key) > 0 and child_key in node.children.keys():
            child = node.children[child_key]
            child.last_access_time = time.monotonic()
            prefix_len = self.key_match_fn(child.key, key)
            if prefix_len < len(child.key):
                new_node = self._split_node(child.key, child, prefix_len)
                if not new_node.evicted:
                    value.append(new_node.value)
                node = new_node
                break
            else:
                if not child.evicted:
                    value.append(child.value)
                node = child
                key = key[prefix_len:]

                if len(key):
                    child_key = self.get_child_key_fn(key)

        return value, node

    def _split_node(self, key: RadixKey, child: TreeNode, split_len: int):
        # child node split into new_node -> child
        new_node = TreeNode(priority=child.priority)
        new_node.children = {self.get_child_key_fn(key[split_len:]): child}
        new_node.parent = child.parent
        new_node.lock_ref = child.lock_ref
        new_node.key = child.key[:split_len]
        new_node.hit_count = child.hit_count

        # split value and host value if exists
        if child.evicted:
            new_node.value = None
        else:
            new_node.value = child.value[:split_len].clone()
            child.value = child.value[split_len:].clone()
        if child.backuped:
            new_node.host_value = child.host_value[:split_len].clone()
            child.host_value = child.host_value[split_len:].clone()

        new_node.hash_value, child.hash_value = split_node_hash_value(
            child.hash_value, split_len, self.page_size
        )
        child.parent = new_node
        child.key = child.key[split_len:]
        new_node.parent.children[self.get_child_key_fn(key)] = new_node
        return new_node

    def insert(self, params: InsertParams) -> InsertResult:
        key = params.key
        value = params.value
        chunked = params.chunked
        priority = params.priority

        if priority is None:
            priority = 0
        key, value = self.maybe_bigram_convert(key, value)

        if len(key) == 0:
            return InsertResult(prefix_len=0)

        if self.is_eagle and value is not None:
            # Make sure the value len equal to the EAGLE bigram key len
            value = value[: len(key)]

        node = self.root_node
        child_key = self.get_child_key_fn(key)
        total_prefix_length = 0

        while len(key) > 0 and child_key in node.children.keys():
            node = node.children[child_key]
            node.last_access_time = time.monotonic()
            node.priority = max(node.priority, priority)
            prefix_len = self.key_match_fn(node.key, key)

            if prefix_len == len(node.key):
                if node.evicted:
                    # change the reference if the node is evicted
                    # this often happens in the case of KV cache recomputation
                    node.value = value[:prefix_len]
                    self.evictable_size_ += len(node.value)
                    self._update_leaf_status(node)
                    self._update_host_leaf_status(node)
                    # update parent status as a new leaf is added into device
                    self._update_leaf_status(node.parent)
                else:
                    self._inc_hit_count(node, chunked)
                    total_prefix_length += prefix_len
            else:
                # partial match, split the node
                new_node = self._split_node(node.key, node, prefix_len)
                # shared-prefix node should also reflect max priority
                new_node.priority = max(new_node.priority, priority)
                if new_node.evicted:
                    new_node.value = value[:prefix_len].clone()
                    self.evictable_size_ += len(new_node.value)
                    self._update_leaf_status(new_node)
                    self._update_host_leaf_status(new_node)
                    # update parent status as a new leaf is added into device
                    self._update_leaf_status(new_node.parent)
                else:
                    self._inc_hit_count(new_node, chunked)
                    total_prefix_length += prefix_len
                node = new_node

            key = key[prefix_len:]
            value = value[prefix_len:]

            if len(key):
                child_key = self.get_child_key_fn(key)

        if len(key):
            new_node = TreeNode(priority=priority)
            new_node.parent = node
            new_node.key = key
            new_node.value = value.clone()
            node.children[child_key] = new_node
            self.evictable_size_ += len(value)
            self._update_leaf_status(node)
            self._update_leaf_status(new_node)

            # Compute hash_value if storage is enabled
            if self.enable_storage:
                new_node.hash_value = compute_node_hash_values(new_node, self.page_size)

            if self.cache_controller.write_policy != "write_back":
                self._inc_hit_count(new_node, chunked)
        return InsertResult(prefix_len=total_prefix_length)

    def release_aborted_request(self, rid: str):
        # Clean up storage hit tracking for aborted request
        self.prefetch_loaded_tokens_by_reqid.pop(rid, None)

        if rid not in self.ongoing_prefetch:
            return

        last_host_node, token_ids, host_indices, operation = self.ongoing_prefetch[rid]
        if operation.host_indices is None:
            return

        completed_tokens, _ = self.cache_controller.terminate_prefetch(operation)
        if self.tp_world_size > 1:
            torch.distributed.barrier(group=self.tp_group)
        last_host_node.release_host()
        del self.ongoing_prefetch[rid]
        self.cache_controller.append_host_mem_release(host_indices[:completed_tokens])
        self.cache_controller.prefetch_tokens_occupied -= len(token_ids)
