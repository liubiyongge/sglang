"""
radix_cache.py - 基数树(Radix Tree)缓存实现

本模块实现了基于基数树(Radix Tree/Patricia Tree)的KV缓存管理数据结构。
基数树是一种压缩的前缀树，用于高效地存储和检索共享前缀的token序列，
这对于大语言模型的推理优化至关重要，因为多个请求通常共享相同的system prompt或前缀。

主要组件：
1. RadixKey - 树节点的键，包含token ID序列和可选的额外命名空间键
2. TreeNode - 树节点，存储KV缓存的索引和元数据
3. RadixCache - 主缓存类，提供前缀匹配、插入、驱逐等功能

核心优势：
- 前缀共享：多个请求共享相同前缀时，只需存储一份KV缓存
- 高效匹配：O(k)时间复杂度匹配最长前缀，k为token数量
- 内存高效：自动合并共享前缀，减少内存占用
- 支持多种驱逐策略：LRU、LFU、FIFO、MRU、FILO、Priority

作者: SGLang Team
版权所有 2023-2024 SGLang Team
许可证: Apache License 2.0
"""

from __future__ import annotations

from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.utils import convert_to_bigram_key

"""原有版权和许可证声明"""
"""
Copyright 2023-2024 SGLang Team
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

"""
The radix tree data structure for managing the KV cache.
基数树数据结构，用于管理KV缓存。
"""

# 标准库导入
import heapq  # 堆队列算法，用于实现优先队列驱逐
import logging
import sys
import time
from collections import defaultdict  # 默认字典，用于构建子节点映射
from functools import lru_cache, partial  # LRU缓存装饰器和偏函数
from typing import TYPE_CHECKING, Any, Iterator, List, Optional, Tuple, Union

import torch  # PyTorch张量库

# 配置模块日志记录器
logger = logging.getLogger(__name__)

# 导入KV缓存事件类型，用于分布式场景下的缓存事件通知
from sglang.srt.disaggregation.kv_events import (
    MEDIUM_GPU,  # GPU存储介质标识
    AllBlocksCleared,  # 所有块清除事件
    BlockRemoved,  # 块移除事件
    BlockStored,  # 块存储事件
)

# 导入前缀缓存基类和相关数据类型
from sglang.srt.mem_cache.base_prefix_cache import (
    BasePrefixCache,  # 前缀缓存抽象基类
    EvictParams,  # 驱逐参数
    EvictResult,  # 驱逐结果
    InsertParams,  # 插入参数
    InsertResult,  # 插入结果
    MatchPrefixParams,  # 前缀匹配参数
    MatchResult,  # 匹配结果
)

# 导入驱逐策略实现
from sglang.srt.mem_cache.evict_policy import (
    EvictionStrategy,  # 驱逐策略基类
    FIFOStrategy,  # 先进先出策略
    FILOStrategy,  # 后进先出策略
    LFUStrategy,  # 最不经常使用策略
    LRUStrategy,  # 最近最少使用策略
    MRUStrategy,  # 最近最常使用策略
    PriorityStrategy,  # 优先级策略
)

# 导入哈希计算工具
from sglang.srt.mem_cache.hicache_storage import get_hash_str, hash_str_to_int64

# 类型检查时导入Req类型，避免运行时循环依赖
if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req


class RadixKey:
    """
    基数树键(RadixKey) - 表示树节点的键

    基数树中的每个节点都由一个RadixKey标识。RadixKey包含：
    - token_ids: token ID序列，是该节点存储的实际数据
    - extra_key: 可选的额外命名空间键，用于区分不同上下文的缓存
      （例如不同的LoRA适配器、不同的缓存版本等）
    - is_bigram: 是否为bigram键（用于EAGLE推测解码）

    设计理念：
    通过extra_key实现命名空间隔离，具有相同token序列但不同extra_key的
    条目会被存储在不同的子树中，互不干扰。这对于多租户场景和不同模型
    配置的缓存隔离非常重要。
    """

    def __init__(
        self,
        token_ids: List[int],
        extra_key: Optional[str] = None,
        is_bigram: bool = False,
    ):
        """
        初始化RadixKey

        Args:
            token_ids: token ID列表，表示该键对应的token序列
            extra_key: 可选的额外键，用于命名空间隔离（如LoRA ID、缓存盐值等）
            is_bigram: 是否为bigram模式的键，用于EAGLE推测解码优化
        """
        # token id序列 - 该节点代表的token列表
        self.token_ids = token_ids
        # 额外键 - 用于命名空间隔离，确保不同上下文的缓存互不干扰
        self.extra_key = extra_key
        # 是否为bigram键 - EAGLE推测解码使用bigram模式
        self.is_bigram = is_bigram

    def __len__(self) -> int:
        """返回token序列的长度"""
        return len(self.token_ids)

    def __iter__(self) -> Iterator[int]:
        """支持迭代，可以遍历token_ids"""
        return iter(self.token_ids)

    def __getitem__(self, idx: Union[int, slice]) -> "RadixKey":
        """
        支持索引和切片操作

        Args:
            idx: 整数索引或切片对象

        Returns:
            新的RadixKey对象，包含切片后的token_ids
        """
        if isinstance(idx, slice):
            return RadixKey(self.token_ids[idx], self.extra_key)
        return RadixKey([self.token_ids[idx]], self.extra_key)

    def __repr__(self) -> str:
        """返回可读的字符串表示，只显示前10个token避免输出过长"""
        preview = self.token_ids[:10]
        return f"RadixKey(extra_key={self.extra_key!r}, token_ids={preview}{'...' if len(self.token_ids) > 10 else ''})"


class TreeNode:
    """
    树节点(TreeNode) - 基数树的基本组成单元

    每个TreeNode代表基数树中的一个节点，存储：
    - key: 该节点对应的token序列（RadixKey）
    - value: KV缓存在GPU内存中的索引位置
    - children: 子节点映射（按token或token块索引）
    - 元数据：锁引用计数、访问时间、优先级等

    树结构说明：
    - 根节点：特殊的空节点，key为空列表
    - 叶节点：没有非evicted子节点的节点
    - 中间节点：既有父节点又有子节点的节点

    内存管理：
    - lock_ref: 锁引用计数，>0表示节点正在被使用，不能被驱逐
    - host_value: 主机（CPU）内存中的KV缓存备份
    - host_ref_counter: 主机值的引用计数保护

    驱逐策略支持：
    - last_access_time: 最后访问时间，用于LRU策略
    - hit_count: 命中次数，用于LFU策略
    - priority: 优先级，用于优先级驱逐策略
    """

    # 类级别的计数器，用于生成唯一节点ID
    counter = 0

    def __init__(self, id: Optional[int] = None, priority: int = 0):
        """
        初始化树节点

        Args:
            id: 可选的节点ID，如果不提供则自动生成
            priority: 节点的初始优先级（用于优先级驱逐策略）
        """
        # 子节点映射：key -> TreeNode
        # 使用defaultdict自动创建TreeNode，但通常我们手动设置
        self.children = defaultdict(TreeNode)

        # 父节点引用，构建树的向上遍历路径
        self.parent: TreeNode = None

        # 该节点存储的token序列键
        self.key: RadixKey = None

        # GPU上的KV缓存索引张量
        # 形状为 [len(key)] 的int64张量，每个元素指向token_to_kv_pool中的位置
        self.value: Optional[torch.Tensor] = None

        # 锁引用计数：
        # - >0: 节点正在被一个或多个请求使用，不能被驱逐
        # - =0: 节点未被使用，可以被驱逐（如果是叶节点）
        self.lock_ref = 0

        # 最后访问时间（单调时钟），用于LRU驱逐策略
        self.last_access_time = time.monotonic()

        # 节点创建时间
        self.creation_time = time.monotonic()

        # 命中计数，用于LFU驱逐策略
        self.hit_count = 0

        # 主机引用计数器：保护主机值不被驱逐
        # 在存储操作引用节点时递增
        self.host_ref_counter = 0

        # 主机值：存储在CPU内存中的KV缓存备份
        # 用于KV缓存卸载(offloading)到CPU内存的场景
        self.host_value: Optional[torch.Tensor] = None

        # 哈希值列表：每个页面一个SHA256哈希
        # 用于分布式场景下的缓存一致性验证
        self.hash_value: Optional[List[str]] = None

        # 优先级：用于优先级感知的驱逐策略
        # 较高的优先级意味着节点更不容易被驱逐
        self.priority = priority

        # 分配唯一ID
        self.id = TreeNode.counter if id is None else id
        TreeNode.counter += 1

    @property
    def evicted(self) -> bool:
        """
        检查节点是否已被驱逐

        Returns:
            True如果节点的value为None（已被驱逐），False否则
        """
        return self.value is None

    @property
    def backuped(self) -> bool:
        """
        检查节点是否有备份（在CPU内存中）

        Returns:
            True如果节点有host_value备份，False否则
        """
        return self.host_value is not None

    def protect_host(self):
        """
        保护主机值不被驱逐

        通过递增host_ref_counter来标记主机值正在被使用，
        防止在存储操作期间被意外释放。
        """
        self.host_ref_counter += 1

    def release_host(self):
        """
        释放主机值的保护

        递减host_ref_counter，当计数归零时，主机值可以被驱逐。
        如果计数已经是0，则抛出运行时错误。
        """
        if self.host_ref_counter > 0:
            self.host_ref_counter -= 1
        else:
            raise RuntimeError("Host reference counter is already zero.")

    def get_last_hash_value(self) -> Optional[str]:
        """
        获取该节点最后一个页面的哈希值

        Returns:
            最后一个页面的哈希字符串，如果没有则返回None
        """
        if self.hash_value is None or len(self.hash_value) == 0:
            return None
        return self.hash_value[-1]

    @lru_cache(maxsize=1)
    def get_prefix_hash_values(self, node: TreeNode) -> List[str]:
        """
        获取从根节点到当前节点的完整哈希值路径

        Args:
            node: 当前节点

        Returns:
            从根到当前节点所有页面的哈希值列表
        """
        if node is None or node.hash_value is None:
            return []

        # 递归获取父节点的哈希值，然后拼接当前节点的哈希值
        return node.get_prefix_hash_values(node.parent) + node.hash_value

    def __lt__(self, other: "TreeNode"):
        """
        小于比较运算符，用于堆排序

        基于last_access_time比较，较早访问的节点"较小"
        这在LRU驱逐策略中用于构建最小堆
        """
        return self.last_access_time < other.last_access_time


def _check_extra_key(key0: RadixKey, key1: RadixKey):
    """
    检查两个键的extra_key是否一致

    用于确保前缀匹配操作只在相同的命名空间内进行。
    具有不同extra_key的键永远不应该匹配彼此的前缀。

    Args:
        key0: 第一个键
        key1: 第二个键

    Raises:
        ValueError: 如果两个键的extra_key不同
    """
    if key0.extra_key != key1.extra_key:
        raise ValueError(
            f"_key_match should be run on the same extra key, but got key0.extra_key={key0.extra_key} != key1.extra_key={key1.extra_key}"
        )


def _key_match_page_size1(key0: RadixKey, key1: RadixKey):
    """
    逐token匹配键（page_size=1的情况）

    找出两个键从开头开始的最长公共前缀长度。
    当page_size=1时，每个token是一个独立的页面。

    Args:
        key0: 第一个键
        key1: 第二个键

    Returns:
        公共前缀的token数量
    """
    _check_extra_key(key0, key1)
    i = 0
    for k0, k1 in zip(key0.token_ids, key1.token_ids):
        if k0 != k1:
            break
        i += 1
    return i


def _key_match_paged(key0: RadixKey, key1: RadixKey, page_size: int):
    """
    按页面匹配键（page_size>1的情况）

    找出两个键从开头开始的最长公共前缀长度，但匹配必须以页面为单位。
    如果某个页面的token不完全匹配，则整个页面不算匹配。

    例如：page_size=4时，前4个token必须全部匹配才算第一页匹配。

    Args:
        key0: 第一个键
        key1: 第二个键
        page_size: 页面大小（每个页面包含的token数量）

    Returns:
        公共前缀的token数量（总是page_size的倍数）
    """
    _check_extra_key(key0, key1)
    min_len = min(len(key0), len(key1))

    i = 0
    while i < min_len:
        # 每次比较一个完整的页面
        if key0.token_ids[i : i + page_size] != key1.token_ids[i : i + page_size]:
            break
        i += page_size

    return i


def get_child_key(key: RadixKey, page_size: int = 1):
    """
    获取用于在children字典中索引子节点的键

    根据page_size的不同，生成不同类型的子节点索引键：
    - page_size=1: 使用单个token ID作为键
    - page_size>1: 使用token块元组作为键

    如果存在extra_key，则返回(extra_key, plain_key)元组，
    这样不同命名空间的节点会被正确隔离。

    Args:
        key: RadixKey对象
        page_size: 页面大小

    Returns:
        子节点字典中的键
    """
    if page_size == 1:
        # 单token模式：使用第一个token ID
        plain_key = key.token_ids[0]
    else:
        # 分页模式：使用前page_size个token组成的元组
        plain_key = tuple(key.token_ids[:page_size])
    if key.extra_key is None:
        return plain_key
    else:
        # 包含命名空间键
        return (key.extra_key, plain_key)


def compute_node_hash_values(node: "TreeNode", page_size: int) -> List[str]:
    """
    计算节点的哈希值列表（用于分布式缓存一致性）

    使用SHA256链式哈希为节点的每个页面计算位置感知的哈希值。
    每个页面的哈希值依赖于：
    1. 该页面的token内容
    2. 前一个页面的哈希值（形成哈希链）

    这种链式哈希确保了相同token在不同位置会产生不同的哈希值，
    这对于分布式KV缓存的正确性至关重要。

    Args:
        node: 要计算哈希的树节点
        page_size: 页面大小

    Returns:
        每个页面的SHA256哈希字符串列表
    """
    hash_values = []

    # 获取父节点的最后一个哈希值作为起始哈希
    parent_hash = None
    if node.parent is not None and node.parent.hash_value is not None:
        # 检查父节点是否为根节点（根节点key长度为0）
        if len(node.parent.key) > 0 and len(node.parent.hash_value) > 0:
            parent_hash = node.parent.hash_value[-1]

    # 遍历节点的每个页面
    for start in range(0, len(node.key), page_size):
        page_tokens = node.key.token_ids[start : start + page_size]
        if not page_tokens:
            continue

        # 使用SHA256链式哈希
        hash_val = get_hash_str(page_tokens, prior_hash=parent_hash)
        hash_values.append(hash_val)
        # 当前页面的哈希成为下一个页面的"prior_hash"
        parent_hash = hash_val

    return hash_values


def split_node_hash_value(
    child_hash_value: Optional[List[str]], split_len: int, page_size: int
) -> tuple[Optional[List[str]], Optional[List[str]]]:
    """
    在节点分裂时分割哈希值列表

    当一个树节点需要分裂时（匹配过程只匹配了节点部分token），
    其哈希值也需要相应地分割给新创建的父节点和更新后的子节点。

    Args:
        child_hash_value: 原子节点的哈希值列表
        split_len: 分裂位置的token数量
        page_size: 页面大小

    Returns:
        元组：(新父节点的哈希值列表, 更新后子节点的哈希值列表)
    """
    if child_hash_value is None:
        return None, None

    # 计算分割点对应的页面数量
    if page_size == 1:
        split_pages = split_len
    else:
        split_pages = split_len // page_size

    # 分割哈希值列表
    new_node_hash = child_hash_value[:split_pages]
    child_hash = child_hash_value[split_pages:]

    return new_node_hash, child_hash


class RadixCache(BasePrefixCache):
    def __init__(self, params: CacheInitParams):
        self.disable = params.disable
        self.req_to_token_pool = params.req_to_token_pool
        self.token_to_kv_pool_allocator = params.token_to_kv_pool_allocator
        self.page_size = params.page_size
        self.enable_kv_cache_events = params.enable_kv_cache_events
        self.is_eagle = params.is_eagle
        self.disable_finished_insert = params.disable_finished_insert
        self.eviction_policy = params.eviction_policy.lower()

        self.kv_event_queue = []

        if params.enable_metrics:
            self.init_metrics_collector()

        if self.token_to_kv_pool_allocator:
            self.device = self.token_to_kv_pool_allocator.device
        else:
            self.device = torch.device("cpu")

        if self.page_size == 1:
            self.key_match_fn = _key_match_page_size1
            self.get_child_key_fn = get_child_key
        else:
            self.key_match_fn = partial(_key_match_paged, page_size=self.page_size)
            self.get_child_key_fn = partial(get_child_key, page_size=self.page_size)

        if self.eviction_policy == "lru":
            self.eviction_strategy: EvictionStrategy = LRUStrategy()
        elif self.eviction_policy == "lfu":
            self.eviction_strategy: EvictionStrategy = LFUStrategy()
        elif self.eviction_policy == "fifo":
            self.eviction_strategy: EvictionStrategy = FIFOStrategy()
        elif self.eviction_policy == "mru":
            self.eviction_strategy: EvictionStrategy = MRUStrategy()
        elif self.eviction_policy == "filo":
            self.eviction_strategy: EvictionStrategy = FILOStrategy()
        elif self.eviction_policy == "priority":
            self.eviction_strategy: EvictionStrategy = PriorityStrategy()
        else:
            raise ValueError(
                f"Unknown eviction policy: {self.eviction_policy}. Supported policies: 'lru', 'lfu', 'fifo', 'mru', 'filo', 'priority'."
            )

        self.evictable_leaves = set()
        self.reset()

    @classmethod
    def create_simulated(
        self,
        disable: bool = False,
        mock_allocator: Optional[Any] = None,
        page_size: int = 1,
        enable_kv_cache_events: bool = False,
    ) -> RadixCache:
        """Init a radix cache without memory pools for simulation purpose."""
        params = CacheInitParams(
            disable=disable,
            req_to_token_pool=None,
            token_to_kv_pool_allocator=mock_allocator,
            page_size=page_size,
            enable_kv_cache_events=enable_kv_cache_events,
        )
        return RadixCache(params)

    ##### Public API #####

    def reset(self):
        # Initialize root with minimum priority so any real priority overrides it
        self.root_node = TreeNode(priority=-sys.maxsize)
        self.root_node.key = RadixKey(token_ids=[], extra_key=None)
        self.root_node.value = []
        self.root_node.host_value = []
        self.root_node.lock_ref = 1
        self.root_node.hash_value = []
        self.evictable_size_ = 0
        self.protected_size_ = 0
        self.evictable_leaves.clear()
        self._record_all_cleared_event()

    def maybe_bigram_convert(
        self, key: RadixKey, value: Optional[torch.Tensor] = None
    ) -> Tuple[RadixKey, Optional[torch.Tensor]]:
        if self.is_eagle and not key.is_bigram:
            key.token_ids = convert_to_bigram_key(key.token_ids)
            if value is not None:
                value = value[: len(key)]

        return key, value

    def match_prefix(self, params: MatchPrefixParams) -> MatchResult:
        """Find the longest cached prefix of ``key`` in the radix tree.

        The logical namespace for prefix matching is determined by both the
        token id sequence and the optional ``extra_key`` carried by ``RadixKey``.
        Entries that share identical leading token ids but have *different*
        ``extra_key`` values are intentionally kept disjoint and never share
        prefix nodes. This is useful to:

        * Isolate KV cache lines for different LoRA / adapter IDs.
        * Separate requests that intentionally should not share state (e.g.,
          different sampling salt, cache version, or retrieval augmentation
          context) by supplying a distinct ``extra_key``.

        Args:
            params (MatchPrefixParams): Parameters containing the lookup key
                with a list of token ids and an optional ``extra_key`` namespace tag.
                If ``page_size > 1`` the length is internally truncated to a multiple
                of ``page_size`` before matching. Passing an empty key returns an
                empty result with the root as the last node.

        Returns:
            MatchResult: ``device_indices`` is a 1-D ``torch.int64`` tensor of
            the concatenated KV cache indices corresponding to the longest
            cached prefix (may be length 0). ``last_device_node`` and
            ``last_host_node`` (currently the same) are the tree node objects
            representing the terminal node of the matched prefix. This method
            may mutate internal structure by splitting an existing node if the
            match ends inside a stored segment.

        Internal updates:
            * Refreshes access metadata (timestamps) used by the
                configured eviction strategy.
            * If the lookup ends inside a stored segment the node is split once
                to expose a precise boundary; this structural refinement improves
                subsequent match efficiency and does not duplicate data.
        """
        key = params.key
        key, _ = self.maybe_bigram_convert(key)

        def empty_match_result():
            return MatchResult(
                device_indices=torch.empty(
                    (0,),
                    dtype=torch.int64,
                    device=self.device,
                ),
                last_device_node=self.root_node,
                last_host_node=self.root_node,
            )

        if self.disable or len(key) == 0:
            return empty_match_result()

        if self.page_size != 1:
            page_aligned_len = len(key) // self.page_size * self.page_size
            key = key[:page_aligned_len]

        if len(key) == 0:
            return empty_match_result()

        value, last_node = self._match_prefix_helper(self.root_node, key)
        if value:
            value = torch.cat(value)
        else:
            value = torch.empty((0,), dtype=torch.int64, device=self.device)
        return MatchResult(
            device_indices=value,
            last_device_node=last_node,
            last_host_node=last_node,
        )

    def insert(self, params: InsertParams) -> InsertResult:
        if self.disable:
            return InsertResult(prefix_len=0)

        key = params.key
        value = params.value
        priority = params.priority

        if value is None:
            value = torch.tensor(key.token_ids, dtype=torch.int64)

        key, value = self.maybe_bigram_convert(key, value)

        prefix_len = self._insert_helper(self.root_node, key, value, priority)
        return InsertResult(prefix_len=prefix_len)

    def _page_align_keys(self, key: list) -> list:
        if self.page_size == 1:
            return key
        page_aligned_len = len(key) // self.page_size * self.page_size
        return key[:page_aligned_len]

    def cache_finished_req(self, req: Req, is_insert: bool = True):
        """Cache request when it finishes."""
        # In deterministic mode, disable finished request insertion to radix cache
        if self.disable_finished_insert:
            is_insert = False

        kv_committed_len = req.pop_committed_kv_cache()
        if self.disable:
            kv_indices = self.req_to_token_pool.req_to_token[
                req.req_pool_idx, :kv_committed_len
            ]
            self.token_to_kv_pool_allocator.free(kv_indices)
            return

        token_ids = (req.origin_input_ids + req.output_ids)[:kv_committed_len]
        kv_indices = self.req_to_token_pool.req_to_token[
            req.req_pool_idx, : len(token_ids)
        ]

        # Maybe convert to bigram keys for EAGLE
        keys = convert_to_bigram_key(token_ids) if self.is_eagle else token_ids
        keys = self._page_align_keys(keys)
        values = kv_indices[: len(keys)].to(dtype=torch.int64, copy=True)
        radix_key = RadixKey(keys, req.extra_key, is_bigram=self.is_eagle)

        # Radix Cache takes one ref in memory pool
        if is_insert:
            priority = getattr(req, "priority", 0) or 0
            result = self.insert(
                InsertParams(key=radix_key, value=values, priority=priority)
            )
            new_prefix_len = result.prefix_len
            # Free the duplicates that were already in the tree
            self.token_to_kv_pool_allocator.free(
                kv_indices[req.cache_protected_len : new_prefix_len]
            )
        else:
            self.token_to_kv_pool_allocator.free(
                kv_indices[req.cache_protected_len : len(keys)]
            )

        # free the unaligned tail
        self.token_to_kv_pool_allocator.free(kv_indices[len(keys) :])

        # Remove req slot release the cache lock
        self.dec_lock_ref(req.last_node)

    def cache_unfinished_req(self, req: Req, chunked=False):
        """Cache request when it is unfinished."""
        if self.disable:
            return

        token_ids = req.fill_ids
        kv_indices = self.req_to_token_pool.req_to_token[
            req.req_pool_idx, : len(token_ids)
        ]

        # Maybe convert to bigram keys for EAGLE
        keys = convert_to_bigram_key(token_ids) if self.is_eagle else token_ids
        keys = self._page_align_keys(keys)
        values = kv_indices[: len(keys)].to(dtype=torch.int64, copy=True)
        radix_key = RadixKey(keys, req.extra_key, is_bigram=self.is_eagle)

        # Radix Cache takes one ref in memory pool
        result = self.insert(
            InsertParams(
                key=radix_key,
                value=values,
                chunked=chunked,
                priority=getattr(req, "priority", 0) or 0,
            )
        )
        new_prefix_len = result.prefix_len

        self.token_to_kv_pool_allocator.free(
            kv_indices[req.cache_protected_len : new_prefix_len]
        )

        # The prefix indices could be updated, reuse it
        match_result = self.match_prefix(MatchPrefixParams(key=radix_key))
        (new_indices, new_last_node) = (
            match_result.device_indices,
            match_result.last_device_node,
        )
        assert len(new_indices) == len(keys), f"{len(new_indices)=}, {len(keys)=}"

        self.req_to_token_pool.write(
            (req.req_pool_idx, slice(req.cache_protected_len, len(new_indices))),
            new_indices[req.cache_protected_len :],
        )

        # The cache_protected_len is not always equal to len(req.prefix_indices)
        # since for page_size > 1, the partial part is added to req.prefix_indices, but that part of kv indices is not added to the tree.
        # It should be freed in the next cache_unfinished_req and final cache_finished_req to avoid memory leak.
        # So we introduce this `cache_protected_len` field to make sure the partial part can be freed correctly.
        req.cache_protected_len = len(new_indices)

        self.dec_lock_ref(req.last_node)
        self.inc_lock_ref(new_last_node)

        # `req.prefix_indices` will be used in `PrefillAdder::add_chunked_req` later
        # - page_size != 1: there is a partial page at the end, keep the full kv_indices
        # - eagle case: bigram keys will only cache len - 1 kv indices
        if len(new_indices) < len(kv_indices):
            req.prefix_indices = torch.cat(
                [new_indices, kv_indices[len(new_indices) :]]
            )
        else:
            req.prefix_indices = new_indices

        req.last_node = new_last_node

    def pretty_print(self):
        self._print_helper(self.root_node, 0)
        print(f"#tokens: {self.total_size()}")

    def total_size(self):
        return self._total_size_helper()

    def evict(self, params: EvictParams) -> EvictResult:
        if self.disable:
            return EvictResult()

        start_time = time.perf_counter()
        num_tokens = params.num_tokens
        leaves = list(self.evictable_leaves)
        eviction_heap = [
            (self.eviction_strategy.get_priority(node), node) for node in leaves
        ]
        heapq.heapify(eviction_heap)

        num_evicted = 0
        while num_evicted < num_tokens and len(eviction_heap):
            _priority, x = heapq.heappop(eviction_heap)

            self.token_to_kv_pool_allocator.free(x.value)
            num_evicted += len(x.value)
            self._delete_leaf(x)

            if len(x.parent.children) == 0 and x.parent.lock_ref == 0:
                new_priority = self.eviction_strategy.get_priority(x.parent)
                heapq.heappush(eviction_heap, (new_priority, x.parent))

            self._record_remove_event(x)

        self.update_eviction_metrics(num_evicted, start_time)
        return EvictResult(num_tokens_evicted=num_evicted)

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
            if node.parent is None:
                assert (
                    node is self.root_node
                ), f"This request holds the node from another tree"
            node = node.parent
        return delta

    def evictable_size(self):
        return self.evictable_size_

    def protected_size(self):
        # protected size refers to the size of the cache that is locked
        return self.protected_size_

    def all_values_flatten(self):
        values = []

        def _dfs_helper(node: TreeNode):
            for _, child in node.children.items():
                values.append(child.value)
                _dfs_helper(child)

        _dfs_helper(self.root_node)
        return torch.cat(values)

    ##### Internal Helper Functions #####

    def _match_prefix_helper(self, node: TreeNode, key: RadixKey):
        access_time = time.monotonic()
        node.last_access_time = access_time

        child_key = self.get_child_key_fn(key)

        value = []
        while len(key) > 0 and child_key in node.children.keys():
            child = node.children[child_key]
            child.last_access_time = access_time
            prefix_len = self.key_match_fn(child.key, key)
            if prefix_len < len(child.key):
                new_node = self._split_node(child.key, child, prefix_len)
                value.append(new_node.value)
                node = new_node
                break
            else:
                value.append(child.value)
                node = child
                key = key[prefix_len:]

                if len(key):
                    child_key = self.get_child_key_fn(key)

        return value, node

    def _split_node(self, key: RadixKey, child: TreeNode, split_len: int):
        # new_node -> child
        # New node inherits child's priority (represents shared prefix)
        new_node = TreeNode(priority=child.priority)
        new_node.children = {self.get_child_key_fn(key[split_len:]): child}
        new_node.parent = child.parent
        new_node.lock_ref = child.lock_ref
        new_node.key = child.key[:split_len]
        new_node.value = child.value[:split_len].clone()
        child.parent = new_node
        child.key = child.key[split_len:]
        child.value = child.value[split_len:].clone()
        new_node.parent.children[self.get_child_key_fn(key)] = new_node

        # Split hash_value if it was already computed, otherwise leave as None
        new_node.hash_value, child.hash_value = split_node_hash_value(
            child.hash_value, split_len, self.page_size
        )

        return new_node

    def _insert_helper(self, node: TreeNode, key: RadixKey, value, priority: int = 0):
        # Convert None priority to 0
        if priority is None:
            priority = 0
        access_time = time.monotonic()
        node.last_access_time = access_time
        # Update priority along the path (take max to propagate higher priority)
        node.priority = max(node.priority, priority)
        if len(key) == 0:
            return 0

        child_key = self.get_child_key_fn(key)

        total_prefix_length = 0
        while len(key) > 0 and child_key in node.children.keys():
            node = node.children[child_key]
            node.last_access_time = access_time
            prefix_len = self.key_match_fn(node.key, key)
            total_prefix_length += prefix_len
            key = key[prefix_len:]
            value = value[prefix_len:]

            if prefix_len < len(node.key):
                new_node = self._split_node(node.key, node, prefix_len)
                new_node.priority = max(new_node.priority, priority)
                node = new_node
            else:
                node.priority = max(node.priority, priority)

            if len(key):
                child_key = self.get_child_key_fn(key)

        if len(key):
            new_node = TreeNode(priority=priority)
            new_node.parent = node
            new_node.key = key
            new_node.value = value.clone()
            node.children[child_key] = new_node
            self.evictable_size_ += len(key)
            self._update_leaf_status(node)
            self._update_leaf_status(new_node)
            # Hash will be computed lazily during event emission
            self._record_store_event(new_node)
        return total_prefix_length

    def _print_helper(self, node: TreeNode, indent: int):
        """Prints the radix tree in a human-readable format."""
        stack = [(node, indent)]
        while stack:
            current_node, current_indent = stack.pop()
            print(
                " " * current_indent,
                len(current_node.key),
                current_node.key.token_ids[:10],
                f"r={current_node.lock_ref}",
            )
            for key, child in current_node.children.items():
                stack.append((child, current_indent + 2))

                assert key == self.get_child_key_fn(
                    child.key
                ), f"{key=}, {self.get_child_key_fn(child.key)=}"

    def _delete_leaf(self, node):
        key = self.get_child_key_fn(node.key)
        v = node.parent.children.pop(key, None)
        assert v == node, f"parent does not have child key, {key}"

        self.evictable_size_ -= len(node.key)
        if node in self.evictable_leaves:
            self.evictable_leaves.remove(node)
        self._update_leaf_status(node.parent)

    def _update_leaf_status(self, node: TreeNode):
        if node.evicted or node.lock_ref > 0:
            if node in self.evictable_leaves:
                self.evictable_leaves.remove(node)
            return

        for child in node.children.values():
            if not child.evicted:
                if node in self.evictable_leaves:
                    self.evictable_leaves.remove(node)
                return

        if node not in self.evictable_leaves:
            self.evictable_leaves.add(node)

    def _total_size_helper(self):
        total_size = 0
        stack = [self.root_node]
        while stack:
            current_node = stack.pop()
            total_size += len(current_node.value)
            for child in current_node.children.values():
                if child.evicted:
                    continue
                stack.append(child)
        return total_size

    def _record_store_event(self, node: TreeNode):
        # One BlockStored per ``page_size`` chunk.
        if self.enable_kv_cache_events:
            # Compute hash_value lazily if not already set
            if node.hash_value is None:
                node.hash_value = compute_node_hash_values(node, self.page_size)

            # Get parent's last hash value for first page
            parent_block_hash = None
            if node.parent is not None and node.parent != self.root_node:
                if (
                    node.parent.hash_value is not None
                    and len(node.parent.hash_value) > 0
                ):
                    parent_block_hash = hash_str_to_int64(node.parent.hash_value[-1])

            page_index = 0
            for start in range(0, len(node.key), self.page_size):
                page_tokens = node.key.token_ids[start : start + self.page_size]
                if not page_tokens:
                    continue

                block_hash = hash_str_to_int64(node.hash_value[page_index])

                self.kv_event_queue.append(
                    BlockStored(
                        block_hashes=[block_hash],
                        parent_block_hash=parent_block_hash,
                        token_ids=page_tokens,
                        block_size=len(page_tokens),
                        lora_id=None,
                        medium=MEDIUM_GPU,
                    )
                )

                parent_block_hash = block_hash
                page_index += 1

    def _record_remove_event(self, node: TreeNode):
        # One BlockRemoved per chunk.
        if self.enable_kv_cache_events:
            # Compute hash_value lazily if not already set (must match what was stored)
            if node.hash_value is None:
                node.hash_value = compute_node_hash_values(node, self.page_size)

            page_index = 0
            for start in range(0, len(node.key), self.page_size):
                page_tokens = node.key.token_ids[start : start + self.page_size]
                if not page_tokens:
                    continue

                block_hash = hash_str_to_int64(node.hash_value[page_index])

                self.kv_event_queue.append(
                    BlockRemoved(block_hashes=[block_hash], medium=MEDIUM_GPU)
                )

                page_index += 1

    def _record_all_cleared_event(self):
        if self.enable_kv_cache_events:
            self.kv_event_queue.append(AllBlocksCleared())

    def take_events(self):
        """Atomically takes all events and clears the queue.

        Returns:
            A list of KV cache events.
        """
        if not self.enable_kv_cache_events:
            return []
        events = self.kv_event_queue
        self.kv_event_queue = []
        return events


if __name__ == "__main__":
    tree = RadixCache.create_simulated()

    # Example token id sequences (as lists of ints)
    tree.insert(InsertParams(key=RadixKey(token_ids=[1, 2, 3], extra_key=None)))
    tree.insert(InsertParams(key=RadixKey(token_ids=[1, 2, 3], extra_key=None)))
    tree.insert(InsertParams(key=RadixKey(token_ids=[1, 2, 4, 5], extra_key=None)))
    tree.insert(
        InsertParams(key=RadixKey(token_ids=[1, 2, 4, 5, 6, 7], extra_key=None))
    )
    tree.insert(
        InsertParams(key=RadixKey(token_ids=[8, 9, 10, 11, 12], extra_key=None))
    )
    tree.pretty_print()

    print(
        tree.match_prefix(
            MatchPrefixParams(key=RadixKey(token_ids=[1, 2, 3, 13, 14], extra_key=None))
        )
    )
