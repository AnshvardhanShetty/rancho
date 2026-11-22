"""Find and copy KV pages using vLLM's actual cache layout"""

from dataclasses import dataclass
from collections.abc import Mapping
import hashlib
import json


@dataclass(frozen=True)
class Snapshot:
    """Cache layout and memory details reported by the worker"""

    layout_sha256: str
    rows_per_group: tuple[tuple[int, int], ...]
    pages: Mapping
    bytes: int


class NativeCachePages:
    """Access the same cache page across every model layer"""

    def __init__(self, config, caches):
        import torch
        from vllm.v1.kv_cache_interface import (
            AttentionSpec,
            MambaSpec,
            UniformTypeKVCacheSpecs,
        )

        self.num_blocks = config.num_blocks
        if self.num_blocks <= 0:
            raise ValueError("A nonempty native allocation is required")
        self.pages, self.layer_groups, self.layout = {}, {}, []
        packed = {
            name
            for tensor in config.kv_cache_tensors
            if getattr(tensor, "block_stride", 0) > 0
            for name in tensor.shared_by
        }
        expected = [name for group in config.kv_cache_groups for name in group.layer_names]
        if len(set(expected)) != len(expected) or set(caches) != set(expected):
            raise ValueError("Registration must contain every native layer exactly once")
        for group_index, group in enumerate(config.kv_cache_groups):
            per_layer = (
                group.kv_cache_spec.kv_cache_specs
                if isinstance(group.kv_cache_spec, UniformTypeKVCacheSpecs)
                else {}
            )
            for name in group.layer_names:
                spec = per_layer.get(name, group.kv_cache_spec)
                tensor = caches[name]
                if not isinstance(tensor, torch.Tensor):
                    raise TypeError("Use registered raw page tensors, not partial state views")
                page = spec.page_size_bytes
                raw_page_view = torch.empty(0, dtype=torch.uint8, device=tensor.device).set_(
                    tensor.untyped_storage()
                )
                offset = tensor.storage_offset() * tensor.element_size()
                if isinstance(spec, AttentionSpec):
                    stride = tensor.stride(0) * tensor.element_size() if name in packed else page
                elif isinstance(spec, MambaSpec):
                    if tensor.dtype != torch.int8 or not tensor.is_contiguous():
                        raise ValueError(
                            "Recurrent state requires the complete registered raw page"
                        )
                    stride = page
                else:
                    raise TypeError("Unsupported cache specification")
                if (
                    stride < page
                    or offset < 0
                    or offset + (self.num_blocks - 1) * stride + page > raw_page_view.numel()
                ):
                    raise ValueError("Page layout exceeds registered allocation")
                self.pages[name] = torch.as_strided(
                    raw_page_view, (self.num_blocks, page), (stride, 1), offset
                )
                self.layer_groups[name] = group_index
                self.layout.append(
                    {
                        "name": name,
                        "group": group_index,
                        "kind": type(spec).__name__,
                        "native_block_tokens": spec.block_size,
                        "page_bytes": page,
                        "stride_bytes": stride,
                        "dtype": str(tensor.dtype),
                    }
                )
        self.groups = frozenset(self.layer_groups.values())
        self.layout_sha256 = hashlib.sha256(
            json.dumps(self.layout, sort_keys=True).encode()
        ).hexdigest()

    def _prepare_page_indices(self, blocks):
        """Check page IDs and prepare them for GPU indexing"""
        if set(blocks) != self.groups:
            raise ValueError("Every hybrid cache group must be included")
        selected = {}
        for group, page_ids in blocks.items():
            page_ids = list(page_ids)
            if not page_ids or any(
                type(page_id) is not int or not 0 <= page_id < self.num_blocks
                for page_id in page_ids
            ):
                raise ValueError("Invalid native page index")
            if len(set(page_ids)) != len(page_ids):
                raise ValueError("Duplicate page index")
            selected[group] = page_ids
        return selected

    def capture_page_data(self, blocks):
        """Copy every required state into an independent, unpublished snapshot."""
        import torch

        selected = self._prepare_page_indices(blocks)
        pages = {}
        for name, view in self.pages.items():
            indices = torch.tensor(
                selected[self.layer_groups[name]], dtype=torch.long, device=view.device
            )
            pages[name] = view.index_select(0, indices).clone()
        return Snapshot(
            self.layout_sha256,
            tuple(sorted((cache_group, len(layers)) for cache_group, layers in selected.items())),
            pages,
            sum(
                page_tensor.numel() * page_tensor.element_size() for page_tensor in pages.values()
            ),
        )

    def restore_page_data(self, snapshot, blocks):
        """Validate the entire transfer before copying; caller handles completion/publication."""
        import torch

        selected = self._prepare_page_indices(blocks)
        if snapshot.layout_sha256 != self.layout_sha256 or set(snapshot.pages) != set(self.pages):
            raise ValueError("Incomplete or incompatible hybrid snapshot")
        if (
            tuple(sorted((cache_group, len(layers)) for cache_group, layers in selected.items()))
            != snapshot.rows_per_group
        ):
            raise ValueError("Source and destination group lengths differ")
        if snapshot.bytes != sum(
            page_tensor.numel() * page_tensor.element_size()
            for page_tensor in snapshot.pages.values()
        ):
            raise ValueError("Snapshot byte accounting changed")
        for name, target in self.pages.items():
            source = snapshot.pages[name]
            count = len(selected[self.layer_groups[name]])
            if (
                source.dtype != torch.uint8
                or source.device != target.device
                or source.shape != (count, target.shape[1])
            ):
                raise ValueError("A layer snapshot has the wrong shape, dtype or device")
        # All validation above precedes the first mutation. GPU completion and
        # cache visibility are deliberately the caller's explicit responsibility.
        for name, target in self.pages.items():
            indices = torch.tensor(
                selected[self.layer_groups[name]],
                dtype=torch.long,
                device=target.device,
            )
            target.index_copy_(0, indices, snapshot.pages[name])


class MemoryProbe:
    """Report the number of cache pages vLLM allocated."""

    def rancho_memory_snapshot(self):
        return {"native_cache_blocks": self.model_runner.kv_cache_config.num_blocks}
