"""Explicit per-layer pages for the installed vLLM hybrid cache layout.

Assumptions: registration receives the same native cache configuration and named
tensors as vLLM's OffloadingConnectorWorker. Each transfer specifies native block
IDs separately for every cache group. Attention pages include padding; recurrent
pages include both convolution and temporal state. No 16-token scientific ID is
silently equated with a native page. This component does not implement RANCHO
admission, LRU, warm-state materialization, or the modeled flash write channel.
"""
from dataclasses import dataclass
from collections.abc import Mapping
import hashlib
import json


@dataclass(frozen=True)
class Snapshot:
    layout_sha256: str
    rows_per_group: tuple[tuple[int, int], ...]
    pages: Mapping
    bytes: int


class NativeCachePages:
    def __init__(self, config, caches):
        import torch
        from vllm.v1.kv_cache_interface import AttentionSpec, MambaSpec, UniformTypeKVCacheSpecs
        self.num_blocks = config.num_blocks
        if self.num_blocks <= 0:
            raise ValueError('A nonempty native allocation is required')
        self.pages, self.layer_groups, self.layout = {}, {}, []
        packed = {name for tensor in config.kv_cache_tensors
                  if getattr(tensor, 'block_stride', 0) > 0 for name in tensor.shared_by}
        expected = [name for group in config.kv_cache_groups for name in group.layer_names]
        if len(set(expected)) != len(expected) or set(caches) != set(expected):
            raise ValueError('Registration must contain every native layer exactly once')
        for gid, group in enumerate(config.kv_cache_groups):
            per_layer = (group.kv_cache_spec.kv_cache_specs
                         if isinstance(group.kv_cache_spec, UniformTypeKVCacheSpecs) else {})
            for name in group.layer_names:
                spec = per_layer.get(name, group.kv_cache_spec)
                tensor = caches[name]
                if not isinstance(tensor, torch.Tensor):
                    raise TypeError('Use registered raw page tensors, not partial state views')
                page = spec.page_size_bytes
                raw = torch.empty(0, dtype=torch.uint8, device=tensor.device).set_(tensor.untyped_storage())
                offset = tensor.storage_offset() * tensor.element_size()
                if isinstance(spec, AttentionSpec):
                    stride = tensor.stride(0) * tensor.element_size() if name in packed else page
                elif isinstance(spec, MambaSpec):
                    if tensor.dtype != torch.int8 or not tensor.is_contiguous():
                        raise ValueError('Recurrent state requires the complete registered raw page')
                    stride = page
                else:
                    raise TypeError('Unsupported cache specification')
                if stride < page or offset < 0 or offset + (self.num_blocks-1)*stride + page > raw.numel():
                    raise ValueError('Page layout exceeds registered allocation')
                self.pages[name] = torch.as_strided(raw, (self.num_blocks, page), (stride, 1), offset)
                self.layer_groups[name] = gid
                self.layout.append({'name': name, 'group': gid, 'kind': type(spec).__name__,
                                    'native_block_tokens': spec.block_size, 'page_bytes': page,
                                    'stride_bytes': stride, 'dtype': str(tensor.dtype)})
        self.groups = frozenset(self.layer_groups.values())
        self.layout_sha256 = hashlib.sha256(json.dumps(self.layout, sort_keys=True).encode()).hexdigest()

    def _ids(self, blocks):
        if set(blocks) != self.groups:
            raise ValueError('Every hybrid cache group must be included')
        selected = {}
        for group, ids in blocks.items():
            ids = list(ids)
            if not ids or any(type(i) is not int or not 0 <= i < self.num_blocks for i in ids):
                raise ValueError('Invalid native page index')
            if len(set(ids)) != len(ids):
                raise ValueError('Duplicate page index')
            selected[group] = ids
        return selected

    def capture(self, blocks):
        """Copy every required state into an independent, unpublished snapshot."""
        import torch
        selected = self._ids(blocks)
        pages = {}
        for name, view in self.pages.items():
            indices = torch.tensor(selected[self.layer_groups[name]], dtype=torch.long, device=view.device)
            pages[name] = view.index_select(0, indices).clone()
        return Snapshot(self.layout_sha256, tuple(sorted((g, len(v)) for g, v in selected.items())),
                        pages, sum(x.numel()*x.element_size() for x in pages.values()))

    def restore(self, snapshot, blocks):
        """Validate the entire transfer before copying; caller handles completion/publication."""
        import torch
        selected = self._ids(blocks)
        if snapshot.layout_sha256 != self.layout_sha256 or set(snapshot.pages) != set(self.pages):
            raise ValueError('Incomplete or incompatible hybrid snapshot')
        if tuple(sorted((g, len(v)) for g, v in selected.items())) != snapshot.rows_per_group:
            raise ValueError('Source and destination group lengths differ')
        if snapshot.bytes != sum(x.numel()*x.element_size() for x in snapshot.pages.values()):
            raise ValueError('Snapshot byte accounting changed')
        for name, target in self.pages.items():
            source = snapshot.pages[name]
            count = len(selected[self.layer_groups[name]])
            if source.dtype != torch.uint8 or source.device != target.device or source.shape != (count, target.shape[1]):
                raise ValueError('A layer snapshot has the wrong shape, dtype or device')
        # All validation above precedes the first mutation. GPU completion and
        # cache visibility are deliberately the caller's explicit responsibility.
        for name, target in self.pages.items():
            indices = torch.tensor(selected[self.layer_groups[name]], dtype=torch.long, device=target.device)
            target.index_copy_(0, indices, snapshot.pages[name])
