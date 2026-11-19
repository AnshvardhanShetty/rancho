"""Inspect this isolated worker's actual CUDA allocation and native KV geometry.

Assumptions: one Qwen3-30B-A3B-Instruct-2507 BF16 GPU worker, no HBF emulation in this component
check. This reports native allocation, not an admission/throughput result. It
never edits tensors, other workers, installed packages, or existing engines.
"""


class MemoryProbe:
    def rancho_memory_snapshot(self):
        import torch
        runner = self.model_runner
        config = runner.kv_cache_config
        groups = []
        for group in config.kv_cache_groups:
            spec = group.kv_cache_spec
            groups.append(dict(kind=type(spec).__name__, layers=list(group.layer_names),
                               block_tokens=spec.block_size,
                               page_bytes_per_layer=spec.page_size_bytes))
        tensors = []
        storages = {}
        for tensor in runner.kv_caches:
            storage = tensor.untyped_storage()
            storages[storage.data_ptr()] = storage.nbytes()
            tensors.append(dict(shape=list(tensor.shape), dtype=str(tensor.dtype),
                                bytes=tensor.numel()*tensor.element_size()))
        native_bytes = sum(storages.values())
        return dict(max_num_seqs=self.vllm_config.scheduler_config.max_num_seqs,max_num_batched_tokens=self.vllm_config.scheduler_config.max_num_batched_tokens,moe_backend=self.vllm_config.kernel_config.moe_backend,model_memory_bytes=int(runner.model_memory_usage),
                    native_cache_blocks=config.num_blocks,
                    native_cache_storage_bytes=native_bytes,
                    native_bytes_per_block=native_bytes//config.num_blocks,
                    groups=groups, tensors=tensors,
                    cuda_allocated_bytes=torch.cuda.memory_allocated(),
                    cuda_reserved_bytes=torch.cuda.memory_reserved(),
                    cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                    cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                    GPU_name=torch.cuda.get_device_name())
