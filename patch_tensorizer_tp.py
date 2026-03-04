"""
Patch vLLM's tensorizer loader to support loading a single (non-sharded)
tensorizer file with tensor-parallel > 1. Without this patch, vLLM requires
per-rank files (model-rank-%03d.tensors) when TP > 1.

With a single model.tensors file, each TP rank loads the full file and
model.load_weights() handles weight distribution — identical to the
safetensors code path.
"""

import site, os, re

site_packages = site.getsitepackages()[0]

# --- Patch 1: tensorizer.py ---
# verify_with_parallel_config: warn instead of raising on non-sharded URI
path1 = os.path.join(site_packages,
    "vllm/model_executor/model_loader/tensorizer.py")

with open(path1) as f:
    src = f.read()

old = '''    def verify_with_parallel_config(
        self,
        parallel_config: "ParallelConfig",
    ) -> None:
        if parallel_config.tensor_parallel_size > 1 and not self._is_sharded:
            raise ValueError(
                "For a sharded model, tensorizer_uri should include a"
                " string format template like '%04d' to be formatted"
                " with the rank of the shard"
            )'''

new = '''    def verify_with_parallel_config(
        self,
        parallel_config: "ParallelConfig",
    ) -> None:
        if parallel_config.tensor_parallel_size > 1 and not self._is_sharded:
            import logging
            logging.getLogger(__name__).warning(
                "Single tensorizer file with TP > 1: each rank will load "
                "the full file and shard weights at load time."
            )'''

assert old in src, "Could not find verify_with_parallel_config to patch"
src = src.replace(old, new)

with open(path1, "w") as f:
    f.write(src)

print(f"Patched {path1}")

# --- Patch 2: tensorizer_loader.py ---
# Skip URI % rank formatting when URI has no format specifier
path2 = os.path.join(site_packages,
    "vllm/model_executor/model_loader/tensorizer_loader.py")

with open(path2) as f:
    src = f.read()

old2 = '''        if parallel_config.tensor_parallel_size > 1:
            from vllm.distributed import get_tensor_model_parallel_rank

            self.tensorizer_config.tensorizer_uri = (
                self.tensorizer_config.tensorizer_uri % get_tensor_model_parallel_rank()
            )'''

new2 = '''        if parallel_config.tensor_parallel_size > 1:
            if self.tensorizer_config._is_sharded:
                from vllm.distributed import get_tensor_model_parallel_rank

                self.tensorizer_config.tensorizer_uri = (
                    self.tensorizer_config.tensorizer_uri % get_tensor_model_parallel_rank()
                )'''

assert old2 in src, "Could not find TP URI formatting to patch"
src = src.replace(old2, new2)

with open(path2, "w") as f:
    f.write(src)

print(f"Patched {path2}")
print("Done: tensorizer TP patches applied")
