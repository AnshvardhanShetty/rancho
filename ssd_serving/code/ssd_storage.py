"""Write KV to a local SSD file and read it back.

Storage keeps one fixed-size file, divided into slots holding full KV blocks.
write_block_to_ssd() copies GPU data through a CPU buffer and writes a slot.
read_block_from_ssd() reads that slot; restore_block_to_gpu() copies the data back into GPU memory.
close() releases the file and staging buffer.

Assumes Linux direct I/O and a caller-selected local ext4 block-device filesystem.
Writes are synchronous. Counters measure host writes, not internal flash wear."""

from collections import OrderedDict
import hashlib
import mmap
import os
from pathlib import Path
import subprocess
import tempfile
import torch


class Storage:
    """Track stored blocks, SSD slots and bytes transferred."""

    def __init__(self, directory, capacity, pages, verify):
        """Check the storage location and create a private temporary KV file."""
        directory = Path(directory).resolve(strict=True)
        mount = subprocess.check_output(
            ["findmnt", "-n", "-o", "TARGET,FSTYPE,SOURCE", "-T", str(directory)],
            text=True,
        ).split()
        if len(mount) != 3 or mount[1] != "ext4" or not mount[2].startswith("/dev/"):
            raise ValueError(f"Expected a local block-device ext4 filesystem, found {mount}")
        self.mount = mount
        self.views = list(pages.pages.values())
        self.page_bytes = sum(layer_view.shape[1] for layer_view in self.views)
        if self.page_bytes != 1572864 or len(self.views) != 48 or pages.groups != {0}:
            raise ValueError("Expected the qualified 48-layer Qwen native geometry")
        self.capacity = capacity
        self.verify = verify
        self.slots = OrderedDict()
        self.digests = {}
        self.writes = self.reads = self.replacements = 0
        self.file = tempfile.TemporaryFile(dir=directory)
        self.fd = os.open(
            f"/proc/self/fd/{self.file.fileno()}", os.O_RDWR | os.O_DIRECT | os.O_DSYNC
        )
        os.ftruncate(self.fd, capacity * self.page_bytes)
        self.staging = mmap.mmap(-1, self.page_bytes)
        self.buffer = torch.frombuffer(self.staging, dtype=torch.uint8)
        self.page_count = pages.num_blocks

    def copy_block_between_gpu_and_cpu(self, page_id, transfer_buffer, writing):
        """Copy one whole KV block between GPU memory and the CPU buffer."""
        if not 0 < page_id < self.page_count:
            raise ValueError("Invalid HBM slot")
        offset = 0
        for layer in self.views:
            layer_page = layer[page_id]
            buffer_slice = transfer_buffer[offset : offset + layer_page.numel()]
            if writing:
                buffer_slice.copy_(layer_page)
            else:
                layer_page.copy_(buffer_slice)
            offset += layer_page.numel()
        assert offset == self.page_bytes

    def write_block_to_ssd(self, key, page_id):
        """Write a new block, replacing the oldest unused SSD copy if full."""
        if key in self.slots:
            raise ValueError("Paid duplicate SSD write")
        if len(self.slots) == self.capacity:
            evicted_block_key, slot = self.slots.popitem(last=False)
            self.digests.pop(evicted_block_key, None)
            self.replacements += 1
        else:
            slot = len(self.slots)
        self.copy_block_between_gpu_and_cpu(page_id, self.buffer, True)
        if os.pwritev(self.fd, [self.staging], slot * self.page_bytes) != self.page_bytes:
            raise OSError("Incomplete direct SSD write")
        self.slots[key] = slot
        self.writes += 1
        if self.verify:
            self.digests[key] = hashlib.sha256(self.staging).digest()

    def read_block_from_ssd(self, key):
        """Read one saved block and optionally verify its contents."""
        slot = self.slots[key]
        if os.preadv(self.fd, [self.staging], slot * self.page_bytes) != self.page_bytes:
            raise OSError("Incomplete direct SSD read")
        if self.verify and hashlib.sha256(self.staging).digest() != self.digests[key]:
            raise ValueError("SSD payload differs from written KV")
        self.reads += 1
        return self.buffer.clone()

    def restore_block_to_gpu(self, page, payload):
        """Copy a staged block into its allocated GPU slot."""
        self.copy_block_between_gpu_and_cpu(page, payload, False)
        if self.verify:
            self.copy_block_between_gpu_and_cpu(page, self.buffer, True)
            if not torch.equal(self.buffer, payload):
                raise ValueError("Restored GPU KV differs from source")

    def close(self):
        """Release the file descriptors and CPU staging buffer."""
        os.close(self.fd)
        self.file.close()
        del self.buffer
        self.staging.close()
