from __future__ import annotations

import logging
from functools import wraps

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)


def patch_rocm_scatter_with_broadcast() -> None:
    if not torch.version.hip:
        return
    if getattr(dist.scatter, "_miles_rocm_broadcast_scatter", False):
        return

    original_scatter = dist.scatter

    @wraps(original_scatter)
    def broadcast_scatter(tensor, scatter_list=None, src=0, group=None, async_op=False):
        if async_op or not tensor.is_cuda:
            return original_scatter(tensor, scatter_list=scatter_list, src=src, group=group, async_op=async_op)

        group_size = dist.get_world_size(group)
        global_rank = dist.get_rank()
        is_src = global_rank == src
        if is_src:
            if scatter_list is None:
                raise ValueError("scatter_list must be provided on the source rank")
            if len(scatter_list) != group_size:
                raise ValueError(f"scatter_list length {len(scatter_list)} != group size {group_size}")

        shape_len = torch.empty((), dtype=torch.long, device=tensor.device)
        for group_rank in range(group_size):
            target_global_rank = dist.get_global_rank(group, group_rank) if group is not None else group_rank
            if is_src:
                send_tensor = scatter_list[group_rank].to(device=tensor.device, dtype=tensor.dtype).contiguous()
                shape_len.fill_(send_tensor.dim())
            else:
                send_tensor = None

            dist.broadcast(shape_len, src=src, group=group)
            if is_src:
                shape = torch.tensor(send_tensor.shape, dtype=torch.long, device=tensor.device)
            else:
                shape = torch.empty(int(shape_len.item()), dtype=torch.long, device=tensor.device)

            dist.broadcast(shape, src=src, group=group)
            if is_src:
                buffer = send_tensor
            else:
                buffer = torch.empty(tuple(int(x.item()) for x in shape), dtype=tensor.dtype, device=tensor.device)

            dist.broadcast(buffer, src=src, group=group)
            if global_rank == target_global_rank:
                if tensor.shape != buffer.shape:
                    raise RuntimeError(f"scatter target shape mismatch: tensor={tensor.shape}, buffer={buffer.shape}")
                tensor.copy_(buffer)

        return None

    broadcast_scatter._miles_rocm_broadcast_scatter = True
    dist.scatter = broadcast_scatter
    logger.warning("Patched torch.distributed.scatter with broadcast-loop scatter for ROCm checkpoint loading")
