# Copyright (c) OpenMMLab. All rights reserved.

from typing import List

import grouped_gemm_cuda
import numpy as np
import torch
from sglang.srt.layers.moe.ep_moe.kernels import (post_reorder_triton_kernel, pre_reorder_triton_kernel,
                                                  run_moe_ep_preproess, silu_and_mul_triton_kernel)
from sglang.srt.layers.moe.ep_moe.layer import GroupedGemmRunner
from torch import distributed as dist

from lmdeploy.pytorch.distributed import get_ep_world_rank
from lmdeploy.pytorch.kernels.dlinfer import moe_gating_topk_softmax

from ..moe import FusedMoEBuilder, FusedMoEImpl, SoftmaxTopKBuilder, SoftmaxTopKImpl

# from vllm.model_executor.layers.fused_moe.deep_gemm_moe import _moe_permute, _moe_unpermute_and_reduce


class DlinferSoftmaxTopKImpl(SoftmaxTopKImpl):
    """Dlinfer softmax topk implementation."""

    def __init__(self, top_k: int, dim: int = -1):
        self.top_k = top_k
        self.dim = dim

    def forward(self, x: torch.Tensor):
        routing_weights, selected_experts = moe_gating_topk_softmax(x, self.top_k)
        return routing_weights, selected_experts


class DlinferSoftmaxTopKBuilder(SoftmaxTopKBuilder):
    """Dlinfer softmax topk implementation builder."""

    @staticmethod
    def build(top_k: int, dim: int = -1):
        """build."""
        return DlinferSoftmaxTopKImpl(top_k, dim)


class DlinferFusedMoEImpl(FusedMoEImpl):
    """Dlinfer fused moe implementation."""

    def __init__(self, top_k: int, num_experts: int, renormalize: bool = False, ep_size: int = 1):
        self.top_k = top_k
        self.num_experts = num_experts
        self.renormalize = renormalize
        self.ep_size = ep_size

        self.ep_size, self.ep_rank = get_ep_world_rank()
        self.num_experts_per_partition = self.num_experts // self.ep_size
        self.start_expert_id = self.ep_rank * self.num_experts_per_partition
        self.end_expert_id = self.start_expert_id + self.num_experts_per_partition - 1
        self.w2_input_scale = None
        self.use_block_quant = False

    def update_weights(self, gate_up_weights: torch.Tensor, down_weights: torch.Tensor):
        """Update weights."""
        device_type = gate_up_weights.device.type
        if device_type in ['npu']:
            return gate_up_weights.transpose(-1, -2).contiguous(), down_weights.transpose(-1, -2).contiguous()
        return gate_up_weights, down_weights

    def support_ep(self):
        """Support expert parallelism."""
        return True

    def ep_expert_list(self, world_size: int, rank: int):
        """Experts list of current rank."""
        expert_per_rank = (self.num_experts + world_size - 1) // world_size
        first_expert = rank * expert_per_rank
        last_expert = min(first_expert + expert_per_rank, self.num_experts)
        return list(range(first_expert, last_expert))

    # from sglang/srt/layers/moe/ep_moe/layer.py class EPMoE forward
    def tmp_forward(self, hidden_states: torch.Tensor, gate_up_weights: torch.Tensor, down_weights: torch.Tensor,
                    topk_weights: torch.Tensor, topk_ids: torch.Tensor):
        self.grouped_gemm_runner = GroupedGemmRunner(hidden_states.device)

        # import pdb; pdb.set_trace()

        N, H = hidden_states.shape
        # device = hidden_states.device
        # if N >= 16:
        #     print("test device:", hidden_states.device, flush=True)
        # topk_weights = topk_weights.reshape(N, self.top_k)
        # topk_ids = topk_ids.reshape(N, self.top_k)

        if not topk_weights.is_contiguous():
            topk_weights = topk_weights.contiguous()

        original_shape = hidden_states.shape
        if len(original_shape) == 3:
            hidden_states = hidden_states.view(-1, hidden_states.shape[-1])

        local_num_experts = self.num_experts // self.ep_size

        # import pdb; pdb.set_trace()
        # moe init routing
        reorder_topk_ids, src2dst, seg_indptr = run_moe_ep_preproess(topk_ids, self.num_experts)
        # reorder_topk_ids, src2dst, seg_indptr = deepep_run_moe_deep_preprocess(topk_ids, self.num_experts)
        expanded_expert_idx = reorder_topk_ids
        # import pdb; pdb.set_trace()

        gateup_input = torch.empty(
            (int(hidden_states.shape[0] * self.top_k), hidden_states.shape[1]),
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )
        # import pdb; pdb.set_trace()
        # if N >= 16:
        # print("before permute topk_ids:", topk_ids, topk_ids.shape, flush=True)
        # print("before permute reorder_topk_ids:", reorder_topk_ids, reorder_topk_ids.shape, flush=True)
        # print("before permute src2dst:", src2dst, src2dst.shape, flush=True)
        # print("before permute seg_indptr:", seg_indptr, seg_indptr.shape, flush=True)
        # print("before permute hidden_states:", hidden_states.shape, flush=True)
        # import pdb; pdb.set_trace()

        assert torch.isnan(hidden_states).any().item() is False
        assert torch.isinf(hidden_states).any().item() is False
        # import pdb; pdb.set_trace()

        # PreReorder
        pre_reorder_triton_kernel[(hidden_states.shape[0], )](
            # deepep_permute_triton_kernel[(hidden_states.shape[0], )](
            hidden_states,
            gateup_input,
            src2dst,
            topk_ids,
            None,
            # self.start_expert_id,
            # self.end_expert_id,
            0,
            self.num_experts,
            self.top_k,
            hidden_states.shape[1],
            BLOCK_SIZE=512,
        )
        # import pdb; pdb.set_trace()

        # #########################################################################################################
        # output = torch.empty_like(gateup_input)
        # assert torch.isnan(gateup_input).any().item() is False
        # assert torch.isinf(gateup_input).any().item() is False
        # assert torch.isnan(topk_weights).any().item() is False
        # assert torch.isinf(topk_weights).any().item() is False

        # import pdb; pdb.set_trace()

        # post_reorder_triton_kernel[(gateup_input.size(0), )](
        # # deepep_post_reorder_triton_kernel[(gateup_input.size(0), )](
        #     gateup_input,
        #     output,
        #     src2dst,
        #     topk_ids,
        #     topk_weights,
        #     # self.start_expert_id,
        #     # self.end_expert_id,
        #     0,
        #     self.num_experts,
        #     self.top_k,
        #     gateup_input.size(1),
        #     BLOCK_SIZE=512,
        # )
        # import pdb; pdb.set_trace()

        # assert torch.isnan(gateup_input).any().item() is False
        # assert torch.isinf(gateup_input).any().item() is False
        # #########################################################################################################

        # import pdb; pdb.set_trace()

        # gateup_input = gateup_input[:N]
        hidden_states = gateup_input
        if N >= 16:
            print('before dispatch hidden_states.shape:', hidden_states.shape, flush=True)

        # dispatch
        global_expert_tokens = torch.bincount(expanded_expert_idx, minlength=self.num_experts)
        scatter_sizes = global_expert_tokens.view(self.ep_size, -1).sum(-1)

        gather_sizes = torch.empty_like(scatter_sizes)

        if N >= 16:
            print('scatter_sizes:', scatter_sizes, scatter_sizes.shape, flush=True)

        dist.all_to_all_single(gather_sizes, scatter_sizes)

        if N >= 16:
            print('scatter_sizes - gather_sizes:', scatter_sizes, gather_sizes, flush=True)
        scatter_size_list = scatter_sizes.cpu().tolist()
        gather_size_list = gather_sizes.cpu().tolist()

        if N >= 16:
            print('expanded_expert_idx1:', expanded_expert_idx, expanded_expert_idx.shape, '!@!', flush=True)
        # print("scatter_size_list - gather_size_list1", scatter_size_list, gather_size_list, flush=True)

        expanded_expert_idx = expanded_expert_idx % local_num_experts
        original_hidden_states = hidden_states

        # print(np.sum(np.array(gather_size_list)), hidden_states.shape[1:])

        hidden_states = original_hidden_states.new_empty((np.sum(np.array(gather_size_list)), ) +
                                                         hidden_states.shape[1:])
        # print(hidden_states.shape, original_hidden_states.shape, original_hidden_states.device)

        # if N >= 16:
        #     print("before alltoall 1", hidden_states.shape, original_hidden_states.shape, flush=True)

        dist.all_to_all_single(hidden_states, original_hidden_states, gather_size_list, scatter_size_list)

        if N >= 16:
            print('after alltoall 1', hidden_states.shape, original_hidden_states.shape, flush=True)
            print('scatter_size_list - gather_size_list2', scatter_size_list, gather_size_list, flush=True)

        local_expert_idx = expanded_expert_idx.new_empty((np.sum(np.array(gather_size_list)), ) +
                                                         expanded_expert_idx.shape[1:])

        # if N >= 16:
        # print("before alltoall 2", local_expert_idx, local_expert_idx.shape, flush=True)
        # print("before alltoall 2", expanded_expert_idx, expanded_expert_idx.shape, flush=True)

        dist.all_to_all_single(local_expert_idx, expanded_expert_idx, gather_size_list, scatter_size_list)
        if N >= 16:
            print('local_expert_idx:', local_expert_idx, local_expert_idx.shape, '!@!', flush=True)
            # print("expanded_expert_idx2:", expanded_expert_idx, expanded_expert_idx.shape, "!@!", flush=True)

        sorted_local_expert_idx, sorted_idx = torch.sort(local_expert_idx)
        if N >= 16:
            print('sorted_local_expert_idx:', sorted_local_expert_idx, sorted_local_expert_idx.shape, '!@!', flush=True)
            # print("sorted_idx:", sorted_idx, sorted_idx.shape, "!@!", flush=True)
            print('hidden_states:', hidden_states.shape, '!@!', flush=True)
            # print("gate_up_weights:", gate_up_weights.shape, "!@!", flush=True)

        assert torch.isnan(hidden_states).any().item() is False
        assert torch.isinf(hidden_states).any().item() is False

        gateup_input = hidden_states[sorted_idx]

        assert torch.isnan(gateup_input).any().item() is False
        assert torch.isinf(gateup_input).any().item() is False

        # if True:
        if N >= 16:
            print('gateup_input:', gateup_input.shape, '!@!', flush=True)
            # print("seg_indptr1:", seg_indptr, seg_indptr.shape, "!@!", flush=True)
            print('start - end:', self.start_expert_id, self.end_expert_id, flush=True)

        _, _, seg_indptr_cur_rank = run_moe_ep_preproess(sorted_local_expert_idx, local_num_experts)

        # seg_indptr_cur_rank = seg_indptr[self.start_expert_id:self.end_expert_id + 1]
        # seg_indptr_cur_rank = seg_indptr[self.start_expert_id:self.end_expert_id + 2]
        # if True:
        if N >= 16:
            # print("seg_indptr2:", seg_indptr, seg_indptr.shape, "!@!", flush=True)
            print('seg_indptr_cur_rank:', seg_indptr_cur_rank, seg_indptr_cur_rank.shape, '!@!', flush=True)
        # weight_indices_cur_rank = torch.arange(
        #     0,
        #     # gateup_input.shape[0],
        #     self.num_experts_per_partition,
        #     device=hidden_states.device,
        #     dtype=torch.int64,
        # )

        # GroupGemm-0
        gateup_output = torch.empty(
            gateup_input.shape[0],
            gate_up_weights.shape[1],
            device=hidden_states.device,
            dtype=hidden_states.dtype,
            # dtype=torch.float32
        )

        assert torch.isnan(gateup_input).any().item() is False
        assert torch.isinf(gateup_input).any().item() is False

        assert torch.isnan(gate_up_weights).any().item() is False
        assert torch.isinf(gate_up_weights).any().item() is False

        if N >= 16:
            print('check1', gateup_input.shape, gate_up_weights.shape, seg_indptr_cur_rank.shape, flush=True)
            # print(gateup_input)
            # print(gate_up_weights)
            print(seg_indptr_cur_rank)

        # gateup_output = self.grouped_gemm_runner(
        #     a=gateup_input,
        #     b=gate_up_weights,
        #     c=gateup_output,
        #     batch_size=self.num_experts_per_partition,
        #     top_k=self.top_k,
        #     weight_column_major=True,
        #     gateup_stage=True,
        #     seg_indptr=seg_indptr_cur_rank,
        #     weight_indices=weight_indices_cur_rank,
        #     block_shape=None,
        # )

        batch_sizes = seg_indptr_cur_rank[1:] - seg_indptr_cur_rank[:-1]
        batch_sizes = batch_sizes.cpu()

        # apex
        grouped_gemm_cuda.gmm(gateup_input, gate_up_weights, gateup_output, batch_sizes, False, True)

        assert torch.isnan(gateup_output).any().item() is False
        assert torch.isinf(gateup_output).any().item() is False

        # Act
        down_input = torch.empty(
            gateup_output.shape[0],
            gateup_output.shape[1] // 2,
            device=gateup_output.device,
            dtype=hidden_states.dtype,
        )

        # tmp_expert_ids = torch.arange(
        #     0,
        #     self.num_experts_per_partition,
        #     device=reorder_topk_ids.device,
        #     dtype=reorder_topk_ids.dtype,
        # )

        # print(down_input.shape, flush=True)

        if self.w2_input_scale is None and not self.use_block_quant:
            self.w2_input_scale = torch.ones(
                self.num_experts_per_partition,
                dtype=torch.float32,
                device=hidden_states.device,
            )

        if N >= 16:
            print('check silu', gateup_output.shape, down_input.shape, sorted_local_expert_idx.shape, flush=True)
            print('sorted_local_expert_idx:', sorted_local_expert_idx)
            print('self.w2_input_scale:', self.w2_input_scale)

        silu_and_mul_triton_kernel[(gateup_output.shape[0], )](
            gateup_output,
            down_input,
            gateup_output.shape[1],
            sorted_local_expert_idx,
            # reorder_topk_ids,
            self.w2_input_scale,
            # self.start_expert_id % local_num_experts,
            # self.end_expert_id % local_num_experts,
            0,
            local_num_experts,
            # self.num_experts,
            # 0,
            # sorted_local_expert_idx.shape[0],
            BLOCK_SIZE=512,
        )
        assert torch.isnan(down_input).any().item() is False
        assert torch.isinf(down_input).any().item() is False

        # GroupGemm-1
        down_output = torch.empty(
            down_input.shape[0],
            down_weights.shape[1],
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )
        # down_output = self.grouped_gemm_runner(
        #     a=down_input,
        #     b=down_weights,
        #     c=down_output,
        #     batch_size=self.num_experts_per_partition,
        #     top_k=self.top_k,
        #     weight_column_major=True,
        #     gateup_stage=False,
        #     seg_indptr=seg_indptr_cur_rank,
        #     weight_indices=weight_indices_cur_rank,
        #     # scale_a=self.w2_input_scale,
        #     block_shape=None,
        # )

        if N >= 16:
            print('check2', down_input.shape, down_weights.shape, down_output.shape, flush=True)
            # print(gateup_input)
            # print(gate_up_weights)
            # print(seg_indptr_cur_rank)
        grouped_gemm_cuda.gmm(down_input, down_weights, down_output, batch_sizes, False, True)

        assert torch.isnan(down_output).any().item() is False
        assert torch.isinf(down_output).any().item() is False

        resorted_idx = torch.argsort(sorted_idx)
        hidden_states = down_output[resorted_idx]

        # print("before  all_to_all_single.shape:", original_hidden_states.shape, hidden_states.shape, flush=True)

        dist.all_to_all_single(original_hidden_states, hidden_states, scatter_size_list, gather_size_list)
        # print("test!!!!")
        # print(hidden_states[0], original_hidden_states[0])
        hidden_states = original_hidden_states
        # print(hidden_states[0], original_hidden_states[0], flush=True)
        # return hidden_states[:N]
        # moe finalize routing
        assert torch.isnan(hidden_states).any().item() is False
        assert torch.isinf(hidden_states).any().item() is False

        # PostReorder
        # output = torch.empty_like(hidden_states,
        #                           device=hidden_states.device,
        #     dtype=hidden_states.dtype,)

        # output = torch.empty(
        #     hidden_states.shape[0],
        #     hidden_states.shape[1],
        #     device=hidden_states.device,
        #     dtype=hidden_states.dtype,
        # )
        output = torch.empty_like(hidden_states)

        # print("before  post_reorder_triton_kernel.shape:", hidden_states.shape, output.shape, flush=True)

        # print(output[0][0])
        # print("test!!!!", self.ep_rank, hidden_states[0])
        # print(hidden_states.shape)
        # print(output.shape)
        # print(topk_weights.shape)
        # print(self.start_expert_id, self.end_expert_id, flush=True)

        assert torch.isnan(hidden_states).any().item() is False
        assert torch.isinf(hidden_states).any().item() is False

        assert torch.isnan(topk_weights).any().item() is False
        assert torch.isinf(topk_weights).any().item() is False

        if N >= 16:
            print('debug info1:')
            print('hidden_states:', hidden_states, hidden_states.shape, '!@!', flush=True)
            print('output:', output.shape, '!@!', flush=True)
            print('src2dst:', src2dst, src2dst.shape, '!@!', flush=True)
            print('topk_ids:', topk_ids.shape, '!@!', flush=True)
            print('topk_weights:', topk_weights.shape, '!@!', flush=True)
            print('self.num_experts:', self.num_experts, '!@!', flush=True)
            print('self.top_k:', self.top_k, '!@!', flush=True)

        # unpermute
        post_reorder_triton_kernel[(hidden_states.size(0), )](
            # deepep_post_reorder_triton_kernel[(hidden_states.size(0), )](
            hidden_states,
            output,
            src2dst,
            topk_ids,
            topk_weights,
            # self.start_expert_id,
            # self.end_expert_id,
            0,
            self.num_experts,
            self.top_k,
            hidden_states.size(1),
            BLOCK_SIZE=512,
        )

        assert torch.isnan(output).any().item() is False
        assert torch.isinf(output).any().item() is False

        if N >= 16:
            print('debug info2:')
            print('hidden_states:', hidden_states, hidden_states.shape, '!@!', flush=True)
            print('output:', output, output.shape, '!@!', flush=True)
            # print("src2dst:", src2dst, src2dst.shape, "!@!", flush=True)
            # print("topk_weights:", topk_weights, topk_weights.shape, "!@!", flush=True)
        # print("output.shape1", N, self.top_k, H, output.shape, flush=True)
        # if len(original_shape) == 3:
        # output = output.view(original_shape)
        # print("output.shape2",output.shape, flush=True)

        # print("checking...", N, self.top_k, H)
        # print(output.stride())
        # print(output.is_contiguous())
        # print(output.numel(), N * self.top_k * H, flush=True)
        # print("output.shape1.5",output.shape, flush=True)
        # output = output.view(N, self.top_k, H)
        if N >= 16:
            print('output.shape2', output[0:2], output.shape, flush=True)
            print('output.shape3', output[0:2], output.shape, flush=True)
        output = output.view(N, self.top_k, H).sum(dim=1)
        return output
        # return output[:N]

    def forward(self,
                hidden_states: torch.Tensor,
                topk_weights: torch.Tensor,
                topk_ids: torch.LongTensor,
                gate_up_weights: torch.Tensor,
                down_weights: torch.Tensor,
                expert_list: List[int] = None):
        """forward."""

        q_seq = hidden_states.shape[0]
        _ = hidden_states.shape[1]

        topk_weights = topk_weights.reshape(q_seq, -1).contiguous()
        topk_ids = topk_ids.reshape(q_seq, -1).contiguous()
        # topk = topk_ids.shape[1]
        # if q_seq != 1:
        # #     print(f"self.top_k: {self.top_k}.", flush=True)
        #     print(f"hidden_states.shape: {hidden_states.shape}.", flush=True)
        #     print(f"gate_up_weights.shape: {gate_up_weights.shape}.", flush=True)
        # #     print(f"down_weights.shape: {down_weights.shape}.", flush=True)
        #     print(f"topk_ids.shape: {topk_ids.shape}.", flush=True)
        #     print(f"topk_weights.shape: {topk_weights.shape}.", flush=True)

        # from lmdeploy.pytorch.kernels.dlinfer import fused_moe
        # return fused_moe(hidden_states, gate_up_weights, down_weights, topk_weights,
        #                  topk_ids, self.top_k, self.num_experts, self.ep_size, self.renormalize, expert_list)

        # import pdb; pdb.set_trace()
        # tp
        # if True:
        # if expert_list is None:
        #     # print("!!!!!!!!!!!!", flush=True)
        #     from lmdeploy.pytorch.kernels.dlinfer import fused_moe
        #     # from lmdeploy.pytorch.backends.cuda.moe import fused_moe
        #     out =  fused_moe(hidden_states, gate_up_weights, down_weights, topk_weights,
        #                  topk_ids, self.top_k, 1, 1, self.renormalize)
        # return out

        # # print("test tmp_forward!", flush=True)
        # test dp + ep
        out = self.tmp_forward(hidden_states, gate_up_weights, down_weights, topk_weights, topk_ids)
        # # print("test tmp_forward end!", flush=True)
        # if self.tp_size > 1:
        #     dist.all_reduce(out, group='tp')

        # dist.all_reduce(out, group='tp')
        # if self.ep_size > 1:
        #     print('running 1', flush=True)
        #     # dist.all_reduce(out, group='dp')
        #     # dist.all_reduce(out, group='tp')
        #     # dist.all_reduce(out, group='ep')
        #     print('running 2', flush=True)
        #     # print("running 3", flush=True)

        print('running after fused_moe...', out.shape, flush=True)
        # out = out.view(q_seq, topk, H).sum(dim=1)
        # return out

        # # dp + ep
        # expert_offset = 0
        # num_experts = None
        # if expert_list is not None and len(expert_list) != self.num_experts:
        #     expert_offset = expert_list[0]
        #     num_experts = self.num_experts
        # from lmdeploy.pytorch.backends.cuda.moe import fused_moe
        # out = fused_moe(hidden_states,
        #                 gate_up_weights,
        #                 down_weights,
        #                 topk_weights=topk_weights,
        #                 topk_ids=topk_ids,
        #                 topk=self.top_k,
        #                 expert_offset=expert_offset,
        #                 num_experts=num_experts,
        #                 renormalize=self.renormalize)
        # return out


class DlinferFusedMoEBuilder(FusedMoEBuilder):
    """Dlinfer fused moe builder."""

    @staticmethod
    def build(top_k: int, num_experts: int, renormalize: bool = False, ep_size: int = 1):
        """Build from mlp."""
        return DlinferFusedMoEImpl(top_k=top_k, num_experts=num_experts, renormalize=renormalize, ep_size=ep_size)
