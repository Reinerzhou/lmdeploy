# Copyright (c) OpenMMLab. All rights reserved.
from contextlib import contextmanager
from dataclasses import dataclass, field, fields
from typing import Any, Dict, List

import torch

from lmdeploy.pytorch.backends import get_backend

from .adapter.adapter import SchedulerAdapter
from .config import CacheConfig


@dataclass
class AdapterInfo:
    ranks: torch.LongTensor
    scalings: torch.Tensor
    rank_offsets: torch.LongTensor
    target_modules: List[str]
    max_rank_per_target: List[int]
    max_rank: int

    @classmethod
    def from_adapters(cls, adapters: List[SchedulerAdapter]):
        """from adapters."""
        if len(adapters) == 0:
            return None
        target_modules = adapters[0].target_modules
        max_rank = adapters[0].max_rank
        ranks = [ada.rank for ada in adapters]
        scalings = [ada.scaling for ada in adapters]
        rank_offsets = [torch.from_numpy(ada.rank_offset) for ada in adapters]
        ranks = torch.tensor(ranks)
        scalings = torch.tensor(scalings)
        rank_offsets = torch.stack(rank_offsets)
        max_rank_per_target = ranks.max(0)[0].tolist()

        return cls(
            ranks=ranks,
            scalings=scalings,
            rank_offsets=rank_offsets,
            target_modules=target_modules,
            max_rank=max_rank,
            max_rank_per_target=max_rank_per_target,
        )

    def split_by_targets(self):
        """split by targets."""
        ret = dict()
        max_rank = self.max_rank
        for idx, target in enumerate(self.target_modules):
            r = self.ranks[:, idx]
            scaling = self.scalings[:, idx]
            r_off_start = idx * max_rank
            r_off_end = r_off_start + max_rank
            rank_offset = self.rank_offsets[:, r_off_start:r_off_end]
            max_rank_per_target = [self.max_rank_per_target[idx]]
            ret[target] = AdapterInfo(
                r,
                scaling,
                rank_offset,
                target_modules=[target],
                max_rank=max_rank_per_target[0],
                max_rank_per_target=max_rank_per_target,
            )
        return ret

    def to_device(self, device: str):
        """to device."""
        out_dict = dict()
        for f in fields(self):
            k = f.name
            v = getattr(self, k)
            if isinstance(v, torch.Tensor):
                v = v.to(device)
            out_dict[k] = v

        return AdapterInfo(**out_dict)


@dataclass
class VisionModelInputs:
    """Vision model inputs."""
    history_lengths: torch.LongTensor = None
    history_image_nums: torch.LongTensor = None
    history_image_token_lengths: torch.LongTensor = None
    input_embeddings: List[List[torch.Tensor]] = None
    input_embedding_ranges: List[torch.LongTensor] = None
    input_embedding_indexing: torch.BoolTensor = None

    def to_device(self, device: str):
        """to device."""
        out_dict = dict()
        for f in fields(self):
            k = f.name
            v = getattr(self, k)
            if isinstance(v, torch.Tensor):
                v = v.to(device)
            elif k == 'input_embedding_ranges' and v is not None:
                v = [e.to(device) for e in v]
            elif k == 'input_embeddings' and v is not None:
                v = [[e.to(device) for e in li] for li in v]
            out_dict[k] = v

        return VisionModelInputs(**out_dict)

    def get_inputs(self, history_lengths: torch.Tensor,
                   seq_lengths: torch.Tensor):
        """get vision embedding inputs."""
        input_embeddings = None
        input_embedding_indexing = None
        if self.input_embeddings is not None and len(
                self.input_embeddings) > 0:
            input_embedding_li = []
            for (his_len, seq_len, embeddings,
                 emb_ranges) in zip(history_lengths, seq_lengths,
                                    self.input_embeddings,
                                    self.input_embedding_ranges):
                for emb, (emb_start, emb_end) in zip(embeddings, emb_ranges):
                    start = max(emb_start, his_len) - emb_start
                    end = min(emb_end, his_len + seq_len) - emb_start
                    if 0 <= start < end:
                        input_embedding_li.append(emb[start:end])
            # has embeddings
            if len(input_embedding_li) > 0:
                input_embeddings = torch.cat(input_embedding_li, dim=0)
                device = input_embeddings.device
                starts = history_lengths - self.history_lengths
                ends = starts + seq_lengths
                input_embedding_indexing = torch.cat([
                    indexing[s:e] for indexing, s, e in zip(
                        self.input_embedding_indexing, starts, ends)
                ],
                                                     dim=0)
                index_ranges = torch.arange(input_embedding_indexing.numel(),
                                            device=device)
                input_embedding_indexing = index_ranges[
                    input_embedding_indexing]
        return input_embeddings, input_embedding_indexing


@dataclass
class ModelInputs:
    """Input of the model."""
    input_ids: torch.LongTensor
    seq_length: torch.LongTensor
    history_lengths: torch.LongTensor
    block_offsets: torch.LongTensor
    max_q_seq_length: int
    max_history_length: int
    is_decoding: bool
    num_ignored_history: torch.LongTensor
    local_adapter_ids: torch.LongTensor = None
    adapter_info: AdapterInfo = None
    meta: Any = None
    vision_inputs: VisionModelInputs = None

    def update(self, input_ids: torch.LongTensor):
        """update input ids."""
        assert self.is_decoding
        self.history_lengths = self.history_lengths + 1
        self.max_history_length = self.max_history_length + 1
        if input_ids.dim() == 1:
            input_ids = input_ids[None, :]
        self.input_ids = input_ids
        return self

    def split(self, split_size: int, block_size: int):
        """split inputs."""
        assert len(
            self.seq_length) == 1, ('Can not perform split on batched input.')
        assert split_size % block_size == 0, (
            'split_size should be multi of block_size.')

        input_ids = self.input_ids
        if input_ids.numel() < split_size:
            return self

        num_blocks = split_size // block_size
        overlap = (self.history_lengths[0] % block_size != 0)
        max_seq_len = self.seq_length[0].item()
        ret = []
        block_start = 0
        for i in range(0, max_seq_len, split_size):
            start = i
            end = min(max_seq_len, i + split_size)
            block_end = block_start + num_blocks
            if overlap:
                block_end += 1

            block_offsets = self.block_offsets
            inp = ModelInputs(
                input_ids=self.input_ids[:, start:end],
                seq_length=input_ids.new_tensor([end - start]),
                block_offsets=block_offsets,
                history_lengths=self.history_lengths + start,
                max_q_seq_length=end - start,
                max_history_length=self.max_history_length + start,
                is_decoding=self.is_decoding,
                num_ignored_history=self.num_ignored_history,
                local_adapter_ids=self.local_adapter_ids,
                adapter_info=self.adapter_info,
                meta=self.meta,
                vision_inputs=self.vision_inputs,
            )
            ret.append(inp)
            block_start += num_blocks

        return ret

    def to_device(self, device: str):
        """to device."""
        out_dict = dict()
        for f in fields(self):
            k = f.name
            v = getattr(self, k)
            if isinstance(v, torch.Tensor):
                v = v.to(device)
            elif isinstance(v, VisionModelInputs):
                v = v.to_device(device)
            elif isinstance(v, AdapterInfo):
                v = v.to_device(device)
            out_dict[k] = v

        return ModelInputs(**out_dict)


@dataclass
class StepContext:
    """context of Model.

    patched model might need extra information to perform inference. This
    dataclass provide these infos and tools.
    """
    inputs: ModelInputs
    block_offsets: torch.LongTensor
    position_ids: torch.LongTensor
    q_start_loc: torch.LongTensor
    attention_mask: torch.LongTensor
    history_lengths: torch.LongTensor
    q_seq_length: torch.LongTensor
    kv_seq_length: torch.LongTensor
    max_q_seq_length: int
    max_kv_seq_length: int
    kv_caches: List
    is_decoding: bool
    world_size: int = 1
    local_adapter_ids: torch.LongTensor = None
    adapter_params: Dict[str, AdapterInfo] = None
    input_embeddings: torch.Tensor = None
    input_embedding_indexing: torch.Tensor = None

    _outputs: Dict = field(default_factory=dict)

    @classmethod
    def new(
        cls,
        inputs: ModelInputs,
        world_size: int = 1,
        kv_caches: List = None,
        cache_config: CacheConfig = None,
    ):
        """build step context.

        Args:
            inputs (ModelInputs): packaged model inputs.
            world_size (int): The distribution world size.
            device (str): The device of the tensors.
        """
        q_seq_length = inputs.seq_length
        max_q_seq_length = inputs.max_q_seq_length
        history_lengths = inputs.history_lengths

        # for vlm
        input_embeddings, input_embedding_indexing = None, None
        if (inputs.vision_inputs is not None
                and inputs.vision_inputs.input_embeddings is not None):
            input_embeddings, input_embedding_indexing = \
                inputs.vision_inputs.get_inputs(history_lengths, q_seq_length)

        batch_size = len(q_seq_length)
        device = q_seq_length.device

        # q_start_loc and kv_seq_length
        if inputs.is_decoding:
            q_start_loc = torch.arange(0, batch_size, device=device)
            attention_mask = torch.ones_like(q_seq_length)[:, None]
            position_ids = history_lengths.unsqueeze(-1)
        else:
            q_start_loc = q_seq_length.cumsum(0) - q_seq_length
            mask_range = torch.arange(max_q_seq_length, device=device)[None, :]
            attention_mask = (mask_range < q_seq_length[:, None]).long()
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids += history_lengths.unsqueeze(-1)

        # position ids 1d
        position_ids = cls.get_position_ids_1d(position_ids,
                                               q_seq_length)[None]
        # seq_len + history_length
        kv_seq_length = q_seq_length + history_lengths
        max_kv_seq_length = max_q_seq_length + inputs.max_history_length

        window_size = getattr(cache_config, 'window_size', 0)
        if window_size > 0:
            kv_seq_length -= inputs.num_ignored_history

        adapter_params = None
        if inputs.adapter_info is not None:
            adapter_params = inputs.adapter_info.split_by_targets()

        ret = StepContext(inputs=inputs,
                          block_offsets=inputs.block_offsets,
                          position_ids=position_ids,
                          input_embeddings=input_embeddings,
                          input_embedding_indexing=input_embedding_indexing,
                          attention_mask=attention_mask,
                          q_start_loc=q_start_loc,
                          history_lengths=inputs.history_lengths,
                          q_seq_length=inputs.seq_length,
                          kv_seq_length=kv_seq_length,
                          max_q_seq_length=max_q_seq_length,
                          max_kv_seq_length=max_kv_seq_length,
                          kv_caches=kv_caches,
                          is_decoding=inputs.is_decoding,
                          world_size=world_size,
                          local_adapter_ids=inputs.local_adapter_ids,
                          adapter_params=adapter_params)

        ret = get_backend().update_step_context(ret)
        return ret

    @classmethod
    def get_position_ids_1d(cls, position_ids: torch.LongTensor,
                            seq_length: torch.LongTensor):
        """get 1d position_ids."""
        if position_ids.size(0) == 1 or position_ids.size(1) == 1:
            position_ids_1d = position_ids.flatten()
        else:
            device = position_ids.device
            position_ids_1d = [
                ids[:l] for ids, l in zip(position_ids.cpu(), seq_length.cpu())
            ]
            position_ids_1d = torch.cat(position_ids_1d).to(device)
        return position_ids_1d


class StepContextManager:

    def __init__(self):
        self._current_ctx = None

    @staticmethod
    def build_context(
        inputs: ModelInputs,
        world_size: int = 1,
        kv_caches: List = None,
        cache_config: CacheConfig = None,
    ):
        """build context."""
        return StepContext.new(
            inputs,
            world_size,
            kv_caches,
            cache_config,
        )

    @contextmanager
    def context(self, ctx: StepContext):
        """context context."""
        self._current_ctx = ctx
        yield ctx
        self._current_ctx = None

    def current_context(self):
        """get current_context."""
        return self._current_ctx
