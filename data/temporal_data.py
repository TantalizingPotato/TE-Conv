import copy

import torch
import numpy as np
from tqdm import tqdm
import time


class TemporalData(object):
    def __init__(self, src=None, dst=None, t=None, msg=None, y=None, useTBatch=True, **kwargs):

        self.src = src
        self.dst = dst
        self.t = t
        self.msg = msg
        self.y = y

        for key, item in kwargs.items():
            self[key] = item

        self.use_t_batch = useTBatch
        self.t_batch_valid = False
        if self.use_t_batch:
            self.prepare_t_batch()

    def prepare_t_batch(self):
        start_time = time.time()
        self.use_t_batch = True
        batchList = []
        batchIdxList = list()
        for i in range(0, self.num_events):
            batchIdxList.append(torch.Tensor())

        maxTS = np.zeros(self.num_nodes)

        milestone_basis = self.num_events // 10
        # for i in tqdm(range(0, self.num_events), desc='[Preparing.]'):
        for i in range(0, self.num_events):
            nowTS = 1 + int(max(maxTS[self[i].src], maxTS[self[i].dst]))
            maxTS[self[i].src] = maxTS[self[i].dst] = nowTS
            batchIdxList[nowTS] = torch.cat((batchIdxList[nowTS], torch.tensor([i])))
            if milestone_basis != 0 and ((i + 1) % milestone_basis == 0 or (i + 1) == self.num_events):
                print(f"{100 * (i + 1) / self.num_events:.2f}% finished for t-batch preparation ...")

        for IdxList in batchIdxList:
            if IdxList.size(0) > 0:
                batchList.append(self[IdxList.to(torch.long)])
        self.batch_list = batchList
        self.t_batch_valid = True
        method_time = time.time() - start_time
        print(f"t-batch preparation finished in {method_time:.2f}s")
        return self

    def __getitem__(self, idx):

        if isinstance(idx, str):  # self[str(idx)]
            return getattr(self, idx, None)
        if isinstance(idx, int):
            idx = torch.tensor([idx])
        if isinstance(idx, (list, tuple)):
            idx = torch.tensor(idx)
        elif isinstance(idx, slice):
            pass
        elif isinstance(idx, torch.Tensor) and (idx.dtype == torch.long
                                                or idx.dtype == torch.bool):
            pass
        else:
            raise IndexError(
                f'Only strings, integers, slices (`:`), list, tuples, and '
                f'long or bool tensors are valid indices (got '
                f'{type(idx).__name__}).')
        # only tensor function implemented here
        data = copy.copy(self)
        for key, item in data:
            if isinstance(item, bool):
                continue
            # if item.shape[0] == self.num_events:
            if len(item) == self.num_events:  # edited by lgh
                data[key] = item[idx]
        # recompute t-batch, added by lgh
        if (isinstance(idx, (list, tuple)) or isinstance(idx, slice) or
            isinstance(idx, torch.Tensor) and (idx.dtype == torch.long or idx.dtype == torch.bool)
            and idx.size(0) > 1) and \
                data.num_events != self.num_events and self.use_t_batch:
            data.t_batch_valid = False
        return data

    def __setitem__(self, key, value):
        """Sets the attribute :obj:`key` to :obj:`value`."""
        setattr(self, key, value)

    @property
    def keys(self):
        return [key for key in self.__dict__.keys() if self[key] is not None]

    def __len__(self):
        return len(self.keys)

    def __contains__(self, key):
        return key in self.keys

    def __iter__(self):
        for key in sorted(self.keys):
            yield key, self[key]

    def __call__(self, *keys):
        for key in sorted(self.keys) if not keys else keys:
            if key in self:
                yield key, self[key]

    @property
    def num_nodes(self):
        return max(int(self.src.max()), int(self.dst.max())) + 1

    @property
    def num_events(self):
        return self.src.size(0)

    @property
    def num_pos_events(self):
        if hasattr(self, 'e_id'):
            return self.src[self.e_id != -1].size(0)
        else:
            return self.src.events(0)

    @property
    def num_batch(self):
        return len(self.batch_list)

    def __apply__(self, item, func):
        if torch.is_tensor(item):
            return func(item)
        elif isinstance(item, (tuple, list)):
            return [self.__apply__(v, func) for v in item]
        elif isinstance(item, dict):
            return {k: self.__apply__(v, func) for k, v in item.items()}
        else:
            return item

    def apply(self, func, *keys):
        r"""Applies the function :obj:`func` to all tensor attributes
        :obj:`*keys`. If :obj:`*keys` is not given, :obj:`func` is applied to
        all present attributes.
        """
        for key, item in self(*keys):
            self[key] = self.__apply__(item, func)
        return self

    def to(self, device, *keys, **kwargs):
        return self.apply(lambda x: x.to(device, **kwargs), *keys)

    def train_val_test_split(self, val_ratio=0.15, test_ratio=0.15):
        val_time, test_time = np.quantile(
            self.t.cpu().numpy(),
            [1. - val_ratio - test_ratio, 1. - test_ratio])

        val_idx = int((self.t <= val_time).sum())
        test_idx = int((self.t <= test_time).sum())

        return self[:val_idx], self[val_idx:test_idx], self[test_idx:]

    # modified by fdf
    def seq_batches(self, batch_size, useTBatch=True):
        # print('seq_batches called')
        # print('TBatch:{}'.format(useTBatch))

        # batchList = []
        if self.use_t_batch == False:
            for start in range(0, self.num_events, batch_size):
                # batchList.append(self[start:start + batch_size])
                yield self[start:start + batch_size]
        # else:
        #     batchIdxList = list()
        #     for i in range(0, self.num_events): batchIdxList.append(torch.Tensor())
        #
        #     maxTS = np.zeros(self.num_nodes)
        #
        #     for i in tqdm(range(0, self.num_events), desc='[Preparing.]'):
        #         nowTS = 1 + int(max(maxTS[self[i].src], maxTS[self[i].dst]))
        #         maxTS[self[i].src] = maxTS[self[i].dst] = nowTS
        #         batchIdxList[nowTS] = torch.cat((batchIdxList[nowTS], torch.tensor([i])))
        #
        #     for IdxList in batchIdxList:
        #         if IdxList.size(0) > 0:
        #             batchList.append(self[IdxList.to(torch.long)])
        #     # print('<finished.>')
        #     # batch : TemporalData(src,dst,t,e_id)
        else:
            assert self.t_batch_valid
            for item in self.batch_list:
                yield item

    def __cat_dim__(self, key, value):
        return 0

    def __inc__(self, key, value):
        return 0

    def __repr__(self):
        cls = str(self.__class__.__name__)
        shapes = ', '.join([f'{k}={list(v.shape)}' for k, v in self])
        return f'{cls}({shapes})'
