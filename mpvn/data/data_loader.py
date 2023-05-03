# MIT License
#
# Copyright (c) 2021 Soohwan Kim
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data.sampler import Sampler

def _collate_fn(is_weakly_s: bool = False):
    def collate_fn(batch, pad_id: int = 0):
        """ functions that pad to the maximum sequence length """
        inputs, r_os, r_cs, scores, utt_ids, is_L1s = zip(*batch)
        
        input_lengths = torch.IntTensor([len(s[0]) for s in batch])
        r_os_lengths = torch.IntTensor([len(s[1])-1 for s in batch])
        
        r_os = [torch.LongTensor(r_o) for r_o in r_os]
        r_cs = [torch.LongTensor(r_c) for r_c in r_cs]
        
        inputs = pad_sequence(inputs, batch_first=True, padding_value=0)
        r_os = pad_sequence(r_os, batch_first=True, padding_value=pad_id)
        r_cs = pad_sequence(r_cs, batch_first=True, padding_value=pad_id)
        L1_list = [idx for idx, is_L1 in enumerate(is_L1s) if is_L1]
        if is_weakly_s:
            scores = [torch.LongTensor(score) for score in scores]
            scores = pad_sequence(scores, batch_first=True, padding_value=-100)
        else:
            scores =  torch.LongTensor(sum(scores, []))
        return inputs, r_os, input_lengths, r_os_lengths, r_cs, scores, utt_ids, L1_list
    return collate_fn


class AudioDataLoader(DataLoader):
    """ Audio Data Loader """
    def __init__(
            self,
            dataset: torch.utils.data.Dataset,
            num_workers: int,
            batch_sampler: torch.utils.data.sampler.Sampler,
            is_weakly_s: bool = False,
            **kwargs,
    ) -> None:
        super(AudioDataLoader, self).__init__(
            dataset=dataset,
            num_workers=num_workers,
            batch_sampler=batch_sampler,
            **kwargs,
        )
        self.collate_fn = _collate_fn(is_weakly_s)


class BucketingSampler(Sampler):
    """ Samples batches assuming they are in order of size to batch similarly sized samples together. """
    def __init__(self, data_source, batch_size: int = 32, drop_last: bool = False) -> None:
        super(BucketingSampler, self).__init__(data_source)
        self.batch_size = batch_size
        self.data_source = data_source
        ids = list(range(0, len(data_source)))
        self.bins = [ids[i:i + batch_size] for i in range(0, len(ids), batch_size)]
        self.drop_last = drop_last

    def __iter__(self):
        for ids in self.bins:
            np.random.shuffle(ids)
            yield ids

    def __len__(self):
        return len(self.bins)

    def shuffle(self, epoch):
        np.random.shuffle(self.bins)
