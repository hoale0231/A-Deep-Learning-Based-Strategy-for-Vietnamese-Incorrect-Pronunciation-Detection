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
from torch.utils.data.sampler import Sampler

def _collate_fn(batch, pad_id: int = 0):
    """ functions that pad to the maximum sequence length """
    # compute maximum lengths
    max_r_o_length = max(len(s[1]) for s in batch)
    max_r_c_length = max(len(s[2]) for s in batch)

    # create tensors
    batch_size = len(batch)
    
    inputs = [s[0] for s in batch]
    r_os = torch.full((batch_size, max_r_o_length), pad_id, dtype=torch.long)
    r_cs = torch.full((batch_size, max_r_c_length), pad_id, dtype=torch.long)
    
    utt_ids = []
    scores = []

    # fill in tensors
    for i, (input, r_o, r_c, score, utt_id) in enumerate(batch):
        r_os[i, :len(r_o)] = torch.LongTensor(r_o)
        r_cs[i, :len(r_c)] = torch.LongTensor(r_c)
        scores.extend(score)
        utt_ids.append(utt_id)
 
    scores = torch.LongTensor(scores)

    return inputs, r_os, r_cs, scores, utt_ids


class AudioDataLoader(DataLoader):
    """ Audio Data Loader """
    def __init__(
            self,
            dataset: torch.utils.data.Dataset,
            num_workers: int,
            batch_sampler: torch.utils.data.sampler.Sampler,
            **kwargs,
    ) -> None:
        super(AudioDataLoader, self).__init__(
            dataset=dataset,
            num_workers=num_workers,
            batch_sampler=batch_sampler,
            **kwargs,
        )
        self.collate_fn = _collate_fn


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
