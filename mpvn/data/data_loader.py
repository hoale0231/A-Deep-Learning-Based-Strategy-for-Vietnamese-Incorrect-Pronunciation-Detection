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
    input_lengths = [len(s[0]) for s in batch]
    r_os_lengths = [len(s[1]) - 1 for s in batch]

    max_inputs_sample = max(batch, key=lambda x: len(x[0]))[0]
    max_r_o_sample = max(batch, key=lambda x: len(x[1]))[1]
    max_r_c_sample = max(batch, key=lambda x: len(x[2]))[2]

    max_input_length = max_inputs_sample.size(0)
    max_r_o_length = len(max_r_o_sample)
    max_r_c_length = len(max_r_c_sample)

    feat_dim = max_inputs_sample.size(1)
    batch_size = len(batch)

    inputs = torch.zeros(batch_size, max_input_length, feat_dim)

    r_os = torch.zeros(batch_size, max_r_o_length).to(torch.long)
    r_os.fill_(pad_id)
    
    r_cs = torch.zeros(batch_size, max_r_c_length).to(torch.long)
    r_cs.fill_(pad_id)
    
    utt_ids = list()
    scores = []

    for x in range(batch_size):
        sample = batch[x]
        input, r_o, r_c, score, utt_id = sample
        seq_length = input.size(0)
        scores += score
        inputs[x].narrow(0, 0, seq_length).copy_(input)
        r_os[x].narrow(0, 0, len(r_o)).copy_(torch.LongTensor(r_o))
        r_cs[x].narrow(0, 0, len(r_c)).copy_(torch.LongTensor(r_c))
        # scores[x].narrow(0, 0, len(score)).copy_(torch.LongTensor(score))
        utt_ids.append(utt_id)
 
    scores = torch.LongTensor(scores)
    input_lengths = torch.IntTensor(input_lengths)
    r_os_lengths = torch.IntTensor(r_os_lengths)

    return inputs, r_os, input_lengths, r_os_lengths, r_cs, scores, utt_ids


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
