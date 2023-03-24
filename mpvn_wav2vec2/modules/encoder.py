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

import torch.nn as nn
from torch import Tensor
from typing import Tuple
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import json

class Wav2Vec2Encoder(nn.Module):
    def __init__(
            self,
            num_classes: int,
            load_pretrain: False
    ):
        super(Wav2Vec2Encoder, self).__init__()
        cache_dir = './cache/'
        self.processor = Wav2Vec2Processor.from_pretrained("nguyenvulebinh/wav2vec2-base-vietnamese-250h", cache_dir=cache_dir)
        if load_pretrain:
            self.wav2vec2_encoder = Wav2Vec2ForCTC.from_pretrained("nguyenvulebinh/wav2vec2-base-vietnamese-250h", cache_dir=cache_dir)
        else:
            self.wav2vec2_encoder = Wav2Vec2ForCTC(json.load(open('config_wav2vec2.json')))
        self.wav2vec2_encoder.lm_head = nn.Linear(in_features=768, out_features=num_classes, bias=True)

    def forward(self, signals, labels):
        input_values = self.processor(
            signals, 
            sampling_rate=16000, # hard code 
            return_tensors="pt",
            padding=True,
        ).input_values
        input_values = input_values.cuda()
        labels = labels + (labels == 0) * -100
        output = self.wav2vec2_encoder(input_values, labels=labels, output_hidden_states=True)
        loss = output.loss
        logits = output.logits
        hidden_states = output.hidden_states[0]
        
        return loss, logits, hidden_states
    
    # loss: Optional[torch.FloatTensor] = None
    # logits: torch.FloatTensor = None
    # hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    # attentions: Optional[Tuple[torch.FloatTensor]] = Non