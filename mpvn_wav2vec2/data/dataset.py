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

import os
import random
import librosa
import soundfile as sf
import torch
import torchaudio
import numpy as np
from torch import Tensor
from torch.utils.data import Dataset
from mpvn.vocabs.vocab import Vocabulary

class AudioDataset(Dataset):
    """
    Dataset for audio & transcript matching

    Note:
        Do not use this class directly, use one of the sub classes.

    Args:
        dataset_path (str): path of librispeech dataset
        audio_paths (list): list of audio path
        transcripts (list): list of transript
        sos_id (int): identification of <sos>
        eos_id (int): identification of <eos>
        sample_rate (int): sampling rate of audio
        num_mels (int): the number of mfc coefficients to retain.
        frame_length (float): frame length for spectrogram (ms)
        frame_shift (float): length of hop between STFT (short time fourier transform) windows.
        freq_mask_para (int): hyper Parameter for freq masking to limit freq masking length
        time_mask_num (int): how many time-masked area to make
        freq_mask_num (int): how many freq-masked area to make
    """

    def __init__(
            self,
            dataset_path: str,
            utt_id: list,
            audio_paths: list,
            transcripts: list,
            score: list,
            vocab: Vocabulary,
            phoneme_map: dict,
            auto_gen_score: list,
            sample_rate: int = 16000,
            num_mels: int = 80,
            frame_length: float = 25.0,
            frame_shift: float = 10.0,
            freq_mask_para: int = 27,
            time_mask_num: int = 4,
            freq_mask_num: int = 2,
    ) -> None:
        super(AudioDataset, self).__init__()
        self.dataset_path = dataset_path
        self.utt_id = list(utt_id)
        self.audio_paths = list(audio_paths)
        self.transcripts = list(transcripts)
        self.score = list(score)
        self.auto_gen_score = list(auto_gen_score)
        self.phone_map = phoneme_map
        self.vocab = vocab
        self.dataset_size = len(self.audio_paths)
        self.sos_id = vocab.sos_id
        self.eos_id = vocab.eos_id
        self.sample_rate = sample_rate
        self.num_mels = num_mels
        self.frame_length = frame_length
        self.frame_shift = frame_shift
        self.freq_mask_para = freq_mask_para
        self.time_mask_num = time_mask_num
        self.freq_mask_num = freq_mask_num
        self.n_fft = int(round(sample_rate * 0.001 * frame_length))
        self.hop_length = int(round(sample_rate * 0.001 * frame_shift))

        self.vowels = """   iə iə2 iəɜ iə4 iə5 iə6
                            iɛ iɛ1 iɛ2 iɛɜ iɛ4 iɛ5 iɛ6
                            i i2 iɜ i4 i5 i6
                            e e1 e2 eɜ e4 e5 e6 e7
                            ɛ ɛ2 ɛɜ ɛ4 ɛ5 ɛ6
                            yə yə2 yəɜ yə4 yə5 yə6
                            y y2 yɜ y4 y5 y6
                            əː əː2 əːɜ əː4 əː5 əː6
                            ə ə1 ə2 əɜ ə4 ə5 ə6
                            aː aː2 aːɜ aː4 aː5 aː6
                            a a2 aɜ a4 a5 a6
                            uə uə2 uəɜ uə4 uə5 uə6
                            u u2 uɜ u4 u5 u6
                            o o2 oɜ o4 o5 o6
                            ɔ ɔ2 ɔɜ ɔ4 ɔ5 ɔ6
                            əɪ əɪ2 əɪɜ əɪ4 əɪ5 əɪ6""".split()
        self.init_consonants = "b ɗ d f ɣ h k x l m n ɲ ŋ p s t̪ t tʃ v z".split()
        self.final_consonants = "j m n ɲ ŋ p k t̪ w c".split()
        
        self.shuffle()

    def _parse_audio(self, audio_path: str) -> Tensor:
        """
        Parses audio.

        Args:
            audio_path (str): path of audio file

        Returns:
            feature (np.ndarray): feature extract by sub-class
        """
        signal, sr = sf.read(audio_path)
        if sr != 16000:
            print(audio_path, file=open('file_invalid_sample_rate.txt', 'a'))
            signal = librosa.resample(signal, orig_sr=sr, target_sr=16000)

        return signal

    def _parse_phonemes(self, phonemes: list) -> list:
        """
        Convert transcript to list phonemes id and add <sos> and <eos> tokens
        """
        return [self.sos_id] + self.vocab.string_to_label(' '.join(phonemes)) + [self.eos_id]
    
    def _random_replace(self, phones: str):
        phones = phones.split('-')
        new_phones = []
        for i, p in enumerate(phones):
            p_ = p
            if np.random.rand() < 0.5:
                if i == 0:
                    if p in self.init_consonants:
                        p_ = np.random.choice(tuple(set(self.init_consonants) - {p}))
                    elif p in self.vowels:
                        p_ = np.random.choice(tuple(set(self.vowels) - {p}))
                elif i == len(phones) - 2:
                    if p in self.vowels:
                        p_ = np.random.choice(tuple(set(self.vowels) - {p}))
                elif i == len(phones) - 1:
                    if p in self.final_consonants:
                        p_ = np.random.choice(tuple(set(self.final_consonants) - {p}))
                    elif p in self.vowels:
                        p_ = np.random.choice(tuple(set(self.vowels) - {p}))
            new_phones.append(p_)
        return '-'.join(new_phones)  
    
    def _random_score(self, phonemes: list, real_score: list, rand_factor: float):   
        replace = np.random.rand(len(phonemes)) < rand_factor
        phonemes_replaced = [self._random_replace(p) if (r and s) else p for p, r, s in zip(phonemes, replace, real_score)]
        score = [int(p == p_ and s) for p, p_, s in zip(phonemes, phonemes_replaced, real_score)]
        return self._parse_phonemes(phonemes_replaced), score
    
    def _parse_score(self, score: str) -> list:
        return [int(s) for s in score.split()]

    def __getitem__(self, idx):
        """
        Return:
            - audio_features: extract from .wav file
            - r_o: canonical phonemes that the speaker was expected to pronounce
            - sent_c: transcripts that the speaker was expected to pronounce
            - r_c: canonical phonemes that the speaker was expected to pronounce
            - score: score of transcripts
            - utt_id: id of sample, help in logging
        """
        audio_path = os.path.join(self.dataset_path, self.audio_paths[idx])
        audio_feature = self._parse_audio(audio_path)
        phonemes = [self.phone_map[word].replace(' ', '-') for word in self.transcripts[idx].split()]
        r_o = self._parse_phonemes(phonemes)
        if self.auto_gen_score[idx]:
            if self.score[idx]:
                score = self._parse_score(self.score[idx])
                rand_factor = 0.4
            else:
                score = [1] * len(phonemes) 
                rand_factor = 0.6
            r_c, score = self._random_score(phonemes, score, rand_factor)
        else:
            r_o = r_c
            score = self._parse_score(self.score[idx])
            if len(self.transcripts[idx].split()) != len(score):
                raise Exception(f"{self.utt_id[idx]} {len(self.transcripts[idx].split())} {len(score)}")
        return audio_feature, r_o, r_c, score, self.utt_id[idx]

    def shuffle(self):
        tmp = list(zip(self.utt_id, self.audio_paths, self.transcripts, self.score, self.auto_gen_score))
        random.shuffle(tmp)
        self.utt_id, self.audio_paths, self.transcripts, self.score, self.auto_gen_score = zip(*tmp)

    def __len__(self):
        return len(self.audio_paths)

    def count(self):
        return len(self.audio_paths)

