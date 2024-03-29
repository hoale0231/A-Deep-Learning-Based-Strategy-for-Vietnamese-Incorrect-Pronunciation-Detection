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
# import pyloudnorm as pyln
# from glob import glob

# noise_file = glob('/home/hoa/mispronunciation-detection-for-vietnamese/Data/noise/*.wav')
# noises = [sf.read(file)[0] for file in noise_file]
# meter = pyln.Meter(16000)
# noise_loundness = [meter.integrated_loudness(noise) for noise in noises]

class AudioDataset(Dataset):
    """
    Dataset for audio & transcript matching

    Note:
        Do not use this class directly, use one of the sub classes.

    Args:
        dataset_path (str): path of librispeech dataset
        audio_paths (list): list of audio path
        transcripts (list): list of transript
        apply_spec_augment (bool): flag indication whether to apply spec augment or not
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
            text_gen: list,
            score: list,
            vocab: Vocabulary,
            phoneme_map: dict,
            auto_gen_score: list,
            apply_spec_augment: bool = False,
            add_noise: bool = False,
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
        self.list_phomemes = list(phoneme_map.values())
        self.vocab = vocab
        self.spec_augment_flags = [False] * len(self.audio_paths)
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
        # self.add_noise = add_noise
        self.n_fft = int(round(sample_rate * 0.001 * frame_length))
        self.hop_length = int(round(sample_rate * 0.001 * frame_shift))
        
        if text_gen is None:
            self.text_gen = [''] * len(self.utt_id)
        else:
            self.text_gen = list(text_gen)

        if apply_spec_augment:
            for idx in range(self.dataset_size):
                self.spec_augment_flags.append(True)
                self.utt_id.append(self.utt_id[idx])
                self.audio_paths.append(self.audio_paths[idx])
                self.transcripts.append(self.transcripts[idx])
                self.score.append(self.score[idx])
                self.auto_gen_score.append(self.auto_gen_score[idx])
                self.text_gen.append(self.text_gen[idx])

        self.vowels = """   a0 a1 a2 a3 a4 a5
                            ă0 ă1 ă2 ă3 ă4 ă5
                            â0 â1 â2 â3 â4 â5
                            ax0 ax1 ax2 ax3 ax4 ax5
                            e0 e1 e2 e3 e4 e5
                            ê0 ê1 ê2 ê3 ê4 ê5
                            i0 i1 i2 i3 i4 i5
                            iê0 iê1 iê2 iê3 iê4 iê5
                            o0 o1 o2 o3 o4 o5
                            ô0 ô1 ô2 ô3 ô4 ô5
                            ơ0 ơ1 ơ2 ơ3 ơ4 ơ5
                            u0 u1 u2 u3 u4 u5
                            ư0 ư1 ư2 ư3 ư4 ư5
                            ưa0 ưa1 ưa2 ưa3 ưa4 ưa5""".split() + ['']
        self.init_vowels = ["w", ""]
        self.init_consonants = "b m v n đ t l s d r k h p g th ph tr ch nh kh ng".split() + ['']
        self.final_consonants = "-p -t -k -m -n -i -w -k -ng".split() + ['']
        self.set_phones = { phones for phones in self.phone_map.values()}
        
        self.set_vowels = set(self.vowels)
        self.set_init_consonants = set(self.init_consonants)
        self.set_final_consonants = set(self.final_consonants)
        
        self.shuffle()

    def _spec_augment(self, feature: Tensor) -> Tensor:
        """
        Provides Spec Augment. A simple data augmentation method for speech recognition.
        This concept proposed in https://arxiv.org/abs/1904.08779
        """
        time_axis_length = feature.size(0)
        freq_axis_length = feature.size(1)
        time_mask_para = time_axis_length / 10      # Refer to "Specaugment on large scale dataset" paper

        # time mask
        for _ in range(self.time_mask_num):
            t = int(np.random.uniform(low=0.0, high=time_mask_para))
            t0 = random.randint(0, time_axis_length - t)
            feature[t0: t0 + t, :] = 0

        # freq mask
        for _ in range(self.freq_mask_num):
            f = int(np.random.uniform(low=0.0, high=self.freq_mask_para))
            f0 = random.randint(0, freq_axis_length - f)
            feature[:, f0: f0 + f] = 0

        return feature

    def _get_feature(self, signal: np.ndarray) -> np.ndarray:
        """
        Provides feature extraction

        Inputs:
            signal (np.ndarray): audio signal

        Returns:
            feature (np.ndarray): feature extract by sub-class
        """
        raise NotImplementedError

    def _parse_audio(self, audio_path: str, apply_spec_augment: bool) -> Tensor:
        """
        Parses audio.

        Args:
            audio_path (str): path of audio file
            apply_spec_augment (bool): flag indication whether to apply spec augment or not

        Returns:
            feature (np.ndarray): feature extract by sub-class
        """
        # print(audio_path)
        signal, sr = sf.read(audio_path)
        if sr != 16000:
            print(audio_path, file=open('file_invalid_sample_rate.txt', 'a'))
            signal = librosa.resample(signal, orig_sr=sr, target_sr=16000)
        
        # if self.add_noise and np.random.rand() > 0.5:
        #     try:
        #         i_noise = random.randrange(len(noises))
        #         rand_val = random.choice([10, 15, 20, 25])
        #         noise = noises[i_noise]
        #         n_loudness = noise_loundness[i_noise]
                
        #         s_loudness = meter.integrated_loudness(signal)
        #         input_snr = s_loudness - n_loudness
                
        #         scale_factor = 10**( (input_snr - rand_val)/20)
                
        #         if len(noise) == len(signal):
        #             signal = signal + noise * scale_factor
        #         elif len(noise) > len(signal):
        #             start = random.randrange(len(noise) - len(signal))
        #             signal = signal + noise[start:start+len(signal)] * scale_factor
        #         else:
        #             start = random.randrange(len(signal) - len(noise))
        #             signal[start:start+len(noise)] = signal[start:start+len(noise)] + noise * scale_factor
        #         if len(signal) == 0:
        #             raise
        #     except:
        #         sf.write(f"/home/hoa/mispronunciation-detection-for-vietnamese/Data/noise/combine/{rand_val}_{i_noise}_{audio_path.split('/')[-1]}", signal, 16000)
        #         signal, sr = sf.read(audio_path)
        
        feature = self._get_feature(signal)

        feature -= feature.mean()
        feature /= np.std(feature)

        feature = torch.FloatTensor(feature).transpose(0, 1)

        if apply_spec_augment:
            feature = self._spec_augment(feature)

        return feature

    def _parse_phonemes(self, phonemes: list) -> list:
        """
        Convert transcript to list phonemes id and add <sos> and <eos> tokens
        """
        return [self.sos_id] + self.vocab.string_to_label(' '.join(phonemes)) + [self.eos_id]
    
    def _random_replace(self, phones: str):
        phones = phones.split('=')
        init_cons, init_vowels, vowels, final_cons = '', '', '', ''
        for p in phones:
            if p in self.set_init_consonants:
                init_cons = p
            elif p in self.init_vowels:
                init_vowels = p
            elif p in self.set_vowels:
                vowels = p
            elif p in self.set_final_consonants:
                final_cons = p
              
        new_phones = []
        while len(new_phones) == 0 or new_phones == phones or ' '.join(new_phones) not in self.set_phones: 
            init_cons_ = init_cons
            init_vowels_ = init_vowels
            vowels_ = vowels
            final_cons_ = final_cons

            if np.random.rand() < 0.5:
                init_cons_ = np.random.choice(self.init_consonants)
            if np.random.rand() < 0.5:
                init_vowels_ = np.random.choice(self.init_vowels)
            if np.random.rand() < 0.5:
                vowels_ = np.random.choice(self.vowels)
            if np.random.rand() < 0.5:
                final_cons_ = np.random.choice(self.final_consonants)
            new_phones = [p for p in [init_cons_, init_vowels_, vowels_, final_cons_] if p != '']
            
        return '='.join(new_phones)
    
    
    def _random_score(self, phonemes: list, real_score: list, rand_factor: float):   
        replace = np.random.rand(len(phonemes)) < rand_factor
        phonemes_replaced = [self._random_replace(p) if (r and s) else p for p, r, s in zip(phonemes, replace, real_score)]
        score = [int(p == p_ and s) for p, p_, s in zip(phonemes, phonemes_replaced, real_score)]
        # for _ in range(5):
        #     if np.random.rand() < rand_factor or len(phonemes_replaced) == 0:
        #         idx_insert = np.random.randint(0, len(phonemes_replaced)+1)
        #         phonemes_insert = np.random.choice(self.list_phomemes).replace(' ', '=')
        #         if phonemes_insert not in phonemes_replaced:
        #             phonemes_replaced.insert(idx_insert, phonemes_insert)
        #             score.insert(idx_insert, 0)
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
        audio_feature = self._parse_audio(audio_path, self.spec_augment_flags[idx])
        phonemes = [self.phone_map[word].replace(' ', '=') for word in self.transcripts[idx].split()]
        r_o = self._parse_phonemes(phonemes)
        if self.auto_gen_score[idx]:
            if self.score[idx]:
                score = self._parse_score(self.score[idx])
                rand_factor = 0.25
            else:
                score = [1] * len(phonemes) 
                rand_factor = 0.5
            if 'lesson' in audio_path or "tôi" in self.transcripts[idx]:
                rand_factor = 0.5
            r_c, score = self._random_score(phonemes, score, rand_factor)
        else:
            if self.text_gen[idx]:
                r_c = self._parse_phonemes(self.phone_map[word].replace(' ', '=') for word in self.text_gen[idx].split())
            else:
                r_c = r_o
            score = self._parse_score(self.score[idx])
            # if len(self.transcripts[idx].split()) != len(score):
            #     raise Exception(f"{self.utt_id[idx]} {len(self.transcripts[idx].split())} {len(score)}")
        return audio_feature, r_o, r_c, score, self.utt_id[idx], self.score[idx] == '' or self.text_gen[idx] != ''

    def shuffle(self):
        tmp = list(zip(self.utt_id, self.audio_paths, self.transcripts, self.spec_augment_flags, self.score, self.auto_gen_score, self.text_gen))
        random.shuffle(tmp)
        self.utt_id, self.audio_paths, self.transcripts, self.spec_augment_flags, self.score, self.auto_gen_score, self.text_gen = zip(*tmp)

    def __len__(self):
        return len(self.audio_paths)

    def count(self):
        return len(self.audio_paths)


class FBankDataset(AudioDataset):
    """ Dataset for filter bank & transcript matching """
    def _get_feature(self, signal: np.ndarray) -> np.ndarray:
        return torchaudio.compliance.kaldi.fbank(
            Tensor(signal).unsqueeze(0),
            num_mel_bins=self.num_mels,
            frame_length=self.frame_length,
            frame_shift=self.frame_shift,
        ).transpose(0, 1).numpy()


class SpectrogramDataset(AudioDataset):
    """ Dataset for spectrogram & transcript matching """
    def _get_feature(self, signal: np.ndarray) -> np.ndarray:
        spectrogram = torch.stft(
            Tensor(signal), self.n_fft, hop_length=self.hop_length,
            win_length=self.n_fft, window=torch.hamming_window(self.n_fft),
            center=False, normalized=False, onesided=True
        )
        spectrogram = (spectrogram[:, :, 0].pow(2) + spectrogram[:, :, 1].pow(2)).pow(0.5)
        spectrogram = np.log1p(spectrogram.numpy())
        return spectrogram


class MelSpectrogramDataset(AudioDataset):
    """ Dataset for mel-spectrogram & transcript matching """
    def _get_feature(self, signal: np.ndarray) -> np.ndarray:
        melspectrogram = librosa.feature.melspectrogram(
            y=signal,
            sr=self.sample_rate,
            n_mels=self.num_mels,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
        )
        melspectrogram = librosa.power_to_db(melspectrogram, ref=np.max)
        return melspectrogram


class MFCCDataset(AudioDataset):
    """ Dataset for MFCC & transcript matching """
    def _get_feature(self, signal: np.ndarray) -> np.ndarray:
        return librosa.feature.mfcc(
            y=signal,
            sr=self.sample_rate,
            n_mfcc=self.num_mels,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
        )
