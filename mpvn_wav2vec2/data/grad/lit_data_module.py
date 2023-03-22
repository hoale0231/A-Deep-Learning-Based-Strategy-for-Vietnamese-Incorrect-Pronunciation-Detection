import json
import logging
import pandas as pd
import pytorch_lightning as pl
from typing import Union, List, Tuple, Optional
from torch.utils.data import DataLoader

from mpvn.vocabs.vocab import Vocabulary
from mpvn.vocabs import GradVocabulary, WordVocabulary

from mpvn_wav2vec2.configs import DictConfig
from mpvn_wav2vec2.data.dataset import AudioDataset

from mpvn_wav2vec2.data.data_loader import (
    BucketingSampler,
    AudioDataLoader,
)

class LightningGradDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning Data Module for Grad Dataset.

    Args:
        configs (DictConfig): configuraion set

    Attributes:
        dataset_path (str): path of grad dataset
        max_epochs (int): the number of max epochs
        batch_size (int): the size of batch samples
        num_workers (int): the number of cpu workers
        sample_rate (int): sampling rate of audio
        num_mels (int): the number of mfc coefficients to retain.
        frame_length (float): frame length for spectrogram (ms)
        frame_shift (float): length of hop between STFT (short time fourier transform) windows.
        freq_mask_para (int): hyper Parameter for freq masking to limit freq masking length
        time_mask_num (int): how many time-masked area to make
        freq_mask_num (int): how many freq-masked area to make
    """
    def __init__(self, configs: DictConfig) -> None:
        super(LightningGradDataModule, self).__init__()
        self.dataset_path = configs.dataset_path
        self.grad_dir = 'Grad'
        self.manifest_paths = [
            f"{configs.dataset_path}/train.csv",
            f"{configs.dataset_path}/dev.csv",
            f"{configs.dataset_path}/test.csv",
            f"{configs.dataset_path}/label_train.csv",
            f"{configs.dataset_path}/label_test.csv",
            f"{configs.dataset_path}/label_valid.csv",
        ]
        self.dataset = dict()
        self.batch_size = configs.batch_size
        self.max_epochs = configs.max_epochs
        self.num_workers = configs.num_workers
        self.sample_rate = configs.sample_rate
        self.num_mels = configs.num_mels
        self.frame_length = configs.frame_length
        self.frame_shift = configs.frame_shift
        self.freq_mask_para = configs.freq_mask_para
        self.time_mask_num = configs.time_mask_num
        self.freq_mask_num = configs.freq_mask_num
        self.train_set = configs.train_set
        self.test_set = configs.test_set
        self.valid_set = configs.valid_set
        self.logger = logging.getLogger(__name__)

        self.audio_dataset = AudioDataset

    def get_vocab(self) -> Vocabulary:
        """
        Get Vocabulary of dataset
        Returns:
            Vocabulary
        """
        self.vocab = GradVocabulary(f"{self.dataset_path}/token.txt")
        self.phone_map = json.load(open(f"{self.dataset_path}/phone_map.json"))
        return self.vocab

    def setup(self, stage: Optional[str] = None) -> None:
        """ Split dataset into train, valid, and test. """
        if not self.vocab:
            self.vocab = GradVocabulary(f"{self.dataset_path}/token.txt")
        
        splits = ['train', 'dev', 'test', 'label_train', 'label_test', 'label_valid']
        for path, split in zip(self.manifest_paths, splits):
            df = pd.read_csv(path)
            utt_id, audio_paths, transcripts, score, gen_score = df.utt_id, df.path, df.text, df.score, df.gen_score
            self.dataset[split] = self.audio_dataset(
                dataset_path=self.dataset_path,
                utt_id=utt_id,
                audio_paths=audio_paths,
                transcripts=transcripts,
                score=score,
                vocab=self.vocab,
                phoneme_map=self.phone_map,
                auto_gen_score=gen_score,
                sample_rate=self.sample_rate,
                num_mels=self.num_mels,
                frame_length=self.frame_length,
                frame_shift=self.frame_shift,
                freq_mask_para=self.freq_mask_para,
                freq_mask_num=self.freq_mask_num,
                time_mask_num=self.time_mask_num,
            )

    def train_dataloader(self) -> DataLoader:
        train_sampler = BucketingSampler(self.dataset[self.train_set], batch_size=self.batch_size)
        return AudioDataLoader(
            dataset=self.dataset[self.train_set],
            num_workers=self.num_workers,
            batch_sampler=train_sampler,
        )

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        val_clean_sampler = BucketingSampler(self.dataset[self.valid_set], batch_size=1)
        return AudioDataLoader(
                dataset=self.dataset[self.valid_set],
                num_workers=self.num_workers,
                batch_sampler=val_clean_sampler,
            )

    def test_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        test_clean_sampler = BucketingSampler(self.dataset[self.test_set], batch_size=1)
        return AudioDataLoader(
                dataset=self.dataset[self.test_set],
                num_workers=self.num_workers,
                batch_sampler=test_clean_sampler,
            )
