from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import pytorch_lightning as pl
import torch

class GradDataModule(pl.LightningDataModule):
    def __init__(self, train_dataset, val_dataset, test_dataset, batch_size=32):
        super().__init__()
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.batch_size = batch_size

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, batch_size=self.batch_size, collate_fn=self._padify, shuffle=True, num_workers=8
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, batch_size=self.batch_size, collate_fn=self._padify, num_workers=8
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset, batch_size=self.batch_size, collate_fn=self._padify, num_workers=8
        )

    def _padify(self, batch):
        _, _, audioId, textId, phonemeId = 0, 1, 2, 3, 4
        audios = [i[audioId][0] for i in batch]
        audio_lens = torch.LongTensor([i.shape[0] for i in audios])
        
        # texts = [torch.IntTensor(i[textId]) for i in batch]
        # text_lens = [i.shape for i in texts]
        
        phonemes = [torch.IntTensor(i[phonemeId]) for i in batch]
        phone_lens = torch.LongTensor([i.shape[0] for i in phonemes]) 
        # print("CC", phone_lens)        
        return (
            (pad_sequence(audios, batch_first=True), audio_lens),
            (textId, 0), 
            (pad_sequence(phonemes, batch_first=True), phone_lens)
        )
            
