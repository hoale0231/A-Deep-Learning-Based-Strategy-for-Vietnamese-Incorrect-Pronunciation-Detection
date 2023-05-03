import pytorch_lightning as pl
from mpvn.configs import DictConfig
from mpvn.modules.decoder import RNNDecoder, WordDecoder
from mpvn.modules.encoder import ConformerEncoder
from mpvn.vocabs.vocab import Vocabulary
import librosa
import numpy as np
import torch
import os

class ConformerRNNModel(pl.LightningModule):
    def __init__(
            self,
            configs: DictConfig,
            num_classes: int,
            vocab: Vocabulary,
            phone_map: dict
    ) -> None:
        super(ConformerRNNModel, self).__init__()
        self.configs = configs
        self.vocab = vocab
        self.phone_map = phone_map

        self.encoder = ConformerEncoder(
            num_classes=num_classes,
            input_dim=configs.num_mels,
            encoder_dim=configs.encoder_dim,
            num_layers=configs.num_encoder_layers,
            num_attention_heads=configs.num_attention_heads,
            feed_forward_expansion_factor=configs.feed_forward_expansion_factor,
            conv_expansion_factor=configs.conv_expansion_factor,
            input_dropout_p=configs.input_dropout_p,
            feed_forward_dropout_p=configs.feed_forward_dropout_p,
            attention_dropout_p=configs.attention_dropout_p,
            conv_dropout_p=configs.conv_dropout_p,
            conv_kernel_size=configs.conv_kernel_size,
            half_step_residual=configs.half_step_residual,
            joint_ctc_attention=configs.joint_ctc_attention,
            half_subsampling=configs.half_subsampling
        )
        
        self.decoder = RNNDecoder(
            num_classes=num_classes,
            hidden_state_dim=configs.encoder_dim,
            eos_id=self.vocab.eos_id,
            space_id=self.vocab.space_id,
            num_heads=configs.num_attention_heads,
            dropout_p=configs.decoder_dropout_p,
            num_layers=configs.num_decoder_layers,
            rnn_type=configs.rnn_type
        )
        
        self.word_decoder = WordDecoder(
            num_classes=2,
            hidden_state_dim=configs.encoder_dim,
            num_heads=configs.num_attention_heads,
            dropout_p=configs.decoder_dropout_p,
        )
        
        os.makedirs('log', exist_ok=True)
        os.makedirs('upload', exist_ok=True)
        
    def load_audio(self, audio_path: str):
        signal, sr = librosa.load(audio_path, sr=self.configs.sample_rate)
        melspectrogram = librosa.feature.melspectrogram(
                    y=signal,
                    sr=self.configs.sample_rate,
                    n_mels=self.configs.num_mels,
                    n_fft=int(round(self.configs.sample_rate * 0.001 * self.configs.frame_length)),
                    hop_length=int(round(self.configs.sample_rate * 0.001 * self.configs.frame_shift)),
                )
        melspectrogram = librosa.power_to_db(melspectrogram, ref=np.max)
        melspectrogram -= melspectrogram.mean()
        melspectrogram /= np.std(melspectrogram)

        melspectrogram = torch.FloatTensor(melspectrogram).transpose(0, 1)
        return torch.stack([melspectrogram])

    def parse_transcript(self, transcript: str):
        words = transcript.lower().split()
        phonemes = [self.phone_map[word].replace(' ', '=') for word in words]
        phonemes = [self.vocab.sos_id] + self.vocab.string_to_label(' '.join(phonemes)) + [self.vocab.eos_id]
        return torch.tensor([phonemes])
            
    def predict(self, audio_path: str, transcript: str):
        audio = self.load_audio(audio_path)
        r_cs = self.parse_transcript(transcript)
        audio_length = audio.shape[1]
        _, encoder_outputs, _ = self.encoder(audio, audio_length)
        pr_out, _, mispronunciation_phone_features = self.decoder(r_cs, encoder_outputs)
        y_hats = pr_out.max(-1)[1]
        
        md_outputs = self.word_decoder(mispronunciation_phone_features)
        md_outputs = torch.exp(md_outputs)
        md_predict = [int(x[1] > 0.8) for x in md_outputs]
        
        # logging
        print(
            audio_path, transcript, ' '.join(map(str,md_predict)), 
            self.vocab.label_to_string(y_hats[0]).replace('   ', '=').replace(' ', '').replace('=', ' '), 
            sep=',', file=open('log/log.csv','a')
        )

        return md_predict


