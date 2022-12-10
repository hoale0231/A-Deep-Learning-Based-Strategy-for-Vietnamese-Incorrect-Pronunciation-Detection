import torch
import torch.nn as nn
from torch import Tensor
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim import Adam, Adadelta, Adagrad, SGD, Adamax, AdamW, ASGD

import pytorch_lightning as pl
from typing import Dict, Union
import matplotlib.pyplot as plt

from mpvn.configs import DictConfig
from mpvn.metric import WordErrorRate
from mpvn.model.decoder import RNNDecoder, WordDecoder
from mpvn.model.encoder import ConformerEncoder
from mpvn.optim import AdamP, RAdam
from mpvn.optim.lr_scheduler import TransformerLRScheduler, TriStageLRScheduler
from mpvn.criterion.criterion import JointCTCCrossEntropyLoss, CrossEntropyLoss
from mpvn.vocabs import GradVocabulary
from mpvn.vocabs.vocab import Vocabulary

class ConformerRNNModel(pl.LightningModule):
    def __init__(
            self,
            configs: DictConfig,
            num_classes: int,
            num_words: int,
            vocab: Vocabulary = GradVocabulary,
            per_metric: WordErrorRate = WordErrorRate,
    ) -> None:
        super(ConformerRNNModel, self).__init__()
        self.configs = configs
        self.gradient_clip_val = configs.gradient_clip_val
        self.teacher_forcing_ratio = configs.teacher_forcing_ratio
        self.vocab = vocab
        self.per_metric = per_metric
        self.criterion = self.configure_criterion(
            ignore_index=self.vocab.pad_id,
            blank_id=self.vocab.blank_id,
            ctc_weight=configs.ctc_weight,
            cross_entropy_weight=configs.cross_entropy_weight,
        )

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
            num_heads=configs.num_attention_heads,
            dropout_p=configs.decoder_dropout_p,
            num_layers=configs.num_decoder_layers,
            rnn_type=configs.rnn_type
        )
        
        self.word_decoder = WordDecoder(
            num_classes=3,
            num_words=num_words,
            hidden_state_dim=configs.encoder_dim,
            num_heads=configs.num_attention_heads,
            dropout_p=configs.decoder_dropout_p,
        )
        
        self.med_criterion = CrossEntropyLoss(self.vocab.pad_id)
        
    def _log_states(self, stage: str, loss: float, cross_entropy_loss: float = None, ctc_loss: float = None, per: float = None) -> None:
        if per:
            self.log(f"{stage}_per", per)
        self.log(f"{stage}_loss", loss)
        if cross_entropy_loss:
            self.log(f"{stage}_cross_entropy_loss", cross_entropy_loss)
        if ctc_loss:
            self.log(f"{stage}_ctc_loss", ctc_loss)
                
    def training_step(self, batch: tuple, batch_idx: int) -> Tensor:
        inputs, targets, input_lengths, target_lengths, trans, phones, score, utt_id = batch
        
        encoder_log_probs, encoder_outputs, encoder_output_lengths = self.encoder(inputs, input_lengths)
        outputs, _, mispronunciation_phone_features = self.decoder(targets, encoder_outputs)
        MED_outputs, attn = self.word_decoder(trans, mispronunciation_phone_features)

        
        max_target_length = targets.size(1) - 1  # minus the start of sequence symbol
        outputs = outputs[:, :max_target_length, :]
        
        # loss, ctc_loss, cross_entropy_loss = self.criterion(
        #     encoder_log_probs=encoder_log_probs.transpose(0, 1),
        #     decoder_log_probs=outputs.contiguous().view(-1, outputs.size(-1)),
        #     output_lengths=encoder_output_lengths,
        #     targets=targets[:, 1:],
        #     target_lengths=target_lengths,
        # )
        
        # # self._log_states('train', loss, cross_entropy_loss, ctc_loss)
        self._log_states('train', loss)
        
        loss = self.med_criterion(MED_outputs.contiguous().view(-1, MED_outputs.size(-1)), score)

        
        return loss

    def validation_step(self, batch: tuple, batch_idx: int) -> Tensor:
        inputs, targets, input_lengths, target_lengths, trans, phones, score, utt_id = batch
        encoder_log_probs, encoder_outputs, encoder_output_lengths = self.encoder(inputs, input_lengths)
        outputs, attn, mispronunciation_phone_features = self.decoder(targets, encoder_outputs=encoder_outputs)
        MED_outputs, attn = self.word_decoder(trans, mispronunciation_phone_features)
        
        max_target_length = targets.size(1) - 1  # minus the start of sequence symbol
        outputs = outputs[:, :max_target_length, :]
        
        # loss, ctc_loss, cross_entropy_loss = self.criterion(
        #     encoder_log_probs=encoder_log_probs.transpose(0, 1),
        #     decoder_log_probs=outputs.contiguous().view(-1, outputs.size(-1)),
        #     output_lengths=encoder_output_lengths,
        #     targets=targets[:, 1:],
        #     target_lengths=target_lengths,
        # )
        
        y_hats = outputs.max(-1)[1]
        y_hats_encoder = encoder_log_probs.max(-1)[1]
        per = self.per_metric(targets[:, 1:], y_hats)
 
        # self._log_states('valid', per, loss, cross_entropy_loss, ctc_loss)
        self._log_states('valid', per, loss)
        loss = self.med_criterion(MED_outputs.contiguous().view(-1, MED_outputs.size(-1)), score)
        
        if batch_idx == 0:
            print("\n1 sample result")
            print("EP:", y_hats_encoder[0].shape, self.vocab.label_to_string(y_hats_encoder[0]).replace('   ', '-').replace(' ', ''))
            print("DP    :", y_hats[0].shape, self.vocab.label_to_string(y_hats[0]).replace('   ', '-').replace(' ', ''))
            print("Target:", targets[0, 1:].shape, self.vocab.label_to_string(targets[0, 1:]).replace('   ', '-').replace(' ', ''))
            print("MED output:", MED_outputs.max(-1)[1][0])
            print("Score:", score[0])
            print("Per:", per)
            print("Attention:", attn.shape)
            attn = torch.sum(attn, dim=0).detach().cpu()
            print(attn.shape)
            plt.imshow(attn, interpolation='none')
            plt.show()
            self.save_attn = attn

        return loss

    def test_step(self, batch: tuple, batch_idx: int) -> Tensor:
        inputs, targets, input_lengths, target_lengths, trans, phones, score, utt_id = batch
        
        encoder_log_probs, encoder_outputs, encoder_output_lengths = self.encoder(inputs, input_lengths)
        outputs, _, mispronunciation_phone_features = self.decoder(targets, encoder_outputs=encoder_outputs)
        MED_outputs = self.word_decoder(trans, mispronunciation_phone_features)
        
        max_target_length = targets.size(1) - 1  # minus the start of sequence symbol
        outputs = outputs[:, :max_target_length, :]
        
        # loss, ctc_loss, cross_entropy_loss = self.criterion(
        #     encoder_log_probs=encoder_log_probs.transpose(0, 1),
        #     decoder_log_probs=outputs.contiguous().view(-1, outputs.size(-1)),
        #     output_lengths=encoder_output_lengths,
        #     targets=targets[:, 1:],
        #     target_lengths=target_lengths,
        # )
        
        y_hats = outputs.max(-1)[1]
        per = self.per_metric(targets[:, 1:], y_hats)
        
        # self._log_states('test', per, loss, cross_entropy_loss, ctc_loss)
        self._log_states('test', per, loss)
        with open('test.result', 'a') as file_result:
            print(
                utt_id,
                per,
                self.vocab.label_to_string(y_hats[0]).replace('   ', '-').replace(' ', ''),
                self.vocab.label_to_string(targets[0, 1:]).replace('   ', '-').replace(' ', ''),
                sep=',' ,
                file=file_result
            )
            
        loss = self.med_criterion(MED_outputs, score)
        

        return loss

    def configure_optimizers(self) -> Dict[str, Union[torch.optim.Optimizer, object, str]]:
        """ Configure optimizer """
        supported_optimizers = {
            "adam": Adam,
            "adamp": AdamP,
            "radam": RAdam,
            "adagrad": Adagrad,
            "adadelta": Adadelta,
            "adamax": Adamax,
            "adamw": AdamW,
            "sgd": SGD,
            "asgd": ASGD,
        }
        assert self.configs.optimizer in supported_optimizers.keys(), \
            f"Unsupported Optimizer: {self.configs.optimizer}\n" \
            f"Supported Optimizers: {supported_optimizers.keys()}"
        optimizer = supported_optimizers[self.configs.optimizer](self.parameters(), lr=self.configs.lr)

        if self.configs.scheduler == 'transformer':
            scheduler = TransformerLRScheduler(
                optimizer,
                peak_lr=self.configs.peak_lr,
                final_lr=self.configs.final_lr,
                final_lr_scale=self.configs.final_lr_scale,
                warmup_steps=self.configs.warmup_steps,
                decay_steps=self.configs.decay_steps,
            )
        elif self.configs.scheduler == 'tri_stage':
            scheduler = TriStageLRScheduler(
                optimizer,
                init_lr=self.configs.init_lr,
                peak_lr=self.configs.peak_lr,
                final_lr=self.configs.final_lr,
                final_lr_scale=self.configs.final_lr_scale,
                init_lr_scale=self.configs.init_lr_scale,
                warmup_steps=self.configs.warmup_steps,
                total_steps=self.configs.warmup_steps + self.configs.decay_steps,
            )
        elif self.configs.scheduler == 'reduce_lr_on_plateau':
            scheduler = ReduceLROnPlateau(
                optimizer,
                patience=self.configs.lr_patience,
                factor=self.configs.lr_factor,
            )
        else:
            raise ValueError(f"Unsupported `scheduler`: {self.configs.scheduler}\n"
                             f"Supported `scheduler`: transformer, tri_stage, reduce_lr_on_plateau")

        return {
            'optimizer': optimizer,
            'scheduler': scheduler,
            'monitor': 'metric_to_track',
        }

    def configure_criterion(
            self,
            ignore_index: int,
            ctc_weight: float = 0.3,
            cross_entropy_weight: float = 0.7,
            blank_id: int = None
    ) -> nn.Module:
        """ Configure criterion """
        return JointCTCCrossEntropyLoss(
            ignore_index=ignore_index,
            reduction="mean",
            blank_id=blank_id,
            cross_entropy_weight=cross_entropy_weight,
            ctc_weight=ctc_weight,
        )

