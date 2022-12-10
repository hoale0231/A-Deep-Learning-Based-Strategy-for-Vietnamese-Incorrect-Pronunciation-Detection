import torch
import torch.nn as nn
from torch import Tensor
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim import Adam, Adadelta, Adagrad, SGD, Adamax, AdamW, ASGD

import pytorch_lightning as pl
from typing import Dict, Union
import matplotlib.pyplot as plt
import pandas as pd

from mpvn.configs import DictConfig
from mpvn.metric import *
from mpvn.model.decoder import RNNDecoder, WordDecoder
from mpvn.model.encoder import ConformerEncoder
from mpvn.optim import AdamP, RAdam
from mpvn.optim.lr_scheduler import TransformerLRScheduler, TriStageLRScheduler
from mpvn.criterion.criterion import JointLoss
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
            md_weight=configs.md_weight,
            pr_weight=configs.pr_weight
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
        
    def _log_states(
        self, 
        stage: str, 
        loss: float, 
        pr_loss: float = None, 
        md_loss: float = None, 
        per: float = None, 
        acc: float = None,
        f1: float = None,
        precision: float = None,
        recall: float = None
    ) -> None:
        self.log(f"{stage}_loss", loss)
        if pr_loss:
            self.log(f"{stage}_pr_loss", pr_loss)
        if md_loss:
            self.log(f"{stage}_md_loss", md_loss)
        if per != None:
            self.log(f"{stage}_per", per)
        if acc != None:
            self.log(f"{stage}_acc", acc)
        if f1 != None:
            self.log(f"{stage}_f1", f1)
        if precision != None:
            self.log(f"{stage}_precision", precision)
        if recall != None:
            self.log(f"{recall}_acc", recall)
          
    def forward(self, inputs, r_os, input_lengths, r_os_lengths, sent_cs, r_cs, scores):        
        encoder_log_probs, encoder_outputs, encoder_output_lengths = self.encoder(inputs, input_lengths)
        pr_outputs, attn_encoder_decoder, _ = self.decoder(r_os, encoder_outputs)
        _, _, mispronunciation_phone_features = self.decoder(r_cs, encoder_outputs)
        md_outputs, md_attn = self.word_decoder(sent_cs, mispronunciation_phone_features)

        max_target_length = r_os.size(1) - 1  # minus the start of sequence symbol
        pr_outputs = pr_outputs[:, :max_target_length, :]
        
        loss, pr_loss, md_loss = self.criterion(
            encoder_log_probs=encoder_log_probs.transpose(0, 1),
            pr_log_probs=pr_outputs.contiguous().view(-1, pr_outputs.size(-1)),
            encoder_output_lengths=encoder_output_lengths,
            r_os=r_os[:, 1:],
            r_os_lengths=r_os_lengths,
            md_log_probs=md_outputs.contiguous().view(-1, md_outputs.size(-1)),
            score=scores
        )
        
        return loss, pr_loss, md_loss, encoder_log_probs, pr_outputs, md_outputs, attn_encoder_decoder, md_attn
                
    def training_step(self, batch: tuple, batch_idx: int) -> Tensor:
        inputs, r_os, input_lengths, r_os_lengths, sent_cs, r_cs, scores, utt_ids = batch
        loss, pr_loss, md_loss, encoder_log_probs, pr_outputs, md_outputs, attn_encoder_decoder, md_attn = self.forward(
            inputs, r_os, input_lengths, r_os_lengths, sent_cs, r_cs, scores
        )
        self._log_states('train', loss=loss, pr_loss=pr_loss, md_loss=md_loss)
        return loss

    def validation_step(self, batch: tuple, batch_idx: int) -> Tensor:
        inputs, r_os, input_lengths, r_os_lengths, sent_cs, r_cs, scores, utt_ids = batch
        loss, pr_loss, md_loss, encoder_log_probs, pr_outputs, md_outputs, attn_encoder_decoder, md_attn = self.forward(
            inputs, r_os, input_lengths, r_os_lengths, sent_cs, r_cs, scores
        )
        
        y_hats = pr_outputs.max(-1)[1]
        y_hats_encoder = encoder_log_probs.max(-1)[1]
        per = self.per_metric(r_os[:, 1:], y_hats)
        
        md_predict = (md_outputs.max(-1)[1] != 1) + 1
        scores_lenghts = torch.sum(scores!=self.vocab.pad_id, axis=1)
        acc = accuracy(y=scores, y_hat=md_predict, length=scores_lenghts)
        f1_ = f1(y=scores, y_hat=md_predict, length=scores_lenghts)
        precision_ = precision(y=scores, y_hat=md_predict, length=scores_lenghts)
        recall_ = recall(y=scores, y_hat=md_predict, length=scores_lenghts)
 
        if batch_idx == 0:
            print("\nSample result")
            print("EP:", y_hats_encoder[0].shape, self.vocab.label_to_string(y_hats_encoder[0]).replace('   ', '-').replace(' ', ''))
            print("PR       :", y_hats[0].shape, self.vocab.label_to_string(y_hats[0]).replace('   ', '-').replace(' ', ''))
            print("Target   :", r_os[0, 1:].shape, self.vocab.label_to_string(r_os[0, 1:]).replace('   ', '-').replace(' ', ''))
            print("Per:", per)
            
            print("MED output   :", md_outputs.max(-1)[1][0])
            print("Score        :", scores[0])
            
            print("Accuracy:", acc)
                    
            md_attn = torch.sum(md_attn, dim=0).detach().cpu()
            print("Attention:", md_attn.shape)
            plt.imshow(md_attn, interpolation='none')
            plt.show()

        self._log_states('valid', loss=loss, per=per, acc=acc, f1=f1_, precision=precision_, recall=recall_)
        return loss

    def test_step(self, batch: tuple, batch_idx: int) -> Tensor:
        inputs, r_os, input_lengths, r_os_lengths, sent_cs, r_cs, scores, utt_ids = batch
        loss, pr_loss, md_loss, encoder_log_probs, pr_outputs, md_outputs, attn_encoder_decoder, md_attn = self.forward(
            inputs, r_os, input_lengths, r_os_lengths, sent_cs, r_cs, scores
        )
        
        y_hats = pr_outputs.max(-1)[1]
        y_hats_encoder = encoder_log_probs.max(-1)[1]
        per = self.per_metric(r_os[:, 1:], y_hats)
        
        scores_lenghts = torch.sum(scores!=self.vocab.pad_id, axis=1)
        md_predict = md_outputs.max(-1)[1]
        acc = accuracy(y=scores, y_hat=md_predict, length=scores_lenghts)
        f1_ = f1(y=scores, y_hat=md_predict, length=scores_lenghts)
        precision_ = precision(y=scores, y_hat=md_predict, length=scores_lenghts)
        recall_ = recall(y=scores, y_hat=md_predict, length=scores_lenghts)
        
        if batch_idx == 0:
            self.df = pd.DataFrame(
                columns= ['utt_id', 'phones', 'phones_predict', 'score', 
                 'score_predict', 'per', 'accuracy', 'f1', 'precision', 'recall']
            )
        
        self.df.loc[len(self.df)] = [
            utt_ids[0], 
            self.vocab.label_to_string(r_os[0, 1:]).replace('   ', '-').replace(' ', ''),
            r_os[0, 1:].shape, self.vocab.label_to_string(r_os[0, 1:]).replace('   ', '-').replace(' ', ''),
            scores[0],
            md_outputs.max(-1)[1][0],
            per, acc, f1_, precision_, recall_
        ]
            
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
            blank_id: int = None,
            md_weight: float = 0.7,
            pr_weight: float = 0.3
    ) -> nn.Module:
        """ Configure criterion """
        return JointLoss(
            ignore_index=ignore_index,
            reduction="mean",
            blank_id=blank_id,
            cross_entropy_weight=cross_entropy_weight,
            ctc_weight=ctc_weight,
            md_weight=md_weight,
            pr_weight=pr_weight
        )
