import torch
import torch.nn as nn
from torch import Tensor
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim import Adam, Adadelta, Adagrad, SGD, Adamax, AdamW, ASGD

import pytorch_lightning as pl
from typing import Dict, Union
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from mpvn.configs import DictConfig
from mpvn.metric import *
from mpvn.modules.decoder import RNNDecoder, WordDecoder
from mpvn.modules.encoder import ConformerEncoder
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
            vocab: Vocabulary = GradVocabulary,
            per_metric: WordErrorRate = WordErrorRate,
    ) -> None:
        super(ConformerRNNModel, self).__init__()
        self.configs = configs
        self.vocab = vocab
        self.per_metric = per_metric
        self.criterion = self.configure_criterion(
            ignore_index=self.vocab.pad_id,
            blank_id=self.vocab.blank_id,
            ctc_weight=configs.ctc_weight,
            cross_entropy_weight=configs.cross_entropy_weight,
            md_weight=configs.md_weight,
            pr_weight=configs.pr_weight,
            gamma=configs.gamma
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
            self.log(f"{stage}_per", per*100)
        if acc != None:
            self.log(f"{stage}_acc", acc*100)
        if f1 != None:
            self.log(f"{stage}_f1", f1*100)
        if precision != None:
            self.log(f"{stage}_precision", precision*100)
        if recall != None:
            self.log(f"{stage}_recall", recall*100)
          
    def forward(self, inputs, r_os, input_lengths, r_os_lengths, r_cs, scores): 
        # Forward encoder   
        encoder_log_probs, encoder_outputs, encoder_output_lengths = self.encoder(inputs, input_lengths)
        
        # Forward phone decoder
        train_md = self.configs.md_weight > 0
        pr_outputs, attn_encoder_decoder, mispronunciation_phone_features = self.decoder(r_os, encoder_outputs, train_md)
        
        # Get mispronunciation_phone_features with r_cs if pronunciation errors are synthetic
        if train_md and torch.any(r_cs != r_os):
            _, _, mispronunciation_phone_features = self.decoder(r_cs, encoder_outputs, train_md)
        
        # Forward word decoder
        md_outputs = self.word_decoder(mispronunciation_phone_features) if train_md else None
        
        # Calc loss
        max_target_length = r_os.size(1) - 1  # minus the start of sequence symbol
        pr_outputs = pr_outputs[:, :max_target_length, :]
        loss, pr_loss, md_loss = self.criterion(
            encoder_log_probs=encoder_log_probs.transpose(0, 1),
            pr_log_probs=pr_outputs.contiguous().view(-1, pr_outputs.size(-1)),
            encoder_output_lengths=encoder_output_lengths,
            r_os=r_os[:, 1:],
            r_os_lengths=r_os_lengths,
            md_log_probs=md_outputs.contiguous().view(-1, md_outputs.size(-1)) if md_outputs != None else None,
            score=scores
        )

        return loss, pr_loss, md_loss, encoder_log_probs, pr_outputs, md_outputs, attn_encoder_decoder
                
    def training_step(self, batch: tuple, batch_idx: int) -> Tensor:
        inputs, r_os, input_lengths, r_os_lengths, r_cs, scores, utt_ids = batch
        loss, pr_loss, md_loss, encoder_log_probs, pr_outputs, md_outputs, attn_encoder_decoder = self.forward(
            inputs, r_os, input_lengths, r_os_lengths, r_cs, scores
        )
        self._log_states('train', loss=loss, pr_loss=pr_loss, md_loss=md_loss)
        return loss

    def validation_step(self, batch: tuple, batch_idx: int) -> Tensor:
        inputs, r_os, input_lengths, r_os_lengths, r_cs, scores, utt_ids = batch
        loss, pr_loss, md_loss, encoder_log_probs, pr_outputs, md_outputs, attn_encoder_decoder = self.forward(
            inputs, r_os, input_lengths, r_os_lengths, r_cs, scores
        )
        
        y_hats = pr_outputs.max(-1)[1]
        y_hats_encoder = encoder_log_probs.max(-1)[1]
        per = self.per_metric(r_os[:, 1:], y_hats)
        
        if self.configs.md_weight > 0:
            scores = scores.cpu()
            md_predict = md_outputs.max(-1)[1].cpu()
  
            acc = accuracy_score(scores, md_predict)
            f1_ = f1_score(scores, md_predict, pos_label=0)
            precision_ = precision_score(scores, md_predict, pos_label=0)
            recall_ = recall_score(scores, md_predict, pos_label=0)
        else:
            scores = md_predict = acc = f1_ = precision_ = recall_ = None
            
        if batch_idx == 0:
            print("\nResult of", utt_ids[0])
            print("EP:", y_hats_encoder[0].shape, self.vocab.label_to_string(y_hats_encoder[0]).replace('   ', '-').replace(' ', ''))
            print("PR:", y_hats[0].shape, self.vocab.label_to_string(y_hats[0]).replace('   ', '-').replace(' ', ''))
            print("Ro:", r_os[0, 1:].shape, self.vocab.label_to_string(r_os[0, 1:]).replace('   ', '-').replace(' ', ''))
            print("Rc:", r_cs[0, 1:].shape, self.vocab.label_to_string(r_cs[0, 1:]).replace('   ', '-').replace(' ', ''))
            print("Per:", per)
            
            if self.configs.md_weight > 0:
                print("MED output   :", md_predict)
                print("Score        :", scores)
                print("Accuracy:", acc)
                
            attn_encoder_decoder = torch.sum(attn_encoder_decoder, dim=0).detach().cpu()
            print("Decoder-Encoder Attention:", attn_encoder_decoder.shape)
            plt.imshow(attn_encoder_decoder, interpolation='none')
            plt.show()

        self._log_states('valid', loss=loss, per=per, acc=acc, f1=f1_, precision=precision_, recall=recall_)
        return loss

    def test_step(self, batch: tuple, batch_idx: int) -> Tensor:
        inputs, r_os, input_lengths, r_os_lengths, r_cs, scores, utt_ids = batch
        loss, pr_loss, md_loss, encoder_log_probs, pr_outputs, md_outputs, attn_encoder_decoder = self.forward(
            inputs, r_os, input_lengths, r_os_lengths, r_cs, scores
        )
        
        y_hats = pr_outputs.max(-1)[1]
        per = self.per_metric(r_os[:, 1:], y_hats)
        
        md_predict = md_outputs.max(-1)[1].cpu()
        scores = scores.cpu()
  
        acc = accuracy_score(scores, md_predict)
        f1_ = f1_score(scores, md_predict, pos_label=0)
        precision_ = precision_score(scores, md_predict, pos_label=0)
        recall_ = recall_score(scores, md_predict, pos_label=0)
        
        if batch_idx == 0:
            self.df = pd.DataFrame(
                columns= ['utt_id', 'phones', 'phone_gen', 'phones_predict', 'score', 
                 'score_predict', 'per', 'accuracy', 'f1', 'precision', 'recall']
            )
            
        self.df.loc[len(self.df)] = [
            utt_ids[0],
            self.vocab.label_to_string(r_os[0, 1:]).replace('   ', '-').replace(' ', ''),
            self.vocab.label_to_string(r_cs[0, 1:]).replace('   ', '-').replace(' ', ''),
            self.vocab.label_to_string(y_hats[0]).replace('   ', '-').replace(' ', ''),
            ' '.join([str(s) for s in scores.cpu().tolist()]),
            ' '.join([str(s) for s in md_outputs.max(-1)[1].cpu().tolist()]),
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
            pr_weight: float = 0.3,
            gamma: float = 1.0
    ) -> nn.Module:
        """ Configure criterion """
        return JointLoss(
            ignore_index=ignore_index,
            reduction="mean",
            blank_id=blank_id,
            cross_entropy_weight=cross_entropy_weight,
            ctc_weight=ctc_weight,
            md_weight=md_weight,
            pr_weight=pr_weight,
            gamma=gamma
        )
