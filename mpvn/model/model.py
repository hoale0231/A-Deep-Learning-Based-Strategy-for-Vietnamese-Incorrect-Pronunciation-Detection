import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch import Tensor
from typing import Dict, Union
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim import Adam, Adadelta, Adagrad, SGD, Adamax, AdamW, ASGD
import numpy as np
import matplotlib.pyplot as plt

from mpvn.configs import DictConfig
from mpvn.metric import WordErrorRate, CharacterErrorRate
from mpvn.model.decoder import SpeechTransformerDecoder, DecoderRNN, DecoderTransformer
from mpvn.model.encoder import ConformerEncoder
from mpvn.optim import AdamP, RAdam
from mpvn.optim.lr_scheduler import TransformerLRScheduler, TriStageLRScheduler
from mpvn.criterion.criterion import JointCTCCrossEntropyLoss
from mpvn.vocabs import GradVocabulary
from mpvn.vocabs.vocab import Vocabulary


class ConformerTransformerModel(pl.LightningModule):
    def __init__(
            self,
            configs: DictConfig,
            num_classes: int,
            vocab: Vocabulary = GradVocabulary,
            per_metric: WordErrorRate = WordErrorRate,
    ) -> None:
        super(ConformerTransformerModel, self).__init__()
        self.configs = configs
        self.gradient_clip_val = configs.gradient_clip_val
        self.teacher_forcing_ratio = configs.teacher_forcing_ratio
        self.vocab = vocab
        self.per_metric = per_metric
        self.criterion = self.configure_criterion(
            ignore_index=self.vocab.pad_id,
            ctc_weight=configs.ctc_weight,
            cross_entropy_weight=configs.cross_entropy_weight,
            blank_id=vocab.blank_id
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
        )
        self.decoder = SpeechTransformerDecoder(
            num_classes=num_classes,
            d_model=configs.encoder_dim,
            d_ff=configs.encoder_dim*configs.feed_forward_expansion_factor,
            num_heads=configs.num_attention_heads,
            num_layers=configs.num_decoder_layers,
            dropout_p=configs.decoder_dropout_p,
            eos_id=vocab.eos_id,
            pad_id=vocab.pad_id
        )

    def _log_states(self, stage: str, loss: float, cross_entropy_loss: float = None, ctc_loss: float = None, per: float = None) -> None:
        if per:
            self.log(f"{stage}_per", per)
        self.log(f"{stage}_loss", loss)
        if cross_entropy_loss:
            self.log(f"{stage}_cross_entropy_loss", cross_entropy_loss)
        if ctc_loss:
            self.log(f"{stage}_ctc_loss", ctc_loss)

    def forward(self, batch: tuple, batch_idx: int) -> Tensor:
        inputs, targets, input_lengths, target_lengths = batch
        
        _, encoder_outputs, encoder_outputs_length = self.encoder(inputs, input_lengths)
        outputs, attn = self.decoder(encoder_outputs, targets, encoder_outputs_length, target_lengths)
        
        return outputs

    def training_step(self, batch: tuple, batch_idx: int) -> Tensor:
        inputs, targets, input_lengths, target_lengths = batch

        encoder_log_probs, encoder_outputs, encoder_output_lengths = self.encoder(inputs, input_lengths)
        outputs, attn = self.decoder(encoder_outputs, targets, encoder_output_lengths, target_lengths)

        max_target_length = targets.size(1) - 1  # minus the start of sequence symbol
        outputs = outputs[:, :max_target_length, :]

        loss, ctc_loss, cross_entropy_loss = self.criterion(
            encoder_log_probs=encoder_log_probs.transpose(0, 1),
            decoder_log_probs=outputs.contiguous().view(-1, outputs.size(-1)),
            output_lengths=encoder_output_lengths,
            targets=targets[:, 1:],
            target_lengths=target_lengths,
        )

        self._log_states('train', loss, ctc_loss, cross_entropy_loss)

        return loss

    def validation_step(self, batch: tuple, batch_idx: int) -> Tensor:
        inputs, targets, input_lengths, target_lengths = batch

        _, encoder_outputs, encoder_outputs_length = self.encoder(inputs, input_lengths)
        outputs, attn = self.decoder(encoder_outputs, targets, encoder_outputs_length, target_lengths)

        encoder_log_probs, encoder_outputs, encoder_output_lengths = self.encoder(inputs, input_lengths)
        outputs, attn = self.decoder(encoder_outputs, targets, encoder_output_lengths, target_lengths)

        max_target_length = targets.size(1) - 1  # minus the start of sequence symbol
        outputs = outputs[:, :max_target_length, :]

        loss, ctc_loss, cross_entropy_loss = self.criterion(
            encoder_log_probs=encoder_log_probs.transpose(0, 1),
            decoder_log_probs=outputs.contiguous().view(-1, outputs.size(-1)),
            output_lengths=encoder_output_lengths,
            targets=targets[:, 1:],
            target_lengths=target_lengths,
        )

        y_hats = outputs.max(-1)[1]
        y_hats_encoder = encoder_log_probs.max(-1)[1]
        per = self.per_metric(targets[:, 1:], y_hats)
        
        if batch_idx == 0:
            print("\n1 sample result")
            print("EP:", y_hats_encoder[0].shape, self.vocab.label_to_string(y_hats_encoder[0]))
            print("DP:", y_hats[0].shape, self.vocab.label_to_string(y_hats[0]))
            print("Target:", targets[0, 1:].shape, self.vocab.label_to_string(targets[0, 1:]))
            print("Attention:", attn.shape)
            attn = torch.sum(attn, dim=1).detach().cpu()
            print(attn.shape)
            plt.imshow(attn, interpolation='none')
            plt.show()

        self._log_states('valid', per, loss, ctc_loss, cross_entropy_loss)

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

class ConformerLSTMModel(pl.LightningModule):
    def __init__(
            self,
            configs: DictConfig,
            num_classes: int,
            vocab: Vocabulary = GradVocabulary,
            per_metric: WordErrorRate = WordErrorRate,
    ) -> None:
        super(ConformerLSTMModel, self).__init__()
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
        )
        
        self.decoder = DecoderRNN(
            num_classes=num_classes,
            max_length=configs.max_length,
            hidden_state_dim=configs.encoder_dim,
            pad_id=self.vocab.pad_id,
            sos_id=self.vocab.sos_id,
            eos_id=self.vocab.eos_id,
            num_heads=configs.num_attention_heads,
            dropout_p=configs.decoder_dropout_p,
            num_layers=configs.num_decoder_layers,
            rnn_type=configs.rnn_type,
            use_tpu=configs.use_tpu
        )
        
    def _log_states(self, stage: str, loss: float, cross_entropy_loss: float = None, ctc_loss: float = None, per: float = None) -> None:
        if per:
            self.log(f"{stage}_per", per)
        self.log(f"{stage}_loss", loss)
        if cross_entropy_loss:
            self.log(f"{stage}_cross_entropy_loss", cross_entropy_loss)
        if ctc_loss:
            self.log(f"{stage}_ctc_loss", ctc_loss)
            
    def forward(self, inputs: Tensor, input_lengths: Tensor) -> Tensor:
        _, encoder_outputs, _ = self.encoder(inputs, input_lengths)
        y_hats = self.decoder(encoder_outputs=encoder_outputs, teacher_forcing_ratio=1.0)
        return y_hats
    
    def training_step(self, batch: tuple, batch_idx: int) -> Tensor:
        """
        Forward propagate a `inputs` and `targets` pair for training.
        Inputs:
            train_batch (tuple): A train batch contains `inputs`, `targets`, `input_lengths`, `target_lengths`
            batch_idx (int): The index of batch
        Returns:
            loss (torch.FloatTensor): Loss for training.
        """
        inputs, targets, input_lengths, target_lengths = batch
        
        encoder_log_probs, encoder_outputs, encoder_output_lengths = self.encoder(inputs, input_lengths)
        outputs, _ = self.decoder(targets, encoder_outputs, teacher_forcing_ratio=self.teacher_forcing_ratio)
        
        max_target_length = targets.size(1) - 1  # minus the start of sequence symbol
        outputs = outputs[:, :max_target_length, :]
        
        loss, ctc_loss, cross_entropy_loss = self.criterion(
            encoder_log_probs=encoder_log_probs.transpose(0, 1),
            decoder_log_probs=outputs.contiguous().view(-1, outputs.size(-1)),
            output_lengths=encoder_output_lengths,
            targets=targets[:, 1:],
            target_lengths=target_lengths,
        )
        self._log_states('train', loss)

        return loss

    def validation_step(self, batch: tuple, batch_idx: int) -> Tensor:
        inputs, targets, input_lengths, target_lengths = batch
        
        encoder_log_probs, encoder_outputs, encoder_output_lengths = self.encoder(inputs, input_lengths)
        outputs, attn = self.decoder(targets, encoder_outputs=encoder_outputs, teacher_forcing_ratio=self.teacher_forcing_ratio)
        
        max_target_length = targets.size(1) - 1  # minus the start of sequence symbol
        outputs = outputs[:, :max_target_length, :]
        
        loss, ctc_loss, cross_entropy_loss = self.criterion(
            encoder_log_probs=encoder_log_probs.transpose(0, 1),
            decoder_log_probs=outputs.contiguous().view(-1, outputs.size(-1)),
            output_lengths=encoder_output_lengths,
            targets=targets[:, 1:],
            target_lengths=target_lengths,
        )
        
        y_hats = outputs.max(-1)[1]
        y_hats_encoder = encoder_log_probs.max(-1)[1]
        per = self.per_metric(targets[:, 1:], y_hats)
        
        self._log_states('valid', per, loss, cross_entropy_loss, ctc_loss)
        
        if batch_idx == 0:
            print("\n1 sample result")
            print("EP:", y_hats_encoder[0].shape, self.vocab.label_to_string(y_hats_encoder[0]))
            print("DP:", y_hats[0].shape, self.vocab.label_to_string(y_hats[0]))
            print("Target:", targets[0, 1:].shape, self.vocab.label_to_string(targets[0, 1:]))
            print("Attention:", attn.shape)
            attn = torch.sum(attn, dim=0).detach().cpu()
            print(attn.shape)
            plt.imshow(attn, interpolation='none')
            plt.show()


        self._log_states('valid', per, loss)

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
        
class Con_Trans_formerModel(pl.LightningModule):
    def __init__(
            self,
            configs: DictConfig,
            num_classes: int,
            vocab: Vocabulary = GradVocabulary,
            per_metric: WordErrorRate = WordErrorRate,
    ) -> None:
        super(Con_Trans_formerModel, self).__init__()
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
        )
        
        self.decoder = DecoderTransformer(
            num_classes=num_classes,
            hidden_state_dim=configs.encoder_dim,
            pad_id=self.vocab.pad_id,
            eos_id=self.vocab.eos_id,
            num_heads=configs.num_attention_heads,
            dropout_p=configs.decoder_dropout_p,
            attention_dropout_p=configs.attention_dropout_p
        )
        
    def _log_states(self, stage: str, loss: float, cross_entropy_loss: float = None, ctc_loss: float = None, per: float = None) -> None:
        if per:
            self.log(f"{stage}_per", per)
        self.log(f"{stage}_loss", loss)
        if cross_entropy_loss:
            self.log(f"{stage}_cross_entropy_loss", cross_entropy_loss)
        if ctc_loss:
            self.log(f"{stage}_ctc_loss", ctc_loss)
            
    def forward(self, inputs: Tensor, targets: Tensor, input_lengths: Tensor, target_lengths: Tensor) -> Tensor:
        encoder_log_probs, encoder_outputs, encoder_output_lengths = self.encoder(inputs, input_lengths)
        outputs, attn = self.decoder(encoder_outputs, targets, encoder_output_lengths, target_lengths)
        max_target_length = targets.size(1) - 1  # minus the start of sequence symbol
        outputs = outputs[:, :max_target_length, :]
        return outputs, attn
    
    def training_step(self, batch: tuple, batch_idx: int) -> Tensor:
        """
        Forward propagate a `inputs` and `targets` pair for training.
        Inputs:
            train_batch (tuple): A train batch contains `inputs`, `targets`, `input_lengths`, `target_lengths`
            batch_idx (int): The index of batch
        Returns:
            loss (torch.FloatTensor): Loss for training.
        """
        inputs, targets, input_lengths, target_lengths = batch
        
        encoder_log_probs, encoder_outputs, encoder_output_lengths = self.encoder(inputs, input_lengths)
        outputs, attn = self.decoder(encoder_outputs, targets, encoder_output_lengths, target_lengths)
        
        max_target_length = targets.size(1) - 1  # minus the start of sequence symbol
        outputs = outputs[:, :max_target_length, :]
        
        loss, ctc_loss, cross_entropy_loss = self.criterion(
            encoder_log_probs=encoder_log_probs.transpose(0, 1),
            decoder_log_probs=outputs.contiguous().view(-1, outputs.size(-1)),
            output_lengths=encoder_output_lengths,
            targets=targets[:, 1:],
            target_lengths=target_lengths,
        )
        self._log_states('train', loss)

        return loss

    def validation_step(self, batch: tuple, batch_idx: int) -> Tensor:
        inputs, targets, input_lengths, target_lengths = batch
        
        encoder_log_probs, encoder_outputs, encoder_output_lengths = self.encoder(inputs, input_lengths)
        outputs, attn = self.decoder(encoder_outputs, targets, encoder_output_lengths, target_lengths)
        
        max_target_length = targets.size(1) - 1  # minus the start of sequence symbol
        outputs = outputs[:, :max_target_length, :]
        
        loss, ctc_loss, cross_entropy_loss = self.criterion(
            encoder_log_probs=encoder_log_probs.transpose(0, 1),
            decoder_log_probs=outputs.contiguous().view(-1, outputs.size(-1)),
            output_lengths=encoder_output_lengths,
            targets=targets[:, 1:],
            target_lengths=target_lengths,
        )
        
        y_hats = outputs.max(-1)[1]
        y_hats_encoder = encoder_log_probs.max(-1)[1]
        per = self.per_metric(targets[:, 1:], y_hats)
        
        self._log_states('valid', per, loss, cross_entropy_loss, ctc_loss)
        
        if batch_idx == 0:
            print("\n1 sample result")
            print("EP:", y_hats_encoder[0].shape, self.vocab.label_to_string(y_hats_encoder[0]))
            print("DP:", y_hats[0].shape, self.vocab.label_to_string(y_hats[0]))
            print("Target:", targets[0, 1:].shape, self.vocab.label_to_string(targets[0, 1:]))
            print("Attention:", attn.shape)
            attn = torch.sum(attn, dim=0).detach().cpu()
            print(attn.shape)
            plt.imshow(attn, interpolation='none')
            plt.show()


        self._log_states('valid', per, loss)

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
        