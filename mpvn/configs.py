from dataclasses import dataclass

@dataclass
class DictConfig:
    # Audio
    sample_rate: int = 16000
    frame_length: float = 25.0
    frame_shift: float = 10.0

    # SpecAugment
    apply_spec_augment: bool = True
    auto_gen_score: bool = (True, True, True, False, False, False) # Train, dev, test, label, train_label, test_label
    freq_mask_para: int = 27
    freq_mask_num: int = 2
    time_mask_num: int = 4

    # Feature extract method
    # num_mels_mfcc: int = 40
    # num_mels_fbank: int = 80
    num_mels: int = 80
    # num_mels_spectrogram: int = 161
    feature_extract_method: str = "melspectrogram"
    # feature_extract_method: str = "spectrogram"
    # feature_extract_method: str = "fbank"
    # feature_extract_method: str = "mfcc"

    # Data
    dataset_path: str = "Data"
    dataset_download: bool = True
    vocab_size: int = 5000
    
    # LRScheduler
    lr: float = 1e-04

    # ReduceLROnPlateauLRScheduler
    lr_patience: int = 1
    scheduler: str = "reduce_lr_on_plateau"
    lr_factor: float = 0.3

    # TriStageLRScheduler
    init_lr: float = 1e-10
    peak_lr: float = 1e-04
    final_lr: float = 1e-07
    init_lr_scale: float = 0.01
    final_lr_scale: float = 0.05
    warmup_steps: int = 10000
    decay_steps: int = 150000
    scheduler: str = "tri_stage"

    # TransformerLRScheduler
    peak_lr: float = 1e-04
    final_lr: float = 1e-07
    final_lr_scale: float = 0.05
    warmup_steps: int = 10000
    decay_steps: int = 150000
    scheduler: str = "transformer"

    # Conformer-Transformer
    encoder_dim: int = 144
    num_encoder_layers: int = 4
    num_decoder_layers: int = 1
    num_attention_heads: int = 4
    feed_forward_expansion_factor: int = 4
    conv_expansion_factor: int = 2
    input_dropout_p: float = 0.1
    feed_forward_dropout_p: float = 0.1
    attention_dropout_p: float = 0.1
    conv_dropout_p: float = 0.1
    decoder_dropout_p: float = 0.1
    conv_kernel_size: int = 31
    half_step_residual: bool = True
    max_length: int = 128
    teacher_forcing_ratio: float = 1.0
    cross_entropy_weight: float = 0.5
    ctc_weight: float = 0.5
    md_weight: float = 0.8
    pr_weight: float = 0.2
    joint_ctc_attention: bool = True
    rnn_type: str = "gru"
    optimizer: str = "adam"
    half_subsampling: bool = False

    # BaseTrainer
    seed: int = 1
    accelerator: str = "cuda"
    accumulate_grad_batches: int = 4
    num_workers: int = 4
    batch_size: int = 8
    check_val_every_n_epoch: int = 1
    gradient_clip_val: float = 5.0
    max_epochs: int = 30
    # auto_scale_batch_size: str = "binsearch"

    # TrainerGPU
    use_cuda: bool = True
    use_tpu: bool = False
    auto_select_gpus: bool = True
    