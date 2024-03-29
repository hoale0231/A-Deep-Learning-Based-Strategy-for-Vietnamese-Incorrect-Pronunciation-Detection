from dataclasses import dataclass

@dataclass
class DictConfig:
    # Audio
    sample_rate: int = 16000
    frame_length: float = 25.0
    frame_shift: float = 10.0

    # SpecAugment
    apply_spec_augment: bool = True
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
    
    # LRScheduler
    lr: float = 1e-04

    # # ReduceLROnPlateauLRScheduler
    # lr_patience: int = 1
    # scheduler: str = "reduce_lr_on_plateau"
    # lr_factor: float = 0.3

    # # TriStageLRScheduler
    # init_lr: float = 1e-10
    # peak_lr: float = 1e-04
    # final_lr: float = 1e-07
    # init_lr_scale: float = 0.01
    # final_lr_scale: float = 0.05
    # warmup_steps: int = 10000
    # decay_steps: int = 150000
    # scheduler: str = "tri_stage"

    # TransformerLRScheduler
    peak_lr: float = 1e-04
    final_lr: float = 1e-07
    final_lr_scale: float = 0.05
    scheduler: str = "transformer"

    # Conformer-Transformer
    mel_channels: int = 128
    mel_units: int = 256
    mel_kernel: int = 5
    mel_padding: int = 2
    mel_stride: int = 1
    mel_dropout_cnn: float = 0.045
    mel_dropout_gru: float = 0.045
    
    phone_channels: int = 64
    phone_units: int = 128
    phone_kernel: int = 5
    phone_padding: int = 2
    phone_stride: int = 1
    phone_dropout_cnn: float = 0.2
    phone_dropout_gru: float = 0.2
    
    attention_dim: int = 256
    num_decoder_layers: int = 1
    num_attention_heads: int = 8
    decoder_dropout_p: float = 0.2
    decoder_units: int = 256
    rnn_type: str = "gru"
    
    max_length: int = 128
    
    cross_entropy_weight: float = 0.5
    ctc_weight: float = 0.5
    
    # stage 0
    # md_weight: float = 0.0
    # pr_weight: float = 1.0
    # train_set = 'train'
    # test_set = 'dev'
    # valid_set = 'test'
    # warmup_steps: int = 16000
    # decay_steps: int = 100000
    
    # stage 1
    # md_weight: float = 0.7
    # pr_weight: float = 0.3
    # train_set = 'train'
    # test_set = 'dev'
    # valid_set = 'test'
    # warmup_steps: int = 8000
    # decay_steps: int = 10000
    
    # combine L1 & L2
    # md_weight: float = 0.7
    # pr_weight: float = 0.3
    # train_set = 'train_total'
    # test_set = 'label_test'
    # valid_set = 'label_valid'
    
    # stage 2
    md_weight: float = 1.0
    pr_weight: float = 0.0
    train_set = 'label_train'
    test_set = 'label_test'
    valid_set = 'label_valid'
    warmup_steps: int = 3000
    decay_steps: int = 5000
    
    gamma: float = 0.0
    joint_ctc_attention: bool = True
    optimizer: str = "adam"
    is_weakly_s: bool = True
    add_noise: bool = False
    # BaseTrainer
    seed: int = 1
    accelerator: str = "cuda"
    accumulate_grad_batches: int = 4
    num_workers: int = 4
    batch_size: int = 16
    check_val_every_n_epoch: int = 1
    gradient_clip_val: float = 5.0
    max_epochs: int = 50
    # auto_scale_batch_size: str = "binsearch"

    # TrainerGPU
    use_cuda: bool = True
    use_tpu: bool = False
    auto_select_gpus: bool = True
    
