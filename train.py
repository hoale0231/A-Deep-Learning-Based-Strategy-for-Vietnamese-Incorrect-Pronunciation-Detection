import warnings
from typing import List
from pathlib import Path
from glob import glob

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from mpvn.data.grad.lit_data_module import LightningGradDataModule
from mpvn.metric import WordErrorRate
from mpvn.model.model import *
from mpvn.configs import DictConfig

checkpoint_callback = ModelCheckpoint(
    save_top_k=3,
    monitor="valid_loss",
    mode="min",
    dirpath="checkpoint/checkpoint_stage_2",
    filename="mpvn-{epoch:02d}-{valid_loss:.2f}-{valid_per:.2f}-{valid_acc:.2f}-{valid_f1:.2f}",
)
early_stop_callback = EarlyStopping(
    monitor="valid_loss", 
    min_delta=0.00, 
    patience=5, 
    verbose=False, 
    mode="min"
)
logger = TensorBoardLogger("tensorboard_2", name="Pronunciation for Vietnamese")

configs = DictConfig()

pl.seed_everything(configs.seed)
warnings.filterwarnings('ignore')

data_module = LightningGradDataModule(configs)
vocab = data_module.get_vocab() 

trainer = pl.Trainer(accelerator=configs.accelerator,
                     devices=[0],
                      logger=logger,
                      max_epochs=configs.max_epochs,
                      callbacks=[checkpoint_callback, early_stop_callback])

def validate():
    for cp in glob('checkpoint/checkpoint_stage_1/*'):
        model = ConformerRNNModel.load_from_checkpoint(
            cp,
            configs=configs,
            num_classes=len(vocab),
            vocab=vocab,
            per_metric=WordErrorRate(vocab)
        )

        print(cp)
        trainer.validate(model, data_module)

def train():
    # checkpoint = 'checkpoint/checkpoint/mpvn-epoch=22-valid_loss=0.28-valid_per=0.07-valid_acc=0.00-valid_f1=0.00.ckpt'
    # model = ConformerRNNModel.load_from_checkpoint(
    #         checkpoint,
    #         configs=configs,
    #         num_classes=len(vocab),
    #         vocab=vocab,
    #         per_metric=WordErrorRate(vocab)
    #     )
    model = average_checkpoints(glob('checkpoint/checkpoint_stage_1/*'))
    trainer.fit(model, data_module)
    

def average_checkpoints(filenames: List[Path], device: torch.device = torch.device("cpu")) -> dict:
    n = len(filenames)

    avg = torch.load(filenames[0], map_location=device)['state_dict']

    # Identify shared parameters. Two parameters are said to be shared
    # if they have the same data_ptr
    uniqued: Dict[int, str] = dict()

    for k, v in avg.items():
        v_data_ptr = v.data_ptr()
        if v_data_ptr in uniqued:
            continue
        uniqued[v_data_ptr] = k

    uniqued_names = list(uniqued.values())

    for i in range(1, n):
        state_dict = torch.load(filenames[i], map_location=device)['state_dict']
        for k in uniqued_names:
            avg[k] += state_dict[k]

    for k in uniqued_names:
        if avg[k].is_floating_point():
            avg[k] /= n
        else:
            avg[k] //= n
            
    model = ConformerRNNModel(
        configs=configs,
        num_classes=len(vocab),
        vocab=vocab,
        per_metric=WordErrorRate(vocab)
    )
    model.load_state_dict(avg)
    
    return model

def val_avg():
    model = average_checkpoints(glob('checkpoint/checkpoint/*'))
    trainer.validate(model, data_module)

train()