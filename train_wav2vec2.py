import warnings
from glob import glob
import argparse
import os

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from mpvn_wav2vec2.data.grad.lit_data_module import LightningGradDataModule
from mpvn_wav2vec2.model.model import Wav2vec2RNNModel
from mpvn_wav2vec2.configs import DictConfig

from mpvn.utils import *
from mpvn.metric import WordErrorRate


def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--action',
        choices=['train', 'val', 'test'],
        required=True
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        default=None,
        help='path to checkpoint or checkpoint dir (in case of avg)'
    )
    parser.add_argument(
        '--savedir',
        type=str,
        default='checkpoint/checkpoint/',
        help='dir to save checkpoint'
    )
    parser.add_argument(
        '--logdir',
        type=str,
        default='tensorboard',
        help='dir to write log'
    )
    parser.add_argument(
        '--gpu',
        type=int,
        default=1,
        help='gpu index to use'
    )
    return parser

if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    
    checkpoint_callback = ModelCheckpoint(
        save_top_k=3,
        monitor="valid_loss",
        mode="min",
        dirpath=args.savedir,
        filename="mpvn-{epoch:02d}-{valid_loss:.2f}-{valid_per:.2f}-{valid_acc:.2f}-{valid_f1:.2f}",
    )
    
    early_stop_callback = EarlyStopping(
        monitor="valid_loss", 
        min_delta=0.00, 
        patience=5, 
        verbose=False, 
        mode="min"
    )
    
    logger = TensorBoardLogger(args.logdir, name="Pronunciation for Vietnamese")
    configs = DictConfig()

    pl.seed_everything(configs.seed)
    warnings.filterwarnings('ignore')

    data_module = LightningGradDataModule(configs)
    vocab = data_module.get_vocab() 

    trainer = pl.Trainer(accelerator=configs.accelerator,
                        devices=[args.gpu],
                        logger=logger,
                        # val_check_interval=0.25,
                        max_epochs=configs.max_epochs,
                        callbacks=[checkpoint_callback, early_stop_callback])
    
    if args.checkpoint != None:
        if os.path.isdir(args.checkpoint):
            print('Checkpoint path is a directory, do avg checkpoint')
            model = Wav2vec2RNNModel(
                configs=configs,
                num_classes=len(vocab),
                vocab=vocab,
                per_metric=WordErrorRate(vocab)
            )
            model = average_checkpoints(model, glob(args.checkpoint+'/*'))
        elif os.path.isfile(args.checkpoint):
            print('Checkpoint path is a file, dont avg checkpoint')

            model = Wav2vec2RNNModel.load_from_checkpoint(
                args.checkpoint,
                configs=configs,
                num_classes=len(vocab),
                vocab=vocab,
                per_metric=WordErrorRate(vocab)
            )
        else:
            raise Exception('Checkpoint path is not file or directory!')
    else:
        model = Wav2vec2RNNModel(
            configs=configs,
            num_classes=len(vocab),
            vocab=vocab,
            per_metric=WordErrorRate(vocab)
        )
        
    # for param in model.encoder.wav2vec2_encoder.wav2vec2.parameters():
    #     param.requires_grad = False
    model.encoder.wav2vec2_encoder.config.vocab_size = 137

    if args.action == 'train':
        trainer.fit(model, data_module)
        
    elif args.action == 'val':
        trainer.validate(model, data_module)
    
    elif args.action == 'test':
        trainer.test(model, data_module)
        score = ' '.join(list(model.df.score))
        predict = ' '.join(list(model.df.score_predict))
        score = [int(i) for i in score.split()]
        predict = [int(i) for i in predict.split()]
        print(args.checkpoint)
        print("Accuracy:", accuracy_score(score, predict) * 100)
        print("F1:", f1_score(score, predict, pos_label=0) * 100)
        print("Precision:", precision_score(score, predict, pos_label=0) * 100)
        print("Recall:", recall_score(score, predict, pos_label=0) * 100)
        
        model.df.to_csv('test_result.csv', index=False)
        
    