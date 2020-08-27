import os
import random
import numpy as np
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from argparse import ArgumentParser
from functools import lru_cache
import segmentation_models_pytorch as smp
from dataset import provider
from metrics import MixedLoss, metric
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateLogger

torch.backends.cudnn.benchmark = True
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.benchmark = False


#https://www.kaggle.com/iafoss/siimacr-pneumothorax-segmentation-data-1024
train_rle_path = 'labels/stage_2_train.csv'
train_image_folder = "size1024/train/train"
train_mask_folder = "size1024/masks"
test_data_folder = "size1024/test"


def get_args():
    parser = ArgumentParser()
    parser.add_argument('--gpus', type=int, default=15)
    parser.add_argument('--epochs', type=int, default=70)
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--train_batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=2.5e-4)
    parser.add_argument('--img_size', type=int, default=1024)
    parser.add_argument('--num_workers', type=int, default=24)

    return parser.parse_args()


class Pneumothorax(pl.LightningModule):

    def __init__(self, hparams):
        super(Pneumothorax, self).__init__()
        self.hparams = hparams
        self.model = smp.Unet("resnet34", encoder_weights="imagenet", activation=None)
        self.loss = MixedLoss(10, 2)
        self.prepare_data()
 
    def forward(self, x):

        x = self.model(x)

        return x

    # @lru_cache(maxsize=None)
    # def len_train_dl(self):
	#     return len(self.train_dataloader())

    def configure_optimizers(self):

        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=3, verbose=True, factor=0.1)
        # scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=self.hparams.lr, steps_per_epoch=self.len_train_dl(), 
		# 												epochs=self.hparams.epochs, pct_start=0.0, base_momentum=0.85, 
		# 												max_momentum=0.95, div_factor=100.0, final_div_factor=1e4)

        scheduler = {'scheduler': scheduler, 'interval' : 'epoch'}

        return {'optimizer': optimizer, 'lr_scheduler': scheduler}

    def train_dataloader(self):
        train_loader = provider(fold=self.hparams.fold, 
                                total_folds=5, 
                                image_folder=train_image_folder, 
                                mask_folder=train_mask_folder, 
                                df_path=train_rle_path,
                                phase='train',
                                size=self.hparams.img_size,
                                mean=(0.485, 0.456, 0.406),
                                std=(0.229, 0.224, 0.225),
                                batch_size=self.hparams.train_batch_size,
                                num_workers=self.hparams.num_workers,
                                shuffle=True)
        return train_loader

    def training_step(self, batch, batch_idx):
        image, mask = batch
        mask = mask.unsqueeze(1)
        output = self.forward(image)
        loss = self.loss(output,mask)
        dice = metric(torch.sigmoid(output), mask, 0.5)
        return {'loss': loss, 'dice': dice}

 

    def training_epoch_end(self, outputs):

        # Compute and log a val loss and dice
        train_loss = torch.stack([x['loss'] for x in outputs]).mean()
        train_dice = torch.stack([x['dice'] for x in outputs]).mean()
        self.logger.experiment.add_scalars("dices", {"Dice/Train": train_dice}, self.current_epoch)
        self.logger.experiment.add_scalars("losses", {"Loss/Train": train_loss}, self.current_epoch)

        return {'train_loss': train_loss}

    def val_dataloader(self):
        valid_loader = provider(fold=self.hparams.fold,
                                total_folds=5,
                                image_folder=train_image_folder,
                                mask_folder=train_mask_folder,
                                df_path=train_rle_path,
                                phase='val',
                                size=self.hparams.img_size,
                                mean=(0.485, 0.456, 0.406),
                                std=(0.229, 0.224, 0.225),
                                batch_size=2,
                                num_workers=self.hparams.num_workers,
                                shuffle=False)
        return valid_loader

    def validation_step(self, batch, batch_idx):

        image, mask = batch
        mask = mask.unsqueeze(1)
        output = self.forward(image)
        loss = self.loss(output, mask)

        dice = metric(torch.sigmoid(output), mask, 0.5)

        return {'val_loss': loss, 'val_dice': dice}
    

    def validation_epoch_end(self, outputs):

        # Compute and print a val loss and dice
        val_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        val_dice = torch.stack([x['val_dice'] for x in outputs]).mean()
 
        self.logger.experiment.add_scalars("dices", {"Dice/Valid": val_dice}, self.current_epoch)
        self.logger.experiment.add_scalars("losses", {"Loss/Valid": val_loss}, self.current_epoch)       

    

     
        return {'val_loss': val_loss, "val_dice": val_dice, 'progress_bar': {'val_loss': val_loss, 'val_dice': val_dice}}


def main(hparams):

    checkpoint = ModelCheckpoint(
        filepath='weights/{epoch}_fold='+str(hparams.fold),
        save_top_k=1,
        monitor = 'val_loss',
        mode = 'min',
        verbose=True,
        prefix=''
    )

    logger = TensorBoardLogger('pn_logs', name = 'fold={fold}'.format(fold=str(hparams.fold)))
    lr_logger = LearningRateLogger(logging_interval='step')
    model = Pneumothorax(hparams)
    trainer = pl.Trainer(
        max_epochs=hparams.epochs,
        gpus=[hparams.gpus], 
        checkpoint_callback=checkpoint,
        accumulate_grad_batches=2,
        logger=logger,
        callbacks=[lr_logger],
        precision=16,
    )

    trainer.fit(model)


if __name__ == '__main__':
    seed_everything(144)
    args = get_args()
    main(args)