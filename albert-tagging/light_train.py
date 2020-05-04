import os

import pytorch_lightning as pl
import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
from tqdm import tqdm
from torch.utils.data import DataLoader

from model import AlbertCrf


class AlbertSeqTag(pl.LightningModule):
    def __init__(self, hparams,
                 train_dataset,
                 val_dataset,
                 bert_path='./albert_base',
                 ):
        super().__init__()
        self.hparams = hparams

        self.train_set = train_dataset
        self.val_set = val_dataset

        self.batch_size = hparams['batch_size']

        self.model = AlbertCrf(hparams['num_classes'], model_path=bert_path)

    def forward(self, inp_ids, seg_ids, mask_ids, trgs):
        return self.model(inp_ids, seg_ids, mask_ids, trgs)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.model.parameters(), lr=self.hparams['lr'])

    @pl.data_loader
    def train_dataloader(self):
        return DataLoader(self.train_set, shuffle=True, batch_size=self.batch_size)

    def training_step(self, batch, batch_idx):
        inp_ids, seg_ids, mask_ids, trgs = batch
        hidden, loss = self.model(inp_ids, seg_ids, mask_ids, trgs)
        # loss = self.model.loss(hidden, mask_ids, trgs)
        loss = loss.mean().cuda()
        # loss.backward()
        # self.optimizer.step()
        if self.logger is not None:
            self.logger.experiment.add_scalar(f'train/loss', loss)
        return {'loss': loss}

    def training_epoch_end(self, outputs):
        loss_mean = torch.stack([x['loss'] for x in outputs]).mean()
        logs = {'loss': loss_mean}
        results = {'progress_bar': logs}
        return results

    @pl.data_loader
    def val_dataloader(self):
        return DataLoader(self.val_set, shuffle=False, batch_size=self.batch_size, num_workers=4)

    def validation_step(self, batch, batch_idx):
        inp_ids, seg_ids, mask_ids, trgs = batch
        hidden, loss = self.model(inp_ids, seg_ids, mask_ids, trgs)
        path_score, best_path = self.model.crf(hidden, mask_ids)
        best_path = best_path[0].detach().tolist()
        inp_ids = inp_ids[0].detach().tolist()
        best_ids = self.convert_path_to_ids(inp_ids, best_path)

        return {'val_loss': loss, 'best_ids': best_ids}

    def validation_epoch_end(self, outputs):
        val_loss_mean = torch.stack([x['val_loss'] for x in outputs]).mean()
        best_ids = outputs[0]['best_ids']

        if self.logger is not None:
            self.logger.experiment.add_scalar(f'val/loss', val_loss_mean)
            self.logger.experiment.add_text('val/best_path', str(best_ids))
        return {'val_loss': val_loss_mean}

    def convert_path_to_ids(self, inp_ids, path):
        ans = []
        for idx in range(len(path)):
            if path[idx] == 1:
                word = []
                while idx < len(path) and path[idx] != 0:
                    word.append(inp_ids[idx])
                    idx += 1
                ans.append(word)
        return ans


if __name__ == '__main__':
    from pytorch_lightning.callbacks import EarlyStopping
    from pytorch_lightning import Trainer
    from pytorch_lightning.loggers import TensorBoardLogger
    from data import KpBioDataset
    from transformers import BertTokenizerFast

    tokenizer = BertTokenizerFast.from_pretrained('./albert_base')
    train_ds = KpBioDataset.from_encoded('./data/train.json', tokenizer)
    test_ds = KpBioDataset.from_encoded('./data/test.json', tokenizer)

    hparams = {'batch_size': 8 * 16,
               'lr': 2e-5,
               'num_classes': 3
               }

    logger = TensorBoardLogger('./logs', name='albertSeqtag')

    # early_stop_callback = EarlyStopping(
    #     monitor='val_loss', min_delta=0.002,  patience=3)

    model = AlbertSeqTag(hparams, train_ds, test_ds)
    trainer = Trainer(logger=logger,
                      default_save_path='./ckpt',
                    #   early_stop_callback=early_stop_callback,
                      val_check_interval=0.5,
                      max_epochs=20,
                      gpus=6
                      )

    # lr_finder = trainer.lr_find(model)
    # print(f'lr: {lr_finder.results}')
    # model.hparams.lr = lr_finder.suggestion()

    trainer.fit(model)
