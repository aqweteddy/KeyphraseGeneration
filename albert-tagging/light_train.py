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
        super(AlbertSeqTag, self).__init__()
        self.hparams = hparams

        self.train_loader = train_dataset
        self.val_loader = val_dataset

        self.batch_size = hparams['batch_size']

        self.model = AlbertCrf(hparams['num_classes'], model_path=bert_path)

    def forward(self, inp_ids, seg_ids, mask_ids, trgs):
        # inp_ids, seg_ids, mask_ids, trgs = batch
        loss, hidden = self.model(inp_ids, seg_ids, mask_ids, trgs)
        return loss, hidden

    def configure_optimizers(self):
        return torch.optim.AdamW(self.model.parameters(), lr=self.hparams['lr'])

    def train_dataloader(self):
        return self.train_loader

    def training_step(self, batch, batch_idx):
        inp_ids, seg_ids, mask_ids, trgs = batch
        loss, _ = self(inp_ids, seg_ids, mask_ids, trgs)
        if self.logger is not None:
            self.logger.experiment.add_scalar(f'train/loss', loss)
        return {'loss': loss}

    def training_epoch_end(self, outputs):
        loss_mean = torch.stack([x['loss'] for x in outputs]).mean()
        logs = {'loss': loss_mean}
        results = {'progress_bar': logs, 'loss': loss_mean}
        return results

    def val_dataloader(self):
        return self.val_loader

    def validation_step(self, batch, batch_idx):
        inp_ids, seg_ids, mask_ids, trgs = batch
        loss, path = self.model.decode(inp_ids, seg_ids, mask_ids, trgs)
        path = path[0]
        inp_ids = batch[0][0].detach().tolist()
        best_ids = self.convert_path_to_ids(inp_ids, path)
        return {'val_loss': loss, 'best_ids': best_ids}

    def validation_epoch_end(self, outputs):
        val_loss_mean = torch.stack([x['val_loss'] for x in outputs]).mean()
        best_ids = outputs[0]['best_ids']

        if self.logger is not None:
            self.logger.experiment.add_scalar('val/loss', val_loss_mean)
            self.logger.experiment.add_text('val/best_path', str(best_ids))
        return {'val_loss': val_loss_mean.cpu()}

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

    hparams = {'batch_size': 32,
               'lr': 2e-5,
               'num_classes': 3
               }
    # small_ds = KpBioDataset.from_encoded('./data/small.json', tokenizer)
    train_l = DataLoader(train_ds, shuffle=True,
                         batch_size=hparams['batch_size'])
    test_l = DataLoader(test_ds, shuffle=False,
                        batch_size=hparams['batch_size'])
    # small_l = DataLoader(small_ds, batch_size=4 * 8)

    logger = TensorBoardLogger('./logs', name='albertSeqtag')

    # model = AlbertSeqTag(hparams, small_l, small_l)
    model = AlbertSeqTag(hparams, train_l, test_l)
    trainer = Trainer(logger=logger,
                      default_save_path='./ckpt',
                      early_stop_callback=True,
                      val_check_interval=0.5,
                    #   distributed_backend='dp',
                      max_epochs=20,
                      gpus=1
                      )

    # lr_finder = trainer.lr_find(model)
    # print(f'lr: {lr_finder.results}')
    # model.hparams.lr = lr_finder.suggestion()

    trainer.fit(model)
