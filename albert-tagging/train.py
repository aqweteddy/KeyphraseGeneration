import os

import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
from tqdm import tqdm

from model import AlbertCrf


class Trainer:
    def __init__(self, bert_path='./albert_base', save_dir='./ckpt', tf_board_dir='./tfboard', num_classes=3, lr=1e-5):
        # tfboard & log
        self.writer = SummaryWriter(tf_board_dir)

        self.model = AlbertCrf(num_classes, model_path=bert_path).cuda()
        self.model = nn.DataParallel(self.model)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
        self.save_dir = save_dir

    def train(self, epochs, train_loader, test_loader=None, check_save_step=800):
        self.model.train()
        global_steps = 0
        prev_best_loss = 1000

        for epoch in range(epochs):
            tot_loss = 0
            size = 0
            for inp_ids, seg_ids, mask_ids, trgs in tqdm(train_loader, desc=f'Epoch {epoch+1}'):
                size += 1
                inp_ids, seg_ids, mask_ids, trgs = inp_ids.cuda(
                ), seg_ids.cuda(), mask_ids.cuda(), trgs.cuda()
                loss = self.__train_one(inp_ids, seg_ids, mask_ids, trgs)

                tot_loss += loss
                global_steps += 1

                # draw train loss scalar
                if global_steps % 10 == 1:
                    self.writer.add_scalar(f'train/loss', tot_loss / size, global_step=global_steps+1)
                    self.writer.flush()


                if test_loader and global_steps % check_save_step == 1 and global_steps > check_save_step:
                    now_loss = self.evaluate(test_loader)
                    
                    # draw test loss
                    self.writer.add_scalar(f'test/loss', now_loss, global_step=global_steps+1)
                    self.writer.flush()

                    if prev_best_loss > now_loss:
                        prev_best_loss = now_loss
                        self.save_model(os.path.join(
                            self.save_dir, f'step_{global_steps+1}.ckpt'))
                        self.model.train()
            print(f'train_loss: {tot_loss / size:.3f}')
            self.save_model(os.path.join(
                self.save_dir, f'epoch_{epoch+1}.ckpt'))
            self.model.train()

    def __train_one(self, inp_ids, seg_ids, mask_ids, trgs):
        self.model.zero_grad()
        hidden, loss = self.model(inp_ids, seg_ids, mask_ids, trgs)
        # loss = self.model.loss(hidden, mask_ids, trgs)
        loss = loss.mean()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def evaluate(self, test_loader):
        self.model.eval()
        loss = 0
        size = 0
        for inp_ids, seg_ids, mask_ids, trgs in tqdm(test_loader, 'Evaluate'):
            inp_ids, seg_ids, mask_ids, trgs = inp_ids.cuda(
            ), seg_ids.cuda(), mask_ids.cuda(), trgs.cuda()
            size += 1
            hidden, l = self.model(inp_ids, seg_ids, mask_ids, trgs)
            loss += l.mean().item()
            path_score, best_path = self.model.module.crf(hidden, mask_ids)

            best_path = best_path[0].detach().tolist()
            inp_ids = inp_ids.detach().tolist()

        best_ids = self.convert_path_to_ids(inp_ids[0], best_path)
        self.writer.add_text('eval/best_path', f'{best_ids}')

        return loss / size
    
    def convert_path_to_ids(self, inp_ids, path):
        ans = []
        for idx in range(len(path)):
            if path[idx] == 1:
                word = []
                while path[idx] != 0:
                    word.append(inp_ids[idx])
                    idx += 1
                ans.append(word)
        return ans


    def save_model(self, filename):
        torch.save(self.model.module.state_dict(),
                   f'{filename}')


if __name__ == '__main__':
    from data import KpBioDataset
    from transformers import BertTokenizerFast
    from torch.utils.data import DataLoader

    BATCH_SIZE = 16 * 8
    EPOCHS = 4

    tokenizer = BertTokenizerFast.from_pretrained('./albert_base')
    train_ds = KpBioDataset.from_encoded('./data/train.json', tokenizer)
    test_ds = KpBioDataset.from_encoded('./data/test.json', tokenizer)
    train_loader = DataLoader(train_ds, shuffle=True, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_ds, shuffle=False, batch_size=BATCH_SIZE)

    trainer = Trainer(lr=2e-5)
    trainer.train(EPOCHS, train_loader, test_loader, 400)
