import torch
import torch.nn as nn
from torchcrf import CRF
from transformers import AlbertModel


class AlbertCrf(nn.Module):
    def __init__(self,  num_classes, model_path='./albert_base'):
        super(AlbertCrf, self).__init__()

        self.albert = AlbertModel.from_pretrained(model_path)
        self.dropout = nn.Dropout(0.1)
        self.fc1 = nn.Linear(self.albert.config.hidden_size, 256)
        self.fc2 = nn.Linear(256, num_classes)
        self.crf = CRF(num_classes, batch_first=True)

    def forward(self, inp_ids, seg_ids, mask_ids, trg=None):
        """forward

        Arguments:
            inp_ids -- [batch_size, seq_len]
            seg_ids  -- [batch_size, seq_len]
            mask_ids -- [batch_size, seq_len]
        """
        batch_size = inp_ids.size(0)
        seq_len = inp_ids.size(1)
        # (batch_size, sequence_length, albert_hidden_size)
        hidden, _ = self.albert(inp_ids, mask_ids, seg_ids)
        hidden = self.dropout(hidden)
        hidden = self.fc1(hidden)  # (batch_size, seq_len, 256)
        hidden = self.fc2(hidden)  # (batch_size, seq_len, num_classes)

        mask_ids = mask_ids.bool()
        if trg is not None:
            loss = -self.crf(hidden, trg, mask_ids,reduction='mean')
            return loss, hidden
        else:
            return hidden

    def decode(self, inp_ids, seg_ids, mask_ids, trg=None):
        with torch.no_grad():
            mask_ids = mask_ids.bool()
            if trg is not None:
                loss, hidden = self(inp_ids, seg_ids, mask_ids, trg)
                return loss, self.crf.decode(hidden, mask_ids)
            else:
                hidden = self(inp_ids, seg_ids, mask_ids)
                return self.crf.decode(hidden)
