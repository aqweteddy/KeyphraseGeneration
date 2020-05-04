import torch
from model import AlbertCrf

inp_ids = [101,  1724,  3621,  7274,  3373,  1319,  1976,  3469,  6268,  3683]
seg_ids = [0] * 10
mask_ids = [1] * 10
trg = [0, 0, 0, 1, 2, 1, 2, 2, 0, 0]

inp_ids = torch.tensor(inp_ids).unsqueeze(0).cuda()
seg_ids = torch.tensor(seg_ids).unsqueeze(0).cuda()
mask_ids = torch.tensor(mask_ids).unsqueeze(0).cuda()
trg = torch.tensor(trg).unsqueeze(0).cuda()

model = AlbertCrf(3, model_path='../albert_base').cuda()
loss, output = model(inp_ids, seg_ids, mask_ids, trg)
print(loss, output)
