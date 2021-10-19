import torch
import torch.nn as nn
import numpy as np
from loss.ssim import MS_SSIM_LOSS, MS_SSIM

class HEM(nn.Module):
    def __init__(self, hard_thre_p=0.5, device='cuda', random_thre_p=0.1):
        super(HEM, self).__init__()
        self.hard_thre_p = hard_thre_p
        self.random_thre_p = random_thre_p
        self.L1_loss = nn.L1Loss()
        self.device = device

    def hard_mining_mask(self, x, y):
        with torch.no_grad():
            b, c, h, w = x.size()

            hard_mask = np.zeros(shape=(b, 1, h, w))
            res = torch.sum(torch.abs(x - y), dim=1, keepdim=True)
            res_numpy = res.cpu().numpy()
            res_line = res.view(b, -1)
            res_sort = [res_line[i].sort(descending=True) for i in range(b)]
            hard_thre_ind = int(self.hard_thre_p * w * h)
            for i in range(b):
                thre_res = res_sort[i][0][hard_thre_ind].item()
                hard_mask[i] = (res_numpy[i] > thre_res).astype(np.float32)

            random_thre_ind = int(self.random_thre_p * w * h)
            random_mask = np.zeros(shape=(b, 1 * h * w))
            for i in range(b):
                random_mask[i, :random_thre_ind] = 1.
                np.random.shuffle(random_mask[i])
            random_mask = np.reshape(random_mask, (b, 1, h, w))

            mask = hard_mask + random_mask
            mask = (mask > 0.).astype(np.float32)

            mask = torch.Tensor(mask).to(self.device)

        return mask

    def forward(self, x, y):
        mask = self.hard_mining_mask(x, y)

        hem_loss = self.L1_loss(x * mask, y * mask)

        return hem_loss

class HEM_MSSIM_L1(nn.Module):
    """ Hard Example Mining for 0.84*(1 - MS-SSIM) + 0.16*L1
    """
    def __init__(self, hard_thre_p=0.5, device='cuda', random_thre_p=0.1, rbg_range=1.0, channel=3):
        super(HEM_MSSIM_L1, self).__init__()
        self.device = device
        self.hard_thre_p = hard_thre_p
        self.random_thre_p = random_thre_p
        self.L1_loss = nn.L1Loss()
        self.rbg_range = rbg_range
        self.channel = channel
        self.MS_SSIM_LOSS = MS_SSIM_LOSS(data_range=rbg_range, channel=channel)
        self.MS_SSIM = MS_SSIM(data_range=rbg_range, channel=channel)

    def hard_mining_mask(self, x, y):
        with torch.no_grad():
            b, c, h, w = x.size()

            #hard_mask = torch.cuda.FloatTensor().resize_(b, 1, h, w).zero_()
            hard_mask = torch.zeros((b, 1, h, w), device=self.device, dtype=torch.float32)
            res = torch.sum(torch.abs(x - y), dim=1, keepdim=True)
            res_value = res
            res_line = res.view(b, -1)
            res_sort = [res_line[i].sort(descending=True) for i in range(b)]
            hard_thre_ind = int(self.hard_thre_p * w * h)
            for i in range(b):
                thre_res = res_sort[i][0][hard_thre_ind]#.detach()
                hard_mask[i] = (res_value[i] > thre_res).float()

            random_thre_ind = int(self.random_thre_p * w * h)
            #random_mask = torch.cuda.FloatTensor().resize_(b, 1 * h * w).zero_()
            random_mask = torch.zeros((b, 1 * h * w), device=self.device, dtype=torch.float32)
            for i in range(b):
                random_mask[i, :random_thre_ind] = 1.
                rand_idx = torch.randperm(h*w)
                random_mask[i] = random_mask[i][rand_idx]
            random_mask = random_mask.view(b, 1, h, w)

            mask = hard_mask + random_mask
            mask = (mask > 0.).float()
            
        return mask


    def forward(self, x, y):
        mask = self.hard_mining_mask(x, y)

        hem_loss = 0.84*(1 - self.MS_SSIM(x * mask, y * mask)) + 0.16*self.L1_loss(x * mask, y * mask)

        return hem_loss
