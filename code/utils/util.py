import os
import numpy as np
import shutil
import torch


def save_checkpoint(state, is_best, save_path, filename='checkpoint.pth.tar'):
    torch.save(state, os.path.join(save_path,filename))
    if is_best:
        shutil.copyfile(os.path.join(save_path,filename), os.path.join(save_path,'model_best.pth.tar'))


class AverageMeter(object):
    """Computes and stores the average and current value
    AverageMeter.sum is a weighted sum
    AverageMeter.avg is a weighted average
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.min = 0
        self.max = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.min = min(self.min, val)
        self.max = max(self.max, val)
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __repr__(self):
        return '{:.5f} ({:.5f})'.format(self.val, self.avg)


def flow2rgb(flow_map, max_value=None):
    #Changed function to center zero around 0.5. Before it was always clipping the values
    flow_map_np = flow_map.detach().cpu().numpy()
    _, h, w = flow_map_np.shape
    # This assumed all values would be between 0 and 1
    #flow_map_np[:,(flow_map_np[0] == 0) & (flow_map_np[1] == 0)] = float('nan')
    rgb_map = np.ones((3,h,w)).astype(np.float32)*0.5
    if max_value is not None:
        normalized_flow_map = flow_map_np / (max_value * 2)
    else:
        normalized_flow_map = flow_map_np / (np.abs(flow_map_np).max() * 2)
    rgb_map[0] += normalized_flow_map[0]
    rgb_map[1] -= 0.5*(normalized_flow_map[0] + normalized_flow_map[1])
    rgb_map[2] += normalized_flow_map[1]
    return rgb_map.clip(0,1)
