import torch
import torch.nn as nn
from model import recons_video
from model import flow_pwc
from utils import utils


def make_model(args):
    device = 'cpu' if args.cpu else 'cuda'
    load_flow_net = True
    load_recons_net = False
    flow_pretrain_fn = args.pretrain_models_dir + 'network-default.pytorch'
    recons_pretrain_fn = ''
    is_mask_filter = True
    return CDVD_TSP(in_channels=args.n_colors, n_sequence=args.n_sequence, out_channels=args.n_colors,
                    n_resblock=args.n_resblock, n_feat=args.n_feat,
                    load_flow_net=load_flow_net, load_recons_net=load_recons_net,
                    flow_pretrain_fn=flow_pretrain_fn, recons_pretrain_fn=recons_pretrain_fn,
                    is_mask_filter=is_mask_filter, device=device)


class CDVD_TSP(nn.Module):

    def __init__(self, in_channels=3, n_sequence=3, out_channels=3, n_resblock=3, n_feat=32,
                 load_flow_net=False, load_recons_net=False, flow_pretrain_fn='', recons_pretrain_fn='',
                 is_mask_filter=False, device='cuda'):
        super(CDVD_TSP, self).__init__()
        print("Creating CDVD-TSP Net")

        self.n_sequence = n_sequence
        self.device = device

        assert n_sequence in [3,5,7], "Only support args.n_sequence=5; but get args.n_sequence={}".format(n_sequence)

        self.is_mask_filter = is_mask_filter
        print('Is meanfilter image when process mask:', 'True' if is_mask_filter else 'False')
        extra_channels = 1
        print('Select mask mode: concat, num_mask={}'.format(extra_channels))

        self.flow_net = flow_pwc.Flow_PWC(load_pretrain = load_flow_net,
                                          pretrain_fn   = flow_pretrain_fn,
                                          device        = device)
        self.recons_net = recons_video.RECONS_VIDEO(in_channels    = in_channels,
                                                    n_sequence     = 3,
                                                    out_channels   = out_channels,
                                                    n_resblock     = n_resblock,
                                                    n_feat         = n_feat,
                                                    extra_channels = extra_channels)
        if load_recons_net:
            self.recons_net.load_state_dict(torch.load(recons_pretrain_fn))
            print('Loading reconstruction pretrain model from {}'.format(recons_pretrain_fn))

    def get_masks(self, img_list, flow_mask_list):
        num_frames = len(img_list)

        img_list_copy = [img.detach() for img in img_list]  # detach backward
        if self.is_mask_filter:  # mean filter
            img_list_copy = [utils.calc_meanFilter(im, n_channel=3, kernel_size=5) for im in img_list_copy]

        delta = 1.
        mid_frame = img_list_copy[num_frames // 2]
        diff = torch.zeros_like(mid_frame)
        for i in range(num_frames):
            diff = diff + (img_list_copy[i] - mid_frame).pow(2)
        del img_list_copy
        del mid_frame
        diff = diff / (2 * delta * delta)
        diff = torch.sqrt(torch.sum(diff, dim=1, keepdim=True))
        luckiness = torch.exp(-diff)  # (0,1)
        del diff

        sum_mask = torch.ones_like(flow_mask_list[0])
        for i in range(num_frames):
            sum_mask = sum_mask * flow_mask_list[i]
        sum_mask = torch.sum(sum_mask, dim=1, keepdim=True)
        sum_mask = (sum_mask > 0).float()
        luckiness = luckiness * sum_mask
        del sum_mask

        return luckiness

    def forward(self, x):
        frame_list = [x[:, i, :, :, :] for i in range(self.n_sequence)]

        # Iteration 1
        warped01, _, _, flow_mask01 = self.flow_net(frame_list[1], frame_list[0])
        warped21, _, _, flow_mask21 = self.flow_net(frame_list[1], frame_list[2])
        one_mask = torch.ones_like(flow_mask01) #Tensor filled with scalar value 1
        frame_warp_list = [warped01, frame_list[1], warped21]
        flow_mask_list = [flow_mask01, one_mask.detach(), flow_mask21]
        del flow_mask01
        del flow_mask21
        luckiness = self.get_masks(frame_warp_list, flow_mask_list)
        concated = torch.cat([warped01, frame_list[1], warped21, luckiness], dim=1)
        del warped01
        del warped21
        recons_11, _ = self.recons_net(concated)

        # Iteration 2
        if self.n_sequence >= 5:
            warped12, _, _, flow_mask12 = self.flow_net(frame_list[2], frame_list[1])
            warped32, _, _, flow_mask32 = self.flow_net(frame_list[2], frame_list[3])
            frame_warp_list = [warped12, frame_list[2], warped32]
            flow_mask_list = [flow_mask12, one_mask.detach(), flow_mask32]
            del flow_mask12
            del flow_mask32
            luckiness = self.get_masks(frame_warp_list, flow_mask_list)
            concated = torch.cat([warped12, frame_list[2], warped32, luckiness], dim=1)
            del warped12
            del warped32
            recons_12, _ = self.recons_net(concated)

            warped23, _, _, flow_mask23 = self.flow_net(frame_list[3], frame_list[2])
            warped43, _, _, flow_mask43 = self.flow_net(frame_list[3], frame_list[4])
            frame_warp_list = [warped23, frame_list[3], warped43]
            flow_mask_list = [flow_mask23, one_mask.detach(), flow_mask43]
            del flow_mask23
            del flow_mask43
            luckiness = self.get_masks(frame_warp_list, flow_mask_list)
            concated = torch.cat([warped23, frame_list[3], warped43, luckiness], dim=1)
            del warped23
            del warped43
            recons_13, _ = self.recons_net(concated)

            warped2_recons12, _, _, flow_mask2_recons12 = self.flow_net(recons_12, recons_11)
            warped2_recons32, _, _, flow_mask2_recons32 = self.flow_net(recons_12, recons_13)
            frame_warp_list = [warped2_recons12, recons_12, warped2_recons32]
            flow_mask_list = [flow_mask2_recons12, one_mask.detach(), flow_mask2_recons32]
            del flow_mask2_recons12
            del flow_mask2_recons32
            luckiness = self.get_masks(frame_warp_list, flow_mask_list)
            concated = torch.cat([warped2_recons12, recons_12, warped2_recons32, luckiness], dim=1)
            del warped2_recons12
            del warped2_recons32
            recons_22, _ = self.recons_net(concated)
        else:
            output_cat = torch.cat([recons_11], dim=1)
            out_image = recons_11
            del recons_11

        # Iteration 3
        if self.n_sequence == 7:
            warped34, _, _, flow_mask34 = self.flow_net(frame_list[4], frame_list[3])
            warped54, _, _, flow_mask54 = self.flow_net(frame_list[4], frame_list[5])
            frame_warp_list = [warped34, frame_list[4], warped54]
            flow_mask_list = [flow_mask34, one_mask.detach(), flow_mask54]
            del flow_mask34
            del flow_mask54
            luckiness = self.get_masks(frame_warp_list, flow_mask_list)
            concated = torch.cat([warped34, frame_list[4], warped54, luckiness], dim=1)
            del warped34
            del warped54
            recons_14, _ = self.recons_net(concated)

            warped45, _, _, flow_mask45 = self.flow_net(frame_list[5], frame_list[4])
            warped65, _, _, flow_mask65 = self.flow_net(frame_list[5], frame_list[6])
            frame_warp_list = [warped45, frame_list[5], warped65]
            flow_mask_list = [flow_mask45, one_mask.detach(), flow_mask65]
            del flow_mask45
            del flow_mask65
            luckiness = self.get_masks(frame_warp_list, flow_mask_list)
            concated = torch.cat([warped45, frame_list[5], warped65, luckiness], dim=1)
            del warped45
            del warped65
            recons_15, _ = self.recons_net(concated)
            
            warped2_recons23, _, _, flow_mask2_recons23 = self.flow_net(recons_13, recons_12)
            warped2_recons43, _, _, flow_mask2_recons43 = self.flow_net(recons_13, recons_14)
            frame_warp_list = [warped2_recons23, recons_13, warped2_recons43]
            flow_mask_list = [flow_mask2_recons23, one_mask.detach(), flow_mask2_recons43]
            del flow_mask2_recons23
            del flow_mask2_recons43
            luckiness = self.get_masks(frame_warp_list, flow_mask_list)
            concated = torch.cat([warped2_recons23, recons_13, warped2_recons43, luckiness], dim=1)
            del warped2_recons23
            del warped2_recons43
            recons_23, _ = self.recons_net(concated)

            warped2_recons34, _, _, flow_mask2_recons34 = self.flow_net(recons_14, recons_13)
            warped2_recons54, _, _, flow_mask2_recons54 = self.flow_net(recons_14, recons_15)
            frame_warp_list = [warped2_recons34, recons_14, warped2_recons54]
            flow_mask_list = [flow_mask2_recons34, one_mask.detach(), flow_mask2_recons54]
            del flow_mask2_recons34
            del flow_mask2_recons54
            luckiness = self.get_masks(frame_warp_list, flow_mask_list)
            concated = torch.cat([warped2_recons34, recons_14, warped2_recons54, luckiness], dim=1)
            del warped2_recons34
            del warped2_recons54
            recons_24, _ = self.recons_net(concated)

            warped3_recons23, _, _, flow_mask3_recons23 = self.flow_net(recons_23, recons_22)
            warped3_recons43, _, _, flow_mask3_recons43 = self.flow_net(recons_23, recons_24)
            frame_warp_list = [warped3_recons23, recons_13, warped3_recons43]
            flow_mask_list = [flow_mask3_recons23, one_mask.detach(), flow_mask3_recons43]
            del flow_mask3_recons23
            del flow_mask3_recons43
            luckiness = self.get_masks(frame_warp_list, flow_mask_list)
            concated = torch.cat([warped3_recons23, recons_13, warped3_recons43, luckiness], dim=1)
            del warped3_recons23
            del warped3_recons43
            recons_33, _ = self.recons_net(concated)

            output_cat = torch.cat([recons_11, recons_12, recons_13, recons_22, recons_14,
                                    recons_15, recons_23, recons_24, recons_33], dim=1)
            out_image = recons_33
            del recons_11
            del recons_12
            del recons_13
            del recons_22
            del recons_14
            del recons_15
            del recons_23
            del recons_24
            del recons_33
        else:
            output_cat = torch.cat([recons_11, recons_12, recons_13, recons_22], dim=1)
            out_image = recons_22
            del recons_11
            del recons_12
            del recons_13
            del recons_22
            #return recons_11, recons_12, recons_13, recons_22

        del frame_list
        del flow_mask_list
        del luckiness
        del concated

        return output_cat, out_image
