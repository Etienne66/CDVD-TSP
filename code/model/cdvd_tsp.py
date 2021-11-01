import torch
import torch.nn as nn
from model import recons_video
from model import flow_pwc
from utils import utils


def make_model(args):
    device = 'cpu' if args.cpu else 'cuda'
# No need to load the pretrained flow network if doing a resume or loading a pretrained deblur model
    load_flow_net = False if args.resume or args.pre_train != '.' else True
    load_recons_net = False
    flow_pretrain_fn = args.pretrain_models_dir / 'network-default.pytorch'
    recons_pretrain_fn = ''
    is_mask_filter = True
    return CDVD_TSP(in_channels        = args.n_colors,
                    n_sequence         = args.n_sequence,
                    out_channels       = args.n_colors,
                    n_resblock         = args.n_resblock,
                    n_feat             = args.n_feat,
                    load_flow_net      = load_flow_net,
                    load_recons_net    = load_recons_net,
                    flow_pretrain_fn   = flow_pretrain_fn,
                    recons_pretrain_fn = recons_pretrain_fn,
                    is_mask_filter     = is_mask_filter,
                    device             = device,
                    use_checkpoint     = args.use_checkpoint)


class CDVD_TSP(nn.Module):

    def __init__(self,
                 in_channels        = 3,
                 n_sequence         = 3,
                 out_channels       = 3,
                 n_resblock         = 3,
                 n_feat             = 32,
                 load_flow_net      = False,
                 load_recons_net    = False,
                 flow_pretrain_fn   = '',
                 recons_pretrain_fn = '',
                 is_mask_filter     = False,
                 device             = 'cuda',
                 use_checkpoint     = False):
        super(CDVD_TSP, self).__init__()
        print("Creating CDVD-TSP Net")

        self.n_sequence = n_sequence
        self.device = device

        assert n_sequence in [3,5,7], "Only support args.n_sequence in [3,5,7]; but get args.n_sequence={}".format(n_sequence)

    # Temporal Sharpness Prior
        self.is_mask_filter = is_mask_filter
        print('Is meanfilter image when process mask:', 'True' if is_mask_filter else 'False')
        extra_channels = 1
        print('Select mask mode: concat, num_mask={}'.format(extra_channels))

    # This is the pytorch-pwc model developed by sniklaus for optical flow estimation
    # The model will be trained as well but their is no direct loss calculation since there are no optical flow estimations
    # ground truth for the DVD, GOPRO or REDS datasets.
        self.flow_net = flow_pwc.Flow_PWC(load_pretrain  = load_flow_net,
                                          pretrain_fn    = flow_pretrain_fn,
                                          device         = device,
                                          use_checkpoint = use_checkpoint)

    # recons_net only works on 3 frames. More frames are done by cascaded training
        self.recons_net = recons_video.RECONS_VIDEO(in_channels    = in_channels,
                                                    n_sequence     = 3,
                                                    out_channels   = out_channels,
                                                    n_resblock     = n_resblock,
                                                    n_feat         = n_feat,
                                                    extra_channels = extra_channels,
                                                    use_checkpoint = use_checkpoint)
        if load_recons_net:
            print('Loading reconstruction pretrain model from {}'.format(recons_pretrain_fn))
            self.recons_net.load_state_dict(torch.load(recons_pretrain_fn))

    def get_masks(self, img_list, flow_mask_list):
    # Temporal Sharpness Prior
    # Which pixels of the adjacent frames are sharp
        num_frames = len(img_list)

        #img_list_copy = [img.detach() for img in img_list]  # detach backward
        if self.is_mask_filter:  # mean filter
            #img_list_copy = [utils.calc_meanFilter(im, n_channel=3, kernel_size=5, device=self.device) for im in img_list_copy]
            img_list_copy = [utils.calc_meanFilter_torch(img.detach(),
                                                         n_channel   = 3,
                                                         kernel_size = 5,
                                                         device      = self.device) for img in img_list]
        else:
            img_list_copy = [img.detach() for img in img_list]  # detach backward

        delta = 1.
        mid_frame = img_list_copy[num_frames // 2]
        diff = torch.zeros_like(mid_frame)
        for i in range(num_frames):
            diff += (img_list_copy[i] - mid_frame).pow(2)
        diff /= 2 * delta * delta
        diff = torch.sqrt(torch.sum(diff, dim=1, keepdim=True))
        luckiness = torch.exp(-diff)  # (0,1)

        sum_mask = torch.ones_like(flow_mask_list[0])
        for i in range(num_frames):
            sum_mask *= flow_mask_list[i]
        sum_mask = torch.sum(sum_mask, dim=1, keepdim=True)
        sum_mask = (sum_mask > 0).float()
        luckiness *= sum_mask

        return luckiness


    def reconstruct(self, Frame0, Frame1, Frame2):
        """ Reconstruct Frame 1 """
        # Optical Flow Estimation
        warped01, _, flow_mask01 = self.flow_net(Frame1, Frame0) # Forward flow of Frame 0 warped into Frame 1
        warped21, _, flow_mask21 = self.flow_net(Frame1, Frame2) # Backward flow of Frame 2 warped into Frame 1

        # Flows used to determine the temporal sharpness of adjacent frames
        one_mask = torch.ones_like(flow_mask01) #Tensor filled with scalar value 1
        frame_warp_list = [warped01, Frame1, warped21]
        flow_mask_list = [flow_mask01, one_mask, flow_mask21]

        # Get the Temporal Sharpness Prior mask
        luckiness = self.get_masks(frame_warp_list, flow_mask_list)

        concated = torch.cat([warped01, Frame1, warped21, luckiness], dim=1)
        reconstructed, _ = self.recons_net(concated)

        return reconstructed


    def forward(self, x):
        # Stage 1
        recons_11 = self.reconstruct(x[:, 0, :, :, :], x[:, 1, :, :, :], x[:, 2, :, :, :])

        if self.n_sequence == 3:
            out_image = recons_11
            output_cat = torch.cat([recons_11], dim=1)
        elif self.n_sequence >= 5:
            recons_12 = self.reconstruct(x[:, 1, :, :, :], x[:, 2, :, :, :], x[:, 3, :, :, :])
            recons_13 = self.reconstruct(x[:, 2, :, :, :], x[:, 3, :, :, :], x[:, 4, :, :, :])

        # Stage 2
            recons_22 = self.reconstruct(recons_11, recons_12, recons_13)

        if self.n_sequence == 5:
            out_image = recons_22
            output_cat = torch.cat([recons_11, recons_12, recons_13,
                                    recons_22], dim=1)
        elif self.n_sequence == 7:
            recons_14 = self.reconstruct(x[:, 3, :, :, :], x[:, 4, :, :, :], x[:, 5, :, :, :])
            recons_15 = self.reconstruct(x[:, 4, :, :, :], x[:, 5, :, :, :], x[:, 6, :, :, :])
            
            recons_23 = self.reconstruct(recons_12, recons_13, recons_14)
            recons_24 = self.reconstruct(recons_13, recons_14, recons_15)

        # Stage 3
            recons_33 = self.reconstruct(recons_22, recons_23, recons_24)

            out_image = recons_33
            output_cat = torch.cat([recons_11, recons_12, recons_13, recons_14, recons_15,
                                    recons_22, recons_23, recons_24,
                                    recons_33], dim=1)

        return output_cat
