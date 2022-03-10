import os
from importlib import import_module
from pathlib import Path

import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, args, ckp):
        super(Model, self).__init__()
        print('Making model...')
        self.args = args
        self.pretrain_models_dir = args.pretrain_models_dir
        epoch = len(ckp.psnr_log)
        self.cpu = args.cpu
        self.device = torch.device('cpu' if args.cpu else 'cuda')
        self.n_GPUs = args.n_GPUs
        self.save_middle_models = args.save_middle_models

        module = import_module('model.' + args.model.lower())
        self.model = module.make_model(args).to(self.device)
        self.original_model = args.original_model
        self.load_flow_net = not(args.skip_load_flow_net)
        
        if not args.cpu and args.n_GPUs > 1:
            # Need to change to DistributedDataParallel
            self.model = nn.DataParallel(self.model, range(args.n_GPUs))

        self.load(
            ckp.dir,
            pre_train = args.pre_train,
            resume    = args.resume,
            cpu       = args.cpu
        )
        
        if args.lr_finder:
            #pass
            self.model.flow_net.moduleNetwork.moduleExtractor.moduleSix = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=128, out_channels=192, kernel_size=3, stride=2, padding=1),
                torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                torch.nn.Conv2d(in_channels=192, out_channels=192, kernel_size=3, stride=1, padding=1),
                torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                torch.nn.Conv2d(in_channels=192, out_channels=192, kernel_size=3, stride=1, padding=1),
                torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
            ).to(self.device)
            #print("reset recons_net and freeze flow_net")
            self.reset_all_weights(self.get_model().recons_net)
            #for param in self.get_model().flow_net.parameters():
            #    param.requires_grad = False

            #print("reset flow_net and freeze recons_net")
            self.reset_all_weights(self.get_model().flow_net)
            #for param in self.get_model().recons_net.parameters():
            #    param.requires_grad = False
            
        #elif False:
        elif (not args.resume and self.load_flow_net) or (self.original_model and args.pre_train is not None):
        # Update Extractor.modeuleSix due to memory leak when 196 channels was used. Because of correlation it needs to be a
        # multiple of 32. Changed 196 to 192
            self.model.flow_net.moduleNetwork.moduleExtractor.moduleSix = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=128, out_channels=192, kernel_size=3, stride=2, padding=1),
                torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                torch.nn.Conv2d(in_channels=192, out_channels=192, kernel_size=3, stride=1, padding=1),
                torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                torch.nn.Conv2d(in_channels=192, out_channels=192, kernel_size=3, stride=1, padding=1),
                torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
            ).to(self.device)
            print('flow_net.moduleNetwork.moduleExtractor.moduleSix updated')
        
        #self.model = self.model.to(self.device)
        

        if not args.lr_finder and epoch == 0:
            print(self.get_model(), file=ckp.config_file)
            ckp.config_file.close()

    def reset_all_weights(self, model: nn.Module) -> None:
        """
        refs:
            - https://discuss.pytorch.org/t/how-to-re-set-alll-parameters-in-a-network/20819/6
            - https://stackoverflow.com/questions/63627997/reset-parameters-of-a-neural-network-in-pytorch
            - https://pytorch.org/docs/stable/generated/torch.nn.Module.html
        """

        @torch.no_grad()
        def weight_reset(m: nn.Module):
            # - check if the current module has reset_parameters & if it's callable, call it on m
            reset_parameters = getattr(m, "reset_parameters", None)
            if callable(reset_parameters):
                m.reset_parameters()

        # Applies fn recursively to every submodule see: https://pytorch.org/docs/stable/generated/torch.nn.Module.html
        model.apply(fn=weight_reset)


    def forward(self, *args):
        return self.model(*args)

    def get_model(self):
        if not self.cpu and self.n_GPUs > 1:
            return self.model.module
        else:
            return self.model

    def state_dict(self, **kwargs):
        target = self.get_model()
        return target.state_dict(**kwargs)

    def save(self, apath, epoch, is_best=False):
        target = self.get_model()
        torch.save(
            target.state_dict(),
            apath / 'model' / 'model_latest.pt'
        )
        if is_best:
            torch.save(
                target.state_dict(),
                apath / 'model' / 'model_best.pt'
            )
        if self.save_middle_models:
            if epoch % 1 == 0:
                torch.save(
                    target.state_dict(),
                    apath / 'model' / 'model_{}.pt'.format(epoch)
                )

    def load(self, apath, pre_train=None, resume=False, cpu=False):  #
        if cpu:
            kwargs = {'map_location': lambda storage, loc: storage}
        else:
            kwargs = {}

    # A resume takes precedence over test_only and test_only takes precedence over pre_train
        if resume:
            print('Loading model from {}'.format(apath / 'model' / 'model_latest.pt'))
            self.get_model().load_state_dict(
                torch.load(apath / 'model' / 'model_latest.pt', **kwargs),
                strict=False
            )
            self.args.load = self.args.save
        elif self.args.test_only:
            print('Loading model from {}'.format(apath / 'model' / 'model_best.pt'))
            self.get_model().load_state_dict(
                torch.load(apath / 'model' / 'model_best.pt', **kwargs),
                strict=False
            )
        elif pre_train is not None:
            pre_train = self.pretrain_models_dir / pre_train
            print('Loading model from {}'.format(pre_train))
            self.get_model().load_state_dict(
                torch.load(pre_train, **kwargs), strict=False
            )
        else:
            pass
        

