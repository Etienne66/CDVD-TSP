# CDVD-TSP

[![LICENSE](https://img.shields.io/badge/license-MIT-green.svg)](https://github.com/csbhr/CDVD-TSP/blob/master/LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.10-%237732a8.svg)](https://pytorch.org/)

#### [Paper](https://arxiv.org/abs/2004.02501) | [Project Page](https://csbhr.github.io/projects/cdvd-tsp/index.html) | [Discussion](https://github.com/csbhr/CDVD-TSP/issues)
### Cascaded Deep Video Deblurring Using Temporal Sharpness Prior[[1](#user-content-citation-1)]
By [Jinshan Pan](https://jspan.github.io/), [Haoran Bai](https://csbhr.github.io/), and Jinhui Tang

## Experimental Results
Our algorithm is motivated by the success of variational model-based methods. It explores sharpness pixels from adjacent frames by a temporal sharpness prior (see (f)) and restores sharp videos by a cascaded inference process. As our analysis shows, enforcing the temporal sharpness prior in a deep convolutional neural network (CNN) and learning the deep CNN by a cascaded inference manner can make the deep CNN more compact and thus generate better-deblurred results than both the CNN-based methods [27, 32] and variational model-based method [12].
![top-result](https://s1.ax1x.com/2020/03/31/GQnfpt.png)

We further train the proposed method to convergence, and get higher PSNR/SSIM than the result reported in the paper.

Quantitative results on the benchmark dataset by Su et al. [24]. All the restored frames instead of randomly selected 30 frames from each test set [24] are used for evaluations. *Note that: Ours * is the result that we further trained to convergence, and Ours is the result reported in the paper.*
![table-1](https://s1.ax1x.com/2020/03/31/GQOAv6.png)

Quantitative results on the GOPRO dataset by Nah et al.[20].
![table-2](https://s1.ax1x.com/2020/03/31/GQYZi8.png)

More detailed analysis and experimental results are included in [[Project Page]](https://csbhr.github.io/projects/cdvd-tsp/index.html).

## Dependencies
- We use a modified version of the implementation of PWC-Net[[3](#user-content-citation-3)] by [[sniklaus/pytorch-pwc]](https://github.com/sniklaus/pytorch-pwc)
- Windows 10
  - Linux (Not tested but should be compatible)
- [Nvidia Cuda Toolkit 11.4](https://developer.nvidia.com/cuda-downloads)
- Python 3.8 (Recommend to use [Anaconda](https://www.anaconda.com/download/))
- [PyTorch 1.9.1](https://pytorch.org/): `conda install pytorch=1.9.1 torchvision torchaudio cudatoolkit=10.2 -c pytorch`
  - *Tried setting to `cudatoolkit=11.1` but the performance decreased*
- numpy: `conda install numpy`
- matplotlib: `conda install matplotlib`
- opencv: `conda install opencv`
  - *Could not find a compatible version for Python 3.9.*
  - *Need to specify the `conda-forge` channel to install a compatible version *
  - `conda install -c conda-forge opencv`
- imageio: `conda install imageio`
- skimage: `conda install scikit-image`
- tqdm: `conda install tqdm`
- [torch-lr-finder](https://github.com/davidtvs/pytorch-lr-finder): `pip install torch-lr-finder`
- [cupy](https://github.com/cupy/cupy/): `conda install -c anaconda cupy`

### Example for Python 3.8
```dos
conda install -y pytorch=1.10 torchvision torchaudio cudatoolkit=11.3 -c pytorch
conda install -y matplotlib
conda install -y opencv
conda install -y imageio
conda install -y scikit-image
conda install -y tqdm
pip install cupy-cuda113
pip install torch-lr-finder
```

### Example for Python 3.9
```dos
conda install -y pytorch=1.10 torchvision torchaudio cudatoolkit=11.3 -c pytorch
conda install -y matplotlib
conda install -y -c conda-forge opencv
conda install -y imageio
conda install -y scikit-image
conda install -y tqdm
pip install cupy-cuda113
pip install torch-lr-finder
```

## Get Started

### Download
- Pretrained models and Datasets can be downloaded [[Here]](https://drive.google.com/drive/folders/1lw_1jITafEQ9DvMys_S6aYwtNApYKWsz?usp=sharing).
  - If you have downloaded the pretrained models，please put them to `./pretrain_models`.
    - CDVD_TSP_DVD_Paper.pt pretrained model that was done for the paper using [DeepVideoDeblurring dataset](https://github.com/shuochsu/DeepVideoDeblurring)
    - CDVD_TSP_GOPRO.pt pretrained model using [GOPRO_Large dataset](https://github.com/SeungjunNah/DeepDeblur_release)
    - CDVD_TSP_DVD_Convergent.pt pretrained model that continued until convergence
    - network-default.pytorch pretrained model for PWC-Net[[3](#user-content-citation-3)]
      - Also available from the PyTorch-PWC author and can be downloaded [[Here]](http://content.sniklaus.com/github/pytorch-pwc/network-default.pytorch).
  - If you have downloaded the datasets，please put them to `./dataset`.
    - [DeepVideoDeblurring](http://www.cs.ubc.ca/labs/imager/tr/2017/DeepVideoDeblurring/)
      - [The following 10 videos should be removed from Training and used as Test to be consistent with prior testing.](https://github.com/shuochsu/DeepVideoDeblurring/issues/2#issuecomment-312954167)

        |              |             |   Videos    |             |             |
        |:------------:|:-----------:|:-----------:|:-----------:|:-----------:|
        |720p_240fps_2 | IMG_0003    | IMG_0021    | IMG_0030    | IMG_0031    |
        |IMG_0032      | IMG_0033    | IMG_0037    | IMG_0039    | IMG_0049    |
    - [GOPRO_Large dataset](https://seungjunnah.github.io/Datasets/gopro.html)
    - [REDS dataset](https://seungjunnah.github.io/Datasets/reds)

### Dataset Organization Form
If you prepare your own dataset, please follow the following form and place it in the `./dataset` directory. `gt` directory is optional:
```
├── your_dataset
    ├── blur
    │   ├── video 1
    │   │   ├── frame 1
    │   │   ├── frame 2
    │   │   │   ⋮
    │   │   └── frame n
    │   ├── video 2
    │   │   ⋮
    │   └── video n
    └── gt
        ├── video 1
        │   ├── frame 1
        │   ├── frame 2
        │   │   ⋮
        │   └── frame n
        ├── video 2
        │   ⋮
        └── video n
```
For the training data, the form is like this for the parent directories
```
├── DVD
│   ├── train
│   │   ├── blur
│   │   │   ├── video 1
│   │   │   │    ⋮
│   │   │   └── video n
│   │   └── gt
│   │       ├── video 1
│   │       │   ⋮
│   │       └── video n
│   └── test
│       ├── blur
│       │   ├── video 1
│       │   │   ⋮
│       │   └── video n
│       └── gt
│           ├── video 1
│           │   ⋮
│           └── video n
├── GOPRO
│   └── ⋮
└── REDS
    └── ⋮
```

### Training
- Paper was trained with an Nvidia RTX Titan with 24GB RAM
  - Took 45 minutes per epoch at 300 Epochs used in the paper. CDVD_TSP_DVD_Convergent.pt was trained for 500 epochs.
  - Batch Size = 8
  - Patch Size = 256
  - Will need to reduce batch size when less RAM available or enable the `--use_checkpoint` switch
- Download the PWC-Net[[3](#user-content-citation-3)] pretrained model.
- Download training dataset, or prepare your own dataset like above form.
- Each epoch has a training phase and a testing phase to determine the PSNR/SSIM to evaluate the results.
- Run the following commands (The list of options is incomplete but these are the ones most pertinent):
```
cd ./code
python main.py --save Experiment_Name --dir_data path/to/train/dataset --dir_data_test path/to/val/dataset --epochs 500 --batch_size 8
  --save           Experiment name to save
                     The experiment result will be in '../experiment/'.
  --dir_data       comma separated list of directories in ../dataset to use for train
  --dir_data_test  comma separated list of directories in ../dataset to use for test
  --n_sequence     Set number of frames to evaluate for 1 output frame. Default 5
  --epochs         The number of training epochs. Default
  --batch_size     The mini batch size. Default 8
  --patch_size     The size to crop the data to during Training. Default 256
  --save_images    Save images during test phase of epoch. Default False
  --load           Experiment name to load
                     The experiment result must be in '../experiment/'.
                     Use --resume to continue from the last epoch completed
  --resume         Resume from the last complete epoch. Must use --load instead of --save.
  --lr             Sets learning Rate. Default 1e-4
  --max_lr         Sets maximum Learning Rate for LRFinder and OneCycleLR. Default 1e-4
  --lrfinder       Only run the LRFinder to determine your lr and max_lr for OneCycleLR. Default False
  --StepLR         Use the original StepLR Scheduler instead of OneCycleLR. Default False
  --Adam           Use the original Adam Optimizer instead of AdamW. Default False
  --LossL1HEM      Use the original 1*L1+2*HEM Loss instead of 0.84*MSL+0.16*L1. Default False
  --original_loss  Combine the losses of all stages the same as the original
  --use_checkpoint Use torch.utils.checkpoint to lower the Memory usage but take longer

```
- #### Examples of Training
  - ##### Find Learning Rate
    `python main.py --save CDVD_TSP_DVD_CLR_720 --dir_data DVD --dir_data_test DVD --epochs 150 --lr 1e-7 --max_lr 0 --use_checkpoint --batch_size 1 --patch_size 720 --lr_finder`
  - ##### Start Training
    `python main.py --save CDVD_TSP_DVD_CLR_720 --dir_data DVD --dir_data_test DVD --epochs 150 --lr 1e-6 --max_lr 1e-4 --use_checkpoint --batch_size 1 --patch_size 720`
    `python main.py --save CDVD_TSP_DVD_GOPRO_REDS_CLR_S7_604 --dir_data DVD,GOPRO,REDS --dir_data_test DVD --pre_train CDVD_TSP_DVD_Convergent.pt --epochs 50 --lr 1e-6 --max_lr 1e-4 --use_checkpoint --batch_size 1 --patch_size 604`
  - ##### Resume Training (Settings should match Start Training)
    `python main.py --resume --load CDVD_TSP_DVD_CLR_720 --dir_data DVD --dir_data_test DVD --epochs 150 --lr 1e-6 --max_lr 1e-4 --use_checkpoint --batch_size 1 --patch_size 720`
    `python main.py --resume --load CDVD_TSP_DVD_GOPRO_REDS_CLR_S7_604 --dir_data DVD,GOPRO,REDS --dir_data_test DVD --epochs 50 --lr 1e-6 --max_lr 1e-4 --use_checkpoint --batch_size 1 --patch_size 604`
  - ##### Start Training with Original settings
    This requires a 24GB GPU because of the batch size of 8.
    `python main.py --save CDVD_TSP_DVD --Adam --StepLR --LossL1HEM --original_loss --dir_data DVD --dir_data_test DVD`
    These can be done with an 8GB GPU. Using Checkpointing allows the default batch size of 8.
    `python main.py --save CDVD_TSP_DVD --Adam --StepLR --LossL1HEM --original_loss --dir_data DVD --dir_data_test DVD --batch_size 3`
    `python main.py --save CDVD_TSP_DVD --Adam --StepLR --LossL1HEM --original_loss --dir_data DVD --dir_data_test DVD --use_checkpoint`

### Testing

#### Quick Test
- Download the pretrained models.
- Download the testing dataset.
- Run the following commands:
```
cd ./code
python inference.py --default_data DVD
  # --default_data: the dataset you want to test, optional: DVD, GOPRO
```
- The deblurred result will be in `./infer_results`.

#### Test Your Own Dataset
- Download the pretrained models.
- Organize your dataset like the above form of a train directory.
- Run the following commands:
```
cd ./code
python inference.py --data_path path/to/data --model_path path/to/pretrained/model
  # --data_path: the path of your dataset.
  # --model_path: the path of the downloaded pretrained model.
```
- The deblurred result will be in `./infer_results`.

## Update History

### [2021-10-19] Added Checkpointing to save GPU memory
  * Implemented `torch.utils.checkpoint`
    * Use `--use_checkpoint` switch to enable
    * Refer to project [pytorch_memonger](https://github.com/prigoyal/pytorch_memonger/)
    * This frees up more GPU Memory allowing a larger `--batch_size` or `--patch_size`.
    * Takes longer if the `--batch_size` is not changed, but if it is increased it can take less time for the same amount of data.
      * For a Stage 2 model with a patch size of 256, an 8GB card can only do a batch size of 3 but with checkpointing it can do 13 (however batch size 10 is faster).
    * Using a larger `--patch_size` results in a more stable total loss and less chance there is a sudden gain in loss that does not recover quickly. This should result in fewer epochs before reaching convergence.
    * As a side note: The `optimizer.pt` and `scheduler.pt` files are about half the size as well.
  * Changed loss calculation to be per stage instead of just one loss for a Stage 2 or Stage 3 model per iteration. See wiki
    * Refer to forum article [What does the parameter retain_graph mean in the Variable's backward() method?](https://stackoverflow.com/a/47174709)
    * Use switch `--original_loss` to use the original `loss.backward` with an average of all losses.
    * For a stage 2 with only 1 loss calculation the Stage 1 losses would swamp out Stage 2.
    * For a stage 3 with only 1 loss calculation the Stage 1 & Stage 2 losses would swamp out Stage 3 almost entirely
    * Use switch `--separate_loss` to enable.
    * Per stage loss calculation is equivalent to the sum of the loss for each stage and compute the `loss.backward` once. This is now the default.
      * Refer to forum article [What exactly does `retain_variables=True` in `loss.backward()` do?](https://discuss.pytorch.org/t/what-exactly-does-retain-variables-true-in-loss-backward-do/3508/25)
  * Added Hard Example Mining for the loss function `0.84*MSL + 0.16*L1` as `HML`
    * `MSL = 1 - MS-SSIM`
    * `HML = `Hard Example Mining for `0.84*MSL + 0.16*L1`
    * `HEM = `Hard Example Mining for `L1`
    * The default loss function is now `0.84*MSL + 0.16*L1 + 2*HML`
    * Use `--LossMslL1` to use loss function `0.84*MSL + 0.16*L1`
    * Use `--LossL1HEM` to use Pan's original loss function `1*L1 + 2*HEM`
  * Added `CyclicLR` with `torch.optim.SGD` because the total loss is not stable enough for `OneCycleLR` when the `--patch_size` is only 256.
    * Default Optimizer is now `SGD`. Use `--OneCycleLR` or `--StepLR` to use `AdamW` or `--Adam` to use the original optimizer.
    * Default Scheduler is now `CyclicLR` which does not work with either `Adam` or `AdamW` optimizers. Use `--OneCycleLR` or `--StepLR` to use other schedulers.
    * It is possible to use `--OneCycleLR` if the `--patch_size` is around 720.
    * Default Scheduler is now `SGD`. Use 
    * Refer to paper [Cyclical Learning Rates for Training Neural Networks](https://arxiv.org/pdf/1506.01186.pdf)
  * Logging changes
    * Changed logging to show the losses and gains for the different stages. This will provide a more accurate picture of what is happening.
    * Breakdown of how much time was spent on each part of the train/test phase printed to screen and written to `log.txt`.
    * When writing to `log.txt` file, it wasn't flushing it to the disk and caused log information to be lost when the computer crashes or loses power.
      * Added `self.log_file.flush()` to flush the buffers to the OS
      * Added `os.fsync(self.log_file.fileno())` to sync all internal buffers with the file.
    * Removed date from filename of `log.txt` and `config.txt`.
    * Moved writing of model to the `config.txt` file.
    * When using `--resume`, append a few arguments to the `config.txt` file instead of writing the same information again. Most arguments should not be changed when resuming.
    * When using `--resume` and a `log.txt` exists there is now a prompt to overwrite the file and continue or abort.
  * Plot changes
    * When saving existing plots in Windows and the plot is open by a viewer, there is a chance that the plot can't be overwritten. A new image is saved with the date/time added to the file name.
    * `loss.png`, `MS-SSIM.png`, `psnr.png`, and `ssim.png` now have separate plot lines for each stage.
    * All plots changed to 4k resolution at the default 100 dpi.
    * Added a graph of the total loss during the epoch per iteration
      * Use `--plot_running_average` to enable
      * Uses `--running_average` with a default of 2000 iterations
      * Saves the graph when the current `(batch + 1) % running_average == 0` or at the end of the current epoch.
      * Saves a new graph for each epoch because it is easier to view than one graph with millions of data points.
  * Added `random.getstate()` to data that is saved at the end of each epoch.
    * Added `random.seed()` in `main.py` but still unable to get 2 runs to be exactly the same.
    * Also save `np.random.get_state` and initialize `np.random.seed(args.seed)` in `main.py`
  * Changed `dir_data` and `dir_data_test` to allow for multiple datasets
    * Now each just needs a comma separated list of folders in the `../dataset/` directory of datasets to be used for training or testing.
    * Default is `DVD` but tested with `DVD,GOPRO,REDS`
  * Made corrections to paths to be more compatible with both Windows and Linux using `pathlib.Path`.
  * Many changes to minimize the transfer from CPU memory to GPU memory
  * Changed the order of precedence to be `--resume`, `--test_only` and `--pre_train` to prevent issues like using `--resume` and `--pre_train`.
    * `--pre_train` should only be used with `--save` and not with `--resume --load` or `--test_only` but now it won't cause a problem if it is done by accident.
  * Stopped loading the pre-trained model for Flow_PWC when doing a `--resume` or `--pre_train` as it is unnecessary. The Flow_PWC model is loaded as part of the RECONS_VIDEO model.
  * Updated `correlation.py` & `flow_pwc.py` to be in sync with [pytorch-pwc](https://github.com/sniklaus/pytorch-pwc)
    * Added several notes to pytorch-pwc code to make it easier to understand.
  * Cleaned up `videodata.py`
    * `VIDEODATA.__getitem__` changed input & gt tuples from 4 dimensions to 3 and back again unnecessarily.
      * `NHWC` -> `N` is number of frames, `H` is height, `W` is width, `C` is channels (Normally 3 channels for Red, Green, & Blue)
      * `NHWC -> HWC` where C is channels and the number of channels is increased to `N*C` when going from 4 to 3 dimensions. i.e. for 5 frames it becomes 15 channels.
      * Keep 4 dimensions and change functions called to support 4 dimensions, `NHWC`.
    * Changed `VIDEODATA.get_patch` to new function `VIDEODATA.get_patch_frames` which supports 4 dimensions
    * Created new supporting functions in `utils.py`: `utils.get_patch_frames` and `util.data_augment_frames` which support `NHWC`
  * Removed Sobel Gradient from `flow_pwc.py` because it is not being used.
    * It appears to have been something Pan tried, but decided against.
    * Code was still wasting time and GPU Memory calculating it.
    * See this article for a better understanding of Sobel Gradient: [Image Processing Project Entry 4: Gradient](https://danilchatuncev.wordpress.com/2018/04/15/image-processing-project-entry-4-gradient/)


### [2021-07-10] Many revisions to code
  * Changed loss function to `0.84*MSL + 0.16*L1`
    * `MSL = (1 - MS-SSIM)` Needed so that it acts like a loss function
    * The research for "Loss Functions For Image Restoration With Neural Networks" indicates that this loss function gives results that are more visually appealing.
    * [Research Paper](https://arxiv.org/pdf/1511.08861.pdf)
    * [Source Code](https://github.com/NVlabs/PL4NN)
    * [Supplementary Material](http://research.nvidia.com/publication/loss-functions-image-restoration-neural-networks)
    * Used MM-SSIM function from [Fast and differentiable MS-SSIM and SSIM for pytorch. ](https://github.com/VainF/pytorch-msssim)
    * Added `--LossL1HEM` switch to use original loss function
  * Changed scheduler from `StepLR` to `OneCycleLR` and changed optimizer from `Adam` to `AdamW` to implement [Super-Convergence](https://towardsdatascience.com/super-convergence-with-just-pytorch-c223c0fc1e51)
    * A better explanation of [The 1cycle policy](https://sgugger.github.io/the-1cycle-policy.html)
    * Implemented [PyTorch learning rate finder](https://github.com/davidtvs/pytorch-lr-finder)
      * For loss function `0.84*MSL+0.16*L1` the ranges is `min_lr = 6e-6` to `max_lr = 2.5e-5`
        * This is with 5 frames, batch size of 3, and patch size of 256
      * For original loss function `1*L1+2*HEM` the ranges is `min_lr = 3e-6` to `max_lr = 8e-6`
        * Original starting `lr = 1e-4` which is far above the range that is stable
      * Added flags `--StepLR` and `--Adam` to change back to original
  * Change logging to not use a buffer. Lines were being lost when power was lost.
  * Followed advice of [Tricks for training PyTorch models to convergence more quickly](https://spell.ml/blog/pytorch-training-tricks-YAnJqBEAACkARhgD)
    * Added `torch.backends.cudnn.benchmark = True` to improve performance
  * Added Save State for Scheduler, Random Number Generator State and a few variables
  * Corrected mistakes with `--resume`
  * Added graph for Learning Rate
  * Added graph for MS-SSIM during training session if used in Loss function
  * Added graph for SSIM during testing phase of training session

### [2021-05-23] Many improvements to code.
- Several warnings and errors were addressed
  - Issues Fixed [#12](https://github.com/csbhr/CDVD-TSP/issues/12), [#22](https://github.com/csbhr/CDVD-TSP/issues/22), [#24](https://github.com/csbhr/CDVD-TSP/issues/24), [#26](https://github.com/csbhr/CDVD-TSP/issues/26)
  - PyTorch 1.8.1, Python 3.8, & Cuda 10.2 has been tested
  - PyTorch 0.4.1, Python 3.7, & Cuda 9.2 should still work
    - However, this hasn't been verified due to test computer unable to run Cuda 9.2
    - It appears that the graphics card is too new for Cuda 9.2
  - Inference no longer requires gt images [#12](https://github.com/csbhr/CDVD-TSP/issues/12), [#22](https://github.com/csbhr/CDVD-TSP/issues/22), [#26](https://github.com/csbhr/CDVD-TSP/issues/26)
  - Added `if __name__ == '__main__':` to `main.py` to resolve [#24](https://github.com/csbhr/CDVD-TSP/issues/24)
  - `optimizer.step()` was added prior to `lr_scheduler.step()`
    - Failure to do this will result in PyTorch skipping the first value of the learning rate schedule. See more details [Here](https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate)
  - Added align_corners=True parameter to `nn.functional.grid_sample` in `flow_pwc.py` since that was the behavior for PyTorch 0.4.1.
    - This behavior was changed in PyTorch 1.3.0 from `True` to `False`. See more details [Here](https://github.com/pytorch/pytorch/releases/tag/v1.3.0)
  - Windows cannot have a colon in a filename.
    - Changed filenames with date time from `YYYY-MM-DD hh:mm:ss` to `YYYY-MM-DDThhmmss` per [ISO 8601](https://en.wikipedia.org/wiki/ISO_8601)
- Changed `log.txt` and `config.txt` files to have the time added to them so that each new run will have a new file. Especially handy for a `--resume`
- `n_frames_per_video` increased from 100 to 200 in order to take full advantage of all of the frames in the training datasets.
- Images are no longer saved by default during test phase of epoch.
  - There was no way to disable with a switch prior to this change
  - Can be enabled with `--save_images` option
- Now gives an estimate of completion time for the current epoch and all epochs
  - 1st training session estimate does not include the 1st test session duration for the 1st epoch.
  - When using `--resume --load`, start time will be recalculated based on the prior elapsed times
- PDF plots no longer created during 1st epoch due to lack of data
  - L1, HEM, & Total Loss plots are now combined in one plot instead of 3
  - PSNR plot no longer has a legend since it was blank
- Inference will handle border situations like this
  - For a video with 5 frames [1, 2, 3, 4, 5] it will use a list of frames [3, 2, 1, 2, 3, 4, 5, 4, 3]
  - Previous handling of the same clip would produce a singe frame [3]
  - The new result will be frames [1, 2, 3, 4, 5] with all 5 frames being deblurred

### [2020-10-22] Inference results on DVD and GOPRO are available [[Here]](https://drive.google.com/drive/folders/1lMpj-fkT89JfMOvTnqXMzfa57zCT8-2s?usp=sharing)!

### [2020-10-10] Metrics(PSNR/SSIM) calculating codes are available [[Here]](https://github.com/csbhr/OpenUtility#chapter-calculating-metrics)!

### [2020-08-04] Inference logs are available [[Here]](https://drive.google.com/drive/folders/1lMpj-fkT89JfMOvTnqXMzfa57zCT8-2s?usp=sharing)!

### [2020-03-07] Paper is available!

### [2020-03-31] We further train the model to convergence, and the pretrained model is available!

### [2020-03-07] Add training code!

### [2020-03-04] Testing code is available!

## Citations
### Citation 1
```
@InProceedings{Pan_2020_CVPR,
    author = {Pan, Jinshan and Bai, Haoran and Tang, Jinhui},
    title = {Cascaded Deep Video Deblurring Using Temporal Sharpness Prior},
    booktitle = {IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month = {June},
    year = {2020}
}
```
### Citation 2
```
@misc{CDVD-TSP,
    author = {Haoran Bai},
    title = {Cascaded Deep Video Deblurring Using Temporal Sharpness Prior},
    year = {2020},
    howpublished = {\url{https://github.com/csbhr/CDVD-TSP}}
}
```
### Citation 3
```
@misc{pytorch-pwc,
    author = {Simon Niklaus},
    title = {A Reimplementation of {PWC-Net} Using {PyTorch}},
    year = {2018},
    howpublished = {\url{https://github.com/sniklaus/pytorch-pwc}}
}
```
