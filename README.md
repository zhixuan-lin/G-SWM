# GSWM

This is an official PyTorch implementation of the GSWM model presented in the following paper:

> [Improving Generative Imagination in Object-Centric World Models](https://proceedings.icml.cc/static/paper_files/icml/2020/4995-Paper.pdf)  
> *[Zhixuan Lin](www.zhixuanlin.com), [Yi-Fu Wu](www.yifuwu.com), [Skand Peri](pvskand.github.io), Bofeng Fu, [Jindong Jiang](www.jindongjiang.me), [Sungjin Ahn](www.sungjinahn.com)*
> *ICML 2020*  
> [Project page](https://sites.google.com/view/gswm)   

## General

Project directories:

* `src`: source code
* `data`: where you should put the datasets
* `output`: anything the program outputs will be saved here. These include
  * `output/checkpoints`: training checkpoints. Also, model weights with the best performance will be saved here
  * `output/logs`: tensorboard event files
  * `output/eval`: quantitative evaluation results
  * `output/vis`: demo gifs
* `scripts`: some useful scripts for downloading things and showing demos
* `pretrained`: where to put downloaded pretrained models


## Dependencies

This project uses Python 3.7 and PyTorch 1.3.0. First, create a conda environment and activate it: 

```
conda create -n gswm python=3.7
conda activate gswm
```

Install PyTorch 1.3.0:

```
pip install torch==1.3.0+cu100 torchvision==0.4.1+cu100 -f https://download.pytorch.org/whl/torch_stable.html
```

Note that this requires CUDA 10.0. If you need CUDA 9.2 then change `cu100` to `cu92`. Depending on your cuda version, you may want to install previous versions of PyTorch.  See [here](https://pytorch.org/get-started/previous-versions/).

Other requirements are in `requirements.txt` and can be installed with

```
pip install -r requirements.txt
```

## Datasets

You can run the following scripts to generated the bouncing ball, maze, and single ball datasets in the paper:

```
sh scripts/gen_data_balls.sh        # (6.4G + 6.4G + 6.6G + 7.4G)
sh scripts/gen_data_maze.sh         # (12G)
sh scripts/gen_data_single_ball.sh  # (1.1G)
```

Note it can take several minutes to generate these datasets.

The 3D dataset can be downloaded from this google drive link: [OBJ3D.zip (7G)](https://drive.google.com/file/d/1XSLW3qBtcxxvV-5oiRruVTlDlQ_Yatzm/view?usp=sharing). Please download it to the `data/` directory and unzip it.
Alternatively, you can download and unzip this dataset with this script:

```
sh scripts/download_data/download_obj3d.sh
```

The `data` directory should look like this:

```
data
├── BALLS_INTERACTION
│   ├── test.hdf5
│   ├── train.hdf5
│   └── val.hdf5
├── ...
├── MAZE
│   ├── test.hdf5
│   ├── train.hdf5
│   └── val.hdf5
└── OBJ3D
    ├── test
    ├── train
    └── val
```


## Visualization with pretrained models

To help you quickly get a feeling of how the code works, we provide pretrained models for all experiments in the paper. These models can be downloaded from this Google Drive directory: [GSWM](https://drive.google.com/drive/folders/1gyuU5u4is37N7m0CBmZ83RKcELPB5wy5?usp=sharing). Please put the downloaded checkpoints (`.pth` files) to the `pretrained` directory.

We also provide scripts for downloading these checkpoints. These scripts are in the `scripts/download_pretrained/` directory. For example, if you need the model checkpoint for the maze dataset, run

```
sh scripts/download_pretrained/download_maze.sh
```

The model checkpoint `maze.pth` will be downloaded to the `pretrained/` directory. 

The `pretrained` directory should look like:

```
pretrained
├── balls_interaction.pth
├── maze.pth
├── obj3d.pth
└── ...
```

Once you have downloaded the corresponding datasets and pretrained models, you can run the following to create some gifs:

```
sh scripts/show_balls.sh      # For the bouncing ball datasets
sh scripts/show_maze.sh       # For the maze dataset
sh scripts/show_3d.sh         # For the 3D interactions dataset
```

These gifs will be saved to the `output/vis`. If you are using a remote server, you can then run `python -m http.server -d output/vis 8080` and go to port 8080 in your browser to view these gifs.


## Training and evaluation

**First, `cd src`.  Make sure you are in the `src` directory for all commands in this section. All paths referred to are also relative to `src`**.

The general command to train the model is (assuming you are in the `src` directory)

```
python main.py --task train --config [PATH TO YAML CONFIG FILE] [OTHER OPTIONS TO OVERWRITE DEFAULT YACS CONFIG...]
```

We provide configuration files for all experiments in the paper. These files are in the `config` directory:

```
configs
├── ablation
│   ├── maze_no_aoe.yaml
│   ├── maze_no_mu.yaml
│   ├── maze_no_sa.yaml
│   └── single_ball_deter.yaml
├── balls_interaction.yaml
├── balls_occlusion.yaml
├── balls_two_layer.yaml
├── balls_two_layer_dense.yaml
├── maze.yaml
├── obj3d.yaml
└── single_ball.yaml
```

For examples, suppose you want to train the model on the 3D dataset, you can run:

```
python main.py --task train --config configs/obj3d.yaml resume True device 'cuda:0'
```

By passing `device 'cuda:0'` we start training on GPU 0. There some useful options that you can specify. For example, if you want to use GPU 5, 6, 7, and 8 and resume from checkpoint `../output/checkpoints/obj3d/model_000008001.pth`, you can run the following:

```
python main.py --task train --config configs/obj3d.yaml \
	resume True resume_ckpt '../output/checkpoints/obj3d/model_000008001.pth' \
	parallel True device 'cuda:5' device_ids '[5, 6, 7, 8]'
```

Other available options are specified in `config.py`.

**Training visualization**. Run the following

```
# Run this from the 'src' directory
tensorboard --bind_all --logdir '../output/logs' --port 8848
```

And visit `http://[your server's address]:8848` in your local browser.

**Evaluation**. We provide scripts to run evaluation with the provided pretrained models (for these two scripts, you should run them from the project root instead of the `src` directory):

```
sh scripts/eval_balls.sh
sh scripts/eval_maze.sh
```

If you are to train the model by yourself, after training is finished, you can run the following to evaluate the performance for the bouncing ball datasets:

```
python main.py --task eval_balls --config configs/balls_interaction.yaml resume True device 'cuda:0' resume_ckpt ../output/eval/balls_interaction/best_med_fisrt_10.pth val.eval_types "['generation']" val.metrics "['med']" val.mode 'test' 

python main.py --task eval_balls --config configs/balls_occlusion.yaml resume True device 'cuda:0' resume_ckpt ../output/eval/balls_occlusion/best_med_fisrt_10.pth val.eval_types "['generation']" val.metrics "['med']" val.mode 'test' 

python main.py --task eval_balls --config configs/balls_two_layer.yaml resume True device 'cuda:0' resume_ckpt ../output/eval/balls_two_layer/best_med_fisrt_10.pth val.eval_types "['generation']" val.metrics "['med']" val.mode 'test' 

python main.py --task eval_balls --config configs/balls_two_layer_dence.yaml resume True device 'cuda:0' resume_ckpt ../output/eval/balls_two_layer_dense/best_med_fisrt_10.pth val.eval_types "['generation']" val.metrics "['med']" val.mode 'test' 
```
and the following for the maze dataset (including ablations):

```
python main.py --task eval_maze --config configs/maze.yaml resume True device 'cuda:0' val.mode 'test' 
python main.py --task eval_maze --config configs/maze_no_sa.yaml resume True device 'cuda:0' val.mode 'test' 
python main.py --task eval_maze --config configs/maze_no_aoe.yaml resume True device 'cuda:0' val.mode 'test' 
python main.py --task eval_maze --config configs/maze_no_mu.yaml resume True device 'cuda:0' val.mode 'test' 
```

The results will saved to the `../output/eval/[exp_name]/` directories in JSON format. Besides, figures plotting the results will also be saved there.

## Acknowledgements

The code structure is inspired (and significantly simplified) by [Mask-RCNN](https://github.com/facebookresearch/maskrcnn-benchmark) (deprecated, with the latest being [Detectron2](https://github.com/facebookresearch/maskrcnn-benchmark)) from Facebook. Google Drive download commands are created with `https://gdrive-wget.glitch.me/` 
MOT metrics are computed with [py-motmetrics](https://github.com/cheind/py-motmetrics). The maze datasets are created with [mazelib](https://github.com/theJollySin/mazelib).


