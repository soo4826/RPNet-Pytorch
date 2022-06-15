#!/bin/bash

#level4 training (1/8 scale)
CUDA_VISIBLE_DEVICES=1 python -W ignore main.py --step=4 --weighting enet --dataset woodscape --dataset-dir /home/happy/AILabDataset/01_Open_Dataset/01_Woodscape/Woodscape --batch-size 4 --learning-rate 5e-4 --epochs 300 --height 720 --width 960 --name enet_v1 --with-unlabeled

#level3 training (1/4 scale)
CUDA_VISIBLE_DEVICES=1 python -W ignore main.py --step=3 --weighting enet --resume --dataset woodscape --dataset-dir /home/happy/AILabDataset/01_Open_Dataset/01_Woodscape/Woodscape --batch-size 4 --learning-rate 5e-4 --epochs 300 --height 720 --width 960 --name enet_v1 --with-unlabeled

#level2 training (1/2 scale)
CUDA_VISIBLE_DEVICES=1 python -W ignore main.py --step=2 --weighting enet --resume --dataset woodscape --dataset-dir /home/happy/AILabDataset/01_Open_Dataset/01_Woodscape/Woodscape --batch-size 4 --learning-rate 5e-4 --epochs 300 --height 720 --width 960 --name enet_v1 --with-unlabeled

#level1 training (original scale)
CUDA_VISIBLE_DEVICES=1 python -W ignore main.py --step=1 --weighting enet --resume --dataset woodscape --dataset-dir /home/happy/AILabDataset/01_Open_Dataset/01_Woodscape/Woodscape --batch-size 4 --learning-rate 5e-4 --epochs 300 --height 720 --width 960 --name enet_v1 --with-unlabeled