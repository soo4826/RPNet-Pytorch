#!/bin/bash

#level4 training (1/8 scale)
python main.py --step=4 --dataset camvid --dataset-dir dataset/CamVid

#level3 training (1/4 scale)
python main.py --step=3 --resume --dataset camvid --dataset-dir dataset/CamVid

#level2 training (1/2 scale)
python main.py --step=2 --resume --dataset camvid --dataset-dir dataset/CamVid

#level1 training (original scale)
python main.py --step=1 --resume --dataset camvid --dataset-dir dataset/CamVid
