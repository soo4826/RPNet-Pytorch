#!/bin/bash

#level4 training (1/8 scale)
# python -W ignore main.py --step=4 --dataset woodscape --dataset-dir data/Woodscape --batch-size 8 --learning-rate 1e-5 --imshow-batch --print-step

#level3 training (1/4 scale)
python -W ignore main.py --step=3 --resume --dataset woodscape --dataset-dir data/Woodscape --batch-size 8 --learning-rate 1e-5 --imshow-batch --print-step

#level2 training (1/2 scale)
python -W ignore main.py --step=2 --resume --dataset woodscape --dataset-dir data/Woodscape --batch-size 8 --learning-rate 1e-5 --imshow-batch --print-step

#level1 training (original scale)
python -W ignore main.py --step=1 --resume --dataset woodscape --dataset-dir data/Woodscape --batch-size 8 --learning-rate 1e-5 --imshow-batch --print-step

