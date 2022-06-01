# PyTorch-RPNet
Implementation of Woodscape Dataset based on [*PyTorch-RPNet*](https://github.com/superlxt/RPNet-Pytorch)

## Package is tested on 
- pytorch = 1.10.2 
- numpy = 1.19.5
- opencv-python = 4.5.5

## Training
### Windows
```ps
$ Set-ExecutionPolicy RemoteSigned -Scope CurrentUser 
$ .\train_woodscape.ps1
```
### Ubuntu
```bash
$ sh train_woodscape.sh
```

## Inference
```ps
python inference.py --model {model path} --dataset-path {dataset path} –image {image name}
```