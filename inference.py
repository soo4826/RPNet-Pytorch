from numpy import size
import torch
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import transforms as ext_transforms
import torchvision.transforms as transforms
from models.rpnet import RPNet

import os
from argparse import ArgumentParser

## Label map (Modified)
# 0: Unlabeled (000)
# 1: road (101)
# 2: lanemarks (011)
# 3: curb (010)
# 4: person (100)
# 5: rider (111)
# 7: bicycle (011)
# 6: vehicles (110)
# 8: motorcycle (10.50.5)
# 9: traffic_sign(0.50.50)

def decode_segmap(label_mask, num_classes):
    label_colors= np.array([
            [0  ,0  ,0  ], # unlabeled
            [255,0,255  ], # road
            [0  ,0  ,255], # lanemarks
            [0,  255,0  ], # curb
            [255,0  ,0  ], # person
            [255,255,255], # rider
            [0  ,255,255], # bicycle
            [255,255,0  ], # vehicles
            [255,128,128], # motorcycle
            [128,128,0  ]  # traffic_sign
    ])

    r = label_mask.copy()
    g = label_mask.copy()
    b = label_mask.copy()
    for ll in range(0, num_classes):
        r[label_mask == ll] = label_colors[ll, 0]
        g[label_mask == ll] = label_colors[ll, 1]
        b[label_mask == ll] = label_colors[ll, 2]
    rgb = np.zeros((label_mask.shape[0], label_mask.shape[1], 3))
    rgb[:, :, 0] = r/255.
    rgb[:, :, 1] = g/255.
    rgb[:, :, 2] = b/255.
    return rgb


def transform(image):
    return transforms.Compose([
         transforms.Resize((height, width),Image.BILINEAR),
         transforms.ToTensor(),
         transforms.Normalize(mean=(0.3257, 0.3690, 0.3223),
            std=(0.2112, 0.2148, 0.2115))
    ])(image)


def transform_gt(image):
    return transforms.Compose([
        transforms.Resize((height,width), Image.BILINEAR),
    ])(image)

if __name__=="__main__":
    # Argparser for inference options
    parser = ArgumentParser()
    
    parser.add_argument(
    "--height",
    type=int,
    default=968,
    help="Hight of input image (default: 968)")
    parser.add_argument(
    "--width",
    type=int,
    default=1280,
    help="width of input image (default: 1280)")
    parser.add_argument(
    "--model",
    type=str,
    default="save/RPNet",
    help="Path to model parameter (default: save/RPNet")
    parser.add_argument(
    "--dataset",
    type=str,
    default="woodscape",
    choices=['valeo', 'woodscape'],
    help="Dataset to use (default: woodscape")
    parser.add_argument(
    "--dataset-path",
    type=str,
    default="data/Woodscape",
    help="Dataset path to use (Default: data/Woodscape")
    parser.add_argument(
    "--image-name",
    type=str,
    default="02958_MVR",
    help="image file name in dataset (default: 02958_MVR)")
    
    
    args = parser.parse_args()

    # Configure input image size
    height, width = args.height, args.width
    
    # Enables GPU if possible
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Load trained model parameter
    checkpoint = torch.load(args.model, map_location=torch.device("cuda:0"))

    # Load model architecture and model weight
    num_classes = 10
    model = RPNet(num_classes=num_classes)
    model.load_state_dict(checkpoint['state_dict'])
    
    # Evaluation mode 
    model.eval() 
    model.to(device)
    torch.set_grad_enabled(False)
    
    if args.dataset == "woodscape":
        # Set inference image path
        img_name = args.image_name + ".png"
        img_path = os.path.join(args.dataset_path, "rgb_images", img_name)
        gt_path = os.path.join(args.dataset_path, "semantic_annotations", "gtLabels", img_name)
        gt = np.array(Image.open(gt_path))
        gt = decode_segmap(gt, num_classes)
        gt = Image.fromarray((gt * 255).astype(np.uint8))
        gt = transform_gt(gt)
        gt = np.array(gt)

        # Load image file
        image = Image.open(img_path)
        img_raw = image
        inputs = transform(image).to(device)
        
        # Inference
        predictions = model(inputs.unsqueeze(0))
        predictions = np.argmax(predictions[0].data.cpu().detach().numpy(), 1)
        predictions = decode_segmap(predictions.squeeze(), num_classes)

        ## Normalized image
        # plt.imshow(transform(img_raw).permute(1,2,0))
        
        # Show image (input_img, pred_image, pred_image)
        plt.figure(figsize=(15, 5))
        plt.subplot(1, 3, 1)
        plt.imshow(img_raw)
        plt.subplot(1, 3, 2)
        plt.imshow(predictions)
        plt.subplot(1, 3, 3)
        plt.imshow(gt)
        plt.show()

    elif args.dataset == "valeo":
        img_base = args.dataset_path
        
        # Infer 100 images in dataset folder (Convert .rec file into .jpg file)
        for i in range (1, 100):
            
            # Set inference image path
            img_path = img_base + str(i).zfill(5) + ".jpg"

            # Load image
            image = Image.open(img_path)
            inputs = transform(image).to(device)
            predictions = model(inputs.unsqueeze(0))
            predictions = np.argmax(predictions[0].data.cpu().detach().numpy(), 1)
            predictions = decode_segmap(predictions.squeeze(), num_classes)

            # Show image (input_img, norm_image, pred_image)
            plt.figure(figsize=(15, 5))
            plt.subplot(1, 3, 1)
            plt.imshow(image)
            plt.subplot(1, 3, 2)
            plt.imshow(transform(image).permute(1,2,0))
            plt.subplot(1, 3, 3)
            plt.imshow(predictions)
            plt.show()
