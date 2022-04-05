import torch
import matplotlib.pyplot as plt
# import torchvision.transforms as tr
import numpy as np
from PIL import Image
import transforms as ext_transforms
import torchvision.transforms as transforms

from models.rpnet import RPNet

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
        transforms.Resize((height,width), Image.BILINEAR), 
        transforms.ToTensor()
    ])(image)

def transform_gt(image):
    return transforms.Compose([
        transforms.Resize((height,width), Image.BILINEAR),
    ])(image)


height, width = 512, 1024
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# checkpoint = torch.load("save/RPNet", map_location=torch.device("cuda:0"))
checkpoint = torch.load("save/RPNet_V4", map_location=torch.device("cuda:0"))
# checkpoint = torch.load("/home/ailab/Project/05_Woodscape/RPNet-RTMaps/save/RPNet",map_location=torch.device("cuda:0"))

num_classes = 10

model = RPNet(num_classes=num_classes)
model.load_state_dict(checkpoint['state_dict'])
model.eval()
model.to(device)

torch.set_grad_enabled(False)

img_name = "02958_MVR"
img_path = "data/Woodscape/rgb_images/" + img_name + ".png"
gt_path = "data/Woodscape/semantic_annotations/gtLabels/" + img_name + ".png"
gt = np.array(Image.open(gt_path))
gt = decode_segmap(gt, num_classes)
gt = Image.fromarray((gt * 255).astype(np.uint8))
gt = transform_gt(gt)
gt = np.array(gt)

image = Image.open(img_path)


inputs = transform(image).to(device)

predictions = model(inputs.unsqueeze(0))
# print(type(predictions[0]))
predictions = np.argmax(predictions[0].data.cpu().detach().numpy(), 1)
# print(predictions.squeeze().shape)


predictions = decode_segmap(predictions.squeeze(), num_classes)

plt.subplot(2, 1, 1)
plt.imshow(predictions)
plt.subplot(2, 1, 2)
plt.imshow(gt)
plt.show()