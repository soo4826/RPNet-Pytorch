import torch
import matplotlib.pyplot as plt
# import torchvision.transforms as tr
import numpy as np
from PIL import Image
import transforms as ext_transforms
import torchvision.transforms as transforms

from models.rpnet import RPNet


def decode_segmap(label_mask, num_classes):
    label_colors= np.array([[0  ,0,  0  ],
                           [255,0,255  ],
                           [255,0,  0  ],
                           [0,  255,0  ],
                           [0,  0  ,255],
                           [255,255,255],
                           [255,255,0  ],
                           [0  ,255,255],
                           [128,128,255],
                           [0  ,128,128]])
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

# from dataloaders.utils import decode_segmap
height, width = 512, 1024
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

checkpoint = torch.load("save/RPNet")
# print(checkpoint)
num_classes = 10
model = RPNet(num_classes=num_classes)

model.eval()

model.to(device)
# def transform(image):
#     return tr.Compose([
#         # tr.Resize(513),
#         # tr.CenterCrop(513),
#         tr.Resize((height, width),Image.BILINEAR),
#         tr.ToTensor(),
#         # tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
#     ])(image)

torch.set_grad_enabled(False)

img_path = "data/Woodscape/rgb_images/00003_FV.png"
# gt_path = ""
image = Image.open(img_path)
def transform(image):
    return transforms.Compose([
        transforms.Resize((height,width), Image.BILINEAR), 
        transforms.ToTensor()
    ])(image)

# image_transform = transforms.Compose(
#     [transforms.Resize((height, width),Image.BILINEAR),
#         transforms.ToTensor()])(image)
inputs = transform(image).to(device)

predictions = model(inputs.unsqueeze(0))
# print(type(predictions[0]))
predictions = np.argmax(predictions[0].data.cpu().detach().numpy(), 1)
# print(predictions.squeeze().shape)




predictions = decode_segmap(predictions.squeeze(), num_classes)
# print(predictions)
plt.imshow(predictions)
plt.show()