# from main import load_dataset
from data import Woodscapes as dataset
import torchvision.transforms as transforms
from PIL import Image
import transforms as ext_transforms
import torch.utils.data as data
import matplotlib.pyplot as plt
import numpy as np

def decode_segmap(label_mask, num_classes):
    # print(label_mask.shape)
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
    rgb[:, :, 0] = r/255.0
    rgb[:, :, 1] = g/255.0
    rgb[:, :, 2] = b/255.0
    return rgb
# from inference import decode_segmap

# dataset_dir = 'data/Woodscape'
# save_dir = 'save'
height, width = 512, 1024
# batch_size = 3
# workers = 4


# print("\nLoading dataset...\n")

# print("Selected dataset:", dataset)
# print("Dataset directory:", dataset_dir)
# print("Save directory:", save_dir)

# image_transform = transforms.Compose(
#     [transforms.Resize((height, width),Image.BILINEAR),
#         transforms.ToTensor()])

label_transform = transforms.Compose([
    transforms.Resize((height, width),Image.NEAREST),
    ext_transforms.PILToLongTensor()
])

# # Get selected dataset
# # Load the training set as tensors
# train_set = dataset(
#     dataset_dir,
#     transform=image_transform,
#     label_transform=label_transform)
# train_loader = data.DataLoader(
#     train_set,
#     batch_size=batch_size,
#     shuffle=True,
#     num_workers=workers)

# img, label = train_set.__getitem__(4)



# filename = 'data/Woodscape/semantic_annotations/rgbLabels/00003_FV.png'
filename = 'data/Woodscape/semantic_annotations/gtLabels/00010_FV.png'
# filename = 'data/Woodscape/rgb_images/00003_FV.png'
img_direct = Image.open(filename)

# img = np.transpose(img_direct, axes=[1, 2, 0])
# label_rgb = decode_segmap()
img_rgb = label_transform(img_direct).numpy()
print(img_rgb.shape, type(img_rgb))
img_rgb = decode_segmap(img_rgb, 10)

plt.imshow(img_rgb)
plt.show()

# plt.imshow(label, cmap='gray_r', vmin=0, vmax=255)
