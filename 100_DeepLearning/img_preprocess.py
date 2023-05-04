#%% package
import torch
from torchvision import transforms
from PIL import Image

# %%
img = Image.open('data/kiki.jpg')
img
# %%
preprocess_steps = transforms.Compose([
    transforms.Resize((300, 225)),
    transforms.RandomRotation(20),
    transforms.CenterCrop(200),
    transforms.Grayscale(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor()
])

x = preprocess_steps(img)
x.shape
# %%
