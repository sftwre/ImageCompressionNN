import numpy as np
import PIL
from models import Encoder, Decoder
from torchvision import transforms
import torch

TEST_TRANSFORMS_256 = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

encoder = Encoder()
decoder = Decoder()

def encode(img, bottleneck):
    """
    Your code here
    img: a 256x256 PIL Image
    bottleneck: an integer from {4096,16384,65536}
    return: a numpy array less <= bottleneck bytes
    """
    img = TEST_TRANSFORMS_256(img).unsqueeze(0)

    with torch.no_grad():
        res = encoder.forward(img.cuda())

    return res


def decode(x, bottleneck):
    """
    Your code here
    x: a numpy array
    bottleneck: an integer from {4096,16384,65536}
    return a 256x256 PIL Image
    """

    img = torch.from_numpy(x).float().cuda().reshape(1, 64, 32, 32)

    with torch.no_grad():
        returned = decoder.forward(img) * bottleneck

    return returned