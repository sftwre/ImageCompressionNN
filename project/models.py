import torch.nn as nn
import torch

class FeatureExtractor(nn.Module):

    def __init__(self):

        # decomposition layer
        self.scales = 6
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv3_bn = nn.BatchNorm2d(64)
        self.alignment_scale = 32

        # interscale alignment layer
        self.downsampleLayers = {
        8: nn.Conv2d(64, 64, kernel_size=3, stride=8, padding=1),
        4: nn.Conv2d(64, 64, kernel_size=3, stride=4, padding=1),
        2: nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
        1: nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        }

        self.upsampleLayers = {
            2: nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2, padding=1),
            4: nn.ConvTranspose2d(64, 64, kernel_size=3, stride=4, padding=1)
        }

        # output layer
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

        self.scale_factor = 0.5
        self.coef_maps = list()


    """
        Performs pyramidal decomposition by extracting
        coefficients from the input scale and
        computing next scale.
    """
    def decompose(self, xm):

        # downsample to next scale
        xm1 = nn.functional.interpolate(xm, mode='bilinear', size=self.scale_factor)

        # extract coefficients
        xm = self.conv1(xm)
        xm = self.conv2(xm)
        xm = self.conv3_bn(xm)

        # return coefficiant and downsampled image
        return xm, xm1

    """
        Performs interscale alignment of features in the
        coef_map. Computes difference between size of coef tensor
        and alignment_scale then passes coef through appropriate
        conv layer. coef_map must contain a tensor for each scale.
        
        :returns sum of coef as a tensor
    """
    def align(self):

        assert(len(self.coef_maps) == self.scales)

        # sum of coefficient tensors
        y = torch.zeros(size=(32, 32, 64), dtype=torch.float32)

        for coef in self.coef_maps:

            if coef.size > self.alignment_scale:
                conv = self.downsampleLayers[coef.size / self.alignment_scale]
            else:
                conv = self.upsampleLayersLayers[self.alignment_scale / self.alignment_scale]

            y += conv(coef)

        return y



    def forward(self, x):

        xm = x


        # perform pyramidal decomposition
        for scale in range(self.scales):
            x, xm = self.decompose(xm)
            print(xm.size)
            self.coef_maps.append(x)

        # perform interscale alignment
        y = self.align()

        # convolve aligned features
        y = self.conv4(y)
        y = self.conv5(y)

        return y
class Quantization(nn.Module):

    B = 6

    def __init__(self):
        super(Quantization, self).__init__()

    def forward(self, y):
        return (1 / pow(2, self.B - 1)) * math.ceil(pow(2, self.B - 1) * y)


class Decoder(nn.Module):

    def __init__(self):
        super(Decoder, self).__init__()

        self.model = nn.Sequential(nn.ConvTranspose2d(64, 64, kernel_size=3, stride=8, padding=1),
                                   nn.LeakyReLU(0.2),
                                   nn.ConvTranspose2d(64, 64, kernel_size=3, stride=8, padding=1),
                                   nn.LeakyReLU(0.2),
                                   nn.ConvTranspose2d(64, 64, kernel_size=3, stride=8, padding=1),
                                   nn.LeakyReLU(0.2),
                                   nn.ConvTranspose2d(64, 64, kernel_size=3, stride=8, padding=1),
                                   nn.LeakyReLU(0.2))

    def forward(self, img):
        return self.model(img).numpy()
    