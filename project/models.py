import torch.nn as nn
import torch
import math.ceil


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, img, encode=True):

        if encode:
            return self.encoder(img)
        else:
            return self.decoder(img)


class Encoder(nn.Module):

    def __init__(self):

        super(Encoder, self).__init__()

        self.scales = 6
        self.alignment_scale = 32

        # TODO put in sequential
        # decomposition layer
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.LeakyReLU(0.2)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.LeakyReLU(0.2)
        self.conv3_bn = nn.BatchNorm2d(64)

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
        self.relu3 = nn.LeakyReLU(0.2)
        self.conv5 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.relu4 = nn.LeakyReLU(0.2)

        self.scale_factor = 0.5
        self.coef_maps = list()


    def decompose(self, xm):
        """
        Performs pyramidal decomposition by extracting
        coefficients from the input scale and computing next scale.

        :param xm:
        :return:
        """

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

    """
        :param x Image that will undergo pyramidal decomposition
        :returns compressed image represented as Tensor
    """
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

        # compressed image
        return y.numpy()


class Quantization(nn.Module):

    B = 6

    def __init__(self):
        super(Quantization, self).__init__()

    def forward(self, y):
        return (1 / pow(2, self.B - 1)) * math.ceil(pow(2, self.B - 1) * y)


class Decoder(nn.Module):

    def __init__(self):
        super(Decoder, self).__init__()

        self.layer1 = nn.Sequential(nn.ConvTranspose2d(64, 64, kernel_size=3, stride=8, padding=1), nn.LeakyReLU(0.2))
        self.layer2 = nn.Sequential(nn.ConvTranspose2d(64, 64, kernel_size=3, stride=8, padding=1), nn.LeakyReLU(0.2))
        self.layer3 = nn.Sequential(nn.ConvTranspose2d(64, 64, kernel_size=3, stride=8, padding=1), nn.LeakyReLU(0.2))
        self.layer4 = nn.Sequential(nn.ConvTranspose2d(64, 64, kernel_size=3, stride=8, padding=1), nn.LeakyReLU(0.2))


    def forward(self, img):

        return self.model(img)
    