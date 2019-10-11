import torch.nn as nn
import torch
import math


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

        # decomposition layer
        self.decompLayer = nn.Sequential(nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
                                         nn.LeakyReLU(0.2),
                                         nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
                                         nn.LeakyReLU(0.2),
                                         nn.BatchNorm2d(64))

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
        self.outputLayer = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
                                         nn.LeakyReLU(0.2),
                                         nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
                                         nn.LeakyReLU(0.2))

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
        xm1 = nn.functional.interpolate(xm, mode='bilinear', scale_factor=self.scale_factor)

        xm = self.decompLayer(xm)

        # return coefficiant and downsampled image
        return xm, xm1


    def align(self):
        """
            Performs interscale alignment of features in the
            coef_map. Computes difference between size of coef tensor
            and alignment_scale then passes coef through appropriate
            conv layer. coef_map must contain a tensor for each scale.

            :returns sum of coef as a tensor
           """

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
        """
            :param x Image that will undergo pyramidal decomposition
            :returns compressed image represented as Tensor
        """

        xm = x

        # perform pyramidal decomposition
        for scale in range(self.scales):
            x, xm = self.decompose(xm)
            self.coef_maps.append(x)

        # perform interscale alignment
        y = self.align()

        # convolve aligned features
        y = self.outputLayer(y)

        # compressed image
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

        self.layer1 = nn.Sequential(nn.ConvTranspose2d(64, 64, kernel_size=3, stride=8, padding=1), nn.LeakyReLU(0.2))
        self.layer2 = nn.Sequential(nn.ConvTranspose2d(64, 64, kernel_size=3, stride=8, padding=1), nn.LeakyReLU(0.2))
        self.layer3 = nn.Sequential(nn.ConvTranspose2d(64, 64, kernel_size=3, stride=8, padding=1), nn.LeakyReLU(0.2))
        self.layer4 = nn.Sequential(nn.ConvTranspose2d(64, 64, kernel_size=3, stride=8, padding=1), nn.LeakyReLU(0.2))


    def forward(self, img):
        """

        :param img: 32x32 compressed image
        :return: 256X256 reconstructed image
        """


        img = self.layer1(img)
        img = nn.functional.interpolate(img, mode="bilinear", size=(64, 64, 64))
        img = self.layer2(img)
        img = nn.functional.interpolate(img, mode="bilinear", size=(128, 128, 64))
        img = self.layer3(img)
        img = nn.functional.interpolate(img, mode="bilinear", size=(256, 256, 64))
        img = self.layer4(img)

        return img
    