from torch import nn
from torchvision import models

class Layer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, bn_features=0):
        super(Layer, self).__init__()
        self.conv2d = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation
        )
        self.relu = nn.ReLU()

        self.bn = None
        if bn_features:
            self.bn = nn.BatchNorm2d(num_features=bn_features)

    def forward(self, input):
        output = self.conv2d(input)
        output = self.relu(output)
        if self.bn:
            output = self.bn(output)

        return output

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.model = nn.Sequential(
            Layer(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1),
            Layer(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            Layer(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            Layer(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1, bn_features=256),
            Layer(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            Layer(in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=1, bn_features=512),
            Layer(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            Layer(in_channels=512, out_channels=512, kernel_size=3, stride=2, padding=1, bn_features=512),
            Layer(in_channels=512, out_channels=256, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, input):
        return self.model(input)

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.model = nn.Sequential(
            Layer(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.Upsample(scale_factor=2),
            Layer(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1),
            Layer(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.Upsample(scale_factor=2),
            Layer(in_channels=32, out_channels=2, kernel_size=3, stride=1, padding=1),
            Layer(in_channels=2, out_channels=2, kernel_size=3, stride=1, padding=1),
            nn.Upsample(scale_factor=2),
        )

    def forward(self, input):
        return self.model(input)

class Fusion(nn.Module):
    def __init__(self, img, embedding):
        super(Fusion, self).__init__()
        self.img = img
        self.embedding = embedding

    def forward(self, input):
        raise('not implement Fusion yet')

class ColorizationNet(nn.Module):
    def __init__(self, encoder, decoder, fusion=None):
        super(ColorizationNet, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.fusion = fusion

    def forward(self, input):
        output = self.encoder(input)
        if self.fusion:
            output = self.fusion(output)
        output = self.decoder(output)
        return output



ColorizationModel = ColorizationNet(Encoder(), Decoder())
