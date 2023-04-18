import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(CNNBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels, 4, stride, 1, bias=False, padding_mode="reflect"
            ),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        return self.conv(x)


class Discriminator(nn.Module):
    def __init__(self, in_channels=1, features=[64,128,256,512]):
        super().__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(
                in_channels * 2,
                features[0],
                kernel_size=1,
                stride=1,
                padding='valid',
                padding_mode="reflect",
            ),
            nn.LeakyReLU(0.2),
        )

        layers = []
        in_channels = features[0]
        for feature in features[1:]:
            layers.append(
                CNNBlock(in_channels, feature, stride=1 if feature == features[-1] else 2),
            )
            in_channels = feature

        self.model = nn.Sequential(*layers)
        
        self.output = nn.Sequential(
            nn.Conv2d(
                in_channels, 1, kernel_size=7, stride=1, padding=0, bias=False
            ),
            nn.Tanh(),
        )

    def forward(self, x, y):
        y = y.squeeze(2)
        x = torch.cat([x, y], dim=1)
        x = self.initial(x)
        x = self.model(x)
        x = self.output(x)
        return x


def test():
    x = torch.randn((1, 1, 32, 32))
    y = torch.randn((1, 1, 32, 32))
    model = Discriminator(in_channels=1)
    preds = model(x, y)
    print(model)
    print(preds.shape)


if __name__ == "__main__":
    test()
