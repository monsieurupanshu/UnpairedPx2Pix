import torch
import torch.nn as nn


class Block(nn.Module):
    def __init__(self, in_channels, out_channels, down=True, act="relu", use_dropout=False):
        super(Block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 4, 2, 1, bias=False, padding_mode="reflect")
            if down
            else nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU() if act == "relu" else nn.LeakyReLU(0.2),
        )

        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(0.5)
        self.down = down

    def forward(self, x):
        x = self.conv(x)
        return self.dropout(x) if self.use_dropout else x


class Generator(nn.Module):
    def __init__(self, in_channels=1, features=4):
        super().__init__()
        self.initial_down = nn.Sequential(
            nn.Conv2d(in_channels, features, 4, 2, 1, padding_mode="reflect"),
            nn.LeakyReLU(0.2),
        )
        self.down1 = Block(features, features * 2, down=True, act="leaky", use_dropout=False)
        self.down2 = Block(
            features * 2, features * 4, down=True, act="leaky", use_dropout=False
        )
        self.down3 = Block(
            features * 4, features * 8, down=True, act="leaky", use_dropout=False
        )
        self.down4 = Block(
            features * 8, features * 8, down=True, act="leaky", use_dropout=False
        )
        self.bottleneck = nn.Sequential(
            nn.Conv2d(features * 8, features * 8, 4, 2, 1), nn.ReLU()
        )

        self.up1 = Block(features * 8, features * 8, down=False, act="relu", use_dropout=True)
        self.up2 = Block(
            features * 8*2 , features * 4, down=False, act="relu", use_dropout=True
        )
        self.up3 = Block(
            features * 8, features * 2, down=False, act="relu", use_dropout=True
        )
        self.up4 = Block(
            features * 4, features * 1, down=False, act="relu", use_dropout=False
        )

        self.final_up = nn.Sequential(
            nn.ConvTranspose2d(features * 2, in_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )

    def forward(self, x):
        x = x.float()
        print("x:",x.shape)
        d1 = self.initial_down(x)
        print("d1:",d1.shape)
        d2 = self.down1(d1)
        print("d2:",d2.shape)
        d3 = self.down2(d2)
        print("d3:",d3.shape)
        d4 = self.down3(d3)
        print("d4:",d4.shape)
        bottleneck = self.bottleneck(d4)
        print("bottleneck:",bottleneck.shape)
        up1 = self.up1(bottleneck)
        print("up1:",up1.shape)
        print("up1+d4:",torch.cat([up1, d4], 1).shape)
        up2 = self.up2(torch.cat([up1, d4], 1))
        print("up2:",up2.shape)
        print("up2+d3:",torch.cat([up2, d3],1).shape)
        up3 = self.up3(torch.cat([up2, d3], 1))
        print("up3:",up3.shape)
        print("up3+d2:",torch.cat([up3, d2], 1).shape)        
        up4 = self.up4(torch.cat([up3, d2], 1))
        print("up4:",up4.shape)
        print("up4+d1:",torch.cat([up4, d1], 1).shape)        
        # up5 = self.up5(torch.cat([up4, d1], 1))
        return self.final_up(torch.cat([up4, d1], 1))


def test():
    x = torch.randn((1, 1, 32, 32))
    model = Generator(in_channels=1, features=4)
    preds = model(x)
    print(preds.shape)


if __name__ == "__main__":
    test()
