import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """(Conv => SiLU) * 2 with Time Injection"""
    def __init__(self, in_ch, out_ch, time_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.time_mlp = nn.Linear(time_dim, out_ch)

    def forward(self, x, t_emb):
        h = F.silu(self.conv1(x))
        # Inject time embedding
        time_val = self.time_mlp(t_emb).unsqueeze(-1).unsqueeze(-1)
        h = h + time_val
        h = F.silu(self.conv2(h))
        return h

class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.time_dim = 64
        self.time_mlp = nn.Sequential(
            nn.Linear(1, self.time_dim),
            nn.SiLU(),
            nn.Linear(self.time_dim, self.time_dim)
        )

        self.down1 = DoubleConv(1, 32, self.time_dim)
        self.pool1 = nn.MaxPool2d(2)
        self.down2 = DoubleConv(32, 64, self.time_dim)
        self.pool2 = nn.MaxPool2d(2)
        self.bot = DoubleConv(64, 128, self.time_dim)
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.up_conv1 = DoubleConv(128, 64, self.time_dim)
        self.up2 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.up_conv2 = DoubleConv(64, 32, self.time_dim)
        self.out_conv = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, x, t):
        t_emb = self.time_mlp(t)
        x1 = self.down1(x, t_emb)
        x2 = self.down2(self.pool1(x1), t_emb)
        x3 = self.bot(self.pool2(x2), t_emb)
        x_up1 = self.up1(x3)
        x4 = self.up_conv1(torch.cat([x_up1, x2], dim=1), t_emb)
        x_up2 = self.up2(x4)
        x5 = self.up_conv2(torch.cat([x_up2, x1], dim=1), t_emb)
        return self.out_conv(x5)