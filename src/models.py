
import torch
import torch.nn as nn
from einops import rearrange
import segmentation_models_pytorch as smp

# ── Double Conv Block ─────────────────────────────
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.conv(x)

# ── U-Net ─────────────────────────────────────────
class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super().__init__()
        self.enc1 = ConvBlock(in_channels, 64)
        self.enc2 = ConvBlock(64, 128)
        self.enc3 = ConvBlock(128, 256)
        self.enc4 = ConvBlock(256, 512)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = ConvBlock(512, 1024)
        self.up4  = nn.ConvTranspose2d(1024,512,2,stride=2)
        self.dec4 = ConvBlock(1024,512)
        self.up3  = nn.ConvTranspose2d(512,256,2,stride=2)
        self.dec3 = ConvBlock(512,256)
        self.up2  = nn.ConvTranspose2d(256,128,2,stride=2)
        self.dec2 = ConvBlock(256,128)
        self.up1  = nn.ConvTranspose2d(128,64,2,stride=2)
        self.dec1 = ConvBlock(128,64)
        self.out  = nn.Conv2d(64,out_channels,1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b  = self.bottleneck(self.pool(e4))
        d4 = self.dec4(torch.cat([self.up4(b),  e4],dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3],dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2],dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1],dim=1))
        return self.out(d1)

# ── Transformer Block ─────────────────────────────
class TransformerBlock(nn.Module):
    def __init__(self, emb_dim=512, num_heads=8,
                 mlp_ratio=4, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(emb_dim)
        self.attn  = nn.MultiheadAttention(
            emb_dim, num_heads,
            dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(emb_dim)
        self.mlp   = nn.Sequential(
            nn.Linear(emb_dim, emb_dim*mlp_ratio),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(emb_dim*mlp_ratio, emb_dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        n = self.norm1(x)
        a,_ = self.attn(n,n,n)
        x = x + a
        x = x + self.mlp(self.norm2(x))
        return x

# ── Patch Embedding ───────────────────────────────
class PatchEmbedding(nn.Module):
    def __init__(self, in_channels=512,
                 patch_size=1, emb_dim=512):
        super().__init__()
        self.patch = nn.Conv2d(
            in_channels, emb_dim,
            kernel_size=patch_size, stride=patch_size)
        self.norm  = nn.LayerNorm(emb_dim)
    def forward(self, x):
        x = self.patch(x)
        x = rearrange(x, "b d h w -> b (h w) d")
        return self.norm(x)

# ── TransUNet ─────────────────────────────────────
class TransUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1,
                 emb_dim=512, num_heads=8,
                 num_transformer_layers=6):
        super().__init__()
        self.enc1 = ConvBlock(in_channels, 64)
        self.enc2 = ConvBlock(64, 128)
        self.enc3 = ConvBlock(128, 256)
        self.enc4 = ConvBlock(256, 512)
        self.pool = nn.MaxPool2d(2)
        self.patch_embed  = PatchEmbedding(512,1,emb_dim)
        self.transformers = nn.Sequential(*[
            TransformerBlock(emb_dim, num_heads)
            for _ in range(num_transformer_layers)])
        self.reshape_proj = nn.Linear(emb_dim, 512)
        self.reshape_conv = nn.Conv2d(512, 512, 1)
        self.up4  = nn.ConvTranspose2d(512,256,2,stride=2)
        self.dec4 = ConvBlock(256+512, 256)
        self.up3  = nn.ConvTranspose2d(256,128,2,stride=2)
        self.dec3 = ConvBlock(128+256, 128)
        self.up2  = nn.ConvTranspose2d(128,64,2,stride=2)
        self.dec2 = ConvBlock(64+128, 64)
        self.up1  = nn.ConvTranspose2d(64,32,2,stride=2)
        self.dec1 = ConvBlock(32+64, 32)
        self.out  = nn.Conv2d(32, out_channels, 1)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b  = self.pool(e4)
        b  = self.patch_embed(b)
        b  = self.transformers(b)
        b  = self.reshape_proj(b)
        H  = W = int(b.shape[1]**0.5)
        b  = rearrange(b,"b (h w) d -> b d h w",h=H,w=W)
        b  = self.reshape_conv(b)
        d4 = self.dec4(torch.cat([self.up4(b),  e4],dim=1))
        d4 = self.dropout(d4)
        d3 = self.dec3(torch.cat([self.up3(d4), e3],dim=1))
        d3 = self.dropout(d3)
        d2 = self.dec2(torch.cat([self.up2(d3), e2],dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1],dim=1))
        return self.out(d1)

# ── ResNet-UNet (pretrained) ──────────────────────
def get_resnet_unet():
    return smp.Unet(
        encoder_name="resnet50",
        encoder_weights="imagenet",
        in_channels=3,
        classes=1,
        activation=None
    )
