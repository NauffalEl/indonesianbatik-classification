import math, torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import efficientnet_b2, EfficientNet_B2_Weights

class ArcMarginProduct(nn.Module):
    def __init__(self, in_features, out_features, s=30.0, m=0.50, easy_margin=False):
        super().__init__()
        self.s, self.m, self.easy_margin = s, m, easy_margin
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        self.cos_m, self.sin_m = math.cos(m), math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m
    def forward(self, emb, label):
        cosine = F.linear(F.normalize(emb), F.normalize(self.weight))
        sine = torch.sqrt(torch.clamp(1.0 - cosine**2, min=1e-9))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin: phi = torch.where(cosine > 0, phi, cosine)
        else:                phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        one_hot = torch.zeros_like(cosine); one_hot.scatter_(1, label.view(-1,1), 1.0)
        return (one_hot * phi + (1.0 - one_hot) * cosine) * self.s

class ChannelAttention(nn.Module):
    def __init__(self, in_ch, ratio=16):
        super().__init__()
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.max = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(in_ch, in_ch // ratio, bias=False), nn.ReLU(inplace=True),
            nn.Linear(in_ch // ratio, in_ch, bias=False)
        )
        self.sig = nn.Sigmoid()
    def forward(self, x):
        b, c, _, _ = x.size()
        a = self.mlp(self.avg(x).view(b, c))
        m = self.mlp(self.max(x).view(b, c))
        att = self.sig(a + m).view(b, c, 1, 1)
        return x * att

class SpatialAttention(nn.Module):
    def __init__(self, k=7):
        super().__init__()
        p = (k - 1) // 2
        self.conv = nn.Conv2d(2, 1, k, padding=p, bias=False)
        self.sig = nn.Sigmoid()
    def forward(self, x):
        avg = torch.mean(x, dim=1, keepdim=True)
        mx, _ = torch.max(x, dim=1, keepdim=True)
        att = self.sig(self.conv(torch.cat([avg, mx], dim=1)))
        return x * att

class CBAM(nn.Module):
    def __init__(self, in_ch, ratio=16, k=7):
        super().__init__()
        self.ca = ChannelAttention(in_ch, ratio)
        self.sa = SpatialAttention(k)
    def forward(self, x): return self.sa(self.ca(x))

class CNNModel(nn.Module):
    def __init__(self, num_classes, embed_dim=512, arc_s=30.0, arc_m=0.50, pretrained=True, drop=0.2):
        super().__init__()
        backbone = efficientnet_b2(weights=EfficientNet_B2_Weights.IMAGENET1K_V1 if pretrained else None)
        self.features = backbone.features
        self.cbam = CBAM(in_ch=1408)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.drop = nn.Dropout(p=drop)
        self.embed = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1408, embed_dim, bias=False),
            nn.BatchNorm1d(embed_dim),
        )
        self.arc = ArcMarginProduct(embed_dim, num_classes, s=arc_s, m=arc_m)
    def forward(self, x, labels=None):
        x = self.features(x)
        x = self.cbam(x)
        x = self.pool(x)
        x = self.drop(x)
        emb = self.embed(x)
        if labels is None: return emb
        logits = self.arc(emb, labels)
        return logits, emb

EffB2_CBAM_Arc = CNNModel
