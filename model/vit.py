import math, torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

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

class ViTModel(nn.Module):
    def __init__(self, num_classes, embed_dim=512, arc_s=30.0, arc_m=0.50, backbone="vit_b_16", pretrained=True, drop=0.1):
        super().__init__()
        if backbone == "vit_b_16":
            vit = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None)
        elif backbone == "vit_b_32":
            vit = models.vit_b_32(weights=models.ViT_B_32_Weights.IMAGENET1K_V1 if pretrained else None)
        else:
            raise ValueError("Unsupported ViT backbone")
        self.patch = vit.conv_proj
        self.cls   = nn.Parameter(vit.class_token.detach().clone())
        self.pos   = nn.Parameter(vit.encoder.pos_embedding.detach().clone())
        self.enc   = vit.encoder
        self.hdim  = vit.hidden_dim
        self.norm  = nn.LayerNorm(self.hdim)
        self.embed = nn.Sequential(
            nn.Linear(self.hdim, embed_dim, bias=False),
            nn.BatchNorm1d(embed_dim),
        )
        self.arc   = ArcMarginProduct(embed_dim, num_classes, s=arc_s, m=arc_m)
        self.drop  = nn.Dropout(drop)

    def _interp_pos(self, pos: torch.Tensor, n_tokens: int) -> torch.Tensor:
        if n_tokens == pos.size(1): return pos
        cls_pos = pos[:, :1, :]; patch_pos = pos[:, 1:, :]
        d = patch_pos.size(-1)
        hw = int(patch_pos.size(1) ** 0.5)
        new_hw = int((n_tokens - 1) ** 0.5)
        pp = patch_pos[0].transpose(0,1).reshape(d, hw, hw).unsqueeze(0)
        pp = torch.nn.functional.interpolate(pp, size=(new_hw, new_hw), mode="bicubic", align_corners=False)
        pp = pp.squeeze(0).reshape(d, -1).transpose(0,1).unsqueeze(0)
        return torch.cat([cls_pos, pp], dim=1)

    def forward(self, x, labels=None):
        B = x.size(0)
        p = self.patch(x).flatten(2).transpose(1, 2)
        cls = self.cls.expand(B, -1, -1)
        tokens = torch.cat([cls, p], dim=1)
        pos = self._interp_pos(self.pos, tokens.size(1))
        h = self.enc(tokens + pos)
        cls_h = self.drop(self.norm(h[:, 0, :]))
        emb = self.embed(cls_h)
        if labels is None: return emb
        logits = self.arc(emb, labels)
        return logits, emb
