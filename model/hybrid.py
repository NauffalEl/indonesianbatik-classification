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

class GatedFusion(nn.Module):
    def __init__(self, dim_cnn, dim_vit, dim_out):
        super().__init__()
        h = dim_cnn + dim_vit
        self.gate = nn.Sequential(nn.Linear(h, h), nn.ReLU(inplace=True), nn.Linear(h, h))
        self.proj = nn.Linear(2 * h, dim_out)
        self.h = h; self.dim_cnn = dim_cnn; self.dim_vit = dim_vit
    def forward(self, f_cnn, f_vit):
        x = torch.cat([f_cnn, f_vit], dim=1)
        g = torch.sigmoid(self.gate(x))
        g_cnn, g_vit = torch.split(g, [self.dim_cnn, self.dim_vit], dim=1)
        cnn_g = g_cnn * f_cnn
        vit_g = g_vit * f_vit
        fused = torch.cat([cnn_g, vit_g], dim=1)
        out = torch.cat([fused, x], dim=1)
        return self.proj(out)

class HybridModel(nn.Module):
    def __init__(self, num_classes, embed_dim=512, arc_s=30.0, arc_m=0.50,
                 cnn_backbone="resnet50", vit_backbone="vit_b_16", pretrained=True, drop=0.2, fusion_dim=1024):
        super().__init__()
        if cnn_backbone == "resnet50":
            cnn = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
            dim_cnn = cnn.fc.in_features
            self.cnn_feat = nn.Sequential(*list(cnn.children())[:-1])  # (B,C,1,1)
        elif cnn_backbone == "resnet18":
            cnn = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
            dim_cnn = cnn.fc.in_features
            self.cnn_feat = nn.Sequential(*list(cnn.children())[:-1])
        else:
            raise ValueError("Unsupported CNN backbone")

        if vit_backbone == "vit_b_16":
            vit = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None)
        elif vit_backbone == "vit_b_32":
            vit = models.vit_b_32(weights=models.ViT_B_32_Weights.IMAGENET1K_V1 if pretrained else None)
        else:
            raise ValueError("Unsupported ViT backbone")
        self.vit_patch = vit.conv_proj
        self.vit_cls   = nn.Parameter(vit.class_token.detach().clone())
        self.vit_pos   = nn.Parameter(vit.encoder.pos_embedding.detach().clone())
        self.vit_enc   = vit.encoder
        dim_vit        = vit.hidden_dim
        self.vit_ln    = nn.LayerNorm(dim_vit)

        self.fusion = GatedFusion(dim_cnn, dim_vit, fusion_dim)
        self.drop   = nn.Dropout(drop)
        self.embed  = nn.Sequential(
            nn.Linear(fusion_dim, embed_dim, bias=False),
            nn.BatchNorm1d(embed_dim),
        )
        self.arc    = ArcMarginProduct(embed_dim, num_classes, s=arc_s, m=arc_m)

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
        f_cnn = self.cnn_feat(x).flatten(1)
        p = self.vit_patch(x).flatten(2).transpose(1, 2)
        cls_tok = self.vit_cls.expand(B, -1, -1)
        tokens = torch.cat([cls_tok, p], dim=1)
        pos = self._interp_pos(self.vit_pos, tokens.size(1))
        tokens = tokens + pos
        h = self.vit_enc(tokens)
        f_vit = self.vit_ln(h[:, 0, :])
        z = self.fusion(f_cnn, f_vit)
        z = self.drop(z)
        emb = self.embed(z)
        if labels is None: return emb
        logits = self.arc(emb, labels)
        return logits, emb
