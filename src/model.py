"""
Two-stage sequence model for sleep staging
================================================================
  Stage-1 per-epoch encoder: MultiScaleCNN + channel attention + mean pooling
  Stage-2 inter-epoch encoder: BiLSTM + Transformer Encoder

Classify each epoch in input (B, L, 3000), producing (B, L, num_classes).
Single-epoch input is supported with L=1, returning (B, num_classes).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


# ==========================================================================
# Stage-1: Multi-scale 1D CNN feature extraction
# ==========================================================================
class SmallScaleBranch(nn.Module):
    """Small kernels capture spindle and beta rhythms."""
    def __init__(self, dropout=0.5):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 64, 50, stride=6, padding=24)
        self.bn1   = nn.BatchNorm1d(64)
        self.pool1 = nn.MaxPool1d(8, 2)
        self.drop  = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(64, 128, 8, padding=4)
        self.bn2   = nn.BatchNorm1d(128)
        self.conv3 = nn.Conv1d(128, 128, 8, padding=4)
        self.bn3   = nn.BatchNorm1d(128)
        self.conv4 = nn.Conv1d(128, 128, 8, padding=4)
        self.bn4   = nn.BatchNorm1d(128)
        self.pool2 = nn.MaxPool1d(4, 4)

    def forward(self, x):
        x = F.gelu(self.bn1(self.conv1(x)))
        x = self.pool1(x); x = self.drop(x)
        x = F.gelu(self.bn2(self.conv2(x)))
        x = F.gelu(self.bn3(self.conv3(x)))
        x = F.gelu(self.bn4(self.conv4(x)))
        x = self.pool2(x)
        return x


class LargeScaleBranch(nn.Module):
    """Large kernels capture slow delta waves."""
    def __init__(self, dropout=0.5):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 64, 400, stride=50, padding=200)
        self.bn1   = nn.BatchNorm1d(64)
        self.pool1 = nn.MaxPool1d(4, 2)
        self.drop  = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(64, 128, 6, padding=3)
        self.bn2   = nn.BatchNorm1d(128)
        self.conv3 = nn.Conv1d(128, 128, 6, padding=3)
        self.bn3   = nn.BatchNorm1d(128)
        self.conv4 = nn.Conv1d(128, 128, 6, padding=3)
        self.bn4   = nn.BatchNorm1d(128)
        self.pool2 = nn.MaxPool1d(2, 2)

    def forward(self, x):
        x = F.gelu(self.bn1(self.conv1(x)))
        x = self.pool1(x); x = self.drop(x)
        x = F.gelu(self.bn2(self.conv2(x)))
        x = F.gelu(self.bn3(self.conv3(x)))
        x = F.gelu(self.bn4(self.conv4(x)))
        x = self.pool2(x)
        return x


class SEBlock(nn.Module):
    """Squeeze-and-excitation channel attention."""
    def __init__(self, c, r=8):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(c, c // r), nn.GELU(),
            nn.Linear(c // r, c), nn.Sigmoid()
        )

    def forward(self, x):  # (B, C, T)
        w = x.mean(dim=-1)
        w = self.fc(w).unsqueeze(-1)
        return x * w


class EpochEncoder(nn.Module):
    """
    Stage-1: Encode epochs (B, 3000) as features (B, feat_dim).
    """
    def __init__(self, feat_dim=256, dropout=0.5):
        super().__init__()
        self.small = SmallScaleBranch(dropout)
        self.large = LargeScaleBranch(dropout)
        self.se    = SEBlock(256)
        self.proj  = nn.Conv1d(256, feat_dim, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (B, 3000) or (B, 1, 3000)
        if x.dim() == 2:
            x = x.unsqueeze(1)
        x1 = self.small(x)
        x2 = self.large(x)
        T = min(x1.size(-1), x2.size(-1))
        x1 = F.adaptive_avg_pool1d(x1, T)
        x2 = F.adaptive_avg_pool1d(x2, T)
        feat = torch.cat([x1, x2], dim=1)                 # (B, 256, T)
        feat = self.se(feat)
        feat = self.proj(feat)                             # (B, feat_dim, T)
        # Average pooling over the within-epoch time dimension.
        feat = feat.mean(dim=-1)                           # (B, feat_dim)
        return self.dropout(feat)


# ==========================================================================
# Stage-2: Inter-epoch BiLSTM + Transformer
# ==========================================================================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() *
                        (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class SequenceEncoder(nn.Module):
    """
    Input and output shape: (B, L, feat_dim).
    BiLSTM models local dynamics; Transformer models sequence dependencies.
    """
    def __init__(self, feat_dim=256, lstm_hidden=128, n_layers=2,
                 n_heads=8, ff_dim=512, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(feat_dim, lstm_hidden,
                            num_layers=1, batch_first=True, bidirectional=True)
        d_model = lstm_hidden * 2
        self.pos_enc = PositionalEncoding(d_model, max_len=512)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=ff_dim, dropout=dropout,
            activation='gelu', batch_first=True
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.out_proj = nn.Linear(d_model, feat_dim)

    def forward(self, x):
        h, _ = self.lstm(x)                               # (B, L, 2*hidden)
        h = self.pos_enc(h)
        h = self.transformer(h)
        return self.out_proj(h)                           # (B, L, feat_dim)


# ==========================================================================
# Main model: two-stage integration
# ==========================================================================
class SeqSleepStagingModel(nn.Module):
    """
    Input: (B, L, 3000), containing L contiguous 30-second epochs.
    Output: (B, L, num_classes), containing per-epoch logits.
    For L=1, output is squeezed to (B, num_classes).
    """
    def __init__(self, num_classes=5,
                 feat_dim=256, lstm_hidden=128,
                 n_trans_layers=2, n_heads=8,
                 cnn_dropout=0.5, attn_dropout=0.1, fc_dropout=0.3):
        super().__init__()
        self.epoch_enc = EpochEncoder(feat_dim, dropout=cnn_dropout)
        self.seq_enc   = SequenceEncoder(feat_dim=feat_dim,
                                         lstm_hidden=lstm_hidden,
                                         n_layers=n_trans_layers,
                                         n_heads=n_heads,
                                         ff_dim=feat_dim * 2,
                                         dropout=attn_dropout)
        self.classifier = nn.Sequential(
            nn.Linear(feat_dim, feat_dim // 2),
            nn.GELU(),
            nn.Dropout(fc_dropout),
            nn.Linear(feat_dim // 2, num_classes),
        )

    def forward(self, x):
        # x: (B, L, 3000) or (B, 3000)
        if x.dim() == 2:
            x = x.unsqueeze(1)
        B, L, T = x.shape
        # Stage-1: Encode each epoch independently with the CNN.
        x_flat = x.reshape(B * L, T)
        feat_flat = self.epoch_enc(x_flat)                # (B*L, feat_dim)
        feat = feat_flat.view(B, L, -1)                   # (B, L, feat_dim)
        # Stage-2: Model dependencies between epochs.
        feat = self.seq_enc(feat)                         # (B, L, feat_dim)
        # classification
        logits = self.classifier(feat)                    # (B, L, num_classes)
        if L == 1:
            logits = logits.squeeze(1)
        return logits


# ==========================================================================
# Focal loss with optional class weights
# ==========================================================================
class FocalLoss(nn.Module):
    """
    Multiclass focal loss supports:
        - class weights (alpha, shape [C])
        - label smoothing
    """
    def __init__(self, alpha=None, gamma=2.0, label_smoothing=0.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.reduction = reduction

    def forward(self, logits, target):
        # logits: (*, C)  target: (*,)
        if logits.dim() > 2:
            logits = logits.reshape(-1, logits.size(-1))
            target = target.reshape(-1)
        C = logits.size(-1)
        logp = F.log_softmax(logits, dim=-1)              # (N, C)
        p = torch.exp(logp)

        if self.label_smoothing > 0:
            # Smoothed one-hot labels.
            with torch.no_grad():
                true_dist = torch.full_like(logp, self.label_smoothing / (C - 1))
                true_dist.scatter_(1, target.unsqueeze(1), 1 - self.label_smoothing)
            # Use target-class probability for the focal factor.
            pt = p.gather(1, target.unsqueeze(1)).squeeze(1).clamp(min=1e-8)
            focal_factor = (1 - pt) ** self.gamma          # (N,)
            loss_per_cls = -true_dist * logp               # (N, C)
            loss = loss_per_cls.sum(dim=-1) * focal_factor
        else:
            logp_t = logp.gather(1, target.unsqueeze(1)).squeeze(1)
            pt = p.gather(1, target.unsqueeze(1)).squeeze(1).clamp(min=1e-8)
            focal_factor = (1 - pt) ** self.gamma
            loss = -focal_factor * logp_t

        if self.alpha is not None:
            at = self.alpha.to(logits.device).gather(0, target)
            loss = at * loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


if __name__ == '__main__':
    m = SeqSleepStagingModel()
    x = torch.randn(2, 20, 3000)
    y = m(x)
    n = sum(p.numel() for p in m.parameters()) / 1e6
    print(f"Parameters: {n:.2f}M")
    print(f"Input: {x.shape} -> Output: {y.shape}")
    # Single-epoch compatibility.
    y2 = m(torch.randn(4, 3000))
    print(f"Single-epoch output: {y2.shape}")
