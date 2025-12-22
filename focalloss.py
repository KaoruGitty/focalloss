import torch
import torch.nn as nn
import torch.nn.functional as F

class FlexibleFocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        """
        Args:
            alpha (list or Tensor, optional): 
                クラスごとの重み。Noneならアルファなし。
                例: [1.0, 1.0, 3.0]
            gamma (float): 
                難易度調整パラメータ。標準は2.0。
            reduction (str): 
                'mean' (平均), 'sum' (合計), 'none' (そのまま)
        """
        super(FlexibleFocalLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction
        
        # alphaがリストなどで渡されたらTensorに変換、Noneならそのまま保持
        if alpha is not None:
            self.alpha = torch.tensor(alpha, dtype=torch.float32)
        else:
            self.alpha = None

    def forward(self, inputs, targets):
        # 1. Softmaxで確率を計算し、正解クラスの確率 pt を抽出
        logpt = F.log_softmax(inputs, dim=1)
        pt = torch.exp(logpt)
        # 正解ラベルのインデックスにある確率だけを取り出す
        pt = pt.gather(1, targets.view(-1, 1)).view(-1)
        logpt = logpt.gather(1, targets.view(-1, 1)).view(-1)

        # 2. Focal項 (1 - pt)^gamma の計算
        focal_term = (1 - pt) ** self.gamma

        # 3. 損失の組み立て
        loss = -focal_term * logpt

        # 4. アルファ（クラス重み）の適用
        if self.alpha is not None:
            # GPU/CPUデバイスを自動調整
            self.alpha = self.alpha.to(inputs.device)
            # 各サンプルの正解ラベルに応じた重みを取得
            at = self.alpha.gather(0, targets)
            loss = loss * at

        # 5. 集計
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

# --- 使い方パターン ---

# パターンA：アルファなし（ガンマのみで「難問」を重視）
# 最初はこれで、モデルが自力でCを見分けられるか試すのがおすすめ
criterion_no_alpha = FlexibleFocalLoss(alpha=None, gamma=2.0)

# パターンB：アルファあり（「希少クラスC」を強制的に3倍重視）
# A, Bの枚数に対してCが少ないことを物理的に補正する
criterion_with_alpha = FlexibleFocalLoss(alpha=[1.0, 1.0, 3.0], gamma=2.0)