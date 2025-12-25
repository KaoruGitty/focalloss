import torch
import torch.nn as nn
import torch.nn.functional as F

class FlexibleFocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        """
        Args:
            alpha (list or Tensor, optional): 
                - 1次元の場合: クラスごとの固定重み [W_日, W_英, W_中]
                - 2次元の場合: コスト行列 [正解クラス, 予測クラス]
                Noneならアルファなし。
            gamma (float): 難易度調整パラメータ。
            reduction (str): 'mean', 'sum', 'none'
        """
        super(FlexibleFocalLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction
        
        if alpha is not None:
            # 常にTensorとして保持
            self.alpha = torch.tensor(alpha, dtype=torch.float32)
        else:
            self.alpha = None

    def forward(self, inputs, targets):
        # デバイスの同期
        if self.alpha is not None:
            self.alpha = self.alpha.to(inputs.device)

        # 1. 確率 P と Log(P) の計算
        log_probs = F.log_softmax(inputs, dim=1)
        probs = torch.exp(log_probs)
        
        # 正解ラベルに対応する P と Log(P) を抽出
        pt = probs.gather(1, targets.view(-1, 1)).view(-1)
        logpt = log_probs.gather(1, targets.view(-1, 1)).view(-1)

        # 2. Focal項 (1 - pt)^gamma
        focal_term = (1 - pt) ** self.gamma

        # 3. アルファ（動的コストまたは固定重み）の計算
        if self.alpha is not None:
            if self.alpha.dim() == 1:
                # パターンB: 従来のクラス別固定重み [C]
                at = self.alpha.gather(0, targets)
            elif self.alpha.dim() == 2:
                # パターンC: コスト行列 [C, C] を使用した動的重み
                # 正解クラスに応じたコストの「行」を取得 -> [Batch, NumClasses]
                batch_cost_matrix = self.alpha[targets]
                # 予測確率分布との加重平均をとり、そのサンプルの「今の危険度」を alpha とする
                at = torch.sum(batch_cost_matrix * probs, dim=1)
            
            # ロスに重みを適用
            loss = -at * focal_term * logpt
        else:
            # アルファなし
            loss = -focal_term * logpt

        # 4. 集計
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

# --- 使い方パターン ---

# パターンA：アルファなし
criterion_no_alpha = FlexibleFocalLoss(alpha=None, gamma=2.0)

# パターンB：従来のアルファ（クラスCを3倍重視）
criterion_fixed_alpha = FlexibleFocalLoss(alpha=[1.0, 1.0, 3.0], gamma=2.0)

# パターンC：コスト行列（「中を日と間違える」のを5倍重く罰する）
# 行：正解クラス, 列：予測クラス [日, 英, 中]
cost_matrix = [
    [1.0, 1.0, 1.0], # 正解が日の時
    [1.0, 1.0, 1.0], # 正解が英の時
    [5.0, 1.0, 1.0]  # 正解が中の時：予測が「日」に寄るほどロスが増大する
]
criterion_cost_sensitive = FlexibleFocalLoss(alpha=cost_matrix, gamma=2.0)
