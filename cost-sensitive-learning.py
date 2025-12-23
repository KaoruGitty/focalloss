import torch
import torch.nn as nn
import torch.nn.functional as F

class CostSensitiveCrossEntropy(nn.Module):
    """
    ペアワイズ・コスト行列を用いた損失関数。
    特定の誤分類（例：日→中）に対して、個別にペナルティ強度を設定可能。
    """
    def __init__(self, cost_matrix: torch.Tensor, reduction="mean"):
        super().__init__()
        # 対角成分（正解）は 0.0、通常のミスは 1.0、厳禁なミスは 10.0 等を想定
        assert cost_matrix.dim() == 2
        self.register_buffer("cost_matrix", cost_matrix)
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor):
        # 1. 数値的安定性のために log_softmax を使用
        log_probs = F.log_softmax(logits, dim=1)  # [Batch, Classes]
        
        # 2. 各サンプルの正解ラベル(y)に基づき、コスト行列の該当行を抽出
        # batch_costs[i] は、正解が y の時に、各クラス(y_hat)と予測した際のコスト
        batch_costs = self.cost_matrix[targets]  # [Batch, Classes]
        
        # 3. コストを重みとした負の対数尤度の計算
        # 各予測クラスへの log_prob にコストを掛けて合計する
        # 正解クラスのコストが 0 であれば、実質的に「不正解クラスへ漏れた確率」への罰則になる
        loss = -(batch_costs * log_probs).sum(dim=1)

        # 標準のCE（正解を伸ばす力） + CSL（間違いを抑える力）
        # standard_loss = F.cross_entropy(logits, targets)
        # csl_penalty = -(batch_costs * log_probs).sum(dim=1).mean()
        # total_loss = standard_loss + csl_penalty
        
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


# デバイス設定
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- コスト行列の定義 (行:実際, 列:予測) ---
# クラス順: 0:日, 1:英, 2:中
# 最初は 10倍 ではなく 5倍 程度から始めるのが安定のコツです
c_matrix = torch.tensor([
    [0.0, 1.0, 5.0],  # 実際「日」: 中(2)へのミスを5倍罰する
    [1.0, 0.0, 1.0],  # 実際「英」: 通常のミス
    [1.0, 1.0, 0.0]   # 実際「中」: 通常のミス
], dtype=torch.float32).to(device)

criterion = CostSensitiveCrossEntropy(cost_matrix=c_matrix)

# --- ハイパーパラメータの推奨設定 ---
# 1. 学習率は通常(例: 1e-3)の半分以下を推奨
optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)

# 2. じっくり収束させるためにスケジューラを併用
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

# --- 学習ループ内での注意点 ---
# 1. 通常通り criterion(logits, targets) で計算
# 2. エポック数は通常より 1.5倍〜2倍 程度長めに確保する