import torch
import torch.nn as nn
import torch.nn.functional as F

class CostSensitiveLoss(nn.Module):
    def __init__(self, cost_matrix):
        super(CostSensitiveLoss, self).__init__()
        # cost_matrix: [正解クラス, 予測クラス] の重み行列
        # 例: [[0.65, 0.1, 0.25], ...]
        self.register_buffer('cost_matrix', torch.tensor(cost_matrix, dtype=torch.float32))

    def forward(self, logits, targets):
        """
        logits:  モデルの出力 (Softmax前の未正規化スコア) [Batch, 3]
        targets: 正解ラベルのインデックス [Batch]
        """
        # 1. 確率分布に変換 (Softmax)
        probs = F.softmax(logits, dim=1)
        
        # 2. 各サンプルに対応するコスト行を取得
        # batch_costs[i] は、i番目のサンプルの正解クラスに基づいた重み [a, b1, b2]
        batch_costs = self.cost_matrix[targets]
        
        # 3. 新案の計算
        # - ( a * log(P_correct) + sum( b_j * log(1 - P_wrong_j) ) )
        
        # 正解クラスの項: -(a * log(P_target))
        p_target = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        loss_target = -batch_costs.gather(1, targets.unsqueeze(1)).squeeze(1) * torch.log(p_target + 1e-8)
        
        # 間違いクラスの項: -(b_j * log(1 - P_j))
        # 全クラスに対して計算し、後で「正解クラス」の分をマスクする
        loss_avoid = -batch_costs * torch.log(1 - probs + 1e-8)
        
        # 正解クラス以外の項（間違いの否定）だけを抽出して合計
        # 対角成分を0にするか、あるいは単に正解以外をsumする
        mask = torch.ones_like(probs).scatter_(1, targets.unsqueeze(1), 0)
        loss_avoid_total = (loss_avoid * mask).sum(dim=1)
        
        # 4. 合計ロス
        total_loss = loss_target + loss_avoid_total
        
        return total_loss.mean()

# --- 使い方 ---
# あなたのコスト行列
my_costs = [
    [0.65, 0.1,  0.25], # 正解が「日」: 中(0.25)を強く拒絶
    [0.1,  0.65, 0.25], # 正解が「英」: 中(0.25)を強く拒絶
    [0.25, 0.1,  0.65]  # 正解が「中」: 日(0.25)を強く拒絶
]

criterion = CostSensitiveLoss(my_costs)
