import torch
import torch.nn as nn
import torch.nn.functional as F

class CostSensitiveFocalLoss(nn.Module):
    def __init__(self, cost_matrix, gamma=2.0, reduction='mean'):
        super(CostSensitiveFocalLoss, self).__init__()
        # コスト行列をTensorとして登録（GPU対応）
        self.register_buffer('cost_matrix', torch.tensor(cost_matrix, dtype=torch.float))
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        inputs: (batch_size, num_classes) - モデルの出力（Logits、Softmax前）
        targets: (batch_size) - 正解ラベル
        """
        # 1. 予測分布 P を取得
        p = F.softmax(inputs, dim=1)
        
        # 2. 正解クラスの確率 P_k を抽出
        p_k = p.gather(1, targets.view(-1, 1)).view(-1)
        
        # 3. 動的な重み α(P) の計算
        # 正解クラスに対応するコスト行列の「行」をバッチ分取得
        # batch_costs: (batch_size, num_classes)
        batch_costs = self.cost_matrix[targets]
        
        # 予測分布 P とコスト行の内積をとる
        # alpha_dynamic: (batch_size)
        alpha_dynamic = torch.sum(p * batch_costs, dim=1)
        
        # 4. Focal項の計算 (1 - P_k)^gamma
        # gamma=0 のときは自動的に 1.0 になり、Focal部がオフになる
        focal_term = (1 - p_k) ** self.gamma
        
        # 5. Cross Entropy項 (logの数値安定性のために正の微小値を加算)
        log_p_k = torch.log(p_k + 1e-9)
        
        # 6. 全体を掛け合わせる
        loss = -alpha_dynamic * focal_term * log_p_k
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

# --- 検証用の設定例 ---

# --- コスト行列の定義 ---

# 基本行列（全クラス 1, 2, 3 で構成）
# [日, 英, 中]
matrix_basic = [
    [1.0, 2.0, 3.0], # 日が正解
    [2.0, 1.0, 3.0], # 英が正解
    [3.0, 2.0, 1.0]  # 中が正解
]

# FL重み融合型行列（基本重み [3, 1, 2] を反映）
# 各行の基本値を [3, 1, 2] とし、地雷箇所をその3倍に設定
matrix_v2_fused = [
    [3.0, 6.0, 9.0], # 日が正解 (基本3, 地雷9)
    [2.0, 1.0, 3.0], # 英が正解 (基本1, 地雷3)
    [6.0, 4.0, 2.0]  # 中が正解 (基本2, 地雷6)
]

# L3用：V2強化版（地雷コストをさらに引き上げ）
matrix_l3_extreme = [
    [3.0,  6.0, 15.0], # 中への誤読に超厳罰
    [2.0,  1.0,  5.0], 
    [15.0, 4.0,  2.0]  # 日への誤読に超厳罰
]

# L1用：αなし（全ての重みを1にする）
matrix_identity = [
    [1.0, 1.0, 1.0],
    [1.0, 1.0, 1.0],
    [1.0, 1.0, 1.0]
]

# --- 各検証用インスタンスの生成 ---

# 【短期検証：数エポック】
# V1: 基本CFL (gamma=2)
criterion_v1 = CostSensitiveFocalLoss(cost_matrix=matrix_basic, gamma=2.0)

# V2: FL重み融合型 (gamma=2)
criterion_v2 = CostSensitiveFocalLoss(cost_matrix=matrix_v2_fused, gamma=2.0)

# V3: 低ガンマ検証 (gamma=1)
criterion_v3 = CostSensitiveFocalLoss(cost_matrix=matrix_basic, gamma=1.0)

# 【長期検証：50エポック】
# L1: ベースライン (純粋なFocal Loss)
criterion_l1 = CostSensitiveFocalLoss(cost_matrix=matrix_identity, gamma=2.0)

# L2: Focalなし検証 (CFL行列 + gamma=0)
criterion_l2 = CostSensitiveFocalLoss(cost_matrix=matrix_basic, gamma=0.0)

# L3: V2強化版
criterion_l3 = CostSensitiveFocalLoss(cost_matrix=matrix_l3_extreme, gamma=2.0)

# L4: CFL本検証 (V1 または V2 のベスト設定を採用)
# 短期検証の結果、V2が良ければこちらを使用
criterion_l4 = CostSensitiveFocalLoss(cost_matrix=matrix_v2_fused, gamma=2.0)
