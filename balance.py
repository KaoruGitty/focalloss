import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score
import numpy as np

# 1. 損失関数の定義（CSL + 通常のCEのハイブリッド）
class HybridCSLLoss(nn.Module):
    def __init__(self, cost_matrix, alpha=0.1):
        super().__init__()
        self.register_buffer('cost_matrix', torch.tensor(cost_matrix).float())
        self.alpha = alpha # CSLの影響力を調整する重み

    def forward(self, logits, targets):
        log_probs = F.log_softmax(logits, dim=1)
        
        # CSL成分: 間違いへの罰金
        batch_costs = self.cost_matrix[targets]
        csl_loss = -(batch_costs * log_probs).sum(dim=1).mean()
        
        # 標準的なCE成分: 正解への報酬（軸がぶれないようにするため）
        standard_ce = F.nll_loss(log_probs, targets)
        
        return standard_ce + self.alpha * csl_loss

# --- 設定 ---
cost_matrix = [
    [0, 1, 3], # GT=日: 英なら1, 中なら3(特にここを減らしたい)
    [1, 0, 1], # GT=英
    [1, 1, 0]  # GT=中
]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = YourModel().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=2e-5) # 低めの学習率
criterion = HybridCSLLoss(cost_matrix, alpha=0.5)

accumulation_steps = 8 # バッチ8 * 8回 = 実質バッチ64
f1_history = []
window_size = 3

# --- 学習ループ ---
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    
    for i, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        
        outputs = model(inputs)
        loss = criterion(outputs, targets) / accumulation_steps
        loss.backward()
        
        if (i + 1) % accumulation_steps == 0:
            # 勾配クリッピングで「日→中」の衝撃を制御
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()

    # --- Validationフェーズ ---
    model.eval()
    val_preds, val_targets = [], []
    
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1)
            
            val_preds.extend(preds.cpu().numpy())
            val_targets.extend(targets.cpu().numpy())

    # スコア計算
    macro_f1 = f1_score(val_targets, val_preds, average='macro')
    f1_history.append(macro_f1)
    
    # 移動平均の計算
    moving_avg = np.mean(f1_history[-window_size:])
    
    print(f"Epoch {epoch}: Macro F1={macro_f1:.4f}, Moving Avg={moving_avg:.4f}")
    
    # 移動平均が最高ならモデル保存
    if moving_avg == max([np.mean(f1_history[max(0, i-window_size+1):i+1]) for i in range(len(f1_history))]):
        torch.save(model.state_dict(), "best_model.pth")
