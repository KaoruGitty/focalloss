import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# 1. コスト行列の設定
# [日, 英, 中]
cost_matrix = np.array([
    [0.65, 0.1,  0.25], # 正解が「日」: 「中」へのミス(0.25)を重く
    [0.1,  0.65, 0.25], # 正解が「英」: 「中」へのミス(0.25)を重く
    [0.25, 0.1,  0.65]  # 正解が「中」: 「日」へのミス(0.25)を重く
])

def get_ternary_data(target_idx, mode="CSL"):
    # 三角形内のグリッド生成
    points = []
    res = 60
    for i in range(res + 1):
        for j in range(res - i + 1):
            p0 = i / res
            p1 = j / res
            p2 = 1.0 - p0 - p1
            
            # log(0)回避
            pp = [max(1e-3, p0), max(1e-3, p1), max(1e-3, p2)]
            
            costs = cost_matrix[target_idx]
            if mode == "CE":
                loss = -np.log(pp[target_idx])
            else:
                # 新案CSL: -(a*log(Pi) + sum( b_j * log(1-Pj) ))
                loss = -costs[target_idx] * np.log(pp[target_idx])
                for k in range(3):
                    if k != target_idx:
                        loss -= costs[k] * np.log(1.0 - pp[k] + 1e-3)
            
            points.append([p0, p1, p2, loss])
    return np.array(points)

# グラフ作成
fig = go.Figure()

labels = ["日", "英", "中"]
for i, label in enumerate(labels):
    data = get_ternary_data(i, "CSL")
    fig.add_trace(go.Scatterternary(
        a=data[:, 0], b=data[:, 1], c=data[:, 2],
        mode='markers',
        marker=dict(
            color=data[:, 3],
            colorscale='Viridis',
            size=4,
            showscale=True,
            cmin=0, cmax=5,
            colorbar=dict(title="Loss (低いほどAIが好む)", x=1.15)
        ),
        visible=(i==0),
        name=f"正解={label}"
    ))

# 切り替えボタン
buttons = []
for i, label in enumerate(labels):
    visible = [False] * 3
    visible[i] = True
    buttons.append(dict(label=f"正解が『{label}』の時", method="update", args=[{"visible": visible}]))

fig.update_layout(
    title="3クラスCSLロス分布（紫・青が低ロス＝AIの目標地点）",
    ternary=dict(
        sum=1,
        aaxis=dict(title="日", tickformat=".0%"),
        baxis=dict(title="英", tickformat=".0%"),
        caxis=dict(title="中", tickformat=".0%"),
    ),
    updatemenus=[dict(type="buttons", direction="down", buttons=buttons, x=0.1, y=1.1)]
)

fig.show()