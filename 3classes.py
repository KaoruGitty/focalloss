import numpy as np
import plotly.graph_objects as go

def get_3class_loss(mode, a, b_eng, b_chn):
    # 三角形内の確率分布を生成
    res = 50
    p_日, p_英 = np.meshgrid(np.linspace(0.01, 0.98, res), np.linspace(0.01, 0.98, res))
    p_中 = 1 - p_日 - p_英
    mask = p_中 > 0.01
    
    p0, p1, p2 = p_日[mask], p_英[mask], p_中[mask]
    
    if mode == "old_bug": # 旧型 (a < b)
        loss = -(a * np.log(p0) + b_chn * np.log(p2))
    elif mode == "old_ok": # 旧型 (a > b)
        loss = -(0.8 * np.log(p0) + 0.1 * np.log(p2))
    else: # 新型
        loss = -(a * np.log(p0) + b_eng * np.log(1-p1) + b_chn * np.log(1-p2))
    
    return p0, p1, p2, loss

# パラメータ (正解=日)
a, b_eng, b_chn = 0.4, 0.1, 0.6  # b_chnを重く設定

fig = go.Figure()

# 各モードのデータを追加
for mode, name, color in [("old_bug", "旧型(a < b)", "red"), ("new", "新型", "blue")]:
    p0, p1, p2, z = get_3class_loss(mode, a, b_eng, b_chn)
    fig.add_trace(go.Scatterternary(
        a=p0, b=p1, c=p2, mode='markers',
        marker=dict(color=z, colorscale='Viridis', size=5, showscale=(mode=="new")),
        name=name, visible=(mode=="new")
    ))

# 切り替えボタン
fig.update_layout(
    updatemenus=[dict(type="buttons", direction="down", x=0.1, y=1.2, buttons=[
        dict(label="新型を表示", method="update", args=[{"visible": [False, True]}]),
        dict(label="旧型(バグ)を表示", method="update", args=[{"visible": [True, False]}])
    ])],
    ternary=dict(aaxis_title="日(正解)", baxis_title="英", caxis_title="中(間違い)")
)
fig.show()
