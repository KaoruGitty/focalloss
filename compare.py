import numpy as np
import plotly.graph_objects as go

# 横軸：間違い（中）への自信度 P_wrong
p_wrong = np.linspace(0.001, 0.999, 100)

# 1. 旧型 (a < b): バグる設定
a_low, b_high = 0.4, 0.6
loss_old_bug = -(a_low * np.log(1 - p_wrong) + b_high * np.log(p_wrong))

# 2. 旧型 (a > b): 正常に見える設定
a_high, b_low = 0.8, 0.2
loss_old_ok = -(a_high * np.log(1 - p_wrong) + b_low * np.log(p_wrong))

# 3. 新型: bを大きくしても安全な設定
a_new, b_new = 0.4, 0.6
loss_new = -(a_new * np.log(1 - p_wrong) + b_new * np.log(1 - p_wrong))

# グラフ作成
fig = go.Figure()

fig.add_trace(go.Scatter(x=p_wrong, y=loss_old_bug, name='旧型 (a < b) : 逆走バグ', line=dict(color='red', width=3)))
fig.add_trace(go.Scatter(x=p_wrong, y=loss_old_ok, name='旧型 (a > b) : 正常（だが不安定）', line=dict(color='orange', width=2)))
fig.add_trace(go.Scatter(x=p_wrong, y=loss_new, name='新型 : 鉄壁', line=dict(color='blue', width=4, dash='dash')))

fig.update_layout(
    title="ロス関数の挙動比較（横軸：間違い P_中 への自信）",
    xaxis_title="間違い（中）への自信度 $P_{中}$",
    yaxis_title="Loss (低い地点を目指す)",
    yaxis=dict(range=[0, 5]),
    xaxis=dict(tickformat=".0%")
)
fig.show()