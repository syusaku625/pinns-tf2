import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import scienceplots

plt.style.use(['science','ieee'])

# テキストファイルからデータを読み込む
with open("detail_loss_log.txt", "r") as file:
    lines = file.readlines()

# 各パラメータのリストを初期化
parameters = [[] for _ in range(10)]

# タイトルのリスト
titles = [
    "data_loss",
    "pde_loss",
    "non-slip_loss",
    "outlet_loss",
    "inlet_loss",
    "e1",
    "e2",
    "e3",
    "e4",
    "e5",
]

# 各行を処理してパラメータを取得
for i, line in enumerate(lines):
    parts = line.split()
    for j, value in enumerate(parts):
        if j==0:
            continue
        elif j==1:
            value = value.replace('data_loss:','')
            value = value.replace(',','')
            parameters[j-1].append(float(value))
        elif j==2:
            value = value.replace('pde_loss:','')
            parameters[j-1].append(float(value))
        elif j==3:
            value = value.replace(',non-slip_loss:','')
            parameters[j-1].append(float(value))
        elif j==4:
            value = value.replace(',outlet_loss:','')
            parameters[j-1].append(float(value))
        elif j==5:
            value = value.replace(',inlet_loss:','')
            parameters[j-1].append(float(value))
        elif j==6:
            value = value.replace(',e1:','')
            parameters[j-1].append(float(value))
        elif j==7:
            value = value.replace(',e2:','')
            parameters[j-1].append(float(value))
        elif j==8:
            value = value.replace(',e3:','')
            parameters[j-1].append(float(value))
        elif j==9:
            value = value.replace(',e4:','')
            parameters[j-1].append(float(value))
        elif j==10:
            value = value.replace(',e5:','')
            parameters[j-1].append(float(value))

# 一つの図に11個のグラフを描画
#plt.figure(figsize=(10, 8))  # 図のサイズを設定
#for i, parameter in enumerate(parameters):
#    plt.subplot(4, 3, i+1)  # 4行3列のサブプロットのi+1番目
#    plt.plot(parameter)
#    plt.title(titles[i])  # タイトルを設定
#    plt.xlabel("Counts")
#    plt.ylabel("Value")
#    plt.yscale('log')  # y軸を対数に設定
plt.plot(parameters[0], color='red', label = 'data loss')
plt.plot(parameters[1], color='blue', label = 'PDE loss')
plt.plot(parameters[2], color='green', label = 'BC loss')
print(len(parameters[0]))
plt.yscale('log')
plt.tight_layout()  # レイアウトを調整して重なりを解消

# グラフのタイトルやラベルの追加
plt.xlabel("learning iteration [-]")
plt.ylabel("Loss value [-]")
plt.legend()

# グラフを表示
plt.savefig('detail_loss.png')