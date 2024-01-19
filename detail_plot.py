import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ファイルのパス
file_path = 'detail_loss_log.txt'  # ファイルの実際のパスに変更してください

# パラメータのリスト
params = [[] for _ in range(9)]

# ファイルを読み込んで各パラメータを抽出
with open(file_path, 'r') as file:
    for line in file:
        values = line.split(',')  # カンマで区切られていると仮定
        for i in range(9):
            if i==0:
                values[i] = values[i].split(' ')[1].replace('data_loss:','')
            elif i==1:
                values[i] = values[i].replace(' pde_loss:','')
            elif i==2:
                values[i] = values[i].replace('bd_loss:','')
            elif i==3:
                values[i] = values[i].replace('initial_loss:','')
            elif i==4:
                values[i] = values[i].replace('e1:','')
            elif i==5:
                values[i] = values[i].replace('e2:','')
            elif i==6:
                values[i] = values[i].replace('e3:','')
            elif i==7:
                values[i] = values[i].replace('e4:','')
            elif i==8:
                values[i] = values[i].replace('e5:','')
            params[i].append(float(values[i]))

#start_index = 3774  # インデックスは0から始まるため、3700番目は3699となります
#params[0][3774:], params[1][3774:] = params[1][3774:], params[0][3774:]

# サブプロットを作成
fig, axs = plt.subplots(3, 3, figsize=(12, 8))
fig.suptitle('Parameters Plot')

# パラメータの名前
param_names = ['data_loss', 'pde_loss', 'bd_loss', 'initial_loss', 'ad_diff_eq_loss', 'x_momentum_loss', 'y_momentum_loss', 'z_momentum_loss', 'continuity_loss']

# 各パラメータのプロット
for i in range(3):
    for j in range(3):
        param_index = i * 3 + j
        x_values = [epoch * 100 for epoch in range(len(params[param_index]))]
        axs[i, j].plot(x_values, params[param_index])
        axs[i, j].set_title(param_names[param_index])
        axs[i, j].set_yscale('log')  # x軸を対数スケールに設定
        axs[i, j].set_xlabel('Epoch [-]')
        axs[i, j].set_ylabel('Loss [-]')

        # x軸のラベルを10の何乗の形式に設定
        axs[i, j].xaxis.set_major_formatter(ticker.ScalarFormatter())
        axs[i, j].xaxis.get_major_formatter().set_scientific(True)
        axs[i, j].xaxis.get_major_formatter().set_powerlimits((0, 0))

# レイアウトの調整
plt.tight_layout(rect=[0, 0, 1, 0.96])


# グラフを表示
plt.savefig('detail_loss.png', dpi=600)