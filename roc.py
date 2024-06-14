import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import os

# コマンドライン引数のパーサーを作成
parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--risk_factor1', default="none", type=str,
                    help='input data')
parser.add_argument('--risk_factor2', default="diabetes_dyslipidemia_pressure_smoking", type=str,
                    help='input data')
args = parser.parse_args()



results_file = '/workspace/results/2024_05/'
# CSVファイルの読み込み
df1_pass = results_file + args.risk_factor1 + "/predscore/normal/" + args.risk_factor1 + "_seed=5_predscore_test.csv"
df2_pass = results_file + args.risk_factor2 + "/predscore/normal/" + args.risk_factor2 + "_seed=5_predscore_test.csv"
df1 = pd.read_csv(df1_pass)
df2 = pd.read_csv(df2_pass)

# 真のクラスラベルと予測スコアを取得
y_true1 = df1.iloc[:, 3].values # 4列目の要素をy_true1に
y_scores1 = df1.iloc[:, 2].values

y_true2 = df2.iloc[:, 3].values
y_scores2 = df2.iloc[:, 2].values

# ROC曲線を計算
fpr1, tpr1, thresholds1 = roc_curve(y_true1, y_scores1)
roc_auc1 = auc(fpr1, tpr1)
fpr2, tpr2, thresholds2 = roc_curve(y_true2, y_scores2)
roc_auc2 = auc(fpr2, tpr2)

# ROC曲線をプロット
plt.figure()  # 8x8インチの正方形のフィギュア
plt.plot(fpr1, tpr1, color='red', linestyle='-', lw=2, label=f'risk_factor : {args.risk_factor1} (AUC = {roc_auc1:.2f})')
plt.plot(fpr2, tpr2, color='green', linestyle='--', lw=2, label=f'risk_factor : {args.risk_factor2} (AUC = {roc_auc2:.2f})')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')  # フォントサイズを14に設定
plt.ylabel('True Positive Rate')  # フォントサイズを14に設定
plt.legend(loc="lower right", fontsize=9)  # フォントサイズを12に設定
plt.xticks(fontsize=12)  # x軸の目盛りのフォントサイズを12に設定
plt.yticks(fontsize=12)  # y軸の目盛りのフォントサイズを12に設定
plt.gca().set_aspect('equal')  # アスペクト比を保持してスケールを合わせる
# PNGファイルとして保存
file_name = f"roc_{args.risk_factor1}-{args.risk_factor2}.png"
plt.savefig(file_name, dpi=300)
print(f"{args.risk_factor1}-{args.risk_factor2}のroc曲線を保存しました")
plt.clf()  # 図をクリア

# 閾値と適合率を計算
precision1, recall1, thresholds1 = precision_recall_curve(y_true1, y_scores1)
precision2, recall2, thresholds2 = precision_recall_curve(y_true2, y_scores2)
# 閾値と適合率の関係をプロット
plt.figure()
plt.plot(thresholds1, precision1[:-1], color='red', linestyle='-', lw=2, label=f'risk_factor : {args.risk_factor1}')
plt.plot(thresholds2, precision2[:-1], color='green', linestyle='--', lw=2, label=f'risk_factor : {args.risk_factor2}')
plt.xlabel('Threshold')
plt.ylabel('Precision')
plt.ylim([0.4, 1.0])  # 縦軸の範囲を0.4〜1.0に設定
plt.xlim([0, 0.98])  # 横軸の表示範囲を0から1.0に設定
plt.legend(fontsize=9)
plt.yticks(fontsize=12)
plt.yticks(fontsize=12)
plt.gca().set_aspect('equal')  # アスペクト比を保持してスケールを合わせる
# PNGファイルとして保存
file_name_precision = f"precision_{args.risk_factor1}-{args.risk_factor2}.png"
plt.savefig(file_name_precision, dpi=300)
plt.clf()  # 図をクリア
print(f"{args.risk_factor1}-{args.risk_factor2}の適合率-閾値曲線を保存しました")


#正診率の比較

# 閾値の設定
thresholds = []
for i in range(101):
  thresholds.append(i / 100)
# 正解率の計算
accuracy1, accuracy2 = [], []
for thresh in thresholds:
    y_pred1 = (y_scores1 >= thresh).astype(int)
    y_pred2 = (y_scores2 >= thresh).astype(int)
    accuracy1.append((y_pred1 == y_true1).mean())
    accuracy2.append((y_pred2 == y_true2).mean())
# グラフの描画
plt.plot(thresholds, accuracy1, color='red', linestyle='-', lw=2, label=f'risk_factor : {args.risk_factor1}')
plt.plot(thresholds, accuracy2, color='green', linestyle='--', lw=2, label=f'risk_factor : {args.risk_factor2}')
# 凡例の追加
plt.legend(fontsize=9)
# 軸ラベルの設定
plt.xlabel('Threshold')
plt.ylabel('accuracy')
plt.ylim([0.4, 1.0])  # 縦軸の範囲を0.4〜1.0に設定
plt.xlim([0, 1.0])  # 横軸の表示範囲を0から1.0に設定
plt.legend(fontsize=9)
plt.yticks(fontsize=12)
plt.yticks(fontsize=12)
plt.gca().set_aspect('equal')  # アスペクト比を保持してスケールを合わせる
# PNGファイルとして保存
file_name_precision = f"accuracy_{args.risk_factor1}-{args.risk_factor2}.png"
plt.savefig(file_name_precision, dpi=300)
plt.clf()  # 図をクリア
print(f"{args.risk_factor1}-{args.risk_factor2}の正診率-閾値曲線を保存しました")
