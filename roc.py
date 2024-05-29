import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import matplotlib.pyplot as plt
import os
import argparse

risk_factor_list = ["none","diabetes","dyslipidemia","pressure","smoking","diabetes_dyslipidemia","diabetes_pressure","diabetes_smoking","dyslipidemia_pressure","dyslipidemia_smoking","pressure_smoking","diabetes_dyslipidemia_pressure","diabetes_dyslipidemia_smoking","diabetes_pressure_smoking","dyslipidemia_pressure_smoking","diabetes_dyslipidemia_pressure_smoking"]
# risk_factor_list = ["none"]
for risk_factor in risk_factor_list:
    results_file = '/workspace/results/2024_05/'+ risk_factor
    #rocディレクトリを作成
    os.makedirs(results_file + '/roc', exist_ok=True)
    # CSVファイルの読み込み
    df1_pass = results_file + "/predscore/normal/" + risk_factor + "_seed=5_predscore_test.csv"
    df2_pass = results_file + "/predscore/mixup/" + risk_factor + "_seed=5mixup_predscore_test.csv"
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
    plt.plot(fpr2, tpr2, color='green', linestyle='-', lw=2, label='ROC with mixup (AUC = %0.2f)' % roc_auc2)
    plt.plot(fpr1, tpr1, color='red', linestyle='--', lw=2, label='ROC without mixup (AUC = %0.2f)' % roc_auc1)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate', fontsize=15)  # フォントサイズを14に設定
    plt.ylabel('True Positive Rate', fontsize=15)  # フォントサイズを14に設定
    plt.legend(loc="lower right", fontsize=12)  # フォントサイズを12に設定
    plt.xticks(fontsize=14)  # x軸の目盛りのフォントサイズを12に設定
    plt.yticks(fontsize=14)  # y軸の目盛りのフォントサイズを12に設定
    plt.gca().set_aspect('equal')  # アスペクト比を保持してスケールを合わせる
    # PNGファイルとして保存
    file_name = results_file + f"/roc/seed=5_roc_curve.png"
    plt.savefig(file_name, dpi=300)
    print(risk_factor + " のroc曲線を保存しました")
    plt.clf()  # 図をクリア
    
    # 閾値と適合率を計算
    precision1, recall1, thresholds1 = precision_recall_curve(y_true1, y_scores1)
    precision2, recall2, thresholds2 = precision_recall_curve(y_true2, y_scores2)
    # 閾値と適合率の関係をプロット
    plt.figure()
    plt.plot(thresholds1, precision1[:-1], color='red', linestyle='-', lw=2)
    plt.xlabel('Threshold', fontsize=15)
    plt.ylabel('Precision', fontsize=15)
    plt.ylim([0.4, 1.0])  # 縦軸の範囲を0.4〜1.0に設定
    plt.xlim([0, 0.98])  # 横軸の表示範囲を0から1.0に設定
    plt.legend(loc="lower left", fontsize=12)
    plt.yticks(fontsize=14)
    plt.gca().set_aspect('equal')  # アスペクト比を保持してスケールを合わせる
    # PNGファイルとして保存
    file_name_precision = results_file + f"/roc/seed=5_precision_recall_curve.png"
    plt.savefig(file_name_precision, dpi=300)
    plt.clf()  # 図をクリア
    print(risk_factor + " の適合率-閾値曲線を保存しました")