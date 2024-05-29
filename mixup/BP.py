import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
import csv

class MLPNet(nn.Module):
    def __init__(self):
        super(MLPNet, self).__init__()
        self.fc1 = nn.Linear(35, 80) 
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(80, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = self.fc2(x)
        x = F.softmax(x, dim=1)  # ソフトマックス関数を適用
        return x

def output(output_file,df,true):
    # 入力の重要度をCSVファイルに出力
    with open(output_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Feature", "Average_Importance"])
        for i, true in enumerate(true):
            writer.writerow([df.drop('result', axis=1).columns[i], true.item()])
    # 重要度の大きい順にソートする
    sorted_df = pd.read_csv(output_file).sort_values(by='Average_Importance', ascending=False)
    # ソートされたデータフレームをCSVファイルとして保存する
    sorted_df.to_csv(output_file, index=False)
    print(f"Sorted input importance for fold saved to {output_file}")

# テストデータを用意する
input_data = "/workspace/data/loading/2024_05/risk_factor.csv"
df = pd.read_csv(input_data, index_col=0) 

# バイナリ特徴量をリストアップする
risk_factors = ['diabetes', 'dyslipidemia', 'pressure', 'smoking', "age", "sex", "受付番号"]

selected = ['none']
# selected = ['diabetes', 'dyslipidemia', 'pressure', 'smoking']
# selected = ['diabetes_dyslipidemia','diabetes_pressure','diabetes_smoking', 'dyslipidemia_pressure','dyslipidemia_smoking', 'pressure_smoking']
# selected = ['diabetes_dyslipidemia_pressure','diabetes_dyslipidemia_smoking', 'dyslipidemia_pressure_smoking']
# selected = ['diabetes_dyslipidemia_pressure_smoking']

for selected in selected:
    # テストデータから選択された特徴量を取得する
    selected_columns = selected.split('_')

    # 選択されたリスクファクターを残し、欠損値を処理する
    for factor in risk_factors:
        if factor in selected_columns:
            df.dropna(subset=[factor], inplace=True)
        else:
            df.drop(factor, axis=1, inplace=True)

    # 説明変数と目的変数を取得
    X = np.array(df.drop('result', axis=1))
    true_label = np.array(df.result)
    # 目的変数を適切な形状に変換
    true_label = true_label.flatten()

    # 目的変数を適切な形状に変換
    true_label = true_label.reshape(-1, 1)
    one_hot_labels = np.zeros((true_label.shape[0], 2))
    one_hot_labels[np.arange(true_label.shape[0]), true_label.flatten()] = 1

    # KFoldを使用してデータを分割し、テストデータをモデルに入力して予測を行う
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    num = 0
    # 入力データの重要度を解析する
    for train_idx, data_idx in kf.split(X):
        num += 1
        # チェックポイントのロード
        checkpoint = torch.load(f'/workspace/results/2024_06/{selected}/model/mixup/{num}_mixup.pth')
        # モデルのインスタンスを作成
        model = MLPNet()
        # モデルの重みを読み込む
        model.load_state_dict(checkpoint['model_state_dict'])
        X_data = X[data_idx]
        print(X_data.shape)
        X_data = torch.Tensor(X_data)
        X_data.requires_grad = True  # 勾配を計算するためにrequires_gradをTrueに設定
        numFeatures = X_data.shape[1]
        input_importance = torch.zeros(numFeatures)  # 入力の重要度を保存するためのテンソルを作成
        outputs = model(X_data)
        # 正解ラベルをone-hotから0次元テンソルに変換
        true_labels = torch.LongTensor([np.argmax(one_hot_labels[i]) for i in data_idx])
        # 正解ラベルが0であるインデックスを取得
        true0 = torch.nonzero(true_labels == 0).squeeze()
        # 正解ラベルが1であるインデックスを取得
        true1 = torch.nonzero(true_labels == 1).squeeze()
        
        loss = F.cross_entropy(outputs, true_labels)
        
        # 勾配の計算と逆伝播
        loss.backward()
        
        # 入力の勾配を取得
        input_grads = X_data.grad
        
        # 説明変数ごとに勾配の平均をとって重要度として使用する
        input_importance = input_grads.mean(dim=0)
        #目的変数が0と1でサンプルをわける
        X_data0 = X_data[true0]
        X_data1 = X_data[true1]
        #サンプルごとの説明変数に勾配をかける
        true0 = X_data0 * input_importance
        true1 = X_data1 * input_importance
        # 説明変数ごとに重要度の和を計算する
        true0 = torch.sum(true0, dim=0)
        true1 = torch.sum(true1, dim=0)
        
        print(true0.shape)
        print(true1.shape)
        output_file = f"/workspace/results/2024_06/{selected}/model/normal/importance_{num}_result0.csv"
        output(output_file,df,true0)
        output_file = f"/workspace/results/2024_06/{selected}/model/normal/importance_{num}_result1.csv"
        output(output_file,df,true1)
        