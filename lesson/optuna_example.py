import optuna
from sklearn.model_selection import KFold
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt

# コマンドライン引数のパーサーを作成
parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--seed', default=0, type=int, help='random seed')
parser.add_argument('--batch-size', default=10, type=int, help='batch size')
parser.add_argument('--epoch', default=200, type=int, help='total epochs to run')
parser.add_argument('--risk_factor', default="", type=str, help='input data')
args = parser.parse_args()

class MLPNet(nn.Module):
    def __init__(self, input_size, num_layers, dropout_rates, num_features):
        super(MLPNet, self).__init__()
        self.input_size = input_size
        self.num_layers = num_layers
        self.num_features = num_features
        self.dropout_rates = dropout_rates
        
        # 入力層
        self.input_layer = nn.Sequential(
            nn.BatchNorm1d(input_size),
            nn.Linear(input_size, int(num_features[0])),  # num_features[0]を整数に変換
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rates[0])
        )

        # 隠れ層のリストを定義
        self.middle_layers = nn.ModuleList()
        for i in range(1, num_layers):
            self.middle_layers.append(nn.BatchNorm1d(int(num_features[i-1])))  # num_features[i-1]を整数に変換
            self.middle_layers.append(nn.Linear(in_features=int(num_features[i - 1]), out_features=int(num_features[i])))  # num_features[i]を整数に変換
            self.middle_layers.append(nn.ReLU())
            self.middle_layers.append(nn.Dropout(dropout_rates[i]))

        self.output_layer = nn.Linear(in_features=int(num_features[num_layers - 1]), out_features=2)  # 出力層の定義

    def forward(self, x):
        x = self.input_layer(x)
        for layer in self.middle_layers:
            x = layer(x)
        x = self.output_layer(x)
        x = F.softmax(x, dim=1)
        return x

# 5分割交差検証の準備
kf = KFold(n_splits=10, shuffle=True, random_state=42)
train_loss_li = []
val_loss_li = []
train_acc_li = []
val_acc_li = []
def objective(trial, train_dataset, valid_dataset, input_size):
    # ハイパーパラメータの探索範囲を定義
    num_layers = trial.suggest_int('num_layers', 1, 10)  # 隠れ層の数
    num_features = [int(trial.suggest_float(f'num_features_{i}', 16, 128, step=16)) for i in range(num_layers)]  # 隠れ層の特徴量数
    dropout_rates = [trial.suggest_float(f'dropout_rates_{i}', 0, 0.7) for i in range(num_layers)]
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-1, log=True)  # 学習率
    optimizer_name = trial.suggest_categorical('optimizer_name', ['Adam', 'RMSprop', 'SGD'])  # オプティマイザー
    weight_decay = trial.suggest_float('weight_decay', 1e-8, 1e-2, log=True)
    batch_size = trial.suggest_int('batch_size', 8, 64)
    input_size = input_size
    
    # DataLoaderを作成
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=len(valid_dataset), shuffle=False)
    
    # モデルを定義
    model = MLPNet(input_size=input_size, num_layers=num_layers, dropout_rates=dropout_rates, num_features=num_features)
    criterion = nn.CrossEntropyLoss()
    
    if optimizer_name == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_name == 'RMSprop':
        optimizer = optim.RMSprop(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    else:
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=0.9)
    
    # モデルをトレーニング
    model.train()
    train_loss_li = []
    train_acc_li = []
    num_epochs = 200  # 例として200エポック
    train_loss = 0
    train_correct = 0
    total = 0
    for epoch in range(num_epochs):
        for batch_idx,(inputs, targets) in train_loader:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            if epoch == num_epochs - 1:
                train_loss += loss.item() # train_loss に結果を蓄積
                total += targets.size(0)
                train_correct += predicted.eq(targets.data).cpu().sum().float()
            optimizer.step()
    train_loss_li.append(train_loss / batch_idx)
    train_acc_li.append(train_correct / total *100)
            
    val_loss_li = []
    val_acc_li = []
    # バリデーションデータでモデルを評価します
    model.eval()
    with torch.no_grad():
        for batch_idx, (inputs, targets) in valid_loader:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total += targets.size(0)
            valid_correct += predicted.eq(targets.data).cpu().sum().float()
    val_loss_li.append(loss.item)
    val_acc_li.append(train_correct / total *100)

    return loss.item()



# 入力データ
input_data = "/workspace/data/loading/2024_05/risk_factor.csv"
df = pd.read_csv(input_data, index_col=0)  # バイナリ
risk_factors = ['diabetes', 'dyslipidemia', 'pressure', 'smoking']
selected_columns = args.risk_factor.split('_')
for factor in risk_factors:
    if factor in selected_columns:
        df.dropna(subset=[factor], inplace=True)
    else:
        df.drop(factor, axis=1, inplace=True)
X = np.array(df.drop('result', axis=1))  # 説明変数
true_label = np.array(df.result)  # 目的変数 (0: '狭心症否定的', 1: '狭心症肯定的')

# true_labelを適切な形状に変換する
true_label = true_label.reshape(-1, 1)  # バッチサイズ x 1の形状に変換
one_hot_labels = np.zeros((true_label.shape[0], 2))  # バッチサイズ x クラス数のゼロ行列を作成
one_hot_labels[np.arange(true_label.shape[0]), true_label.flatten()] = 1  # 正解クラスの位置に1を設定

num_epochs = 300  # トレーニングのエポック数
i=0
for train_idx, test_idx in kf.split(X):
    i+=1
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = one_hot_labels[train_idx], one_hot_labels[test_idx]

    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    # データセットをPyTorchのTensorに変換
    train_dataset = torch.utils.data.TensorDataset(torch.Tensor(X_train), torch.FloatTensor(y_train))
    valid_dataset = torch.utils.data.TensorDataset(torch.Tensor(X_valid), torch.FloatTensor(y_valid))
    test_dataset = torch.utils.data.TensorDataset(torch.Tensor(X_test), torch.FloatTensor(y_test))
    
    train_loss_li = []
    val_loss_li = []
    train_acc_li = []
    val_acc_li = []
    
    # Optunaを使用してハイパーパラメータを最適化
    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: objective(trial, train_dataset, valid_dataset, X_train.shape[1]), n_trials=200)
    
    # 最適なハイパーパラメータを取得
    best_params = study.best_params
    print("Best Params:", best_params)

    # 最適なハイパーパラメータを使用して最終モデルをトレーニング
    num_layers = best_params['num_layers']
    num_features = [best_params[f'num_features_{i}'] for i in range(num_layers)]
    dropout_rates = [best_params[f'dropout_rates_{i}'] for i in range(num_layers)]  # ドロップアウト率を取得
    learning_rate = best_params['learning_rate']
    optimizer_name = best_params['optimizer_name']
    weight_decay = best_params['weight_decay']
    batch_size = best_params['batch_size']
    
    # 最適なハイパーパラメータを使用して最終モデルをトレーニング
    best_model = MLPNet(input_size=X_train.shape[1], num_layers=num_layers, dropout_rates=dropout_rates, num_features=num_features)
    criterion = nn.CrossEntropyLoss()
    
    # OptunaのstudyオブジェクトからDataFrameを取得
    df = study.trials_dataframe()

    # DataFrameをCSVファイルとして保存
    df.to_csv(f'optuna_trials{i}.csv', index=False)
    exit(1)
