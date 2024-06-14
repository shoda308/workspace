#!/usr/bin/env python3 -u
# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree.

# 必要なモジュールのインポート
from __future__ import print_function
import argparse
import csv
import os
import pandas as pd
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset
import datetime
from sklearn.metrics import confusion_matrix,precision_score,f1_score,recall_score,accuracy_score,roc_auc_score
from utils import progress_bar
import models
from utils import progress_bar
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from statistics import mean

# コマンドライン引数のパーサーを作成
parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
parser.add_argument('--model', default="MLP", type=str,
                    help='model type (default: MLP)')
parser.add_argument('--name', default='0', type=str, help='name of run')
parser.add_argument('--seed', default=0, type=int, help='random seed')
parser.add_argument('--batch-size', default=10, type=int, help='batch size')
parser.add_argument('--epoch', default=200, type=int,
                    help='total epochs to run')
parser.add_argument('--no-augment', dest='augment', action='store_false',
                    help='use standard augmentation (default: True)')
parser.add_argument('--decay', default=1e-4, type=float, help='weight decay')
parser.add_argument('--alpha', default=1., type=float,
                    help='mixup interpolation coefficient (default: 1)')
parser.add_argument('--risk_factor', default="", type=str,
                    help='input data')
args = parser.parse_args()

use_cuda = False #GPUを使わない
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
model_num = 0
prediction_valid, prediction_test, outputs_valid, outputs_test, targets_valid, targets_test,age_valid, age_test, sex_valid, sex_test, number_valid, number_test= [], [], [], [], [], [], [], [], [], [], [], []  # outputs_listを初期化
# 入力データの読み込み
input_data = "/workspace/data/loading/2024_05/risk_factor.csv"
results_file = '/workspace/results/2024_06/'+ args.risk_factor
saved_model_pth = '/workspace/results/2024_06/'+ args.risk_factor + '/model/normal/'

os.makedirs(results_file, exist_ok=True)
os.makedirs(results_file + '/log', exist_ok=True)
os.makedirs(results_file + '/log/normal', exist_ok=True)
os.makedirs(results_file + '/predscore', exist_ok=True)
os.makedirs(results_file + '/predscore/normal', exist_ok=True)
os.makedirs(results_file + '/model/', exist_ok=True)
os.makedirs(saved_model_pth, exist_ok=True)


if args.seed != 0:
    torch.manual_seed(args.seed)

print('==> Preparing data..')

if not os.path.isdir('results'):
    os.mkdir('results')
dt_now = datetime.datetime.now()
logname = (results_file +'/log/normal/seed=' + str(args.seed) + '_log_' + dt_now.strftime('%Y_%m_%d_%H_%M_%S') + args.risk_factor + '.csv')
criterion = nn.CrossEntropyLoss()


def train(epoch, X, Y):
    X = X[:, 1:-2]
    dataset = torch.utils.data.TensorDataset(X, Y)
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size,shuffle=False, num_workers=2)
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    train_correct = 0
    true_label, pred_label = [],[]
    total = 0
    for batch_idx, (inputs, targets) in enumerate(loader):
        optimizer.zero_grad()# 勾配リセット
        outputs = net(inputs)# 順伝播の計算
        targets = targets.float()
        # 損失関数を計算
        loss = criterion(outputs, targets)
        loss.backward()# 逆伝播の計算  
        optimizer.step()# 重みの更新
        train_loss += loss.item() # train_loss に結果を蓄積
        _, predicted = torch.max(outputs.data, 1)
        _, targets = torch.max(targets.data, 1)
        total += targets.size(0)
        # print(f"predicted:{predicted.shape}")
        train_correct += predicted.eq(targets.data).cpu().sum().float()
        # print("predicted:",predicted.tolist(),"targets.data",targets.data.tolist())
        
        pred_label += predicted.data.tolist()
        true_label += targets.data.tolist()
        progress_bar(batch_idx, len(loader),
                     'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss / (batch_idx + 1), 
                        100. * train_correct / total, train_correct, total))
    acc, precision, specificity, recall, f1 = evaluation(true_label, pred_label)
    # 最終エポックでモデルを保存する
    if epoch == start_epoch + args.epoch - 1:
        global model_num  # 関数内でグローバル変数を使用するために必要
        model_num += 1
        save_model(net, optimizer, epoch, saved_model_pth +str(model_num)+'_normal.pth')
        print("true_label:",np.sum(true_label))
    # print("batch_idx",batch_idx)
    return (train_loss / batch_idx, acc, precision, specificity, recall, f1)

def valid(epoch, X, Y):
    global age_valid
    global number_valid
    global sex_valid
    global prediction_valid
    global outputs_valid
    global targets_valid
    print('\nEpoch: %d' % epoch)
    dataset = torch.utils.data.TensorDataset(X, Y)
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size,shuffle=False, num_workers=2)
    net.eval()
    reg_loss = 0
    reg_correct = 0
    total = 0
    true_label, pred_label,predictions = [],[],[]
    with torch.no_grad():  # 必要のない計算を停止
        for batch_idx, (inputs, targets) in enumerate(loader):       
            inputs_use, targets = Variable(inputs[:, 1:-2]), Variable(targets)
            outputs = net(inputs_use)# 順伝播の計算
            loss = criterion(outputs, targets)# 損失関数を計算
            _, predicted = torch.max(outputs.data, 1)
            _, targets = torch.max(targets.data, 1)
            total += targets.size(0)
            print(f"predicted:{predicted.shape}")
            reg_correct += predicted.eq(targets.data).cpu().sum().float()
            predictions.extend(outputs.tolist())  # 予測確率をリストに追加
            pred_label += predicted.data.tolist()
            true_label += targets.data.tolist()
            # 適切な閾値を決定（例えば、F1スコアを最大化する閾値など）
            threshold = determine_threshold_from_roc_curve([item[1] for item in predictions], true_label)
            if epoch == start_epoch + args.epoch - 1:
                number_valid.extend(float(item) for item in inputs[:, :1].cpu().numpy())
                age_valid.extend(float(item) for item in inputs[:, -1:].cpu().numpy())
                sex_valid.extend(float(item) for item in inputs[:, -2:-1].cpu().numpy())
                prediction_valid += outputs.data.tolist()
                outputs_valid += predicted.data.tolist()
                targets_valid += targets.data.tolist()
    acc, precision, specificity, recall, f1 = evaluation(true_label, pred_label)
    return (loss.item(), acc, precision, specificity, recall, f1,threshold)

def test(epoch,X, Y,threshold):
    global best_acc
    global prediction_test
    global outputs_test #予測スコアのリスト
    global targets_test #正解クラスのリスト
    global age_test
    global number_test
    global sex_test
    dataset = torch.utils.data.TensorDataset(X, Y)
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size,shuffle=False, num_workers=2)
    net.eval()
    test_loss = 0
    correct = 0
    true_label,predictions= [],[]
    total = 0
    with torch.no_grad():  # 必要のない計算を停止
        for batch_idx, (inputs, targets) in enumerate(loader):
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            inputs_use, targets = Variable(inputs[:, 1:-2], volatile=True), Variable(targets)
            outputs = net(inputs_use)
            targets = targets.float()
            predictions.extend(outputs.tolist())  # 予測確率をリストに追加
            true_label += targets.data.tolist() # 真のラベルをリストに追加
            loss = criterion(outputs,targets)
            test_loss += loss.item()
            # 予測確率に対して閾値を適用して、予測を行う(予測ラベル)
            binary_predictions = [1 if item >= threshold else 0 for item in [item[1] for item in predictions]]
            _, targets = torch.max(targets.data, 1)
            total += targets.size(0)#予測した回数
            print(binary_predictions)
            print([item[1] for item in true_label])
            correct = sum(1 for p, t in zip(binary_predictions, [item[1] for item in true_label]) if p == t)#正解した回数
            if epoch == start_epoch + args.epoch - 1:
                number_test.extend(float(item) for item in inputs[:, :1].cpu().numpy())
                age_test.extend(float(item) for item in inputs[:, -1:].cpu().numpy())
                sex_test.extend(float(item) for item in inputs[:, -2:-1].cpu().numpy())
                prediction_test += outputs.data.tolist()
                outputs_test += binary_predictions
                targets_test += targets.data.tolist()
            progress_bar(batch_idx, len(loader),
                        'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                        % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
        if epoch == start_epoch + args.epoch - 1:
            a.extend(float(item) for item in X[:, :1].cpu().numpy())
    acc, precision, specificity, recall, f1 = evaluation([item[1] for item in true_label], binary_predictions)
    if epoch == start_epoch + args.epoch - 1 or acc > best_acc:
        checkpoint(acc, epoch)
    if acc > best_acc:
        best_acc = acc
    print(f"batch_idx: {batch_idx}")
    return (loss.item(), acc, precision, specificity, recall, f1)

def evaluation(true_label, pred_label):
    # ユニークなクラスラベルを取得
    classes = np.unique(true_label)
    # 混同行列の計算
    cm = confusion_matrix(true_label, pred_label, labels=classes)
    # 行名と列名を設定した混同行列のデータフレームを作成
    cm_df = pd.DataFrame(cm, index=classes, columns=classes)
    # 混同行列の表示
    print("Confusion Matrix:")
    print(cm_df)
    tn, fp, fn, tp= confusion_matrix(true_label, pred_label).ravel()
    print("tn:",tn,"fn:",fn,"fp",fp,"tp:",tp)
    acc = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    specificity = fp / (fp + tn)
    f1 = 2 * (precision * recall) / (precision + recall)
    return acc, precision, recall, specificity, f1

def determine_threshold_from_roc_curve(predictions, true_labels):
        fpr, tpr, thresholds = roc_curve(true_labels, predictions)
        # 左上に一番近い点を探す
        min_distance = float('inf')
        best_threshold = None
        
        for i in range(len(fpr)):
            distance = (fpr[i])**2 + (1 - tpr[i])**2  # 左上からの距離を計算
            if distance < min_distance:
                min_distance = distance
                best_threshold = thresholds[i]
        return best_threshold

def checkpoint(acc, epoch):
    # チェックポイントを保存
    print('Saving..')
    state = {
        'net': net,
        'acc': acc,
        'epoch': epoch,
        'rng_state': torch.get_rng_state()
    }
    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')
    torch.save(state, './checkpoint/ckpt.t7' + args.name + '_' + str(args.seed))

def save_model(model, optimizer, epoch, path):
    """
    学習済みモデルを保存する関数
    """
    torch.save({
        'net': net,
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'rng_state': torch.get_rng_state(),
    }, path)

# ログファイルのヘッダを書き込む
if not os.path.exists(logname):
    with open(logname, 'w') as logfile:
        logwriter = csv.writer(logfile, delimiter=',')
        logwriter.writerow(['epoch', 'train loss', 'reg loss', 'train acc','reg acc', 'test loss', 'test acc',
                            'train precision','reg precision','test precision','train specificity','reg specificity','test specificity',
                            'train recall','reg recall','test recall','train f1','reg f1','test f1' ])
        
df = pd.read_csv(input_data,index_col=0) # バイナリ
risk_factors = ['diabetes','dyslipidemia','pressure','smoking']
selected_columns = args.risk_factor.split('_')
for factor in risk_factors:
    if factor in selected_columns:
        df.dropna(subset=[factor], inplace=True)
    else:
        df.drop(factor, axis=1, inplace=True)
X = np.array(df.drop('result', axis=1))#説明変数
true_label = np.array(df.result)# 目的変数(0: '狭心症否定的', 1: '狭心症肯定的')
# true_labelを適切な形状に変換する
true_label = true_label.reshape(-1, 1)  # バッチサイズ x 1の形状に変換
one_hot_labels = np.zeros((true_label.shape[0], 2))  # バッチサイズ x クラス数のゼロ行列を作成
one_hot_labels[np.arange(true_label.shape[0]), true_label.flatten()] = 1  # 正解クラスの位置に1を設定
# n_splits分割（ここでは5分割）してCV
kf = KFold(n_splits=5, shuffle=True, random_state=42)
a, b =[],[]
# 学習とテストを実行
train_loss_list_ave, train_acc_list_ave,val_loss_list_ave, val_acc_list_ave, test_loss_list_ave, test_acc_list_ave = [], [], [], [],[],[]
for train_idx, data_idx in kf.split(X):
    X_train = X[train_idx]
    y_train = one_hot_labels[train_idx]
    X_data = X[data_idx]
    y_data = one_hot_labels[data_idx]
    X_train = torch.Tensor(X_train)
    y_train = torch.FloatTensor(y_train)
    X_data = torch.Tensor(X_data)
    y_data = torch.FloatTensor(y_data)
    X_valid1, X_test1, y_valid1, y_test1 = train_test_split(X_data, y_data, test_size=0.5, random_state=2023)
    X_valid2 = X_test1
    y_valid2 = y_test1
    X_test2 = X_valid1
    y_test2 = y_valid1
    # モデルの作成
    if args.resume:
        # チェックポイントのロード
        print('==> Resuming from checkpoint..')
        assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
        checkpoint = torch.load('./checkpoint/ckpt.t7' + args.name + '_' + str(args.seed))
        net = checkpoint['net']
        best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch'] + 1
        rng_state = checkpoint['rng_state']
        torch.set_rng_state(rng_state)
    else:
        print('==> Building model..')
        net = models.__dict__[args.model]()
    # 重みの正則化
    weight_decay = args.decay
    if weight_decay > 0:
        reg_loss = 0
        for param in net.parameters():
            reg_loss += torch.sum(param.pow(2))
        reg_loss *= weight_decay
    else:
        reg_loss = 0
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.decay)
    train_loss_list, train_acc_list,val_loss_list, val_acc_list, test_loss_list, test_acc_list = [], [], [], [],[],[]
    for epoch in range(start_epoch, args.epoch):
        train_loss, train_acc, train_precision, train_specificity, train_recall, train_f1= train(epoch, X_train, y_train)
        valid_loss, valid_acc,valid_precision, valid_specificity, valid_recall, valid_f1, threshold1 = valid(epoch, X_valid1,y_valid1)
        test_loss, test_acc, test_precision, test_specificity, test_recall, test_f1 = test(epoch, X_test1, y_test1, threshold1)
        print("train_loss:",train_loss,"train_acc:",train_acc,"train_precision:",train_precision,"train_specificity:",train_specificity,"train_recall:",train_recall,"train_f1:",train_f1)
        print("valid_loss:",valid_loss,"valid_acc:",valid_acc,"valid_precision:",valid_precision,"valid_specificity:",valid_specificity,"valid_recall:",valid_recall,"valid_f1:",valid_f1)
        print("test_loss:",test_loss,"test_acc:",test_acc,"test_precision:",test_precision,"test_specificity:",test_specificity,"test_recall:",test_recall,"test_f1:",test_f1)
        train_loss_list.append(train_loss)
        train_acc_list.append(train_acc)
        val_loss_list.append(valid_loss)
        val_acc_list.append(valid_acc)
        test_loss_list.append(test_loss)
        test_acc_list.append(test_acc)
        valid_loss, valid_acc,valid_precision, valid_specificity, valid_recall, valid_f1, threshold2 = valid(epoch, X_valid2, y_valid2)
        test_loss, test_acc, test_precision, test_specificity, test_recall, test_f1 = test(epoch, X_test2, y_test2, threshold2)
        print("valid_loss:",valid_loss,"valid_acc:",valid_acc,"valid_precision:",valid_precision,"valid_specificity:",valid_specificity,"valid_recall:",valid_recall,"valid_f1:",valid_f1)
        print("test_loss:",test_loss,"test_acc:",test_acc,"test_precision:",test_precision,"test_specificity:",test_specificity,"test_recall:",test_recall,"test_f1:",test_f1)
        train_loss_list.append(train_loss)
        train_acc_list.append(train_acc)
        val_loss_list.append(valid_loss)
        val_acc_list.append(valid_acc)
        test_loss_list.append(test_loss)
        test_acc_list.append(test_acc)
        with open(logname, 'a') as logfile:
            logwriter = csv.writer(logfile, delimiter=',')
            logwriter.writerow([epoch, train_loss, valid_loss, train_acc,valid_acc, test_loss, test_acc,
                            train_precision,valid_precision,test_precision,train_specificity,valid_specificity,test_specificity,
                            train_recall,valid_recall,test_recall,train_f1,valid_f1,test_f1])
    train_loss_list_ave.append(train_loss_list)
    train_acc_list_ave.append(train_acc_list)
    val_loss_list_ave.append(val_loss_list)
    val_acc_list_ave.append(val_acc_list)
    test_loss_list_ave.append(test_loss_list)
    test_acc_list_ave.append(test_acc_list)
print(number_test)
print(number_valid)
train_loss_list_ave = np.mean(train_loss_list_ave, axis=0)
train_acc_list_ave = np.mean(train_acc_list_ave, axis=0)
val_loss_list_ave = np.mean(val_loss_list_ave, axis=0)
val_acc_list_ave = np.mean(val_acc_list_ave, axis=0)
test_loss_list_ave = np.mean(test_loss_list_ave, axis=0)
test_acc_list_ave = np.mean(test_acc_list_ave, axis=0)
print(epoch+1)
print(train_acc_list)
print(number_valid, prediction_valid, targets_valid,outputs_valid, sex_valid, age_valid)
print("number_valid：",len(number_valid)," prediction_valid:",len(prediction_valid)," targets_valid:",len(targets_valid)," outputs_valid:",len(outputs_valid)," sex_valid:",len(sex_valid)," age_valid:",len(sex_valid))
result = [[a] + x + [y] + [z] + [b] + [c] for a, x, y, z, b, c in zip(number_valid, prediction_valid, targets_valid,outputs_valid, sex_valid, age_valid)]
with open(results_file +"/predscore/normal/"+ args.risk_factor +'_seed='  + str(args.seed) + 'predscore_valid.csv', "w", newline="") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerows(result)

result = [[a] + x + [y] + [z] + [b] + [c] for a, x, y, z, b, c in zip(number_test, prediction_test, targets_test,outputs_test, sex_test, age_test)]
with open(results_file +"/predscore/normal/"+ args.risk_factor +'_seed='  + str(args.seed) + 'predscore_test.csv', "w", newline="") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerows(result)


plt.figure()
plt.plot(range(epoch+1), train_loss_list_ave, color='blue', linestyle='-', label='train_loss')
plt.plot(range(epoch+1), val_loss_list_ave, color='green', linestyle='--', label='valid_loss')
plt.plot(range(epoch+1), test_loss_list_ave, color='red', linestyle='-', label='test_loss')
plt.legend()
plt.xlabel('epoch')
plt.ylabel('loss')
plt.title('Training and validation and Test loss')
plt.grid()
plt.savefig('loss.png')

plt.figure()
plt.plot(range(epoch+1), train_acc_list_ave, color='blue', linestyle='-', label='train_acc')
plt.plot(range(epoch+1), val_acc_list_ave, color='green', linestyle='--', label='valid_acc')
plt.plot(range(epoch+1), test_acc_list_ave, color='red', linestyle='-', label='test_acc')
plt.legend()
plt.xlabel('epoch')
plt.ylabel('acc')
plt.title('Training and validation and Test accuracy')
plt.grid()
plt.savefig('accuracy.png')