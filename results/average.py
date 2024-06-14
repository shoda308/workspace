import glob
import pandas as pd
import numpy as np
import csv
import glob
import os
# plot graph
import matplotlib.pyplot as plt
import pandas as pd
import argparse

results_dir = []
risk_factors = []

def average(risk_factor):
    results_file = '/workspace/results/2024_06_data315/' + risk_factor
    results_dir.append(results_file)
    risk_factors.append(risk_factor)
    #averageディレクトリを作成
    os.makedirs(results_file + '/average', exist_ok=True)
    # ファイルパスのパターン
    file_pattern = results_file + "/log/normal/seed=*"
    # ファイルパスの一覧を取得
    csv_files = glob.glob(file_pattern)
    for file_path in csv_files:
        # 処理するファイル名を取得
        file_name = file_path.split("/")[-1]
        print(file_name)

        # ファイルを上書きするため一時ファイルを生成
        temp_file = f"temp_{file_name}"

        # with open(file_path, 'r') as f_in, open(temp_file, 'w', newline='') as f_out:
        #     reader = csv.reader(f_in)
        #     writer = csv.writer(f_out)

        #     for row in reader:
        #         new_row = []
        #         for item in row:
        #             if item.startswith("tensor(") and item.endswith(")"):
        #                 item = float(item[7:-1])
        #             new_row.append(item)
        #         writer.writerow(new_row)

        # # 一時ファイルを元のファイルにリネームして上書き
        # os.replace(temp_file, file_path)

        # print(f"ファイル {file_path} を処理し、上書きしました。")

    # データを格納するリスト
    data = []
    final_df = pd.DataFrame()
    # ファイルごとにデータを読み込んでリストに追加
    for file_path in csv_files:
        df = pd.read_csv(file_path)
        # dfを5つの等分に分割
        split_size = len(df) // 5
        split_dfs = [df[i:i+split_size] for i in range(0, len(df), split_size)]
        for split_df in split_dfs:
            final_df = final_df.append(split_df.iloc[-1], ignore_index=True)
            data.append(split_df.values)
    # CSVファイルとして出力
    final_df.to_csv(results_file + '/average/' + risk_factor + ".csv", index=False)

    # データの平均値を計算
    if len(data) > 0:
        averages = np.mean(data, axis=0)

        # 平均値をDataFrameに変換
        averages_df = pd.DataFrame(averages, columns=df.columns)

        print(averages_df)

        # 結果をCSVファイルとして保存する
        averages_df.to_csv(results_file + '/average/averages_' + risk_factor + ".csv", index=False)

        print("averages_" + risk_factor + ".csv ファイルが作成されました。")
        print(len(csv_files))

        train_loss_list_ave = list(averages_df['train loss'])
        val_loss_list_ave = list(averages_df['reg loss'])
        test_loss_list_ave = list(averages_df['test loss'])
        train_acc_list_ave = list(averages_df['train acc'])
        val_acc_list_ave = list(averages_df['reg acc'])
        test_acc_list_ave = list(averages_df['test acc'])

        plt.figure()
        plt.plot(range(len(averages_df)), train_loss_list_ave, color='blue', linestyle='-', label='train_loss')
        plt.plot(range(len(averages_df)), val_loss_list_ave, color='green', linestyle='--', label='valid_loss')
        plt.plot(range(len(averages_df)), test_loss_list_ave, color='red', linestyle='-', label='test_loss')
        plt.ylim(0.58, 0.77)
        plt.legend()
        plt.xlabel('epoch')
        plt.ylabel('loss_average')
        plt.title('Training and validation and Test loss')
        plt.grid()
        plt.savefig(results_file + '/average/loss_average_' + risk_factor + '.png')

        plt.figure()
        plt.plot(range(len(averages_df)), train_acc_list_ave, color='blue', linestyle='-', label='train_acc')
        plt.plot(range(len(averages_df)), val_acc_list_ave, color='green', linestyle='--', label='valid_acc')
        plt.plot(range(len(averages_df)), test_acc_list_ave, color='red', linestyle='-', label='test_acc')
        plt.legend()
        plt.xlabel('epoch')
        plt.ylabel('acc')
        plt.title('Training and validation and Test accuracy')
        plt.grid()
        plt.savefig(results_file + '/average/accuracy_average_' + risk_factor + '.png')

        
def plt_average():
    averages_df_1 = pd.read_csv(results_dir[0] + "/average/averages_"+ risk_factors[0] +".csv", index_col=0)
    averages_df_2 = pd.read_csv(results_dir[1] + "/average/averages_"+ risk_factors[1] +".csv", index_col=0)

    train_loss_1 = list(averages_df_1['train loss'])
    test_loss_1 = list(averages_df_1['test loss'])
    train_acc_1 = list(averages_df_1['train acc'])
    test_acc_1 = list(averages_df_1['test acc'])

    train_loss_2 = list(averages_df_2['train loss'])
    test_loss_2 = list(averages_df_2['test loss'])
    train_acc_2 = list(averages_df_2['train acc'])
    test_acc_2 = list(averages_df_2['test acc'])

    # グラフサイズを設定
    fig1, ax1 = plt.subplots(figsize=(7, 5))
    ax2 = ax1.twinx()

    ax1.plot(range(len(averages_df_1)), train_loss_1, color='blue', linestyle='-', label='train_loss')
    ax1.plot(range(len(averages_df_1)), test_loss_1, '+', label='test_loss')
    ax2.plot(range(len(averages_df_1)), train_acc_1, color='red', linestyle='--', label='train_acc')
    ax2.plot(range(len(averages_df_1)), test_acc_1, color='red', linestyle=':', label='test_acc')
    ax1.set_xlabel('No. epochs', fontsize=14)
    ax1.set_ylabel('Loss Function', fontsize=14)
    ax2.set_ylabel('Accuracy of tests [%]', fontsize=14)

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc='lower right', fontsize=12)

    # x軸の目盛りを1, 30, 60, 90, 120, 150と表示
    x_tick_positions = [1, 60,120,180,240,300]
    x_tick_labels = [str(x) for x in x_tick_positions]
    ax1.set_xticks(x_tick_positions + [len(averages_df_1) - 1])
    ax1.set_xticklabels(x_tick_labels + [str(len(averages_df_1))])

    # x軸のメモリのサイズを大きく
    ax1.tick_params(axis='x', labelsize=12)
    # y軸のメモリのサイズを大きく
    ax1.tick_params(axis='y', labelsize=12)
    ax2.tick_params(axis='y', labelsize=12)
    # plt.tight_layout()
    plt.savefig(results_dir[0] + '/average/test_average.png')

average("diabetes_dyslipidemia_pressure_smoking")
average("none")
plt_average()

