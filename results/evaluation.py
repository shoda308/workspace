import glob
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from scipy import stats
import numpy as np

results_dir, risk_factors = [],[]
def calculate_metrics(risk_factor,threshold):
    results_file = '/workspace/results/2024_06_data315/' + risk_factor
    results_dir.append(results_file)
    risk_factors.append(risk_factor)
    # ファイルパターンを指定します
    file_pattern = results_file + '/predscore/normal/' + risk_factor + '_seed=*predscore_test.csv'
    # 指定されたパターンに一致するファイルをすべて取得します
    all_files = glob.glob(file_pattern)
    print(all_files)
    # 結果を格納するリストを準備します
    results = []
    # 各ファイルに対して処理を行います
    for file_path in all_files:
        # CSVファイルを読み込みます
        df = pd.read_csv(file_path)
        
        # 予測スコアを基に予測ラベルを再計算します
        pred_scores = df.iloc[:, 2].values
        pred_labels = (pred_scores >= threshold).astype(int)
        
        # 正解ラベルを取得します
        true_labels = df.iloc[:, 3].values
        
        # 各評価指標を計算します
        accuracy = accuracy_score(true_labels, pred_labels)
        precision = precision_score(true_labels, pred_labels)
        sensitivity = recall_score(true_labels, pred_labels)  # sensitivityはrecallと同じです
        f_score = f1_score(true_labels, pred_labels)
        
        # Specificityの計算
        tn, fp, fn, tp = confusion_matrix(true_labels, pred_labels).ravel()
        specificity = tn / (tn + fp)
        
        # ファイル名からseedを抽出
        seed_str = file_path.split('seed=')[1].split('predscore_test')[0]
        seed = int(seed_str)
        
        # 結果をリストに追加します
        results.append([seed, accuracy, precision, specificity, sensitivity, f_score])
    
    # 結果をデータフレームに変換します
    results_df = pd.DataFrame(results, columns=['Seed', 'Accuracy', 'Precision', 'Specificity', 'Sensitivity', 'F-Score'])
    
    # 結果をCSVファイルに保存します
    results_df.to_csv(results_file + "/" + risk_factor + '_evaluation_metrics.csv', index=False)
    print("Results saved to diabetes_evaluation_metrics.csv")

# 信頼区間を計算する関数
def mean_confidence_interval(data, confidence=0.95):
    n = len(data)
    mean = np.mean(data)
    stderr = stats.sem(data)
    h = stderr * stats.t.ppf((1 + confidence) / 2., n-1)
    return mean, h

def t_test(risk_factors):
    summary_results = []
    for risk_factor in risk_factors:
        results_file = '/workspace/results/2024_06_data315/' + risk_factor
        # データをDataFrameに変換する
        df = pd.read_csv(results_file + "/" + risk_factor + '_evaluation_metrics.csv')
        metrics = ['Accuracy', 'Precision', 'Specificity', 'Sensitivity', 'F-Score']
        file_results = [risk_factor]
        for metric in metrics:
            mean, ci = mean_confidence_interval(df[metric])
            file_results.append(f"{mean:.4f}±{ci:.4f}")
        summary_results.append(file_results)

    # 結果をデータフレームに変換してCSVに保存
    summary_df = pd.DataFrame(summary_results, columns=['file', 'accuracy', 'precision', 'specificity', 'sensitivity', 'f-score'])
    summary_df.to_csv('summary_results.csv', index=False)
    

risk_factors = ["none", "diabetes", "dyslipidemia", "pressure", "smoking", "diabetes_dyslipidemia_pressure_smoking"]
for risk_factor in risk_factors:
    calculate_metrics(risk_factor,0.8)
t_test(risk_factors)