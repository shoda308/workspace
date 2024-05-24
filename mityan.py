import re
import subprocess
import argparse

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--risk_factor', default="none", type=str,
                    help='input data')
parser.add_argument('--threshold', default="0.1", type=str,
                    help='input data')
args = parser.parse_args()

threshold = args.threshold
risk_factor = args.risk_factor # 追加したリスクファクター
risk_factor_columns = ["none"]
# risk_factor_columns = ["diabetes","dyslipidemia","pressure","smoking"]
# risk_factor_columns = ["diabetes_dyslipidemia","diabetes_pressure","diabetes_smoking","dyslipidemia_pressure","dyslipidemia_smoking","pressure_smoking"]
# risk_factor_columns = ["diabetes_dyslipidemia_smoking","diabetes_dyslipidemia_pressure","diabetes_pressure_smoking","dyslipidemia_pressure_smoking"]
# risk_factor_columns = ["diabetes_dyslipidemia_pressure_smoking"]
for i in risk_factor_columns:
    seed = 5
    command = f" CUDA_VISIBLE_DEVICES=0 /usr/local/bin/python /workspace/mixup-cifar10-main/train_mixup.py --lr=0.001 --decay=5e-04 --batch-size=10 --epoch=300 --seed={seed} --risk_factor="+ i
    subprocess.run(command, shell=True)
    
for i in risk_factor_columns:
    seed = 5
    command = f" CUDA_VISIBLE_DEVICES=0 /usr/local/bin/python /workspace/mixup-cifar10-main/train.py --lr=0.001 --decay=5e-04 --batch-size=10 --epoch=300 --seed={seed} --risk_factor=" + i
    subprocess.run(command, shell=True)

