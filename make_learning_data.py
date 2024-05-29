from sklearn.datasets import load_iris
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder
import csv


def binary_data(df):#入力値バイナリのもの
    use_column_index = []
    column_list = df.columns.values
    use_column_list = ['刺すような痛み','鈍い痛み','圧迫されるような痛み','締め付けるような痛み','左胸部','右胸部','前胸部','心窩部','背部','首や肩に放散', '局所的',
                       '瞬間的', '5分以内', '5分〜20分', '20分以上', 'それ以上', '動いた時', '安静時', '体位を変えたとき', '呼吸との関係',
                        '食中もしくは食後', '透析中','はい', 'いいえ', '朝方', '日中', '夜', '寝ているとき','動悸', '呼吸困難', '発熱', '冷汗', '吐き気、嘔吐', '倦怠感', '失神・前失神','result']
    #use_column_listに含まれている要素のindexをuse_column_indexに格納する
    for i,use_column in enumerate(use_column_list):
        for k,column in enumerate(column_list):
            if column == use_column:
                use_column_index.append(k)
    #使用する項目以外を削除する   
    dataset = df.iloc[:,use_column_index]
    #Nanを0に変換する
    dataset = dataset.fillna(0)
    return dataset

#カテゴリー単位のベクトル作成、二次元にしてから平均をとって作成
def make_lerning1_category_data():
    df1 = pd.read_csv("/Users/shoda/labo/learning/data/loading/learning_in_result.csv") # データセットの読み込み
    #配列宣言
    x = []
    y = []
    result =[]
    with open("/Users/shoda/labo/learning/data/loading/uni_vector_2d.csv") as f:
        #csvファイルのデータをループ
        reader = csv.reader(f)
        for row in reader:

            #A列を配列へ格納
            x.append(float(row[0]))

            #B列を配列へ格納
            y.append(float(row[1]))
        # del x[0]
        # del y[0]
    i=0
    print(len(x))
    print(len(y))
    # カテゴリーに含まれるベクトル
    unit_vec = []
    # カテゴリー単位のベクトル
    average_vec  = []
    # 問診内容
    lerning1_category_data = []
    print(len(df1))
    # 欠損値がある行を除去
    df1 = df1.dropna(axis=0).reset_index(drop=True)
    df1 = df1.drop(df1.columns[[0]], axis=1)
    #カテゴリーごとにベクトルの平均値を求める
    print(len(df1))
    for index, data in df1.iterrows():
        print(index)
        j=0
        # print(index)
        for column_name, item in data.iteritems():
            # print(item)
            if column_name == "result":
                result.append(item)
            elif item == 1:
                i = i + 1
                # print([unit_vector.iloc[i,0],unit_vector.iloc[i,1]])
                unit_vec.append([x[j],y[j]])
            # print(unit_vec)
            # print(column_name)
            if column_name == "締め付けるような痛み" or column_name == "首や肩に放散" or column_name == "それ以上" or column_name == "食中もしくは食後" or column_name == "寝ているとき" :
                print(column_name)
                if i >1:
                    aveX = np.mean(unit_vec, axis=0)[0]
                    aveY = np.mean(unit_vec, axis=0)[1]
                    average_vec.append(aveX)
                    average_vec.append(aveY)
                elif i == 1:
                    average_vec.append(unit_vec[0][0])
                    average_vec.append(unit_vec[0][1])
                else:
                    average_vec.append(0)
                    average_vec.append(0)
                unit_vec = []
                aveX=0
                aveY=0
                i=0
            if column_name == "失神・前失神":
                if i >1:
                    aveX = np.mean(unit_vec, axis=0)[0]
                    aveY = np.mean(unit_vec, axis=0)[1]
                    average_vec.append(aveX)
                    average_vec.append(aveY)
                elif i == 1:
                    average_vec.append(unit_vec[0][0])
                    average_vec.append(unit_vec[0][1])
                else:
                    average_vec.append(0)
                    average_vec.append(0)
                print(j)
                lerning1_category_data.append(average_vec)
                # print(lerning1_category_data)
                unit_vec = []
                aveX=0
                aveY=0
                i=0
                average_vec = []
            j = j + 1
    
    a = np.array(lerning1_category_data)
    print(a)
    np.savetxt('category.csv', a, delimiter=',')

if __name__ == "__main__":
    #data_in_result.xlsxから診断情報を読み込む
    df = pd.read_excel("/workspace/data/excel/data_in_result3.xlsx",skiprows=1)
    print(df)
    df.to_csv('/workspace/data/loading/2024_05/bi3.csv')
    binary_data(df).to_csv('/workspace/data/loading/2024_05/binary3.csv')
