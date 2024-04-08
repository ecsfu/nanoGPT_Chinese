import os

import pandas as pd
import requests
import tiktoken
import numpy as np

# download the tiny shakespeare dataset
input_file_path = os.path.join(os.path.dirname(__file__), 'input.txt')
import csv

def prepare_datatxt():
    for txt in  ['medQA.test.txt','medQA.train.txt','medQA.valid.txt']:
        with open(txt, 'r', encoding='utf-8') as f:
                # 打开或创建CSV文件
            with open(f'{txt}.csv', 'w', newline='',encoding='utf-8') as csv_file:
                # 创建CSV写入器对象
                column_names = ['department','label','qaid','question','answer']
                writer = csv.writer(csv_file)
                writer.writerow(column_names)
                # 逐行读取文本文件
                for line in f:
                    # 假设每行文本由逗号分隔的值组成
                    # 将这些值写入CSV文件
                    writer.writerow(line.strip().split('\t'))
    df_cMedQA2 = pd.DataFrame()
    for csv_file in ['medQA.test.txt.csv','medQA.train.txt.csv','medQA.valid.txt.csv']:
        df = pd.read_csv(csv_file)
        df = df[df['label']==1]
        df_cMedQA2 = pd.concat([df_cMedQA2,df],ignore_index=True)


    datalist = df_cMedQA2[['question','answer']].values
    txtdata=""
    for data in datalist:
        txtdata+='\n'.join(data)+'\n'
    print(len(df_cMedQA2),len(txtdata))

    return txtdata
def prepare_datacsv():
    df_q = pd.read_csv('question.csv')
    df_q = df_q.rename(columns={'content':'question'})
    df_a = pd.read_csv('answer.csv')
    df_a = df_a[['question_id','content']]
    df_a = df_a.rename(columns={'content': 'answer'})
    df_aq = pd.merge(df_q,df_a,how='left',on='question_id')


    datalist = df_aq[['question', 'answer']].values
    txtdata = ""
    for data in datalist:
        txtdata += '\n'.join(data) + '\n'
    print(len(df_aq), len(txtdata))
    return txtdata
def prepare_data():
    data = prepare_datatxt()+prepare_datacsv()
    with open('input.txt', 'wt', encoding='utf-8') as f:
        f.write(data)
if __name__=='__main__':
    prepare_data()
    input_file_path = 'input.txt'
    with open(input_file_path, 'r', encoding='utf-8') as f:
        data = f.read()
    n = len(data)
    train_data = data[:int(n*0.9)]
    val_data = data[int(n*0.9):]
    print(data)
    # gpt4 编码
    enc = tiktoken.get_encoding("cl100k_base")
    train_ids = enc.encode_ordinary(train_data)
    val_ids = enc.encode_ordinary(val_data)
    print(f"train has {len(train_ids):,} tokens")
    print(f"val has {len(val_ids):,} tokens")

    # export to bin files
    train_ids = np.array(train_ids, dtype=np.uint16)
    val_ids = np.array(val_ids, dtype=np.uint16)
    train_ids.tofile(os.path.join(os.path.dirname(__file__), 'train.bin'))
    val_ids.tofile(os.path.join(os.path.dirname(__file__), 'val.bin'))

# train has 58,766,056 tokens
# val has 6,571,043 tokens
