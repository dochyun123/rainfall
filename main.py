import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix, recall_score, accuracy_score, precision_score, f1_score
from tensorflow.keras.models import load_model


# LSTM_class = load_model('LSTM_class.keras')
# data -> 0,1 labeling 
# import numpy as np

# Testing with whole train dataset
def preprocess(df):
    df.columns = df.columns.str.replace('rainfall_train.', '', regex=False)
    df.columns = df.columns.str.replace('rainfall_test.', '', regex=False)
    df = df[df['vv']>=0] # 결측치 제거
    columns = ['stn4contest','ef_month','dh','v01', 'v02', 'v03','v04', 'v05', 'v06', 'v07', 'v08', 'v09', 'vv', 'class_interval']
    df = df[columns]
    df = pd.get_dummies(df,columns = ['stn4contest','ef_month'])
    df['v01'] = df['v01']/100
    df['v02'] = df['v02']/100
    df['v03'] = df['v03']/100
    df['v04'] = df['v04']/100
    df['v05'] = df['v05']/100
    df['v06'] = df['v06']/100
    df['v07'] = df['v07']/100
    df['v08'] = df['v08']/100
    df['v09'] = df['v09']/100
    df['dh'] = df['dh']/240
    df['label'] = np.where(df['class_interval'] == 0, 0, 1)
    return df

def binomial_test(df):   # preprocess 실행한 dataframe을 입력받음
    model_xgb = XGBClassifier()
    model_xgb.load_model('xgb_binary.json')
    model = model_xgb
    X = df.drop(columns=['label','class_interval','vv'])
    y = df['label'] # 0 : 강수없음 1 : 강수있음
    y_pred = model.predict(X)
    return y,y_pred

def evaluation(y,y_pred):
    print('Accuracy:', accuracy_score(y, y_pred))
    print('Precision:', precision_score(y, y_pred))
    print('Recall:', recall_score(y, y_pred))
    print('F1:', f1_score(y, y_pred))
    print('Confusion Matrix:', confusion_matrix(y, y_pred))

'''
train data => 0,1 label => binary classificaiton
'''
train_df = pd.read_csv('train.csv')
preprocess_df = preprocess(train_df)
print(preprocess_df.columns)
y,y_pred = binomial_test(preprocess_df)
evaluation(y,y_pred)

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from sklearn.preprocessing import OneHotEncoder



def seqeuential(train_df):
    train_df.columns = train_df.columns.str.replace('rainfall_train.', '', regex=False)
    train_df.columns = train_df.columns.str.replace('rainfall_test.', '', regex=False)
    train_df = train_df[train_df['vv']>=0] # 결측치 제거
    columns = ['stn4contest','ef_year','ef_month','ef_day','ef_hour','dh','v01', 'v02', 'v03','v04', 'v05', 'v06', 'v07', 'v08', 'v09', 'vv', 'class_interval']
    train_df = train_df[columns]
    train_df['v01'] = train_df['v01']/100
    train_df['v02'] = train_df['v02']/100
    train_df['v03'] = train_df['v03']/100
    train_df['v04'] = train_df['v04']/100
    train_df['v05'] = train_df['v05']/100
    train_df['v06'] = train_df['v06']/100
    train_df['v07'] = train_df['v07']/100
    train_df['v08'] = train_df['v08']/100
    train_df['v09'] = train_df['v09']/100
    train_df['dh'] = train_df['dh']/240    
    seq_df = train_df.sort_values(by=['stn4contest','ef_year','ef_month','ef_day','ef_hour'])
    columns_drop = ['vv']
    seq_df = seq_df.drop(columns = columns_drop)
    split_datasets = {station: seq_df[seq_df['stn4contest'] == station] for station in seq_df['stn4contest'].unique()}
    grouped_counts = seq_df.groupby(['stn4contest','ef_year', 'ef_month', 'ef_day', 'ef_hour']).size()
    grouped_counts = grouped_counts.reset_index(name='count')
    start_idx = 0
    xs = []
    ys = []

    for i in grouped_counts['count']:
        seq_length = i
        end_idx = start_idx + seq_length
        if end_idx <= len(seq_df):
            x = seq_df.iloc[start_idx:end_idx].drop(columns = ['class_interval','stn4contest','ef_year'], axis=1).values
            y = seq_df.iloc[end_idx - 1]['class_interval']
            xs.append(x)
            ys.append(y)
        start_idx = end_idx

    return xs,ys

def zero_padding(xs):
    max_seq_length = max(len(seq) for seq in xs)
    padded_sequences = pad_sequences(xs, maxlen=max_seq_length, dtype='float32', padding='post')
    return padded_sequences


def sequential_test(xs):
    LSTM_class = load_model('LSTM_class_ver2.keras')
    y_pred = LSTM_class.predict(xs)
    return y_pred

def seq_eval(ys,y_pred):
    y_pred_class = np.argmax(y_pred,axis=1)
    accuracy = accuracy_score(ys, y_pred_class)
    cm = confusion_matrix(ys, y_pred_class)
    f1 = f1_score(ys, y_pred_class, average='macro')
    print("Confusion Matrix:")
    print(cm)
    print("Accuracy:", accuracy)
    print("F1 Score:", f1)

train_df = pd.read_csv('train.csv')
xs,ys = seqeuential(train_df)
xs = zero_padding(xs)
y_pred = sequential_test(xs) 
seq_eval(ys,y_pred)



