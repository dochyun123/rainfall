import pandas as pd
import numpy as np

data_df = pd.read_csv("sequential.csv")
columns_drop = ['class_interval','vv']
data_df.drop(columns = columns_drop)
split_datasets = {station: data_df[data_df['stn4contest'] == station] for station in data_df['stn4contest'].unique()}

X_total = []
y_total = []

def create_sequences(data, seq_length):
    xs = []
    ys = []
    for i in range(len(data)-seq_length):
        x = data.iloc[i:(i+seq_length)].drop('label', axis=1).values
        y = data.iloc[i+seq_length]['label']
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)


for i in range(1, 10):
    X, y = create_sequences(split_datasets[f'STN00{i}'], 10)
    X_total.extend(X)
    y_total.extend(y)

def remove_strings(sequence):
    return [[element for element in step if not isinstance(element, str)] for step in sequence]

for i in range(10, 21):
    X, y = create_sequences(split_datasets[f'STN0{i}'], 10)
    X_total.extend(X)
    y_total.extend(y)

X_total = [remove_strings(sequence) for sequence in X_total]
X_total = np.array(X_total)
y_total = np.array(y_total)

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, recall_score, accuracy_score, precision_score, f1_score


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Defining the LSTM model
model = Sequential([
    LSTM(50, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True),
    LSTM(50),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Training the model
history = model.fit(X_train, y_train, epochs=30, validation_split=0.2, batch_size=32, verbose=1)

y_pred = (model.predict(X_test) > 0.5).astype(int)
conf_matrix = confusion_matrix(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)

print('Training Loss:', history.history['loss'][-1])
print('Validation Loss:', history.history['val_loss'][-1])
print('Confusion Matrix:\n', conf_matrix)
print('F1 Score:', f1)
print('Accuracy:', accuracy)

