# coding=UTF-8
# 基于LSTM的股票价格预测

'''
@File: stock_price_predict
@Author: WeiWei
@Time: 2022/12/3
@Email: weiwei_519@outlook.com
@Software: PyCharm
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import os
import tensorflow as tf
from tensorflow import keras

# 设置日志级别
# os.environ["TF_CPP_MIN_LOG_LEVEL"]='1' # 默认设置，显示所有信息
# os.environ["TF_CPP_MIN_LOG_LEVEL"]='2' # 只显示 warning 和 Error
# os.environ["TF_CPP_MIN_LOG_LEVEL"]='3' # 只显示 Error
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# set GPU
# 设置GPU的使用率可按需增长
# tf.debugging.set_log_device_placement(True)
# gpus = tf.config.experimental.list_physical_devices('GPU')
# tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
# for gpu in gpus:
#     tf.config.experimental.set_memory_growth(gpu, True)
# print(len(gpus))
# logical_gpus = tf.config.experimental.list_logical_devices('GPU')
# print(len(logical_gpus))

# read datasets
# [index_code	date	open	close	low	high	volume	money	change	label]
df = pd.read_csv('Datasets/time_predict/stock_price.csv')
ds_columns = ['open', 'close', 'low', 'high', 'volume', 'money', 'change', 'label']
df = df[ds_columns]
df.info()
pd_value = df.values

# 设置参数
batch_size = 1  # 一批次含数据的多少
features = 7  # 特征个数
units = df.shape[1] - features

# 原始价格数据展示
plt.plot(df['label'], label='Price')
plt.ylabel('Price')
plt.legend()
plt.show()

# 数据归一化到[0,1]
mms = MinMaxScaler(feature_range=(0, 1))
pd_value_norm = mms.fit_transform(pd_value)
print("pd_value after normalized: {0}".format(pd_value_norm[0:10, :]))

# 切分数据，创建训练集和测试集
train_size = int(len(pd_value_norm) * 0.8)  # 80%的训练集
train_ds = pd_value_norm[:train_size]
test_ds = pd_value_norm[train_size:]


# 不能使用sklearn.model_selection.train_test_split 来分割训练集和测试集，因为该方法会打乱数据集的顺序
# train_ds, test_ds = train_test_split(pd_value, test_size=0.2)


# X(t) = [open(t), close(t), low(t), high(t), volumn(t), money(t), change(t)]
# H(t) = [label(t)]
# 创建训练数据集跟测试数据集，以batch_size天作为窗口期来创建我们的训练数据集跟测试数据集，一般以一天为单位
def create_dataset(dataset, batch_size, features):
    X, H = [], []
    ds_X = dataset[:, 0:features]
    ds_H = dataset[:, features:dataset.shape[1]]
    for i in range(len(dataset) - batch_size - 1):
        a = ds_X[i:(i + batch_size)]
        X.append(a)
        H.append(ds_H[i + batch_size])
    return np.array(X), np.array(H)


train_X, train_H = create_dataset(train_ds, batch_size, features)
test_X, test_H = create_dataset(test_ds, batch_size, features)

# ========== set dataset ======================
# train_X = np.reshape(train_X, (train_X.shape[0], train_X.shape[1], features))
# test_X = np.reshape(test_X, (test_X.shape[0], test_X.shape[1], features))

# create and fit the LSTM network
model = tf.keras.Sequential()
# LSTM中，1，2，4的激活函数用sigmoid, 第三个默认是tanh
model.add(tf.keras.layers.LSTM(units=32, activation='tanh', recurrent_activation='sigmoid', return_sequences=True,
                               input_shape=(batch_size, features)))
model.add(tf.keras.layers.Dropout(0.2))
# model.add(tf.keras.layers.LSTM(32, activation='tanh', recurrent_activation='sigmoid'))
model.add(tf.keras.layers.Dense(units, activation='relu'))
# model.compile(optimizer='adam', loss='mse')
# model.compile(metrics=['accuracy'], loss='mean_squared_error', optimizer='adam')
model.compile(metrics=['accuracy'], loss='mae', optimizer='adam')  # mae：Mean Absolute Error 平均绝对误差
model.summary()
epochs = 30
history = model.fit(train_X, train_H, validation_data=(test_X, test_H), epochs=epochs, verbose=2).history
model.save("./model/stock_price_predict/stock_price_predict_model.h5py")

# plt.plot(history['loss'], linewidth=2, label='Train')
# plt.plot(history['val_loss'], linewidth=2, label='Test')
# plt.legend(loc='upper right')
# plt.title('Model loss')
# plt.ylabel('Loss')
# plt.xlabel('Epoch')
# # plt.ylim(ymin=0.70,ymax=1)
# plt.show()

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(history['accuracy'], label='Training Accuracy')
plt.plot(history['val_accuracy'], label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(history['loss'], label='Training Loss')
plt.plot(history['val_loss'], label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
# plt.savefig('./model/stock_price_predict/training.png')
plt.show()

# 测试集预测及绘图
test_X, test_H = create_dataset(test_ds, batch_size, features)
predict_H = model.predict(test_X)
predict_H = np.reshape(predict_H, (predict_H.shape[0], predict_H.shape[1]))
# 对预测结果进行反归一化
predict_ds = np.concatenate((test_X.squeeze(), predict_H), axis=1)
predict_H = mms.inverse_transform(predict_ds)[:, features:predict_ds.shape[1]]
plt.plot(predict_H, label='predict price')
# 获取原始价格测试集
test_ds = pd_value[train_size:]
test_X, test_H = create_dataset(test_ds, batch_size, features)
plt.plot(test_H, label='orignal price')
plt.legend()
plt.show()
