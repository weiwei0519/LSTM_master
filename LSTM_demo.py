# coding=UTF-8
# LSTM dim try

'''
@File: LSTM_demo
@Author: WeiWei
@Time: 2022/12/3
@Email: weiwei_519@outlook.com
@Software: PyCharm
'''

import tensorflow as tf

## tf.keras.layers.LSTM(
# units,
# activation=“tanh”,
# recurrent_activation=“sigmoid”,#用于重复步骤的激活功能
# use_bias=True,#，是否图层使用偏置向量
# kernel_initializer=“glorot_uniform”,#kernel权重矩阵的 初始化程序，用于输入的线性转换
# recurrent_initializer=“orthogonal”,#权重矩阵的 初始化程序，用于递归状态的线性转换
# bias_initializer=“zeros”,#偏差向量的初始化程序
# unit_forget_bias=True,#则在初始化时将1加到遗忘门的偏置上
# kernel_regularizer=None,＃正则化函数应用于kernel权重矩阵
# recurrent_regularizer=None,＃正则化函数应用于 权重矩阵
# bias_regularizer=None,＃正则化函数应用于偏差向量
# activity_regularizer=None,＃正则化函数应用于图层的输出（其“激活”）
# kernel_constraint=None,＃约束函数应用于kernel权重矩阵
# recurrent_constraint=None,＃约束函数应用于 权重矩阵
# bias_constraint=None,＃约束函数应用于偏差向量
# dropout=0.0,＃要进行线性转换的输入单位的分数
# recurrent_dropout=0.0,＃为递归状态的线性转换而下降的单位小数
# return_sequences=False,＃是否返回最后一个输出。在输出序列或完整序列中
# return_state=False,＃除输出外，是否返回最后一个状态
# go_backwards=False,＃如果为True，则向后处理输入序列并返回反向的序列
# stateful=False,＃如果为True，则批次中索引i的每个样本的最后状态将用作下一个批次中索引i的样本的初始状态。
# time_major=False,
# unroll=False,＃如果为True，则将展开网络，否则将使用符号循环。展开可以加快RNN的速度，尽管它通常会占用更多的内存。展开仅适用于短序列。
# )
#
# 一般情况下只使用三个参数
#
# units 输出空间的维度
# input_shape (timestep, input_dim),timestep可以设置为None,由输入决定,input_dime根据具体情况
# activation 激活函数,默认tanh


inputs = tf.random.normal([32, 10, 8])
lstm = tf.keras.layers.LSTM(4)
output = lstm(inputs)
print(output.shape)
#(32, 4)
lstm = tf.keras.layers.LSTM(4, return_sequences=True, return_state=True)
whole_seq_output, final_memory_state, final_carry_state = lstm(inputs)
print(whole_seq_output.shape)
#(32, 10, 4)
print(final_memory_state.shape)
#(32, 4)
print(final_carry_state.shape)
#(32, 4)