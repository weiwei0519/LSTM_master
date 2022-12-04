# coding=UTF-8
# 利用tensorflow2自带keras搭建BiLSTM+CRF的序列标注模型，完成中文的命名实体识别任务

'''
@File: NER_BiLSTM_CRF_tf2
@Author: WeiWei
@Time: 2022/12/4
@Email: weiwei_519@outlook.com
@Software: PyCharm
'''

import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing import sequence
import numpy as np

# 查看tensorflow和tensorflow-addons的版本号
print(tf.__version__)
print(tfa.__version__)
training_testing = "testing"  # training训练模式，testing测试模式

# 加载数据集
char_vocab_path = "./Datasets/NER/char_vocabs.txt"  # 字典文件
train_data_path = "./Datasets/NER/train_data"  # 训练数据
test_data_path = "./Datasets/NER/test_data"  # 测试数据

special_words = ['<PAD>', '<UNK>']  # 特殊词表示

# "BIO"标记的标签
label2idx = {"O": 0,
             "B-PER": 1, "I-PER": 2,
             "B-LOC": 3, "I-LOC": 4,
             "B-ORG": 5, "I-ORG": 6
             }
# 索引和BIO标签对应
idx2label = {idx: label for label, idx in label2idx.items()}

# 读取字符词典文件
with open(char_vocab_path, "r", encoding="utf8") as fo:
    char_vocabs = [line.strip() for line in fo]
char_vocabs = special_words + char_vocabs

# 字符和索引编号对应
idx2vocab = {idx: char for idx, char in enumerate(char_vocabs)}
vocab2idx = {char: idx for idx, char in idx2vocab.items()}

# 模型参数设置
EPOCHS = 20
BATCH_SIZE = 64
EMBED_DIM = 128
HIDDEN_SIZE = 64
MAX_LEN = 100
VOCAB_SIZE = len(vocab2idx)
CLASS_NUMS = len(label2idx)


# CRF模型
class CRF(layers.Layer):
    def __init__(self, label_size):
        super(CRF, self).__init__()
        self.trans_params = tf.Variable(
            tf.random.uniform(shape=(label_size, label_size)), name="transition")

    @tf.function
    def call(self, inputs, labels, seq_lens):
        log_likelihood, self.trans_params = tfa.text.crf_log_likelihood(
            inputs, labels, seq_lens,
            transition_params=self.trans_params)
        loss = tf.reduce_sum(-log_likelihood)
        return loss


K.clear_session()
# 构建模型
inputs = layers.Input(shape=(MAX_LEN,), name='input_ids', dtype='int32')
targets = layers.Input(shape=(MAX_LEN,), name='target_ids', dtype='int32')
seq_lens = layers.Input(shape=(), name='input_lens', dtype='int32')
# embedding层，对输入层进行编码
x = layers.Embedding(input_dim=VOCAB_SIZE, output_dim=EMBED_DIM, mask_zero=True)(inputs)
x = layers.Bidirectional(layers.LSTM(units=HIDDEN_SIZE, return_sequences=True))(x)
logits = layers.Dense(CLASS_NUMS)(x)
loss = CRF(label_size=CLASS_NUMS)(logits, targets, seq_lens)

model = models.Model(inputs=[inputs, targets, seq_lens], outputs=loss)
print(model.summary())


class CustomLoss(keras.losses.Loss):
    def call(self, y_true, y_pred):
        loss, pred = y_pred
        return loss


# 自定义Loss
# model.compile(loss=CustomLoss(), optimizer='adam')
# 或者使用lambda表达式
model.compile(loss=lambda y_true, y_pred: y_pred, optimizer='adam')


# 读取训练语料
def read_corpus(corpus_path, vocab2idx, label2idx):
    words, labels = [], []
    with open(corpus_path, encoding='utf-8') as fr:
        lines = fr.readlines()
    sent_, tag_ = [], []
    for line in lines:
        if line != '\n':
            char, label = line.strip().split()
            sent_.append(char)
            tag_.append(label)
        else:
            sent_ids = [vocab2idx[char] if char in vocab2idx else vocab2idx['<UNK>'] for char in sent_]
            tag_ids = [label2idx[label] if label in label2idx else 0 for label in tag_]
            words.append(sent_ids)
            labels.append(tag_ids)
            sent_, tag_ = [], []
    return words, labels


# 加载训练集
train_words, train_labels = read_corpus(train_data_path, vocab2idx, label2idx)
# 加载测试集
test_words, test_labels = read_corpus(test_data_path, vocab2idx, label2idx)
## 为了实现的简便，keras只能接受长度相同的序列输入。因此如果目前序列长度参差不齐，这时需要使用pad_sequences()。
# 该函数是将序列转化为经过填充以后的一个长度相同的新序列新序列，返回值：np.array()
# maxlen：大于此长度的序列将被截短，小于此长度的序列将在后部填0.
# padding: 'pre'或'post'，确定当需要补0时，在序列的起始还是结尾补.
train_words = sequence.pad_sequences(train_words, maxlen=MAX_LEN)  # maxlen：大于此长度的序列将被截短，小于此长度的序列将在后部填0.
train_labels = sequence.pad_sequences(train_labels, maxlen=MAX_LEN)
train_seq_lens = np.array([MAX_LEN] * len(train_labels))
labels = np.ones(len(train_labels))
# train_labels = keras.utils.to_categorical(train_labels, CLASS_NUMS)

print(np.shape(train_words), np.shape(train_labels))

if training_testing == "training":
    # 训练模型
    model.fit(x=[train_words, train_labels, train_seq_lens], y=labels,
              validation_split=0.1, batch_size=BATCH_SIZE, epochs=EPOCHS)

    # 保存
    model.save("./model/bilstm_crf_ner.h5py")
elif training_testing == "testing":
    # 加载模型
    model = models.load_model("./model/bilstm_crf_ner.h5py", custom_objects={'<lambda>': lambda y_true, y_pred: y_pred},
                              compile=True)

trans_params = model.get_layer('crf').get_weights()[0]
print(trans_params)
# 获得BiLSTM的输出logits
sub_model = models.Model(inputs=model.get_layer('input_ids').input, outputs=model.get_layer('dense').output)


# 模型预测
def predict(model, inputs, input_lens):
    logits = sub_model.predict(inputs)
    # 获取CRF层的转移矩阵
    # crf_decode：viterbi解码获得结果
    pred_seq, viterbi_score = tfa.text.crf_decode(logits, trans_params, input_lens)
    return pred_seq


sentence = "中华人民共和国国务院总理周恩来在外交部长陈毅的陪同下，连续访问了埃塞俄比亚等非洲10国以及阿尔巴尼亚。"
sent_chars = list(sentence)
sent2id = [vocab2idx[word] if word in vocab2idx else vocab2idx['<UNK>'] for word in sent_chars]
sent2id_new = np.array([[0] * (MAX_LEN - len(sent2id)) + sent2id[:MAX_LEN]])
test_lens = np.array([100])

pred_seq = predict(model, sent2id_new, test_lens)
print(pred_seq)
y_label = pred_seq.numpy().reshape(1, -1)[0]
print(y_label)
y_ner = [idx2label[i] for i in y_label][-len(sent_chars):]
print(sent2id)
print(y_ner)


# 对预测结果进行命名实体解析和提取
def get_valid_nertag(input_data, result_tags):
    result_words = []
    start, end = 0, 1  # 实体开始结束位置标识
    tag_label = "O"  # 实体类型标识
    for i, tag in enumerate(result_tags):
        if tag.startswith("B"):
            if tag_label != "O":  # 当前实体tag之前有其他实体
                result_words.append((input_data[start: end], tag_label))  # 获取实体
            tag_label = tag.split("-")[1]  # 获取当前实体类型
            start, end = i, i + 1  # 开始和结束位置变更
        elif tag.startswith("I"):
            temp_label = tag.split("-")[1]
            if temp_label == tag_label:  # 当前实体tag是之前实体的一部分
                end += 1  # 结束位置end扩展
        elif tag == "O":
            if tag_label != "O":  # 当前位置非实体 但是之前有实体
                result_words.append((input_data[start: end], tag_label))  # 获取实体
                tag_label = "O"  # 实体类型置"O"
            start, end = i, i + 1  # 开始和结束位置变更
    if tag_label != "O":  # 最后结尾还有实体
        result_words.append((input_data[start: end], tag_label))  # 获取结尾的实体
    return result_words


result_words = get_valid_nertag(sent_chars, y_ner)
for (word, tag) in result_words:
    print("".join(word), tag)
