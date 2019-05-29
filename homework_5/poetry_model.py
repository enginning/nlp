# *-* coding:utf-8 *-*
'''
@author: enginning
@date:   2019/5/27 14:00
@Ref:    https://github.com/ioiogoo/poetry_generator_Keras
'''
import random
import os

import keras
import numpy as np
from keras.callbacks import LambdaCallback
from keras.models import Input, Model, load_model
from keras.layers import LSTM, Dropout, Dense, Flatten, Bidirectional, Embedding, GRU
from keras.utils.vis_utils import plot_model
from keras.optimizers import Adam
from data_utils import *


class PoetryModel(object):
    def __init__(self, config):
        self.model = None
        self.do_train = True
        self.loaded_model = False
        self.config = config

        # 文件预处理
        self.word2numF, self.num2word, self.words, self.files_content = preprocess_file(self.config)

        # 如果模型文件存在则直接加载模型，否则开始训练
        if os.path.exists(self.config.weight_file) and self.config.flag:
            self.model = load_model(self.config.weight_file)
            self.model.summary()
        else:
            self.train()
        self.do_train = False
        self.loaded_model = True

    def build_model(self):
        '''建立模型'''

        # 输入的dimension
        # 嵌入层embedding用在网络的开始层，将输入转换成分布式词向量
        input_tensor = Input(shape=(self.config.max_len,))
        embedd = Embedding(len(self.num2word) + 1, 300, input_length=self.config.max_len)(input_tensor)
        # Bidirectional: RNN 的双向封装器，对序列进行前向和后向计算
        # 因为单向 RNN，是根据前面的信息推出后面的，但有时只看前面的词是不够的
        lstm = Bidirectional(GRU(128, return_sequences=True))(embedd)
        # dropout = Dropout(0.6)(lstm)
        # lstm = LSTM(256)(dropout)
        # dropout = Dropout(0.6)(lstm)
        flatten = Flatten()(lstm)
        dense = Dense(len(self.words), activation='softmax')(flatten)
        self.model = Model(inputs=input_tensor, outputs=dense)
        optimizer = Adam(lr=self.config.learning_rate)
        self.model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        plot_model(self.model, to_file='Network.svg', show_shapes=True)

    def sample(self, preds, temperature=1.0):
        '''
        当temperature=1.0时，模型输出正常
        当temperature=0.5时，模型输出比较open
        当temperature=1.5时，模型输出比较保守
        在训练的过程中可以看到temperature不同，结果也不同
        '''
        preds = np.asarray(preds).astype('float64')
        preds = np.log(preds) / temperature
        exp_preds = np.exp(preds)
        preds = exp_preds / np.sum(exp_preds)
        probas = np.random.multinomial(1, preds, 1)
        return np.argmax(probas)

    def generate_sample_result(self, epoch, logs):
        '''训练过程中，每个epoch打印出当前的学习情况'''
        # if epoch % 5 != 0:
        #     return
        print("\n==================Epoch {}=====================".format(epoch))
        for diversity in [0.5, 1.0, 1.5]:
            print("------------Diversity {}--------------".format(diversity))
            # random.randint: 用于生成一个指定范围内的随机整数
            start_index = random.randint(0, len(self.files_content) - self.config.max_len - 1)
            generated = ''
            sentence = self.files_content[start_index: start_index + self.config.max_len]
            generated += sentence
            for i in range(20):
                x_pred = np.zeros((1, self.config.max_len))
                for t, char in enumerate(sentence[-6:]):
                    x_pred[0, t] = self.word2numF(char)

                preds = self.model.predict(x_pred, verbose=0)[0]
                next_index = self.sample(preds, diversity)
                next_char = self.num2word[next_index]

                generated += next_char
                sentence = sentence + next_char
            print(sentence)


    def predict(self, text):
        # '''根据给出的文字，生成诗句'''
        generated = ''
        sentence = text
        generated += sentence

        for i in range(20):
            x_pred = np.zeros((1, self.config.max_len))
            for t, char in enumerate(sentence[-6:]):
                x_pred[0, t] = self.word2numF(char)

            preds = self.model.predict(x_pred, verbose=0)[0]
            next_index = self.sample(preds, 1.0)
            next_char = self.num2word[next_index]

            generated += next_char
            sentence = sentence + next_char
        return sentence


    def data_generator(self):
        '''生成器生成数据'''
        i = 0
        while 1:
            x = self.files_content[i: i + self.config.max_len]
            y = self.files_content[i + self.config.max_len]

            puncs = [']', '[', '（', '）', '{', '}', '：', '《', '》', ':', '_', '-']
            if len([i for i in puncs if i in x]) != 0:
                i += 1
                continue
            if len([i for i in puncs if i in y]) != 0:
                i += 1
                continue

            y_vec = np.zeros(
                shape=(1, len(self.words)),
                dtype=np.bool
            )
            y_vec[0, self.word2numF(y)] = 1.0

            x_vec = np.zeros(
                shape=(1, self.config.max_len),
                dtype=np.int32
            )

            for t, char in enumerate(x):
                x_vec[0, t] = self.word2numF(char)
            yield x_vec, y_vec
            i += 1



    def train(self):
        '''训练模型'''
        number_of_epoch = len(self.files_content) // self.config.batch_size

        if not self.model:
            self.build_model()

        self.model.summary()
        
        if os.path.exists(self.config.weight_file):
            self.model.load_weights(self.config.weight_file)
            # 若成功加载前面保存的参数，输出下列信息
            print("checkpoint_loaded")

        self.model.fit_generator(
            generator=self.data_generator(),
            verbose=True,
            steps_per_epoch=self.config.batch_size,
            epochs=number_of_epoch,
            callbacks=[
                keras.callbacks.ModelCheckpoint(self.config.weight_file, save_weights_only=False, period=1),
                LambdaCallback(on_epoch_end=self.generate_sample_result)
            ]
        )


if __name__ == '__main__':
    from config import Config

    model = PoetryModel(Config)
    generate_line = 1
    while generate_line:
        text = "床前明月光，"
        sentence = model.predict(text)
        print(sentence)
        generate_line -= 1