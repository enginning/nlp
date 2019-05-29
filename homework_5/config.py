# *-* coding:utf-8 *-*
'''
@author: enginning
@date:   2019/5/26 16:00
@Ref:    https://github.com/ioiogoo/poetry_generator_Keras
'''


class Config(object):
    poetry_file = 'poetry_generator/poetry.txt'
    weight_file = 'poetry_generator/poetry_model.h5'
    
    # 根据前六个字预测第七个字
    flag = 1
    max_len = 6
    batch_size = 512
    learning_rate = 0.001
