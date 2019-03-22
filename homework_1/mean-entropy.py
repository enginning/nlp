# -*- coding: utf-8 -*-
# by enginning for nlp homework_1

import os
import re
import jieba  
import math

print("Current path: {}".format( os.path.abspath(os.curdir)))
Dict = {}
word_count = 0
char_count = 0
jieba.load_userdict('./data/dict.txt')  

# 仅留中文
def remove_punctuation(line):
    #rule = re.compile("[^\u4e00-\u9fa5^.^a-z^A-Z^0-9]")   
    rule = re.compile("[^\u4e00-\u9fa5，。、《》（）；;：:]")   # 正则表达式，分句，同时保留中文
    line = re.sub(rule, '', line) 
    global char_count
    char_count += len(line)
    return line

# 创建停用词list  
def stopwordslist(filepath):  
    # file.readlines():  依次读取每行, 直至EOF; 
    # line.strip():      去掉每行头尾空白  
    stopwords = [line.strip() for line in open(filepath, 'r', encoding='utf-8').readlines()]  
    return stopwords  

# 对句子进行分词  
def seg_sentence(sentence):  
    sentence_seged = jieba.cut(sentence.strip(), cut_all=False, HMM=True)  
    stopwords = stopwordslist('./data/stopword.txt')    # 分词后，遍历一下停用词表，去掉停用词
    outstr = ''  
    for word in sentence_seged:  
        if word not in stopwords:   
            if (Dict.get(word) == None):        # 若查找不到键值，则添加到字典    
                Dict[word] = 1
            else:
                Dict[word] = Dict[word] + 1     # 找到则value + 1
            global word_count
            word_count  += 1
            outstr += word  
            outstr += " " 
    return outstr  

# 计算汉语平均熵
def mean_entropy (Dict):
    entropy = 0
    for key in Dict:
        Dict[key] /= word_count
        entropy += Dict[key] * math.log((1 / Dict[key]), 2)
    #entropy /= len(Dict)
    return entropy

with open('./data/santi.txt', 'r', encoding='utf-8') as inputs:
    with open('./data/output.txt', 'w', encoding='utf-8') as outputs:
        for line in inputs:  
            line_solve = remove_punctuation(line)   # 过滤
            line_seg = seg_sentence(line_solve)     # 分词, 这里的返回值是字符串  
            outputs.write(line_seg + '\n')  
        outputs.write(str(Dict))
        result = mean_entropy(Dict)      

print("\nCalculation is done successfully! The concrete details are in ./data/output.txt") 
print("Total number of chars: {0}\nTotal number of words: {1}\nMean entropy: {2:.2f}".format(char_count, \
    len(Dict), result))