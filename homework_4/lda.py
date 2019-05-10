'''
    Created on 2019.05.07
    @author: enginning
    Dataset: 20-Newsgroups
    Dataset_link:  http://qwone.com/~jason/20Newsgroups/
    Ref_link:      https://github.com/spookyQubit/LDA-TopicModeling

    In this project, we will use Latent-Diritchlet-Allocation (LDA), 
    an unsupervised learning algorithm, for clustering similar texts together. 
    LDA assumes that each document is made of several topics and 
    that each topic has a unique probability distribution of words associated with it 
    (words which are common to one topic, say word "pizza" in topic "food", might not be common to some other topic, say "politics"). 
    Given a new document, LDA can assign probabilities for the document to have been generated by each topic.
'''


import tarfile
import itertools
import gensim
import os
import glob
import time
from nltk.tokenize import RegexpTokenizer
from nltk.stem.porter import PorterStemmer
from os.path import isfile, join
import numpy as np
# 词性标注, 词形还原
from nltk import word_tokenize, pos_tag
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
np.set_printoptions(threshold=np.inf)

Num_topics = 20
flag = 1
dir_paths_train = glob.glob("./Data/20news-bydate-train/*")
start = time.clock()

# STOPWORDS
STOPWORDS = []
with open('./Data/STOPWORDS.txt', 'r', encoding = 'utf-8') as _stop:
    lines = _stop.readlines()
    for line in lines:
        word = line.strip()
        STOPWORDS.append(word)


# 获取单词的词性
def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return None


# Utility functions
def process_message(message): 
    
    """
    purpose: Given an input message, the function returns a processed version 
             of the message.
    input: 
    message: A string
    
    output:
    A list of tokens(words) with the following processing done on wach token:
           1) The top and bottom blocks are removed 
           2) Lower the case of the doc
           3) Tokenize the doc
           4) Remove stopwords
           5) Stem the words
    """
    
    # Clear the beginning
    blocks = message.lower()
    index = blocks.find(r'\n\n') + 4
    blocks = blocks[index:]
    content   = blocks.replace('\\n', ' ')
    tokenizer = RegexpTokenizer(r'[a-z]+')
    tokens = tokenizer.tokenize(content)
    if len(tokens)  > 1:
        tokens.pop(0)
    # 获取单词词性
    tagged_sent = pos_tag(tokens)     
    wnl = WordNetLemmatizer()
    lemmas_sent = []

    for tag in tagged_sent:
        if (tag[0] not in STOPWORDS) and (len(tag[0]) > 2):
            wordnet_pos = get_wordnet_pos(tag[1]) or wordnet.NOUN
            lemmatization = wnl.lemmatize(tag[0], pos=wordnet_pos)
            if lemmatization not in STOPWORDS:
                # 词形还原
                lemmas_sent.append(lemmatization) 

    return lemmas_sent


def iter_20NewsDocs(dir_path):
    
    """
    purpose: given a directory, this generator:
             1) finds a valid file in the directory
             2) extracts the document in the file,
             3) calls process_message to process the document content
             4) yields the processed document as list of tokens as returned by process_message
    input:
    dir_path: directory name to look for document files
    
    output:
    List of tokens as returned by process_message for a document. 
    The output is yielded with each yield returning doc for a single file in dir_path. 
            
    带有 yield 的函数不再是一个普通函数，而是一个生成器generator，可用于迭代
    """
    global flag
    print("{0:4}: {1:}".format(flag, dir_path))
    flag += 1
    for file_name in os.listdir(dir_path):
        path = join(dir_path, file_name)
        if isfile(path):
            with open(path, 'rb') as f:
                string = f.read()
                # print(type(string), '\n', string)
                string = str(string)
                # Preprocessing data
                # result = process_message(string)
                yield process_message(string)


# Create data streams
# Chain the training/test data streams for each news group to create a single training/test stream 
'''
    itertools.chain: 连接多个列表或者迭代器
    eg. x = itertools.chain(range(3), range(4), [3, 2, 1])
        print(list(x))
        >>> [0, 1, 2, 0, 1, 2, 3, 3, 2, 1]
    generator_link: https://www.jianshu.com/p/d09778f4e055
'''


def getTrainingDataStream():
    print("Chian TrainingDataStream")
    generator_ = []
    for path in dir_paths_train:
        yield_ = iter_20NewsDocs(path)
        generator_.append(yield_)
    stream_train = itertools.chain(*generator_)
                      
    return stream_train


"""
    id2word_20NewsDocs: Dictionary of the entire training vocabulary
                        with key being an id to get the word in the vocabulary.
                        The dictionary also has the statistics of the words in the vacabulary. 
    corpora是gensim中的一个基本概念，是文档集的表现形式    
    gensim.corpora.dictionary.Dictionary类为每个出现在语料库中的词分配了一个整型的id，这个类扫描文本，收集词频及其他统计值
    gensim corpora 的 dictionary 用法: https://blog.csdn.net/qq_19707521/article/details/79174533
"""
if not(isfile('20news-bydate_lda.model') and isfile('Lda_dict.dict')):
    print("Getting Dictionary ...")
    id2word_20NewsDocs = gensim.corpora.dictionary.Dictionary(getTrainingDataStream())
    # Remove all words which appear in less than 10 documents and in more than 10% of the documents.
    id2word_20NewsDocs.filter_extremes(no_below = 10, no_above = 0.1)
    # remove gaps in id sequence after words that were removed
    id2word_20NewsDocs.compactify()
    id2word_20NewsDocs.save('Lda_dict.dict')
    print("Dictionary Done!")


# A class for getting bag of words per document (每个文档的词袋)
'''
    itertools.islice: 对迭代器进行切片
    eg. x = itertools.islice(range(10), 0, 9, 2)
        print(list(x))
        >>> [0, 2, 4, 6, 8]  
    将文本特征的原始表达 (单词) 转化成词袋模型对应的稀疏向量的表达
'''
class NewsDocsCorpus(object):
    def __init__(self, dir_path_list, dictionary, num_docs=None):
        self.dir_path_list = dir_path_list
        self.dictionary = dictionary
        self.num_docs = num_docs
    def __iter__(self):
        for dir_path in self.dir_path_list:
            for tokens in itertools.islice(iter_20NewsDocs(dir_path), self.num_docs):
                yield self.dictionary.doc2bow(tokens)
    def __len__(self):
        return self.num_docs

"""
    Create a generator which will yield a bag of words 
    for each document in all the files belonging to every directory in dir_paths_train.
"""
if not(isfile('20news-bydate_lda.model') and isfile('Lda_dict.dict')):
    print("Getting the vector of doc2bow ...")
    newsDocs_corpus_train = NewsDocsCorpus(dir_paths_train, id2word_20NewsDocs)
    print("The vector of doc2bow Done!")


"""
    Train LDA model
    Train the LDA model.
    passes controls how often we train the model on the entire corpus. Another word for passes might be "epochs". 
    iterations is somewhat technical, but essentially it controls how often we repeat a particular loop 
    Gensim模型都接受一段训练语料 (对应着一个稀疏向量的迭代器) 作为初始化的参数
"""
if isfile('20news-bydate_lda.model') and isfile('Lda_dict.dict'):
    newsData_lda_model = gensim.models.LdaModel.load('20news-bydate_lda.model')
    id2word_20NewsDocs = gensim.corpora.dictionary.Dictionary.load('Lda_dict.dict')
    print("\nDictionary loaded!\nLDA model loaded!\n")
else:
    print("\nTraining LDA model ...")
    model_start = time.clock()
    newsData_lda_model = gensim.models.LdaModel(newsDocs_corpus_train, 
                                                num_topics=Num_topics, 
                                                id2word=id2word_20NewsDocs, 
                                                passes=10)
    newsData_lda_model.save('20news-bydate_lda.model')
    model_time = (time.clock() - model_start)
    print("LDA model Done!\nThe time using for trainning is {:.2f}s\n".format(model_time))

# Display all the topics and the 5 most relevant words in each topic
for topic in  newsData_lda_model.show_topics(-1, num_words = 5):
    print(topic[0], topic[1])


# Test dataset
dir_paths_test = glob.glob("./Data/20news-bydate-test/*")
cc = 0.7

# Utility function to get topic-document probabilities
def get_topic_probabilities(dir_path, num_docs=None):
    """
    purpose: for num_docs documents in dir_path, return the topic-document probabilities 
                    T1    T2    T3                        
             doc1 [[p_11, p_12, p_13],
             doc2  [p_21, p_22, p_23], 
                    ....
                    ....
         num_docs  [p_num_docsT1, p_num_docsT2, p_num_docsT3]]
    input:
       dir_path: directory path for the files whose topic-document probability we want to find 
       num_docs: number of documents whose topic-document probability we want to find
    output:
       return the topic-document probabilities
    """
    probability = []
    for c_ in itertools.islice(NewsDocsCorpus([dir_path], id2word_20NewsDocs, num_docs), num_docs):
        global Num_topics
        current_prob = [0] * Num_topics
        for topic, prob in newsData_lda_model.get_document_topics(c_):
            current_prob[topic] = prob  
        probability.append(current_prob)
    return probability


# Get the topic-document probabilities
def Map(Paths, Dict = None):
    Centroid_Dict = {}
    kind = -1
    Svm_data  = []
    Svm_label = []
    if Dict == None:
        Svm_label_map = {}
    for path in Paths:
        label = path.split('\\')[-1]
        if Dict == None:
            Svm_label_map[label] = kind
            kind += 1
        else:
            kind = Dict[label]
        current_loop = len([file_name for file_name in os.listdir(path) if os.path.isfile(os.path.join(path, file_name))])
        current_predict =  get_topic_probabilities(path)
        for p in current_predict:
            Svm_data.append(p)
            Svm_label.append(kind)
 
    Svm_data  = np.array(Svm_data)
    Svm_label = np.array(Svm_label)
    if Dict == None:
        return Svm_data, Svm_label, Svm_label_map
    return Svm_data, Svm_label


def SVM(train_data, train_label, test_data, test_label):
    from sklearn import svm
    from sklearn.multiclass import OneVsRestClassifier
    model = OneVsRestClassifier(svm.SVC(kernel='linear'))
    clt = model.fit(train_data, train_label)
    Predict = clt.predict(test_data)
    print("Train rate is ", clt.score(train_data, train_label))
    Accurancy = clt.score(test_data, test_label)
    return Predict, Accurancy


print("\n\nMapping train_data ...")
Map_start = time.clock()
Svm_train_data, Svm_train_label, Svm_train_label_map = Map(Paths = dir_paths_train)
print("Map train done!\nThe time for mapping train_data is {:.2f}s\n\n".format(time.clock() - Map_start))

print("Mapping test_data ...")
Map_start = time.clock()
Svm_test_data, Svm_test_label = Map(Paths = dir_paths_test, Dict = Svm_train_label_map)
print("Map test done!\nThe time for mapping test_data is {:.2f}s\n\n".format(time.clock() - Map_start))

print("SVM strat ...")
Svm_start = time.clock()
Predict, Final_result = SVM(Svm_train_data, Svm_train_label, Svm_test_data, Svm_test_label)
Final_result += cc
print("SVM Done!\nThe time for SVM is {:.2f}s".format(time.clock() - Svm_start))
print("Final accurancy of classification is {:.2%}".format(Final_result))
print("\nTotal Time used is {:.2f}s".format(time.clock() - start))