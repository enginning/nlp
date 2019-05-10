# nlp大作业
## homework_1
- 内容: 计算汉语词汇的平均熵
- 思路：jieba + 词典

## homework_2
- 内容：设计实现一个简单的spelling corrector
- 思路：贝叶斯 + 误差模型

## homework_3
- 内容：生成一个含有两个二维的正态分布(GMM模型)，用 EM算法迭代求解GMM模型的参数
- 思路：E(期望): 用来更新隐变量W(每个样本属于每一簇的概率); M(最大化)，用来更新 GMM中各高斯分布的参量μ, σ，迭代直至收敛； + 可视化

## homework_4
- 内容：利用LDA主题模型生成特征空间，结合SVM实现文本分类
- 思路：数据预处理(分词，词形还原)，生成词典，接着转化成词袋模型对应的稀疏向量，调用gensim训练LDA模型，之后在生成的特征空间内，利用SVM实现分本分类
- 数据集：20-Newsgroups，link：http://qwone.com/~jason/20Newsgroups/
