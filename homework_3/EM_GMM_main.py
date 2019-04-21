'''
    By enginning
    For buaa nlp homework_3
    内容: 生成G MM模型，然后用 EM算法迭代求出 GMM模型的参数
    掌握：隐变量、隐含模型、EM算法、GMM模型、二维正态分布、协方差矩阵
    Ref: http://sofasofa.io/tutorials/gmm_em/
    
    其中，均值表征各维变量的中心，协方差矩阵(对称阵)正对角为各维变量的方差，斜对角为各维变量间的协方差（相关性)
    GMM算法的两个前提: 1. 数据服从高斯分布; 2. 已知分类数 k
    当然，如果分类数未知的话，则需要先进行 k-means，给出相应的初值，再用 EM算法进行求解
    https://github.com/enginning/nlp/

    核心是 EM算法，同时也加入可视化的模块，其中，椭圆的长、短轴采用经验公式: long_axis = 3 * sigma_big, 
        short_axis = 3 * sigma_small * rho，
        椭圆长轴的夹角为 arctan(rho * (sigma_2 / sigma_1))，可能和理论推导出来的有一定的差别

'''

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from scipy.stats import multivariate_normal
from scipy.interpolate import spline
from itertools import chain
plt.style.use('seaborn')


# 生成数据
def generate_X(true_Mu, true_Var):
    # 生成一个多元正态分布矩阵
    # 第一簇的数据
    # 隐变量概率
    N = 1000
    ratio = [0.6, 0.4]
    Node = [(int)(x * N) for x in ratio]
    num1, mu1, var1 = Node[0], true_Mu[0], true_Var[0]
    X1 = np.random.multivariate_normal(mu1, var1, num1)
    # 第二簇的数据
    num2, mu2, var2 = Node[1], true_Mu[1], true_Var[1]
    X2 = np.random.multivariate_normal(mu2, var2, num2)

    # 合并在一起(垂直按行堆叠)
    X = np.vstack((X1, X2))
    # 显示数据
    plt.figure(figsize=(10, 8))
    plt.axis([-10, 15, -10, 15])
    plt.scatter(X1[:, 0], X1[:, 1], s=5)
    plt.scatter(X2[:, 0], X2[:, 1], s=5)
    plt.savefig("generate.svg")
    plt.show()
    return X


# 更新W
def update_W(X, Mu, Var, Pi):
    n_points, n_clusters = len(X), len(Pi)
    pdfs = np.zeros(((n_points, n_clusters)))
    for i in range(n_clusters):
        pdfs[:, i] = Pi[i] * multivariate_normal.pdf(X, Mu[i], Var[i])
    W = pdfs / pdfs.sum(axis=1).reshape(-1, 1)
    # print(W, type(W), W.shape)
    return W


# 更新pi
def update_Pi(W):
    Pi = W.sum(axis=0) / W.sum()
    return Pi


# 计算log似然函数(目标函数极大化)
def logLH(X, Pi, Mu, Var):
    n_points, n_clusters = len(X), len(Pi)
    pdfs = np.zeros(((n_points, n_clusters)))
    for i in range(n_clusters):
        pdfs[:, i] = Pi[i] * multivariate_normal.pdf(X, Mu[i], Var[i])
    return np.mean(np.log(pdfs.sum(axis=1)))


# 画出聚类图像
def plot_clusters(X, Mu, Var, Mu_true=None, Var_true=None, picture_name=None):
    colors = ['b', 'g']
    n_clusters = len(Mu)
    plt.figure(figsize=(10, 8))
    plt.axis([-10, 15, -10, 15])
    plt.scatter(X[:, 0], X[:, 1], s=5)
    ax = plt.gca()
    angle_list, rho_list = ellipse_angle(n_clusters, true_Var)
    angles = ellipse_angle(n_clusters, Var)
    
    for i in range(n_clusters):
        plot_args = {'fc': 'None', 'lw': 2, 'edgecolor': colors[i], 'ls': ':'}
        axis = find_axis(Var[i])
        if rho_list[i] > 0.7:
            axis[1] *= rho_list[i]
        ellipse = Ellipse(Mu[i], 3 * axis[0], 3 * axis[1], **plot_args, angle=angles[0][i])
        ax.add_patch(ellipse)

    if (Mu_true is not None) & (Var_true is not None):
        for i in range(n_clusters):
            axis = find_axis(Var_true[i])
            if rho_list[i] > 0.7:
                axis[1] *= rho_list[i]
            plot_args = {'fc': 'None', 'lw': 2,
                         'edgecolor': colors[i], 'alpha': 0.7}
            ellipse = Ellipse(Mu_true[i], 3 * axis[0], 3 * axis[1], **plot_args, angle=angle_list[i])
            ax.add_patch(ellipse)
    if picture_name is not None:
        plt.savefig("{}.svg".format(picture_name))
    plt.show()


# 计算椭圆长轴的角度
def ellipse_angle(n_clusters, Var):
    angle_list = []
    rho_list = []
    for i in range(n_clusters):
        a1 = Var[i].tolist()[0][0]
        a2 = Var[i].tolist()[0][1]
        a3 = Var[i].tolist()[1][1]
        rho = a2 / (np.sqrt(a1 * a3))
        angle = np.arctan2(a3, a1)
        angle = 180 * rho * angle / np.pi
        angle_list.append(angle)
        rho_list.append(rho)
    return angle_list, rho_list


# 方差比较，找出长、短轴，以方便画椭圆
def find_axis(Var):
    tmp_1 = Var.tolist()[0][0]
    tmp_2 = Var.tolist()[1][1]
    axis = [tmp_1, tmp_2]
    if axis[0] < axis[1]:
        axis[0] = axis[0] + axis[1]
        axis[1] = axis[0] - axis[1]
        axis[0] = axis[0] - axis[1]
    return axis

# 误差曲线
def plot_error(loglh):
    plt.figure(figsize=(10, 8))
    # plt.plot([x for x in range(len(loglh))], loglh, marker='o', linestyle='solid')
    R = np.array([x for x in range(len(loglh))])
    F = np.array(loglh)
    new = np.linspace(R.min(), R.max(), 300)
    Smooth = spline(R, F, new)
    plt.plot(new, Smooth)
    plt.xlabel("Round number")
    plt.ylabel("log-likelihood function")
    plt.title("EM for GMM")
    plt.savefig("log_error_curve.svg")
    plt.show()

# 更新Mu
def update_Mu(X, W):
    n_clusters = W.shape[1]
    Mu = np.zeros((n_clusters, 2))
    for i in range(n_clusters):
        Mu[i] = np.average(X, axis=0, weights=W[:, i])
    return Mu


# 更新Var
def update_Var(X, Mu, W, Pi):
    n_clusters = W.shape[1]
    Var = []
    for i in range(n_clusters):
        a1 = np.average((X[:, 0] - Mu[i][0]) ** 2, axis=0, weights=W[:, i])
        a3 = np.average((X[:, 1] - Mu[i][1]) ** 2, axis=0, weights=W[:, i])
        a2 = np.dot(W[:, i] * (X[:, 0] - Mu[i][0]), (X[:, 1] - Mu[i][1])) / (len(X) * Pi[i])
        # a2 = np.dot(W[:, i] * (X[:, 0] - Mu[i][0]), (X[:, 1] - Mu[i][1])) / np.sum(W[:, i])
        # np.sum(W[:, i]) = len(X) * Pi[i]
        Var.append(np.mat([[a1, a2], [a2, a3]]))
    return Var


# 打印最后的输出结果
def final_result(Pi, true_Mu, true_Var, Mu, Var):
    print("The final result".center(100, '-'))
    np.set_printoptions(precision=3)
    print("\nThe proportion of samples is: ", Pi)
    print("\nThe True_Mu is ", true_Mu)
    print("The Pridict Mu is ", Mu.reshape(1, -1))
    
    tmp_1 = list(chain(*true_Mu))
    tmp_2 = list(chain(*(Mu.tolist())))
    Cal_diff(tmp_1, tmp_2, flag="Mu")

    print("\n\nThe True_Var is \n matrix_1: {} \n matrix_2: {}"
          .format(true_Var[0].reshape(1, -1), true_Var[1].reshape(1, -1)))
    print("The Predict Var is \n matrix_1: {} \n matrix_2: {}"
          .format(Var[0].reshape(1, -1), Var[1].reshape(1, -1)))

    for i in range(len(Var)):
        tmp_1 = list(chain(*true_Var[i].tolist()))
        tmp_2 = list(chain(*(Var[i].tolist())))
        Cal_diff(tmp_1, tmp_2, flag="Var")


def Cal_diff(list_1, list_2, flag):
    if flag == "Mu":
        print("\nThe accurancy for predicting Mu is (ordered by row) ")
    else:
        print("\nThe accurancy for predicting Var is (ordered by row) ")
    for i in range(len(list_1)):
        if (list_1[i] != 0): 
            result = 1 - (abs(list_1[i] - list_2[i]) / abs(list_1[i]))
            print("%.3f " % result, end='')
        else:
            print("%s " % "inf", end='')


if __name__ == '__main__':
    # 生成数据
    true_Mu = [[0.5, 0.5], [1, 7]]
    # true_Var = [np.mat("1, 0; 0, 3"), np.mat("6, 0; 0, 2")]
    true_Var = [np.mat("3, 4; 4, 7"), np.mat("6, 3; 3, 2")]
    # true_Var = [[1, 3], [6, 2]]
    X = generate_X(true_Mu, true_Var)

    '''
    初始化(预测)
    n_clusters 聚类数
    n_points   样本点数
    Mu  每个高斯分布的均值
    Var 每个高斯分布的方差
    W   隐变量，每个样本属于每一簇的概率，初始均分可能性
    Pi  每簇的比重
    '''

    n_clusters = 2
    n_points = len(X)
    Mu = [[0, 0], [3, 9]]
    Var = [np.mat("1, 0; 0, 1"), np.mat("1, 0; 0, 1")]
    Pi = [1 / n_clusters] * 2
    W = np.ones((n_points, n_clusters)) / n_clusters
    Pi = W.sum(axis=0) / W.sum()
    
    # 迭代
    loglh = []
    i = 0
    print('\n')
    while((i < 3) or ((loglh[-1] - (loglh[-2] + loglh[-3]) / 2) > 0.001)):
        # 下面一行取消注释的话，可以进行 EM迭代求解的可视化
        # plot_clusters(X, Mu, Var, true_Mu, true_Var)
        loglh.append(logLH(X, Pi, Mu, Var))
        W = update_W(X, Mu, Var, Pi)
        Pi = update_Pi(W)
        Mu = update_Mu(X, W)
        print("{0:4}: log-likehood: {1:.4f}".format(i, loglh[-1]))
        Var = update_Var(X, Mu, W, Pi)
        i += 1
    
    # 收工
    plot_clusters(X, Mu, Var, true_Mu, true_Var, picture_name='EM_GMM_fit')
    plot_error(loglh)
    final_result(Pi, true_Mu, true_Var, Mu, Var)
