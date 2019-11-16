import numpy as np


def kl_divengence(p1,p2,p1weights=None,p2weights=None):
    '''
    KL散度计算
    :param p1:        Union(list,array)  分布1的样本采样概率
    :param p2:        Union(list,array)  分布2的样本采样概率
    :param p1weights: Union(list,array)  分布1样本的权重，其长度应与p1相同
    :param p2weights: Union(list,array)  分布2样本的权重，其长度应与p2相同
    :return:
    '''
    length = len(p1)

