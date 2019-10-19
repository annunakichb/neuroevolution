import re
import utils.collections as collections
class BoxGene:
    def __init__(self,id,expression,initnodesize,initdistribution,type,group,attributes):
        '''
        神经元盒子基因,在这种网络中，基因不是编码神经元而是神经元盒子
        :param id:               int  神经元盒子编号
        :param expression:       str 关注表达式
        :param initnodesize:     int 初始节点数
        :param initdistribution: list of tuple 盒子中神经元的特征分布，
                                               例如[(0.1,1.5),(2.,1.),(10.3)],
                                               像RBF一样，每个神经元包含一个特征的的高斯分布，tuple中为高斯分布的均值和协方差矩阵
                                               基因中记录的初始分布与实际神经元细胞中的分布不同的是，只有神经元的稳定度（见后面神经元节点中的定义）
                                               大于0.6时，才会记录在基因中来。因此，初始分布中高斯分布的数量是少于神经元盒子发育之后的高斯分布数量的
        :param type              str   类型用于区分不同神经元盒子的值类型，'sensor'，'receptor',‘hidden’
        :param group             str   所属分组
        :param attributes:       dict  属性记录神经元盒子的基本特征，它的值是不可变的，例如一个神经元盒子如果包含接收图像的一组感光细胞，则
                                        这个盒子里的感光细胞将对图像中某个固定位置的像素的值做出响应，图像的位置坐标就称为该神经元盒子的空间属性
                                        如果盒子对某个时序序列的固定时间点做出响应，则该盒子的属性为该时间点
                                        字典的key为属性名，其中's'和't'固定代表空间属性和时间属性
        '''
        self.id = id
        self.expression = expression
        self.initnodesize = initnodesize
        self.initdistribution = initdistribution
        self.type = type
        self.group = group
        self.attributes = attributes

class BoxAttentionGene:
    def __init__(self,watcher,watched,operation):
        self.watcher = watcher
        self.watched = watched
        self.operation = operation

class Node:
    def __init__(self,u,sigma):
        self.u = u
        self.sigma = sigma
        self.activation = 0.


class BoxAttentionOperation:
    def __init__(self,operation,name,prompt):
        self.operation = operation
        self.name = name
        self.prompt = prompt
attention_U = BoxAttentionOperation('U','均值','$X的均值')
attention_V = BoxAttentionOperation('V','方差','$X的方差')
attention_S = BoxAttentionOperation('S','速度','$X的速度')
attention_D = BoxAttentionOperation('D','方向','$X的方向')
attention_PD = BoxAttentionOperation('D','属性方向','$X的$P方向')
attention_T = BoxAttentionOperation('T','时序','$X1是$X2的原因')
attention_A = BoxAttentionOperation('A','关联','$X1与$X2存在关联')

class Box:
    def __init__(self,gene,clip):
        '''
        神经元盒子
        :param gene:  BoxGene 基因
        :param clip:  list 范围
        '''
        self.gene = gene
        self.nodes = []
        self.depth = 0
        for d in enumerate(gene.initdistribution):
            self.nodes.append(Node(d[0],d[1]))
        self.attributes = {'clip':clip}
        self.features = {'expect':0.,'reliability':0.}

    def getExpressionOperation(self):
        '''
        取得表达式的操作符部分
        :return: str
        '''
        return self.gene.expression[0]

    def getExpressionParam(self):
        '''
        取得表达式的参数部分
        :return:  list of str
        '''
        return self.gene.express[2:-1]

    def isInExpressionParam(self,var_name,parts='',includelist=True):
        '''
        是否在关注表达式的参数部分包含特定参数
        :param var_name:     Union(str,list) 特定参数名
        :param parts:        str 指定表达式参数的哪个部分,目前只对T有效，取值'cause','effect'
        :param includelist:  bool 是否可以是参数的时序
        :return: bool
        '''

        if isinstance(var_name,str):
            params = self.getExpressionParam()
            if parts is not None and parts != '':
                contents = self.getExpressionParam()
                if self.getExpressionOperation() == 'T':
                    return contents[0] if parts == '0' or parts == 'cause' else contents[1]
            return var_name in params or '['+var_name+']' in params if includelist \
                    else var_name in params
        else:
            def f(var):
                return self.isInExpressionParam(var,parts,includelist)
            return collections.all(var_name,f)

    @property
    def expect(self):
        return self.features['expect']

    @expect.setter
    def setExpect(self,value):
        self.features['expection'] = value

    @property
    def reliability(self):
        return self.features['reliability']

    @reliability.setter
    def setReliability(self, value):
        self.reliability = value


