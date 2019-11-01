import re
import numpy as np
import utils.collections as collections
from utils.properties import PropertyInfo
from brain.elements import Neuron
import math

#region Gene
class BoxGene:
    '''
    盒子基因
    '''
    type_sensor = 'sensor'         # 感知器类型
    type_receptor = 'effector'     # 效应器类型
    type_attention = 'attention'   # 关注类型
    type_action = 'action'         # 动作类型

    def __init__(self, id,type,**kwargs):
        self.id = id
        self.expression = kwargs['expression']
        self.initsize = kwargs['initsize']
        self.initdistribution = [] if 'initdistribution' not in kwargs else kwargs['initdistribution']
        self.type = type
        self.group = kwargs['group']
        self.clip = kwargs['clip']
        self.caption = kwargs['caption']
        self.attributes = {} if 'attributes' not in kwargs else kwargs['attributes']
        self.dimension = 1 if 'dimension' not in kwargs else kwargs['dimension']

    def __str__(self):
        return self.caption

class BoxAttentionGene:
    def __init__(self,watcherid,watchedids,operation):
        '''
        关注基因
        :param watcher:    int 关注者id
        :param watched:    list of int  被关注者id
        :param operation:  str          操作名
        '''
        self.watcherid = watcherid
        self.watchedids = watchedids
        self.operation = operation
    @property
    def id(self):
        return self.watcherid+"_"+ ".".join(self.watchedids)


class BoxActionConnectionGene:
    def __init__(self,action_box_id,attention_box_ids,receptor_box_id):
        '''
        输出连接基因，为权重连接网络
        :param action_box_id:
        :param activation_box_ids:
        '''
        self.action_box_id  = action_box_id
        self.attention_box_ids = attention_box_ids
        self.receptor_box_id = receptor_box_id
    @property
    def id(self):
        return self.action_box_id + "_" + ".".join(self.attention_box_ids)



#endregion

#region element
class FeatureNeuron(Neuron):
    def __init__(self,id,layer,boxid,featureValue,sigma,birth,modelConfiguration,coord=None):
        '''
        特征神经元，每个神经元表达一个特征值
        :param id:                  int  id
        :param layer:               int  所属层
        :param boxid:               int  所属盒子id
        :param featureValue:        list 特征值
        :param sigma:               list of list 方差
        :param birth:               int  出生年代
        :param modelConfiguration:  dict 模型配置
        :param coord:               list 坐标
        '''
        Neuron.__init__(self,id,layer,birth,modelConfiguration,coord)
        self.state['activation'] = False
        self.state['liveness'] = 0.
        self.state['features'] = featureValue
        self.state['sigma'] = sigma
        self.state['time'] = 0
        self.boxid = boxid


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
attention_operations = [attention_U,attention_V,attention_S,attention_D,attention_PD,attention_T,attention_A]

class Box:
    def __init__(self,gene):
        '''
        神经元盒子
        :param gene:  BoxGene 基因
        '''
        self.id = gene.id
        self.gene = gene
        self.nodes = []
        self.depth = 0
        self.features = {'expection':0.,'reliability':0.,'stability':0.}
        self.inputs = []
        self.outputs = []

    def __str__(self):
        return self.gene.expression

    def put_input_boxes(self,inputs):
        self.inputs = []
        if inputs is None:return self.inputs
        elif isinstance(inputs,Box):self.inputs.append(inputs)
        elif isinstance(inputs,list):self.inputs.extend(inputs)
        return self.inputs
    def add_input_boxes(self,inputs):
        if inputs is None:return self.inputs
        elif isinstance(inputs,Box):self.inputs.append(inputs)
        elif isinstance(inputs,list):self.inputs.extend(inputs)
        return self.inputs
    def put_output_boxes(self,outputs):
        self.outputs = []
        if outputs is None:return self.outputs
        elif isinstance(outputs,Box):self.outputs.append(outputs)
        elif isinstance(outputs,list):self.outputs.extend(outputs)
        return self.outputs
    def add_output_boxes(self,outputs):
        if outputs is None:return self.outputs
        elif isinstance(outputs,Box):self.inputs.append(outputs)
        elif isinstance(outputs,list):self.inputs.extend(outputs)
        return self.outputs


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
    def expection(self):
        return self.features['expection']

    @expection.setter
    def expection(self,value):
        self.features['expection'] = value

    @property
    def reliability(self):
        return self.features['reliability']

    @reliability.setter
    def setReliability(self, value):
        self.reliability = value

    def random(self):
        return  np.random.rand()*self.gene.clip[0]+(self.gene.clip[1]-self.gene.clip[0])


    def norm_pdf_multivariate(x, mu, sigma):
        size = len(x)
        det = np.linalg.det(sigma)
        if det == 0:
            raise NameError("The covariance matrix can't be singular")
        norm_const = 1.0 / (math.pow((2 * math.pi), float(size) / 2) * math.pow(det, 1.0 / 2))
        #x_mu = np.matrix(x - mu)
        x_mu = np.array(x - mu)
        inv = sigma.I
        result = math.pow(math.e, -0.5 * (x_mu * inv * x_mu.T))
        return norm_const * result

    def redistribution(self):
        pass









#endregion