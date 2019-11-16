import re
import numpy as np
import utils.collections as collections
from utils.properties import PropertyInfo
from brain.elements import Neuron
import math
import brain.networks as networks
import utils.collections as collections
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
    max_history_memory_length = 5
    def __init__(self,id,layer,boxid,u,sigma,birth,modelConfiguration,coord=None):
        '''
        特征神经元，每个神经元表达一个特征值
        :param id:                  int  id
        :param layer:               int  所属层
        :param boxid:               int  所属盒子id
        :param u:                   list 中心特征值
        :param sigma:               list of list 方差
        :param birth:               int  出生年代
        :param modelConfiguration:  dict 模型配置
        :param coord:               list 坐标
        '''
        Neuron.__init__(self,id,layer,birth,modelConfiguration,coord)
        self.state['activation'] = 0.
        #self.state['clock'] = 0
        self.features = {}
        self.features['liveness'] = 0.
        self.variables['u'] = u
        self.variables['sigma'] = sigma
        self.boxid = boxid
        self.history_states = []
    def activation(self,intensity,clock):
        if 'clock' in self.states:
            self.history_states.append((self.state['activation'],self.state['clock']))
        while len(self.history_states)>FeatureNeuron.max_history_memory_length:
            self.history_states.pop(0)

        self.state['activation'] = intensity
        self.state['clock'] = clock
    def reset(self):
        self.state['activation'] = 0.
        self.states.pop('clock')


class Box:

    #region 初始化
    max_activation_feature_history = 5
    def __init__(self,gene,net):
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
        self.net = net
        self.grids =[]
        self.energy_list = []
        self.activation_feature_history = []
        self.attributes = []

    def __str__(self):
        return self.gene.expression

    #endreigon

    #region 输入和输出盒子管理

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

    #endregion

    #region 关注表达式管理
    def getExpressionOperation(self):
        '''
        取得表达式的操作符部分
        :return: str
        '''
        index = self.gene.expression.indexOf('(')
        return self.gene.expression[:index]

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
    #endregion

    #region 期望(目标)特征管理
    @property
    def expection(self):
        return self.features['expection']

    @expection.setter
    def expection(self,value):
        self.features['expection'] = value

    def random(self):
        return  np.random.rand()*self.gene.clip[0]+(self.gene.clip[1]-self.gene.clip[0])

    #endregion

    #region 可靠度和稳定度管理
    @property
    def reliability(self):
        return self.features['reliability']

    @reliability.setter
    def setReliability(self, value):
        self.reliability = value

    #endregion



    #region 值与节点的距离、激活强度计算

    def mahalanobia_distance(self,x,mu,sigma):
        '''
        某点到正态分布的马氏距离
        :param x:     list   点
        :param mu:    list   正态分布均值
        :param sigma: array  正态分布协方差矩阵
        :return:
        '''
        size = len(x)
        det = np.linalg.det(sigma)
        if det == 0:
            raise NameError("The covariance matrix can't be singular")
        x_mu = np.array(x - mu)
        inv = sigma.I
        result = math.sqrt(x_mu * inv * x_mu.T)
        return result

    def norm_pdf_multivariate(x, mu, sigma):
        '''
        计算x的多元高斯分布函数
        ：param x:
        :param mu:
        :param sigma:
        :return:
        '''
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

    @property
    def activation_max(self):
        '''
        计算每个节点的最大激活值
        :return: list 每个节点的最大激活值
        '''
        return [self.norm_pdf_multivariate(node.u,node.u,node.sigma) for node in self.nodes]

    def activation_feature(self):
        '''
        计算激活特征值
        :return: list
        '''
        features = [node.u * node.states['activation'] if 'clock' in node.states and node.states['clock']==self.net.clock else 0. for node in self.nodes]
        return np.sum(features)

    def history_activation_features(self):
        return self.activation_feature_history

    def activation_intensity(self, values):
        '''
        计算输入值对每个节点的激活强度
        :param values: list 激活强度
        :return:
        '''
        intensity = [self.norm_pdf_multivariate(values,node.u,node.sigma) for node in self.nodes]
        activation_range = [self.norm_pdf_multivariate(node.u,node.u,node.sigma) for node in self.nodes]
        activation_range = [np.min(activation_range),np.max(activation_range)]
        intensity = (intensity-activation_range[0])/(activation_range[1]-activation_range[0])
        return intensity

    def get_max_activation_intersity(self,values):
        '''
        取得最大激活强度
        :param values:  样本值
        :return: tuple  最大激活强度，对应节点序号
        '''
        intensity = [self.norm_pdf_multivariate(values, node.u, node.sigma) for node in self.nodes]
        intensity = intensity / self.activation_range
        index = np.argsort(intensity)
        return intensity[index[-1]],index[-1]

    #endregion

    #region 核心算法，完成关注操作
    def do_attention(self):
        operation = self.getExpressionOperation()
        return operation.do_attention(self)

    def activate(self,values):
        '''
        :param values list 新输入的值
        重新分布中心算法满足以下原则：
        合理的能耗范围和尽可能大的识别率（这两个约束本身是选择的结果，猜测最终会导致激活节点的稀疏性特性产生）
        一个神经元盒子的瞬时能耗(energy)定义为其节点个数*节点基准能耗 + sum(所有节点的瞬时激活强度)*exp(激活的节点数量)
        exp(激活节点数量)是使得激活节点数量越少越好，最好只有一个
        激活节点是指激活强度大于强度阈值的节点，激活强度的计算
        总之，节点个数越少，激活的节点数量越少，能耗越低，能耗最低原则最好是就一个节点来表达所有分布
        一个神经元盒子的瞬时识别率(recognition_rate)是对所有节点按照激活强度从高到底排序，第一个记为m1，第二个记为mk
        假设其中前q个的激活强度是大于0的
        瞬时识别率的计算为：sum(mi+1到mk与mi的距离绝对值之和）除以q，其中i从1到q
        这样，q越大识别率越差，即每个输入最好只有一个激活，
             激活频率差别越大越好
        节点分布调整算法的目标是：在满足energy在一定范围内的约束下，最大化识别率的均值
        能量范围是自适应的，如果获得的正向奖励多，可用能量的上限就提升，能量的下限总是0
        可以采用基于黑盒优化的自然进化策略方法，也可以采用基于在线密度和网格混合聚类的方法
        方法参数：
        1.距离上限（距离节点中心的马氏距离上限，若超过该距离，认为不在该节点内）
        2.溢出数量（大于该数量意味着需要对节点进行分裂）
        3.调整数量（某个网格内的值数量上限，大于该值将对节点的中心和方差进行调整）
        3.瞬时能耗范围[0,max_engegy]
        算法描述：
        输入：当前节点分布[(u1,sigma1),(u2,sigma2),...,(uk,sigmak)]、新输入点values，上次瞬时能耗=0
        输出：新的节点分布
        总过程：1）计算新输入点相对于各个节点激活强度
             2）若激活强度最大值小于上限（即没有节点被激活），则以该节点为中心生成新节点，执行调整操作
             3）若最近节点中的容纳数量超过溢出数量，以eplison概率执行分裂操作，以1-eplison执行融合操作
                eplison=（1-上次瞬时能耗）/max_engegy
             4）若最近节点中的某个网格的容纳数量超过调整数量，则对该节点的均值和协方差进行调整
             5) 重新计算所有节点分
        :return:
        '''
        #计算值所在网格
        activation_intensity_list = self.activation_intensity(values)
        grid,no,center,max_intersity,node_index = self._put_values_in_grid(values,activation_intensity_list)

        idGenerator = networks.idGenerators.find(self.net.genome.definition.idGenerator)
        # 如果没有实际激活的节点，则创建一个新节点
        if max_intersity < self.net.definition.box.activation_threadshold:
            # 创建节点，没有做连接是以为初节点与上层是全连接
            node_sigma = self.net.definition.box.init_lambda*np.identity(len(values))
            node = FeatureNeuron(idGenerator.getNeuronId(),self.depth,self.id,values,node_sigma,None)
            self.nodes.append(node)
        else:
            node = self.nodes[node_index]
            # 如果激活节点的样本容量大于分裂阈值，以eplison概率进行分裂，以1-eplison概率进行融合
            if node.cur_capcity_count >= self.net.definition.box.overflow_count:
                eplison =(self.net.definition.box.max_engegy - self.energy_list[-1]) / self.net.definition.box.max_engegy
                if np.random()<= eplison:
                    self._node_spilt(grid,node,values)
                elif np.random() <= 1-eplison:
                    self._node_merge(grid,node,values)
            # 如果激活节点的样本容量大于调整阈值，执行节点调整
            elif node.cur_capcity_count >= self.net.definition.box.adjust_count:
                self._node_adjust(grid,node,values)
        self._redistribution()

        # 计算各个节点激活强度
        activation_intensity_list = self.activation_intensity(values)
        for index,activation_intensity in enumerate(activation_intensity_list):
            self.nodes[index].activation(activation_intensity,self.net.clock)

        activation_feature = self.activation_feature()
        self.activation_feature_history.append((activation_feature,self.net.clock))
        while len(self.activation_feature_history)>Box.activation_feature_history:
            self.activation_feature_history.pop(0)

        self.compute_energy()

    def compute_energy(self):
        activation_nodes = [node.states['activation'] for node in self.nodes if node.states['activation']>0]
        # 节点个数*节点基准能耗 + sum(所有节点的瞬时激活强度)*exp(激活的节点数量)
        energy = len(self.nodes)*self.net.definition.box.benchmark_energy +\
                 np.sum(activation_nodes)*np.exp(len(activation_nodes))
        self.energy_list.append((energy,self.net.clock))

    def _node_adjust(self,grid,node,values):
        '''
        对节点进行均值和协方差调整
        :param grid:   tuple 值所属网格
        :param node:   FeatureNeuron 值所属节点
        :param values: list 值
        :return:
        '''
        grids = self._get_grids_by_node(node)
        u,sigma = self._compute_multivariant_pdf_by_grids(grids)
        node.u,node.sigma = u,sigma
    def _node_merge(self,grid,node,values):
        '''
        合并操作
        :param grid:    新加入值所属网格
        :param node:    新加入值所属节点
        :param values:  新加入值
        :return: None
        '''
        if len(self.nodes)<=2:return
        # 计算所有节点之间的KL距离最小值，找到距离最近的两个节点
        dis = np.zeros(shape=(len(self.nodes),len(self.nodes)))
        for i in range(dis.shape[0]):
            for j in range(dis.shape[1]):
                if i == j:continue
                dis[i][j] = self._get_node_distance(self.nodes[0],self.nodes[1])
        n1,n2 = np.argmin(dis)
        grids = self._get_grids_by_node(self.nodes[n1])
        grids.extend(self._get_grids_by_node(self.nodes[n2]))
        u,sigma = self._compute_multivariant_pdf_by_grids(grids)
        idGenerator = networks.idGenerators.find(self.net.genome.definition.idGenerator)
        node = FeatureNeuron(idGenerator.getNeuronId(),self.depth,self.id,u,sigma,0,None)

    def _node_spilt(self,grid,node,values):
        '''
        节点分裂
        :param grid:   tuple 值所属网格
        :param node:   FeatureNeuron 值所属节点
        :param values: list 值
        :return: tuple (FeatureNeuron,[FeatureNeuron]) 原节点，新节点列表
        '''
        # 所有属于node的网格
        grids = [grid for grid in self.grids if self.activation_intensity(grid[1])>=self.net.definition.box.activation_threadshold]
        if len(grids)<=1:return node,[]
        # 计算所有网格的邻域密度，进行改进的密度聚类
        ## 所有网格按照密度进行排序
        grid_asc_index = np.argsort(grids,order='samples_count')
        ## 对所有网格进行密度聚类
        grids_classes = {}
        for i in len(grids):
            grid_index = np.argsort([grid['grid_xh'] - np.sum(np.abs(grids[i]['grid_xh']-grid['grid_xh']))  for grid in grids])
            grid_index = grid_index[-1]
            if grid_index not in grids_classes:
                grids_classes[grid_index] = (0,[])
            grids_classes[grid_index][0] += 1
            grids_classes[grid_index][1].append(grids[i])
        # 根据聚类结果进行节点分裂
        if len(grids_classes)<=1:return node,[]
        idGenerator = networks.idGenerators.find(self.net.genome.definition.idGenerator)
        r = []
        for index,grid_class in grids_classes.items():
            x = np.array([x['center'] for x in grid_class[1]]).T
            fweights = np.array([x['samples_count'] for x in grid_class[1]])
            sigma = np.cov(x, fweights=fweights)
            n = FeatureNeuron(idGenerator.getNeuronId(),self.depth,self.id,self.grids[index],sigma,None)
            self.nodes.append(node)
            r.append(node)
        self.nodes.remove(node)
        return node,r


    def _get_grid_by_value(self,values,activation_intensity_list,createifnotexisted=True):
        '''
        计算值在特征空间所属的网格
        :param values:              Union(list,array)   值
        :param activation_intensity_list: list values对所有节点的激活强度列表
        :param createifnotexisted:  Bool 若网格不存在则创建一个
        :return: tuple grid元组,
                  list grid各维的序号
                  list grid中心点值
                  float values最大激活强度
                  int   values最大激活强度对应的节点序号
        '''
        dimension = len(values)
        no, center = [], []
        for i in range(dimension):
            clip = self.gene.clip[i] if len(self.gene.clip) > i else [0., 1.]
            grid_width = (clip[1] - clip[0]) / self.net.definition.box.grid_size
            index = int((values[i] - clip[0]) / grid_width)
            no.append(index)
            center.append((index * grid_width + (index + 1) * grid_width) / 2.0)
        grid = collections.first(self.grids, lambda x: np.equal(x['grid_xh'], no))
        if grid is None and createifnotexisted:
            node_index = -1
            if len(activation_intensity_list)>0:
                asc_index = np.argsort(activation_intensity_list)
                max_intersity, node_index = activation_intensity_list[asc_index[-1]],asc_index[-1]
            grid = {'grid_xh': no, 'grid_center': center, 'samples_count': 0, 'node_id':self.nodes[node_index].id}
            self.grids.append(grid)
        grid['samples_count'] += 1
        return grid,no,center,max_intersity,node_index

    def _put_values_in_grid(self,values):
        '''
        将值放入所属网格
        :param values:  样本值
        :return:   tuple  网格元祖，最大激活强度，最大激活强度对应的节点序号
        '''
        return self._get_grid_by_value(values)

    def _get_grids_by_node(self,node):
        '''
        在grids成员中寻找所有属于node的网格，这里只是寻找，并不重新计算
        :param node:  FeatureNeuron
        :return: list of grid
        '''
        return [grid for grid in self.grids if grid['node_id'] == node.id]

    def _compute_multivariant_pdf_by_grids(self,grids):
        '''
        根据网格信息计算节点均值和方差
        :param grids:  list 属于某个节点的网格
        :return:  tuple 均值，方差
        '''
        u = np.sum([x.center * x.samples_count for x in grids]) / np.sum(
            [x.samples_count for x in grids])
        x = np.array([x.center for x in grids]).T
        fweights = np.array([x.samples_count for x in grids])
        sigma = np.cov(x, fweights=fweights)
        return u,sigma

    def _redistribution(self):
        '''
        根据网格情况对节点的方差和均值进行调整,包括两步：
        1.根据网格中心点值判断其所属节点
        2.根据节点中网格重新计算节点的中心和协方差
        :return: None
        '''
        # 判断每一个网格所在节点
        node_grid_records = []*len(self.nodes)
        for grid in self.grids:
            # 当前网格信息
            grid_xh,grid_center,samples_count,node_id = grid['grid_xh'],grid['grid_center'],grid['samples_count'],grid['node_id']
            # 以当前网格中心对应每个节点的激活值，最大激活值节点索引，最大激活值节点
            intensity = self.activation_intensity(grid_center)
            max_intensity_index = np.argmax(intensity)
            max_intensity = intensity[max_intensity_index]
            node = self.nodes[max_intensity_index]
            if max_intensity < self.net.definition.box.activation_threadshold:# 不属于任何节点
                grid['node_id'] = None
                continue
            grid['node_id'] = node.id
            node_grid_records[max_intensity_index].append(grid)
        # 对每一个节点重新计算分布
        for index,node_grid_record in enumerate(node_grid_records):
            u,sigma = self._compute_multivariant_pdf_by_grids(node_grid_record)
            self.nodes[index].u = u
            self.nodes[index].sigma = sigma
            self.nodes[index].cur_capcity_count = np.sum([samples_count for x in node_grid_record])









#endregion