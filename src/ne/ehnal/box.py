

class BoxGene:
    def __init__(self,id,inputboxs,outputboxs,initdistribution,type,attributes):
        '''
        神经元盒子基因,在这种网络中，基因不是编码神经元而是神经元盒子
        :param id:               int  神经元盒子编号
        :param inputboxs:        list of int 输入盒子的编号集
        :param outputboxs:       list of int 输出盒子的编号集
        :param initdistribution: list of tuple 盒子中神经元的特征分布，
                                               例如[(0.1,1.5),(2.,1.),(10.3)],
                                               像RBF一样，每个神经元包含一个特征的的高斯分布，tuple中为高斯分布的均值和协方差矩阵
                                               基因中记录的初始分布与实际神经元细胞中的分布不同的是，只有神经元的稳定度（见后面神经元节点中的定义）
                                               大于0.6时，才会记录在基因中来。因此，初始分布中高斯分布的数量是少于神经元盒子发育之后的高斯分布数量的
        :param type              str   类型用于区分不同神经元盒子的值类型，type不可变，例如感知自身力气的神经元盒子与感知自身坐标的盒子，类型不同
                                       类型用于神经元进化的变异过程，同类型的盒子之间优先建立连接
        :param attributes:       dict  属性记录神经元盒子的基本特征，它的值是不可变的，例如一个神经元盒子如果包含接收图像的一组感光细胞，则
                                        这个盒子里的感光细胞将对图像中某个固定位置的像素的值做出响应，图像的位置坐标就称为该神经元盒子的空间属性
                                        如果盒子对某个时序序列的固定时间点做出响应，则该盒子的属性为该时间点
                                        字典的key为属性名，其中's'和't'固定代表空间属性和时间属性
        '''
        self.id = id
        self.inputs = inputboxs
        self.outputs = outputboxs
        self.initdistribution = initdistribution
        self.attributes = attributes

class Box:
    def __init__(self,gene):
        self.gene = gene
        self.elements = []