from utils.properties import NameInfo
import utils.collections as collections

class BoxModel:
    nameInfo = NameInfo('box', cataory='box')
    __initStates = {}
    __variables = []

    def __init__(self, **configuration):
        '''
        普通输入模型
        :param configuration: 模型缺省配置
        '''
        self.nameInfo = BoxModel.nameInfo
        self.configuration = configuration if not collections.isEmpty(configuration) else {}
        self.initStates = BoxModel.__initStates
        self.variables = BoxModel.__variables

    def execute(self, box, net, **context):
        feature = box.doAttention(net,context)

