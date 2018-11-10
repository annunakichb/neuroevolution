from enum import Enum
import json
from .properties import *
from .properties import Registry
from utils import collections

__all__ = ['FORMAT','isVaild','vaild','equals','getName','format','ExtendJsonEncoder']


#region 性质判定

#是否有效字符串
def isVaild(str):
    '''
    验证字符串是否有效
    :param str: 字符串，None，''和只有空格的字符串都是无效字符串
    :return:
    '''
    if str is None:return False
    if str == '':return False
    if str.strip() == '':return False
    return True

# 字符串有效性处理
def vaild(str,default=''):
    '''对字符串做验证，若字符串无效，则返回default，否则返回去前后空格的str'''
    if not isVaild(str):return default
    return str.strip()

# 字符串是否相等
def equals(str1,str2,ignoreCase = True):
    '''
    判断两个字符串是否相等
    :param str1:  字符串1
    :param str2:  字符串2
    :param ignoreCase:  忽略大小写，缺省True
    :return:
    '''
    if str1 is None and str2 is None:return True
    elif str1 is None or str2 is None:return False

    if ignoreCase and str1.lower() == str2.lower():
        return True
    elif not ignoreCase and str1 == str2:
        return True


    return False

#endregion

#region 对象的序列化

def getName(obj):
    '''
    取得对象名称，按照以下顺序查找名称字符串:对象是否有getName方法，是否有__name__字段，是否有name字段，是否有nameInfo字段
                以上字段如果类型是str，则直接返回，如果是NameInfo，则返回其中的name字段
    如果该对象类型本身就是字符串，则直接返回
    其它情况将得到''
    :param obj:
    :return:
    '''
    if obj is None:return ''
    if obj is str:return str(obj)

    if hasattr(obj,'getName'):
        return obj.getName()
    nameAttr = None
    if hasattr(obj,'__name__'):
        nameAttr = obj.__name__
    elif hasattr(obj,'name'):
        nameAttr = obj.name
    elif hasattr(obj,'nameInfo'):
        nameAttr = obj.nameInfo

    if nameAttr is None:return ''
    elif nameAttr is str:return str(nameAttr)
    elif nameAttr is NameInfo:return nameAttr.name
    else: return str(nameAttr)


# 存储格式说明
FORMAT = {
                  'sep' : ',',              #分割符
                  'format' : '${P}=${V}',   # 属性格式
                  'float' :  '.2f'          # 浮点数格
         }

def format(obj,**fmts):
    '''
    对象转字符串
    :param obj:  any 对象
    :param fmts: dict 转换格式
                      对于整数，用radix表示进制,2,b,bin,binary为二进制;8,0,oct为八进制；16，h，hex为十六进制，缺省十进制
                      对于浮点数，用float表示格式，如'.2f'表示小数点保留两位，若obj值等于整数(如3.0)，则缺省以整数方式显示
                      对于日期时间，用datetime表示格式,缺省为'yyyyMMdd hh24misi'
                      对于一般对象（非集合类型），如果有实现format(**fmts)函数，则直接调用；若
    :return:
    '''
    raiseWhenNone = False if fmts is None else fmts.get('raiseWhenNone',False)
    if obj is None:
        if raiseWhenNone: raise RuntimeError('无法格式化对象:对象为None')
        else: return ''
    elif isinstance(obj,str): return str(obj)
    elif isinstance(obj,int):
        radix = 10 if fmts is None else fmts.get('radix',10)
        if radix == 2 or radix == 'b' or radix == 'bin' or radix == 'binary':return "{0:b}".format(obj)
        elif radix == 8 or radix == 'o' or radix == 'oct' : return "{0:o}".format(obj)
        elif radix == 16 or radix == 'h' or radix == 'hex' : return "{0:h}".format(obj)
        else: return str(obj)
    elif isinstance(obj,float):
        if float(obj) == int(obj): return str(int(obj))
        else:
            fmt = '.2f' if fmts is None else fmts.get('float','.2f')
            return ('{0:'+fmt+'}').format(obj)
    else:
        return str(obj)
        #raise RuntimeError('功能没有实现')

#endregion

#region command

class Command:
    def __init__(self,id,parser):
        '''
        命令，这里约定命令的格式总是：动作 对象 动作参数，例如：show task log
        :param id:              str     id
        :param parser:          any     解析器，包含cmdMatch(text,**context)函数,cmdCompletions函数,cmdExecute(action,obj,*param,text,**context)函数，cmdHelp函数
        '''
        self.id = id
        self.parser = parser
        self.help = help
    def match(self,text):
        '''
        命令匹配
        :param text: str 匹配文本
        :return:     (bool,action,obj,list)
        '''
        if not isVaild(text):return False
        return self.parser.cmdMath(text)

cmdRegistry = Registry()

#endregion

#region json扩展
class ExtendJsonEncoder(json.JSONEncoder):
    ignoreTypes = []
    autostrTypes = []
    def default(self, obj):
        if isinstance(obj,Enum):
            return str(obj)
        if collections.any(ExtendJsonEncoder.ignoreTypes,lambda t:isinstance(obj,t)):
            return None
        if collections.any(ExtendJsonEncoder.autostrTypes,lambda t:isinstance(obj,t)):
            return str(obj)
        return json.JSONEncoder.default(self, obj)

#endregion