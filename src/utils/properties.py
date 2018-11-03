
from functools import reduce
import re
from .collections import *
from .strs import *

__all__ = ['NameInfo','Range','PropertyInfo','Variable']

class NameInfo:
    __slots__ = ['name','caption','description','cataory','alias']

    def __init__(self,name,caption='',description='',cataory='',*alias):
        '''
        名称信息
        :param name:         名称
        :param caption:      显示名
        :param description:  描述
        :param cataory:      分类
        :param alias:        别名list
        '''
        self.name = vaild(name)
        self.caption = vaild(caption,self.name)
        self.description = vaild(description,self.name)
        self.cataory = cataory
        self.alias = alias

    def hasName(self,name,ignoreCase=True,allName=True):
        '''
        是否包含名称
        :param name:       名称
        :param ignoreCase: 忽略大小写
        :param allName:    只检查name字段还是包括caption和别名
        :return:
        '''
        if name is None:return False
        if equals(name,self.name):return True
        if not allName:return False
        if equals(name,self.caption):return True
        if self.alias is None or len(self.alias)<=0:return False
        return reduce(lambda x, y: x or y, map(lambda x: equals(x,name), self.alias))

    def __eq__(self, other):
        '''
        是否相等
        :param other: str or NameInfo
        :return: 相等只对other的name字段进行判断
        '''
        if other is None:return False
        if other is NameInfo:
            return self.hasName(other.name)
        elif other is str:
            return self.hasName(other)
        else:
            return self.hasName(str(other))

    def __str__(self):
        '''
        用于输出给客户
        :return:
        '''
        return self.name

    def __repr__(self):
        '''
        用于调试打印详细信息
        :return:
        '''
        return self.name + '' if not isVaild(self.description) else '(' + self.description + ')'

    storeformat1 = '{"name":"{$name}"' + '{?({$name}!={$caption})({#sep}"caption":"{$caption?hide}")}'\
                                       + '{?({$name}!={$description})({#sep}"description":"{$description?hide}")}'\
                                       + '({#sep}"cataory":"{$cataory?hide}")'\
                                       + '("alias":[ + {$alias?hide}})'

    def store(self,formats=FORMAT):
        return "{name = '" + self.name + ''




class Range:
    #? 正则定义有问题
    regax = '(\S*)' + '([|(){1}' + '([-]\d+\.\d+){1}' + ':' + '([-]\d+\.\d+){1}' + '[:([-]\d+\.\d+){1}]' + '(]|)){1}'
    pattern = re.compile(regax)
    def __init__(self,range):
        '''
        值范围
        :param range: str，格式如[0:1] (0:1) [0:1:0.1] (0:1:0.1) uniform[0:1] normal[0,1]，其中[]与()为数值区间
        '''
        self.distributionName = 'uniform'
        self.begin = 0
        self.includeBegin = True
        self.end = 1
        self.IncludeEnd = True
        self.step = 0.1
        self.__list = []
        self.__stepMode = 'step'

        if not isVaild(range):return
        m = Range.pattern.match(range)    #？未校验错误
        self.distributionName = m.group(0)
        self.begin = m.group(1)
        self.end = m.group(2)
        self.step = m.group(3)  #？有错误



class PropertyInfo:
    def __init__(self,xh,name,type,default,storeformats={},range = None,getter=None,setter=None,**props):
        '''
        属性信息，描述一个对象的属性字段
        :param xh:                int 序号
        :param name:              str或者NameInfo 名称
        :param type:              type           类型
        :param default:           any            缺省值
        :param storeformats:      dict           存储格式
        :param range:             str or Range   取值范围
        :param getter:            func           getter
        :param setter:            func           setter
        :param props:             dict           扩展属性
        '''
        self.xh = xh
        self.nameInfo = name if name is NameInfo else NameInfo(str(name))
        self.type = type
        self.default = default
        self.range = range
        self.getter = getter
        self.setter = setter
        self.storeformats = storeformats
        self.props = props if props is not None else {}

class Variable(PropertyInfo):
    def __init__(self,xh,nameInfo,type,default,value=None,storeformats={},range = None,getter=None,setter=None,**props):
        '''
        变量,参数意义参见PropertyInfo
        :param xh:
        :param nameInfo:
        :param type:
        :param default:
        :param value:
        :param storeformats:
        :param range:
        :param getter:
        :param setter:
        :param props:
        '''
        super(xh,nameInfo,type,default,getter,setter,storeformats,range,props)
        self.value = value

    def __str__(self):
        return format(self.value)

    def __repr__(self):
        return self.nameInfo.name + '=' + format(self.value)

# 注册表对象
class Registry:
    def __init__(self):
        '''
        注册表对象
        '''
        self.__tables__ = {}

    def register(self,obj,name=''):
        '''
        注册
        :param obj:  注册对象
        :param name: 对象名称，如果无效，则尝试调用strs.getName(obj)取得
        :return: None
        '''
        if obj is None:return
        if name == '':name = getName(obj)
        if not isVaild(name):return
        self.__tables__[name] = obj

    def find(self,name,default=None,setifnotexist=True):
        '''
        查找对象
        :param name:           str 对象名称
        :param default:        any 缺省对象
        :param setifnotexist:  如果找不到，是否将default记录（下次就可以找到）
        :return: 注册的对象
        '''
        if not isVaild(name):return default
        if name in self.__tables__.keys():
            return self.__tables__[name]

        if not setifnotexist:return default
        self.__tables__[name] = default
        return default