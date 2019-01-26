import numpy as np
from functools import reduce
import re
from utils import strs
import copy
__all__ = ['NameInfo','Range','PropertyInfo','Variable','Properties']

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
        self.name = strs.vaild(name)
        self.caption = strs.vaild(caption,self.name)
        self.description = strs.vaild(description,self.name)
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
        if strs.equals(name,self.name):return True
        if not allName:return False
        if strs.equals(name,self.caption):return True
        if self.alias is None or len(self.alias)<=0:return False
        return reduce(lambda x, y: x or y, map(lambda x: strs.equals(x,name), self.alias))

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
        return self.name + '' if not strs.isVaild(self.description) else '(' + self.description + ')'

    def clone(self):
        return NameInfo(self.name,self.caption,self.description,self.caption,[].extend(self.alias))
class Range:
    regax = '(\S*)'  + '(\[|\(){1}' + '((\+|-)?\d+(\.\d+)?){1}'+ '\:' + '((\+|-)?\d+(\.\d+)?){1}' + '(\:((\+|-)?\d+(\.\d+)?))?' + '(\]|\)){1}'
    pattern = re.compile(regax)

    def __init__(self,range):
        '''
        值范围
        :param range: str，格式如[0:1] (0:1) [0:1:0.1] (0:1:0.1) uniform[0:1] normal[0,1]，其中[]与()为数值区间
        '''
        self.rangeStr = range

        self.distributionName = 'uniform'
        self.begin = 0
        self.includeBegin = True
        self.end = 1
        self.IncludeEnd = True
        self.step = 0.1
        self.__list = []
        self.__stepMode = 'step'

        #m = re.compile('(\S*)').match('uniform')
        #m = re.compile('(\[|\(){1}').match('[')
        #m = re.compile('((\+|-)?\d+(\.\d+)?){1}').match('-30.0')
        #m = re.compile('\:').match(':')
        #m = re.compile('(\:(\+|-)?\d+(\.\d+)?)?').match(':-30.0')
        #m = re.compile('(\]|\)){1}').match(']')

        if not strs.isVaild(range):return
        m = Range.pattern.match(range)
        self.distributionName = m.groups()[0]
        self.begin = float(m.groups()[2])
        self.end = float(m.groups()[5])
        self.step = (self.end - self.begin)/10 if m.groups()[9] is None else float(m.groups()[9])  #？有错误

    def sample(self,size=1):
        if self.distributionName == 'uniform':
            return np.random.uniform(self.begin,self.end,size)
        elif self.distributionName == 'normal':
            return np.random.normal(self.begin,self.end,size)
        return None

    def clone(self):
        return Range(self.rangeStr)

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
    def clone(self):
        return PropertyInfo(self.xh,self.nameInfo,self.type,self.default,self.storeformats,self.range,self.getter,self.setter,**self.props)

class Variable(PropertyInfo):

    def __init__(self,nameInfo,type=float,default=0.0,xh=1,value=None,storeformats={},range = None,getter=None,setter=None,**props):
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
        super(Variable, self).__init__(xh, nameInfo, type, default, storeformats, range, getter, setter , **props)
        self.value = value

    def __str__(self):
        return format(self.value)

    def __repr__(self):
        if self.value is None:return ''
        return self.nameInfo.name + '=' + format(self.value)

    def clone(self):
        return Variable(self.nameInfo.clone(),self.type,self.default,self.xh,np.array(0),self.storeformats, \
                        None if self.range is None else self.range.clone(),self.getter,self.setter,**copy.deepcopy(self.props))

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
        if name == '':name = strs.getName(obj)
        if not strs.isVaild(name):return
        self.__tables__[name] = obj

    def find(self,name,default=None,setifnotexist=True):
        '''
        查找对象
        :param name:           str 对象名称
        :param default:        any 缺省对象
        :param setifnotexist:  如果找不到，是否将default记录（下次就可以找到）
        :return: 注册的对象
        '''
        if not strs.isVaild(name):return default
        if name in self.__tables__.keys():
            return self.__tables__[name]

        if not setifnotexist:return default
        self.__tables__[name] = default
        return default

    def keys(self):
        return self.__tables__.keys()

#region 字典扩展
class Properties(dict):
    def __init__(self, *args, **kwargs):
        super(Properties, self).__init__(*args, **kwargs)

    def __getattr__(self, name):
        value = self[name]
        if isinstance(value, dict):
            value = Properties(value)
        return value

    def __setattr__(self, key, value):
        self[key] = value

    def __getitem__(self, item):
        value = super(Properties,self).__getitem__(item)
        if isinstance(value,dict) and not isinstance(value,Properties):
            return Properties(value)
        return value




#endregion