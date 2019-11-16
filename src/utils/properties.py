# -*- coding: UTF-8 -*-

import numpy as np
from functools import reduce
import re
from utils import strs
import copy
__all__ = ['NameInfo','Range','PropertyInfo','Variable','Properties','Registry','getName']

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

    def __init__(self,xh,name,type,default=None,catalog="",desc='',alias='',required='optional',range = None,getter=None,setter=None,**props):
        '''
        属性信息，描述一个对象的属性字段
        :param xh:                int 序号
        :param name:              str或者NameInfo 名称
        :param type:              type           类型
        :param default:           any            缺省值
        :param catalog            Union(str,list)分类目录
        :param desc               str            描述
        :param alias              str             名称（存储名称）
        :param required           str            是否必须（‘optional’,'necessary'）
        :param range:             str or Range   取值范围
        :param getter:            func           getter
        :param setter:            func           setter
        :param props:             dict           扩展属性
                                        'format.formatname'  str
        '''
        self.xh = xh
        self.nameInfo = name if name is NameInfo else NameInfo(str(name))
        self.type = type
        self.default = default
        self.catalog = catalog
        self.desc = desc
        self.range = range
        self.getter = getter
        self.setter = setter
        self.props = props if props is not None else {}
        self.alias = alias
        self.required = required

    def __str__(self):
        return (self.alias if self.alias is not None and self.alias != '' else self.nameInfo.name) + \
               (' type='+self.type if self.type is not None and self.type != object else '') + \
               (' required=' + str(self.required)) + ('' if self.required else str(self.default))



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
        if name == '':name = getName(obj)
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
        #if name in self.__tables__.keys():
        if name in self.__tables__:
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

    @classmethod
    def create_from_dict(cls,param_meta_infos,**kwargs):
        '''
        从字段对象创建Properties
        :param cls                type of Properties
        :param param_meta_infos:  dict of PropertyInfo 字段参数元信息，key是PropertyInfo的name，value是PropertyInfo对象，规定kwargs应该有哪些参数
        :param kwargs:            dict                  字典数据
        :return:                  Properties            属性集对象
        '''
        results = {}
        kw_key = ''
        for key, param_info in param_meta_infos.items():
            # 取得参数值,没有则选择缺省
            value = param_info.default
            if key not in kwargs and param_info.alias not in kwargs:
                if param_info.required == 'necessary':
                    raise RuntimeError(key + ' is necessary in population parameters')
            else:
                kw_key = key if key in kwargs else param_info.alias
                value = kwargs[kw_key]
            # 别名转换成list,并根据别名生成popParam的多级key
            catalogs = param_info.catalog.split('.')
            t = results
            for a in catalogs[1:-1]:
                if a not in t:
                    t[a] = {}
                t = t[a]
            t[catalogs[-1] if len(catalogs)>0 else kw_key] = value
            if kw_key != '' and kw_key in kwargs:kwargs.pop(kw_key)

        results.update(kwargs)
        return Properties(results)

    def __getattr__(self, name):
        value = self[name]
        if isinstance(value, dict) and not isinstance(value,Properties):
            value = Properties(value)
            self[name] = value
        return value

    def __setattr__(self, key, value):
        self[key] = value

    def __getitem__(self, item):
        value = super(Properties,self).__getitem__(item)
        if isinstance(value,dict) and not isinstance(value,Properties):
            return Properties(value)
        return value

    def __getstate__(self):
        """Return state values to be pickled."""
        dicts = {}
        for key in self.keys():
            value = self[key]
            if isinstance(value, Properties):
                value = value.__getstate__()
            dicts[key] = value
        return dicts

    def __setstate__(self, state):
        """Restore state from the unpickled state values."""
        dicts = state
        for key in dicts:
            self[key] = dicts[key]




        #endregion