import json
import  numpy as np
import  copy
from functools import reduce
from enum import Enum

import utils.strs as strs

__all__ = ['first','findall','count','foreach','isEmpty','listtostr','listtostr','dicttostr','rangefeature',
            'dicttostr','dict_getattr','dict_setattr','equals','all','ExtendJsonEncoder']

#region 字段扩展

def dict_getattr(self, item):
    return self[item]

def dict_setattr(self,item,value):
    self[item] = value

#endregion

#region 集合中查询

# list中取满足条件的第一个元素
def first(the_iterable, condition = lambda x: True,defaultvalue=None):
    '''
    取得list中第一个满足条件的元素
    :param the_iterable:
    :param condition:
    :param defaultvalue:
    :return:
    '''
    if the_iterable is None or len(the_iterable)<=0:
        return defaultvalue
    for i in the_iterable:
        if condition(i):
            return i
    return defaultvalue

# list中取满足条件的所有元素
def findall(the_iterable, condition = lambda x: True):
    '''
    取得list满足条件的所有元素
    :param the_iterable:  迭代器
    :param condition:     条件
    :return:
    '''
    if the_iterable is None or len(the_iterable)<=0:
        return []
    r = []
    for i in the_iterable:
        if condition(i):
            r.append(i)
    return r

#取得满足条件的元素个数
def count(the_iterable, condition = lambda x: True):
    '''
    取得满足条件的元素个数
    :param the_iterable:  迭代器
    :param condition:     条件
    :return:
    '''
    if the_iterable is None or len(the_iterable)<=0:
        return 0
    count = 0
    for i in the_iterable:
        if condition(i):
            count += 1
    return count

def foreach(the_iterable,operation):
    if the_iterable is None or len(the_iterable)<=0:
        return
    for i in the_iterable:
        operation(i)

def mapreduce(the_iterable,reducefunc,mapfunc=None,default=''):
    '''
    mapreduce操作
    :param the_iterable: iterable 迭代对象
    :param reducefunc:   func  reduce函数，如果为None，则返回结果由下一个参数决定
    :param mapfunc:      func  map函数，如果为None，则返回结果只做reduce操作
    :param default:      the_iterable无效的时候返回的值
    :return:
    '''
    if isEmpty(the_iterable):return default
    if mapfunc is None and reducefunc is None:return default
    r1 = the_iterable if mapfunc is None else map(mapfunc,the_iterable)
    return r1 if reducefunc is None else reduce(reducefunc,r1)
#endregion

#region 集合性质判定
def isEmpty(the_iterable):
    '''
    判断集合是否空
    :param the_iterable:
    :return:
    '''
    return the_iterable is None or len(the_iterable) <= 0

# 比较两个list是否相等
def equals(list1,list2,lenmatch=True):
    '''
     比较两个list是否相等
    :param list1:
    :param list2:
    :param lenmatch:  是否要求长度必须匹配
    :return:
    '''
    if not isinstance(list1,list) and not isinstance(list2,list):return list1 == list2
    if not isinstance(list,list) or not  isinstance(list2,list):return False
    if isEmpty(list1) and isEmpty(list2): return True
    elif isEmpty(list1) or isEmpty(list2):return False

    if lenmatch and len(list1) != len(list2):return False
    len = min(len(list1),len(list2))

    for index,e in enumerate(list1):
        if index >= len:break
        if e != list2[index]:return  False

    return True


def all(the_iterable, condition = lambda x: True):
    '''
    判断集合是否满足某性质
    :param the_iterable:
    :param condition:
    :return:
    '''
    if the_iterable is None or len(the_iterable)<=0:
        return False
    for i in the_iterable:
        if not condition(i):return False
    return True

def any(the_iterable, condition = lambda x: True):
    '''
    判断集合是否满足某性质
    :param the_iterable:
    :param condition:
    :return:
    '''
    if the_iterable is None or len(the_iterable)<=0:
        return False
    for i in the_iterable:
        if condition(i):return True
    return False

#endregion

#region 集合序列化

#list转字符串
def listtostr(list,**props):
    '''
        字典转字符串
        :param list:  list
        :param props: 转换参数
                      'format' str 转换格式，可以是'csv'(缺省),'{$P}={$V}'（缺省）,'{$V}','json',其中{$P}要求list中元包含name或者nameInfo属性或者getName函数
                      'sep'    字典项之间的分割符,可以是',','\n'等
        :return:
        '''
    if isEmpty(list):return ''
    #format = 'csv' if props is None or 'format' not in props.keys() else props['format']
    format = 'csv' if props is None or 'format' not in props else props['format']
    if format == 'json':
        return json.dumps(list)

    #sep = ',' if props is None or not props.keys().__contains__('sep') else props['sep']
    sep = ',' if props is None or 'sep' not in props else props['sep']
    ss = ''
    for value in list:
        if len(ss)>0:ss += sep    #添加分隔符
        if format == 'csv':
            ss += strs.format(value)
        else:
            ss += format.replace('{$P}', strs.getName(value)).replace('{$V}', strs.format(value))
    return ss

#字段转字符串
def dicttostr(dict,**props):
    '''
    字典转字符串
    :param dict:  字典
    :param props: 转换参数
                  'format' str 转换格式，可以是'{$P}={$V}'（缺省）,'{$V}','json'
                  'sep'    字典项之间的分割符,可以是',','\n'等
    :return:
    '''
    if isEmpty(dict):return ''

    #format = '{$P}={$V}' if props is None or 'format' not in props.keys() else props['format']
    format = '{$P}={$V}' if props is None or 'format' not in props else props['format']
    if format == 'json':
        return json.dumps(dict)
    #sep = ',' if props is None or 'sep' not in props.keys() else props['sep']
    sep = ',' if props is None or 'sep' not in props else props['sep']
    s = []
    for key, value in dict.items():
        s.append(format.replace('{$P}',str(key)).replace('{$V}',strs.format(value)))
    return listtostr(s,format='csv',sep=sep)

#endregion

#region 值类型list的计算

# 计算list的平均值，最大和最小
def rangefeature(list):
    if isEmpty(list):return 0.0,0.,0.
    sum = reduce(lambda x,y:x+y,list)
    stdev = np.std(list, ddof=1)
    return max(list),sum / len(list),min(list),stdev

#endregion


#region json扩展
class ExtendJsonEncoder(json.JSONEncoder):
    ignoreTypes = []
    autostrTypes = []
    def default(self, obj):
        if isinstance(obj,Enum):
            return str(obj)
        if any(ExtendJsonEncoder.ignoreTypes,lambda t:isinstance(obj,t)):
            return None
        if any(ExtendJsonEncoder.autostrTypes,lambda t:isinstance(obj,t)):
            return str(obj)
        return json.JSONEncoder.default(self, obj)

#endregion