import collections
import json
from .strs import *

__all__ = ['first','findall','listtostr','dicttostr']


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


#endregion

#region 集合性质判定
def isEmpty(the_iterable):
    '''
    判断集合是否空
    :param the_iterable:
    :return:
    '''
    return the_iterable is not None and len(the_iterable)>0

# 比较两个list是否相等
def equals(list1,list2,lenmatch=True):
    '''
     比较两个list是否相等
    :param list1:
    :param list2:
    :param lenmatch:  是否要求长度必须匹配
    :return:
    '''
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
        if condition(i):return False
    return True

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
    format = 'csv' if props is None or not props.keys().__contains__('format') else props['csv']
    if format == 'json':
        return json.dumps(list)

    sep = ',' if props is None or not props.keys().__contains__('sep') else props['sep']
    strs = ''
    for value in list:
        if len(strs)>0:strs += sep    #添加分隔符
        if format == 'csv':
            str += format(value)
        else:
            strs += format.append(format.replace('{$P}', getName(value)).replace('{$V}', format(value)))
    return strs

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

    format = '{$P}={$V}' if props is None or not props.keys().__contains__('format') else props['format']
    if format == 'json':
        return json.dumps(dict)
    sep = ',' if props is None or not props.keys().__contains__('sep') else props['sep']
    strs = []
    for key, value in dict.items():
        strs.append(format.replace('{$P}',str(key)).replace('{$V}',str(value)))
    return listtostr(strs,format='csv',sep=sep)

#endregion