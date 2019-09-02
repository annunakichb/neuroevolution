# -*- coding: UTF-8 -*-

import os
import utils.strs as strs
import utils.collections as collections
from functools import reduce

def getFullFileName(filename,dir=''):
    '''
    取得全文件名
    :param filename: 如果文件名是全路径，则结果不变，否则以py所在路径+dir为文件所在路径
    :param dir:
    :return:
    '''
    # 如果文件存在，仍返回文件名
    if(os.path.exists(filename)):return filename
    # 如果文件名包含全路径，直接返回文件名(windows操作系统中是包含:，unix家族是/开始)
    if not strs.isVaild(filename):return filename
    if filename.startWith('/') or filename.contains(':'):return filename

    p = os.path.dirname(os.path.abspath(__file__)+dir)
    if not p.endswith('/'):p += '/'
    return p + filename

def getFullDirectory(dir):
    '''
    得到全路径，并以/结尾
    :param dir:
    :return:
    '''
    if (os.path.exists(dir)):return dir
    if not strs.isVaild(dir):return dir
    if dir.startWith('/') or dir.contains(':'):return dir
    p = os.path.dirname(os.path.abspath(__file__) + dir)
    if not p.endswith('/'): p += '/'

def spiltfilename(filename):
    '''
    分解文件名
    :param filename: 带路径的完整文件名字符串
    :return:         路径,文件名(无扩展名),扩展名
    '''
    (filepath,tempfilename) = os.path.split(filename)
    (shotname,extension) = os.path.splitext(tempfilename)
    return filepath,shotname,extension

def serachPath(root,match_method,childs = False):
    '''
    搜索路径
    :param root:
    :param match_method:
    :param childs:
    :return:
    '''
    items = os.listdir(root)
    results = []
    for item in items:
        file = os.path.join(root, item)
        if match_method(item):
            results.append(file)
        if not childs:
            continue
        if os.path.isdir(file):
            r = serachPath(file,match_method,childs)
            results += r
    return results




def writeLines(filename,lines,encode='utf-8',raiseError=False):
    try:
        f = open(filename,'w')
        for s in lines:
            f.write(str(s)+'\n')
    except IOError as e:
        if raiseError:raise  e
        return
    finally:
        f.close()

def writeText(filename,text,encode='utf-8',newline=True,raiseError=False):
    try:
        f = open(filename,'w')
        f.write(text+'\n' if newline else text)
    except IOError as e:
        if raiseError:raise  e
        return
    finally:
        f.close()

def appendLines(filename,lines,encode='utf-8',raiseError=False):
    try:
        f = open(filename,'wa')
        for s in lines:
            f.write(str(s)+'\n')
    except IOError as e:
        if raiseError:raise  e
        return
    finally:
        f.close()

def appendText(filename,text,encode='utf-8',newline=True,raiseError=False):
    try:
        f = open(filename,'wa')
        f.write(text+'\n' if newline else text)
    except IOError as e:
        if raiseError:raise  e
        return
    finally:
        f.close()

def readLines(filename,limit=None,encode='utf-8',raiseError=False):
    try:
        f = open(filename,'r')
        return f.readlines(limit)
    except IOError as e:
        if raiseError:raise  e
        return
    finally:
        f.close()

def readText(filename,encode='utf-8',raiseError=False):
    lines = readLines(filename,None,encode,raiseError)
    if collections.isEmpty(lines): return  ''
    return reduce(lambda x,y:x+'\n'+y,lines)


def get_data_path():
    curpath = os.path.split(os.path.realpath(__file__))[0]
    srcpath = os.path.abspath(os.path.dirname(curpath))
    rootpath = os.path.abspath(os.path.dirname(srcpath))
    datapath = rootpath + '\\datas'
    return datapath