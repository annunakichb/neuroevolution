
import numpy as np

class SelectionBase:
    def __init__(self):
        pass
    def roulette(self,inds,selectcount,featureKey='fitness'):
        '''
        轮盘赌算法实现
        :param inds:       list 所有待选择个体
        :param selectcount int  选择数量，缺省是inds的数量
        :param featureKey: str  用于比较的特征，缺省是适应度
        :return:
        '''
        if selectcount <= 0: selectcount = len(inds)

        if len(inds) <=0 : return []
        elif len(inds) == 1:
            return [inds[0]]*selectcount

        feature_values = [ind[featureKey] for ind in inds]
        total = sum(feature_values)
        feature_probs = [value/total for value in feature_values]

        ms = [np.random.random() for i in range(selectcount)]
        ms.sort()
        fitin,selected_count,selected = 0,0,[]
        while selected_count < selectcount:
            if (ms[selected_count] < feature_probs[fitin]):
                selected.append(inds[fitin])
                selected_count = selected_count + 1
            else:
                fitin = fitin + 1
        return selected
