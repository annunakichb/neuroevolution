import utils.collections as collections
from functools import reduce

class Coordination:
    def __init__(self,*values):
        if values is None:
            self.values = []
        else:
            self.values = list(values)

    @property
    def X(self):
        return self.__getValue(0)

    @X.setter
    def X(self,value):
        self.__setValue(0,value)

    @property
    def Y(self):
        return self.__getValue(1)

    @Y.setter
    def Y(self, value):
        self.__setValue(1,value)

    @property
    def Z(self):
        return self.__getValue(2)

    @Z.setter
    def Z(self, value):
        self.__setValue(2, value)

    def __getValue(self,index):
        return 0 if len(self.values) <= index else self.values[index]

    def __setValue(self, index, value):
        if self.values is None: self.values = []
        if len(self.values)>(index+1):
            self.values[index] = value
        else:self.values.extend([0.0]*(index-len(self.values))).append(value)

    def __eq__(self, other):
        if isinstance(other,Coordination):return False
        return collections.equals(self.values,other.values)

    def __str__(self):
        if self.values is None:self.values = []
        if len(self.values)<=0:return ''
        return  reduce(lambda x,y:x+","+y,map(lambda v: "{:d}".format(v) if v == int(v) else "{:.2f}".format(v),self.values))

    def center(cls,coord1,coord2):
        vals = []
        for i in range(min(coord1.values,coord2.values)):
            vals.append((coord1.values[i]+coord2.values[i])/2)
        c = Coordination(vals)