

class Coordination:
    def __init__(self,*values):
        self.values = [].extend(values)

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
        if len(self.values)>(index+1):
            self.values[index] = value
        else:self.values.extend([0.0]*(index-len(self.values))).append(value)