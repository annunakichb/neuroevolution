import  numpy as np

class BoxAttentionOperation:
    def __init__(self,operation,name,prompt):
        self.operation = operation
        self.name = name
        self.prompt = prompt
    def do_attention(self,box):
        pass
class BoxAttentionOperationU(BoxAttentionOperation):
    def do_attention(self,box):
        if box.inputs is None:return None
        if len(box.inputs)>1:
            activation_features = [input.activation_feature() for input in box.inputs]
        else:
            activation_features = box.history_activation_features()
        return np.average(activation_features)

class BoxAttentionOperationV(BoxAttentionOperation):
    def do_attention(self,box):
        if box.inputs is None:return None
        if len(box.inputs)>1:
            activation_features = [input.activation_feature() for input in box.inputs]
        else:
            activation_features = box.history_activation_features()
        return np.cov(activation_features)

class BoxAttentionOperationS(BoxAttentionOperation):
    def do_attention(self,box):
        if box.inputs is None:return None
        if len(box.inputs)>1:
            activation_features = [input.activation_feature() for input in box.inputs]
        else:
            activation_features = box.history_activation_features()
        dis = 0
        for i in range(len(activation_features)-1):
            dis += np.sqrt(np.sum(np.square(activation_features[i] - activation_features[i+1])))
        return dis/len(activation_features)

class BoxAttentionOperationS(BoxAttentionOperation):
    def do_attention(self,box):
        if box.inputs is None:return None
        if len(box.inputs)>1:
            activation_features = [input.activation_feature() for input in box.inputs]
        else:
            activation_features = box.history_activation_features()
        dis = 0
        for i in range(len(activation_features)-1):
            dis += np.sqrt(np.sum(np.square(activation_features[i] - activation_features[i+1])))
        return dis/len(activation_features)


class BoxAttentionOperationD(BoxAttentionOperation):
    def do_attention(self, box):
        if box.inputs is None: return None
        if len(box.inputs) > 1:
            activation_features = [input.activation_feature() for input in box.inputs]
        else:
            activation_features = box.history_activation_features()
        directions = []
        for i in range(len(activation_features) - 1):
            directions.append(activation_features[i+1]-activation_features[i])
        return (directions - np.mean(directions, axis=0)) / np.std(directions, axis=0)

class BoxAttentionOperationPD(BoxAttentionOperation):
    def do_attention(self, box):
        if box.inputs is None: return None
        if len(box.inputs) > 1:
            activation_features = [input.activation_feature() for input in box.inputs]
        else:
            activation_features = box.history_activation_features()
        directions = []
        for i in range(len(activation_features) - 1):
            directions.append(activation_features[i+1]-activation_features[i])
        return (directions - np.mean(directions, axis=0)) / np.std(directions, axis=0)

class BoxAttentionOperationF(BoxAttentionOperation):
    def do_attention(self, box):
        if box.inputs is None: return None
        if len(box.inputs) > 1:
            activation_features = [input.activation_feature() for input in box.inputs]
        else:
            activation_features = box.history_activation_features()
        return activation_features



attention_U = BoxAttentionOperation('U','均值','$X的均值')
attention_V = BoxAttentionOperation('V','方差','$X的方差')
attention_S = BoxAttentionOperation('S','速度','$X的速度')
attention_D = BoxAttentionOperation('D','方向','$X的方向')
attention_PD = BoxAttentionOperation('PD','属性方向','$X的$P方向')
attention_T = BoxAttentionOperation('T','时序','$X1是$X2的原因')
attention_A = BoxAttentionOperation('A','关联','$X1与$X2存在关联')
attention_operations = [attention_U,attention_V,attention_S,attention_D,attention_PD,attention_T,attention_A]
attention_operations_name = [a.name for a in attention_operations]