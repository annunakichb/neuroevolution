import  numpy as np
import utils.collections as collections

class BoxAttentionOperation:
    def __init__(self,operation,name,prompt):
        self.operation = operation
        self.name = name
        self.prompt = prompt
    def do_attention(self,box):
        pass
    @classmethod
    def getAttention(cls,name):
        return collections.first(attention_operations,lambda x:x.name == name)

class BoxAttentionOperationU(BoxAttentionOperation):
    def do_attention(self,box):
        if box.inputs is None:return None
        if len(box.inputs)>1:
            activation_features = [input.activation_feature() for input in box.inputs]
        else:
            activation_features = box.history_activation_features()
        return np.average(activation_features)

    def do_expection(self,box):
        if box.inputs is None:return None
        if not collections.all(box.inputs,lambda b:b.expection is None):
            return None
        activation_features = [input.expection for input in box.inputs]
        return np.average(activation_features)

class BoxAttentionOperationV(BoxAttentionOperation):
    def do_attention(self,box):
        if box.inputs is None:return None
        if len(box.inputs)>1:
            activation_features = [input.activation_feature() for input in box.inputs]
        else:
            activation_features = box.history_activation_features()
        return np.cov(activation_features)
    def do_expection(self,box):
        if box.inputs is None:return None
        activation_features = [input.expection for input in box.inputs]
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
    def do_expection(self,box):
        if box.inputs is None:return None
        activation_features = [input.expection for input in box.inputs]
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
    def do_expection(self, box):
        if box.inputs is None: return None
        activation_features = [input.expection for input in box.inputs]
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
    def do_expection(self, box):
        if box.inputs is None: return None
        activation_features = [input.expection for input in box.inputs]
        directions = []
        for i in range(len(activation_features) - 1):
            directions.append(activation_features[i+1]-activation_features[i])
        return (directions - np.mean(directions, axis=0)) / np.std(directions, axis=0)


class BoxAttentionOperationT(BoxAttentionOperation):
    def do_attention(self, box):
        if box.inputs is None: return None
        if len(box.inputs) > 1:
            activation_features = [input.activation_feature() for input in box.inputs]
        else:
            activation_features = box.history_activation_features()
        return activation_features
    def do_expection(self, box):
        if box.inputs is None: return None
        activation_features = [input.expection for input in self.inputs]
        dis = []
        for node in self.nodes:
            x = [f if f is not None else u for f,u in zip(activation_features,node.u)]
            dis.append(box.norm_pdf_multivariate(x,node.u,node.sigma))
        dis = [(d-np.min(dis))/(np.max(dis)-np.min(dis)) for d in dis]
        dis = [d if d >= box.net.definition.box.activation_threadshold else 0. for d in dis]
        dis = reversed(dis)
        index = reversed(np.argsort(dis))
        expection_features = []
        expection_features_scale = []
        for i in index:
            if dis[i]<=0:continue
            f = [None if input.expection is not None else self.nodes[i].u[j] for j,input in enumerate(self.inputs)]
            expection_features.append(f*dis[i])

        expection_features = np.sum(expection_features)/np.sum(dis)
        return expection_features



class BoxAttentionOperationA(BoxAttentionOperation):
    def do_attention(self, box):
        if box.inputs is None: return None
        if len(box.inputs) > 1:
            activation_features = [input.activation_feature() for input in box.inputs]
        else:
            activation_features = box.history_activation_features()
        return activation_features
    def do_expection(self, box):
        if box.inputs is None: return None
        activation_features = [input.expection for input in self.inputs]
        dis = []
        for node in self.nodes:
            x = [f if f is not None else u for f,u in zip(activation_features,node.u)]
            dis.append(box.norm_pdf_multivariate(x,node.u,node.sigma))
        dis = [(d-np.min(dis))/(np.max(dis)-np.min(dis)) for d in dis]
        dis = [d if d >= box.net.definition.box.activation_threadshold else 0. for d in dis]
        dis = reversed(dis)
        index = reversed(np.argsort(dis))
        expection_features = []
        expection_features_scale = []
        for i in index:
            if dis[i]<=0:continue
            f = [None if input.expection is not None else self.nodes[i].u[j] for j,input in enumerate(self.inputs)]
            expection_features.append(f*dis[i])

        expection_features = np.sum(expection_features)/np.sum(dis)
        return expection_features

class BoxAttentionOperationP(BoxAttentionOperation):
    def do_attention(self, box):
        if box.inputs is None: return None
        if len(box.inputs) > 1:
            activation_features = [input.activation_feature() for input in box.inputs]
        else:
            activation_features = box.history_activation_features()
        return activation_features

    def do_expection(self, box):
        pass
class BoxAttentionOperationC(BoxAttentionOperation):
    def do_attention(self, box):
        if box.inputs is None: return None
        if len(box.inputs) > 1:
            activation_features = [input.activation_feature() for input in box.inputs]
        else:
            activation_features = box.history_activation_features()
        return activation_features

    def do_expection(self, box):
        pass
class BoxAttentionOperationSY(BoxAttentionOperation):
    def do_attention(self, box):
        if box.inputs is None: return None
        if len(box.inputs) > 1:
            activation_features = [input.activation_feature() for input in box.inputs]
        else:
            activation_features = box.history_activation_features()
        return activation_features

    def do_expection(self, box):
        pass
class BoxAttentionOperationPS(BoxAttentionOperation):
    def do_attention(self, box):
        if box.inputs is None: return None
        if len(box.inputs) > 1:
            activation_features = [input.activation_feature() for input in box.inputs]
        else:
            activation_features = box.history_activation_features()
        return activation_features

    def do_expection(self, box):
        pass


attention_U = BoxAttentionOperationU('U','均值','$X的均值')
attention_V = BoxAttentionOperationV('V','方差','$X的方差')
attention_S = BoxAttentionOperationS('S','速度','$X的速度')
attention_D = BoxAttentionOperationD('D','方向','$X的方向')
attention_C = BoxAttentionOperationC('C','周期','$X的周期')
attention_P = BoxAttentionOperationP('P','投影','$X的投影')
attention_SY = BoxAttentionOperationSY('SY','合成','$Xi的合成')
attention_PS = BoxAttentionOperationPS('PS','属性速度','$X的$P速度')
attention_PD = BoxAttentionOperationPD('PD','属性方向','$X的$P方向')
attention_T = BoxAttentionOperationT('T','因果','$X1是$X2的原因')
attention_A = BoxAttentionOperationA('A','关联','$X1与$X2相关')
attention_operations = [attention_U,attention_V,attention_S,attention_D,attention_PD,attention_T,attention_A]
attention_operations_name = [a.name for a in attention_operations]