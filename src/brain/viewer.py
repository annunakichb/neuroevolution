
import graphviz

class NetworkView:
    def __init__(self):
        self.styles = self.getDefaultNodeConfig()
        self.node_attrs = {
        'shape': 'circle',
        'fontsize': '9',
        'height': '0.2',
        'width': '0.2'}

    def getDefaultNodeConfig(self):
        inputNodeConfig = {
            'style': 'filled',
            'shape': 'box',
            'fillcolor' :'lightgray'
        }
        hiddenNodeConfig = {
            'shape': 'circle',
            'fontsize': '9',
            'height': '0.2',
            'width': '0.2',
            'text':'id',
            'fillcolor':'white'
        }
        outputNodeConfig = {
            'style': 'filled',
            'shape': 'box',
            'fillcolor': 'lightblue'
        }

        return {'input':inputNodeConfig,'hidden':hiddenNodeConfig,'output':outputNodeConfig,'connection':{}}


    def drawNet(self,net,fmt='svg',filename=None,view=True):
        dot = graphviz.Digraph(format=fmt, node_attr=self.node_attrs)
        #画输入节点
        ns = net.neurons[0]
        for i,input in enumerate(ns):
            #text =str(i+1)
            text = str(input.id)
            dot.node(text,_attributes=self.styles['input'])
        # 画输出节点
        ns = net.neurons[-1]
        for i,output in enumerate(ns):
            #text= str(i+1)
            text = str(output.id)
            dot.node(text, _attributes=self.styles['output'])

        # 画所有中间节点
        ns = net.getHiddenNeurons()
        for i,hidden in enumerate(ns):
            style = self.styles['hidden']
            text = ''
            if style['text'] == 'id':text = str(hidden.id)
            elif style['text'] == 'birth':text = str(int(hidden.birth))
            elif style['text'] == 'bias':text = "%。2f"%hidden['bias']
            text = str(hidden.id)
            dot.node(text,_attributes=style)

        # 画连接
        synapses = net.getSynapses()
        connectionConfig = self.styles['connection']
        for i,s in enumerate(synapses):
            #style = 'solid' if cg.enabled else 'dotted'
            color = 'green' if s['weight'] > 0 else 'red'
            width = str(0.1 + abs(s['weight'] / 5.0))
            connectionConfig['color'] = color
            connectionConfig['width'] = width

            dot.edge(str(s.fromId), str(s.toId), _attributes=connectionConfig)


        dot.render(filename, view=view)


