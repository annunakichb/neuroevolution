import  domains.vee.env as env
import  evolution
from evolution.session import EvolutionTask
import numpy as np
import copy
import evolution.agent as agent
from evolution.agent import Individual
from evolution.agent import individualTypes
from evolution.agent import Population

#region parameters
# length of gene
gene_bits = 4
# chromosome of ancestor
init_genome = '0000'*10
# length of phenotype
phenotype_bits = 4


def development(genome):
    postfix_length = phenotype_bits - gene_bits
    max_postfix = pow(10, postfix_length)
    gene_count = len(genome)//gene_bits
    phenome = ''
    for i in range(gene_count):
        gene = genome[i*gene_bits:(i+1)*gene_bits]
        postfix = np.random.uniform(low=0,high=99)
        if len(postfix)<2:postfix = '0'+str(postfix)
        phenome += gene+str(postfix)
    return phenome
#endregion

#region env

# Fitness Landscape
class FitnessLandscape:
    def __init__(self,center_count=100):
        # 随机选择中心
        self.centers = np.array([np.random.uniform(low=0,high=9999,size=center_count),np.random.uniform(low=0,high=99,size=center_count)])
        # 随机权重
        self.weights = np.array([1.0]*center_count) #np.random.rand(center_count)
        self.weights /= np.sum(self.weights)
        # 随机协方差
        self.covs = np.array([np.random.rand(2,2) for i in range(center_count)])

    def fitness(self,ind):
        count = len(ind['phenotype'])/phenotype_bits
        props = []
        for i in range(count):
            x = ind['phenotype'][i*gene_bits:(i+1)*gene_bits]
            y = ind['phenotype'][(i+1)*gene_bits:(i+1)*gene_bits+phenotype_bits-gene_bits]
            v = np.array([x,y])
            p = 0.
            for j in range(self.centers):
                covdet = np.linalg.det(self.cov[j]+np.eye(2)*0.0001)
                covinv = np.linalg.inv(self.cov[j]+np.eye(2)*0.0001)
                xdiff = (v - self.centers[j]).reshape(1,2)
                p += self.weights[j]*np.exp(-0.5*xdiff.dot(covinv).dot(xdiff.T))[0][0]
            props.append(p)
        return np.average(props)

fitnessLandscape = FitnessLandscape()


#endregion

#region geomoe factory
class VEEGenomeFactory:
    def __init__(self):
        self.root = None

    def create(self,popParam):
        return [init_genome]*popParam.size



genomeFactory = VEEGenomeFactory()
indType =  individualTypes.find('dict')
indType.genomeDecoder = genomeFactory.decode

def genome_distance(g1,g2):
    dis = 0
    for i in range(max(len(g1),len(g2))):
        if i < len(g1) and i < len(g2):
            dis += abs(int(g1[i]) - int(g2[i]))
        elif i >= len(g1):
            dis += g2[i] + 1
        elif i >= len(g2):
            dis += g1[i] + 1
    return dis
def phenome_spilt(p):
    gene_count = len(p) // (gene_bits+phenotype_bits)
    s1,s2=[],[]
    for i in range(gene_count):
        t = p[i * (gene_bits+phenotype_bits):(i + 1) * (gene_bits+phenotype_bits)]
        s1.append(p[:gene_bits])
        s2.append(p[gene_bits:])
    return s1,s2
def phenome_distance(p1,p2):
    p11,p12 = phenome_spilt(p1)
    p21,p22 = phenome_spilt(p2)
    return np.sqrt(np.power(genome_distance(p11,p21),2)+np.power(p12-p22))

def genome_mutate(g):
    index = np.random.uniform(low=0,high=len(g))
    oper = np.random.uniform(low=1,hight=3)
    v = int(g[index])
    if v == 0:v += 1
    elif v == 9:v -=1
    elif oper == 1:v += 1
    else: v -= 1
    newg = g
    newg[index] = str(v)
    return newg
def phenome_mutate(p):
    g,pt = phenome_spilt(p)
    newg = genome_mutate(g)
    postfix = np.random.uniform(low=0, high=99)
    if len(postfix) < 2: postfix = '0' + str(postfix)
    return newg + postfix

#endregion

#region init
popParam = evolution.createPopParam(size=10000,evaluators=fitnessLandscape.fitness,genomeFactory=VEEGenomeFactory())
runParam = evolution.createRunParam(maxIterCount=1000,operations='')

generation,max_generation = 0,10000
pop = Population.create(popParam)

class Node:
    def __init__(self,genome,level,inds,childs,parent):
        self.genome = genome
        self.level = level
        self.inds = inds
        self.childs = childs
        self.parent = parent
        self.features = {}

root = Node(init_genome,0,pop.inds,[],None)
activenodes = {root}

#endregion

# region evolution
min_fitness = 0.1
min_node_genome_dis = 4

while generation < max_generation:
    # 计算适应度,并对节点适应度从高到低排序
    totalfitness, totalind = 0.0, 0
    for node in activenodes:
        node_fitness = 0.
        for ind in node.inds:
            fitness = fitnessLandscape.fitness(ind)
            ind['fitness'] = fitness
            node_fitness += fitness
        node.inds = sorted(node.inds,lambda x,y:x['fitness'] > y['fitness'])
        node['features']['fitness'] = node_fitness/len(node.inds)
        totalfitness += node_fitness
        totalind += len(node.inds)
    averagefitness = totalfitness / totalind
    activenodes = sorted(activenodes,lambda x,y:x['features']['fitness']>y['features']['fitness'])

    # 过低适应度的淘汰
    remove_ind_count = 0
    for node in activenodes:
        ind_count = len(node.inds)
        node.inds = [ind for ind in node.inds if ind['fitness']>min_fitness]
        remove_ind_count += ind_count - len(node.inds)
    activenodes = [node for node in activenodes if len(node.inds)>0]
    print('%d代淘汰个体数=%d' % (generation,remove_ind_count))

    # 按比例计算每个节点是否需要增加个体
    t = 0.
    for node in activenodes:
        if node['features']['fitness']<averagefitness:continue
        node['features']['addcount'] = np.exp(node['features']['fitness']/averagefitness)
        t += node['features']['addcount']
    c = 0
    for node in activenodes:
        if node['features']['fitness']<averagefitness:continue
        node['features']['addcount'] = int(node['features']['addcount']/t)
        c += node['features']['addcount']
        if c>remove_ind_count:
            node['features']['addcount'] -= (c - remove_ind_count)

    # 执行个体变异过程
    for node in activenodes:
        if 'addcount' not in node['features']: continue
        if node['features']['addcount'] <= 0: continue
        p = [ind['fitness'] for ind in node.inds]
        p /= sum(p)
        inds = []
        for i in range(node['features']['addcount']+len(node.inds)):
            ind = np.random.choice(node.inds,size=1,p=p)
            genome = genome_mutate(ind.genome)
            if genome_distance(genome,node['genome']) < min_node_genome_dis:
                inds.append(Individual(genome,generation,genome,'dict',[ind.genome]))
            elif node.parent is not None and genome_distance(node.parent.genome,genome)<min_node_genome_dis:
                node.parent.inds.append(Individual(genome,generation,genome,'dict',[ind.genome]))
                if node.parent not in activenodes:activenodes.append(node.parent)
            else:
                ind = Individual(genome,generation,genome,'dict',[ind.genome])
                c = [node for node in node.childs if genome_distance(genome,node['genome'])<min_node_genome_dis]
                c = Node(genome,node.level+1,[ind],[],node) if len(c)<=0 else c[0]
                if c not in node.childs:node.childs.append(c)
                if c not in activenodes:activenodes.append(c)
        node.inds = inds

    # 计算每个节点的群落多样性和遗传多样性
    for node in activenodes:
        pass


def callback(event,monitor):
    pass

evolutionTask = EvolutionTask(1,popParam,callback)
evolutionTask.execute(runParam)


class Selection:
    # selection:eliminate individuals whose fitness is below the threshold
    def execute(self, session):
        pop = session.pop

class Evolution:
    # evolution:choose an evolutionary branch
    def execute(self, session):
        pass