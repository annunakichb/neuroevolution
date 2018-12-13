#!/usr/bin/python3

'''
EHNAL
Evolutionary Heterogeneous Neural Network Based on Attention Logic
一 基本概念
1. 变量：
1.1 定义：
    变量是突触和神经元中定义的一组可变值，并在突触或神经元计算中使用。
    如果变量值是系统配置中设置的，在运行时不再改变，则称为参数变量。
    如果变量值是在系统运行中按照一定的学习规则改变的，则称为学习变量。
1.2 变量由id，元信息，值，取值方式和范围（方式包括固定取值，随机取值(高斯分布，均匀分布)）五部分组成
    变量元信息包括名称，类型，缺省值，缺省取值方式和范围

二. 突触
2.1 突触的构成：每个突触由上游神经元(只能一个)，下游神经元（只能一个），一组状态，一组学习变量和参数变量，一个计算模型组成
2.2 突触维持的内部状态包括：输入状态值，计算状态值，状态时间。当连接上游神经元激活，突触状态值才有效，状态时间小于0表示突触状态值无效
2.3 突触内部变量包括学习变量和参数变量
A 权重：既可能是学习变量，也可能是参数变量
B 延时：既可能是学习变量，也可能是参数变量
C 抑制: 参数变量，取值非0表示该突触为抑制型突触
D 强度：学习变量，取值0-10之间，与权重不同，强度是指该刺激输入期望被关注的程度，其含义由接收神经元的计算模型决定
E 权重可学习：参数变量，取值0表示权重是参数变量，不是学习变量
F 延时可学习：参数变量，取值0则表示延时是参数变量，不是学习变量
E 计算模型id：参数变量，决定使用的是哪个计算模型
2.4 计算模型：
A  计算状态值 = 权重 * 输入状态值
   状态时间 = 输入状态时间 + 延时
2.5 学习模型：
2.5.1 权重学习模型：BP,遗传算法，强化学习
2.5.2 延时学习模型：？
2.5.3 强度学习模型：？

三 神经元
3.1 神经元的构成：多个突触组或一个突触集，一组状态，一组学习变量和参数变量，一个计算模型组成。其中一个突触组表示一个树突分支(一种简化的房室模型)
3.2 神经元的状态包括：值，激活频率，激活状态，激活时间。
3.3 神经元的变量根据计算模型的不同而不同，可能的计算模型：
3.3.1 感知机模型:
3.3.1.1 变量：
        A  激活函数：可以是gauss，S函数，Step函数等
        B  激活函数变量：不同激活函数有不同的变量，有些变量是学习变量，有些变量是参数变量，例如gauss激活函数中的变量可以作为学习变量
3.3.1.2 计算方式：
        状态值 = 所有输入突触状态值求和并计算激活函数
3.3.1.3 学习变量学习方式：BP，遗传算法，强化学习
3.3.2 关注逻辑计算模型：
3.3.2.1 变量：
       A 关注变量名：学习变量，它指神经元关注输入的哪个特征变量，可以是时间，值，频率，激活状态四种
       B 关注逻辑表达式：以下四种类型中的一个
                       阈值逻辑：si.value in [v11,v12] and|or sj.value in [v21.v22] ..... ，and和or只能取一个
                       高斯逻辑：gauss[u,sigma](si.value,sj.value,....)，高斯逻辑的输出作为激活概率
                       时间逻辑：si.time-sj.time <=t1 and sj.time - sk.time < t2
                       频率逻辑：si.frenccy > [$v1] or sj.frecny > [$v2]
                       激活逻辑：si.activation and sj.activation....  and和or只能取一个
         逻辑表达式的设定有三种方式：预先设定：在神经元创建的时候人为设定好
                                 生成设定：在神经元第一次进入网络，由输入信号的特征决定
                                 统计设定：在一段运行时间内，由输入的统计特征决定
       C 关注逻辑变量：学习变量，以上各种逻辑中的变量都属于学习变量
3.3.2.2 学习变量的学习方式：学习是为了寻找对完成任务有帮助的关注模式
3.3.2.2.1 遗传算法
3.3.2.2.2 协作学习：
3.3.2.2.2.1 以最小的能量消耗，学习到尽可能多的特征。

                      (s1.value in [v11,v12] and s2.value in [v21,v22]) or (s1.time - s2.time < -1)



B.突触计算模型：
B.1 变量['权重'] * 状态值
C 内部变量：
C.1 学习变量：权重，延时，强度
C.2 参数变量：权重，延时，权重变量是否可学习，延时变量是否可学习
D 学习模型：
D.1 权重变量学习模型：BP，随机进化，强化学习方法
D.2 延时变量学习模型：随机进化
# 神经元维持一组内部状态，包括当前值，激活频率，激活频率形式，激活状态，激活时间
# 神经元输入：突触状态值和突触时间
'''
