

####################################################
# 机械手学习推动一个箱子一端距离后松手，让其滑动到指定目标    #
# 地面摩擦力和推动距离固定但是未知                       #
# 以箱子初始位置为原点，箱子移动方向为坐标轴正向            #
# 网络输入为箱子位置（只有初始和最终停止），目标位置，力大小  #
# 网络输出为力大小                                    #
####################################################
from ne.senal.box import BoxGene
import brain
import evolution
from evolution.session import Session

netdef = brain.createNetDef()
netdef.inputboxs = [
    {'expression':'f','initsize':1,'group':'body.hand.output.f','range':[0.,1.],'attributes':{}},  # 对力量输出的感知
    {'expression':'s','initsize':1,'group':'env.s','attributes':{}},   # 对箱子位置的感知
    {'expression':'o','initsize':1,'group':'env.o','attributes':{}}    # 对目标位置的感知
    ]
netdef.outputboxs = [{'expression':'f_','initsize':1,'group':'hand.f_','attributes':{}}]

popParam = evolution.createPopParam(size=100)
runParam = evolution.createRunParam()

session = Session()
monitor = session.run()










