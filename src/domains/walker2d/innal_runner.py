
import brain
import ne.senal as ehnal
import evolution
from evolution.session import Session

def run():
    # ehnal网络初始化
    senal.ehnal_init()
    netdef = brain.createNetDef(neuronCounts=[30, 6])
    pop_param = evolution.createPopParam(size=50)
    run_param = evolution.createRunParam()

    session = Session()
    monitor = session.run()


