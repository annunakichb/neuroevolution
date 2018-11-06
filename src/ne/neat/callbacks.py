
import utils.collections as collections
from brain.viewer import NetworkView

def neat_callback(event,monitor):
    # session结束的时候将精英个体的图形写入文件
    if event == 'session.end':
        filename = 'session'+monitor.evoTask.curSession.taskxh+".ind"
        eliest = monitor.evoTask.curSession.pop.eliest
        if collections.isEmpty(eliest):return

        viewer = NetworkView()
        collections.foreach(eliest,lambda ind : viewer.drawNet(filename=filename+ind.id+'.svg',view=False))

    return True
