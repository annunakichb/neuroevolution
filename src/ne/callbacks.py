
import utils.collections as collections
from brain.viewer import NetworkView
from evolution.viewer import EvolutionViewer


__all__ = ['neat_callback']
def neat_callback(event,monitor):
    # session结束的时候将精英个体的图形写入文件
    if event == 'session.end':
        filename = 'session'+monitor.evoTask.curSession.taskxh+".ind"
        eliest = monitor.evoTask.curSession.pop.eliest
        if collections.isEmpty(eliest):return

        netviewer = NetworkView()
        collections.foreach(eliest,lambda ind : netviewer.drawNet(filename=filename+ind.id+'.svg',view=False))

        evolutionView = EvolutionViewer()
        evolutionView.drawSession(monitor,monitor.evoTask.curSession,'fitness')

    return True
