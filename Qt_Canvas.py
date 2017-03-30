import matplotlib, random

matplotlib.use('Qt5Agg')
from PyQt5 import QtCore, QtWidgets
import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

class PlotCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=50):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        self.xMax = 50
        self.xn = [];
        self.yn = []
        self.init_fig()
        FigureCanvas.__init__(self, self.fig)
        self.setParent(parent)
        FigureCanvas.setSizePolicy(self,
                                   QtWidgets.QSizePolicy.Expanding,
                                   QtWidgets.QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)

    def init_fig(self):
        pass


class DynamicCanvas(PlotCanvas):
    def __init__(self, data, *args, **kwargs):
        PlotCanvas.__init__(self, *args, **kwargs)
        timer = QtCore.QTimer(self)
        timer.timeout.connect(self.update_figure)
        self.data = data
        self.xMax = 100
        self.xn = np.zeros(self.xMax)
        self.yn = np.zeros(self.xMax)
        timer.start()

    #     def begin(self):
    #         for n in range(0,len(self.data)-1):
    #             self.update_figure()
    #             time.sleep(0.1)

    def init_fig(self):
        self.axes.step(self.xn, self.yn, 'r')

    def scale_fig(self):
        self.xn = np.append(self.xn, self.xn[-1] + 1)
        self.yn = np.append(self.yn, self.data.pop(0))
        self.xn = np.delete(self.xn, 0)
        self.yn = np.delete(self.yn, 0)

    def update_figure(self):
        self.xn = np.append(self.xn, self.xn[-1] + 1)
        self.yn = np.append(self.yn, self.data.pop(0))
        self.xn = np.delete(self.xn, 0)
        self.yn = np.delete(self.yn, 0)
        #         self.scale_fig()
        if self.xn[-1] <= self.xMax:
            self.axes.cla()
            self.axes.step(self.xn, self.yn, 'r')
            self.fig.gca().set_xlim([0, self.xMax])
            self.fig.gca().set_ylim([0, max(self.data) + 1])
            self.draw()
        else:
            self.axes.cla()
            self.axes.step(self.xn, self.yn, 'r')
            self.fig.gca().set_xlim([self.xn[0], self.xn[-1] + 1])
            self.fig.gca().set_ylim([0, max(self.data) + 1])
            self.draw()

