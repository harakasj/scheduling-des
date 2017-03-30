from __future__ import unicode_literals
import os, sys
from PyQt5 import QtCore, QtWidgets
from Qt_Canvas import DynamicCanvas

progname = os.path.basename(sys.argv[0])
progversion = "0.1"

'''
This runs like total SHIT because the plot should update on a timeout 
event from the Env() class, but I didn't like it properly, so instead it runs
choppy updating everytime it gets new data, and blocks the simulation from
continuing, making it even slower.
'''


class ApplicationWindow(QtWidgets.QMainWindow):
    def __init__(self, data1, data2):
        QtWidgets.QMainWindow.__init__(self)
        self.resize(600, 400)
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        self.setWindowTitle("Graph")
        self.file_menu = QtWidgets.QMenu('&File', self)
        self.file_menu.addAction('&Quit', self.fileQuit,
                                 QtCore.Qt.CTRL + QtCore.Qt.Key_Q)
        self.menuBar().addMenu(self.file_menu)
        self.help_menu = QtWidgets.QMenu('&Help', self)
        self.menuBar().addSeparator()
        self.menuBar().addMenu(self.help_menu)
        self.main_widget = QtWidgets.QWidget(self)
        l = QtWidgets.QVBoxLayout(self.main_widget)
        self.canvas1 = DynamicCanvas(data1, self.main_widget, width=5, height=4, dpi=100)
        self.canvas2 = DynamicCanvas(data2, self.main_widget, width=5, height=4, dpi=100)
        l.addWidget(self.canvas1)
        l.addWidget(self.canvas2)
        self.main_widget.setFocus()
        self.setCentralWidget(self.main_widget)
        self.statusBar().showMessage("", 2000)

    def fileQuit(self): self.close()

    def closeEvent(self, se): self.fileQuit()

# def data_loop(callback):
#     env = Env()
#     env.data_signal.connect( callback )
#     env.start()
#
# qApp = QtWidgets.QApplication(sys.argv)
# aw = ApplicationWindow()
# aw.setWindowTitle("RR Sim")
# aw.show()
# sys.exit(qApp.exec_())
#






