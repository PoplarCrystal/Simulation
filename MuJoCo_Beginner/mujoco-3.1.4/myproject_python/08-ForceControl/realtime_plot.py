import numpy as np
import math
import pyqtgraph as pg
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import QTimer
import threading
import sys
import time


class Plot:
    def __init__(self):
        # 4. data init
        self.data1, self.new_data1 = [0.] * 1000, 0
        self.data2, self.new_data2 = [0.] * 1000, 0
        # 2. thread
        self.plot_job = threading.Thread(target=self.thread_job)
        self.plot_job.start()

    def thread_job(self):
        # 1. set the win of plot
        app = QApplication([])
        win = pg.GraphicsLayoutWidget(show=True)
        win.setWindowTitle('Data curve')
        win.resize(800, 500)
        # 2. curve init
        plot1 = win.addPlot(title="Position-X")
        plot1.showGrid(x=True, y=True)
        self.curve_1 = plot1.plot(pen='y')
        win.nextRow()
        plot2 = win.addPlot(title="Position-Z")
        plot2.showGrid(x=True, y=True)
        self.curve_2 = plot2.plot(pen='y')
        # 3. timer init
        timer = QTimer()
        # timer.timeout.connect(self.callback1)
        # timer.timeout.connect(self.callback2)
        timer.timeout.connect(self.callback)
        timer.start(100)
        # 4. start plot
        app.exec_()

    def callback(self):
        self.data1[:-1] = self.data1[1:]
        self.data1[-1] = self.new_data1
        self.curve_1.setData(self.data1)
        self.data2[:-1] = self.data2[1:]
        self.data2[-1] = self.new_data2
        self.curve_2.setData(self.data2)

    def callback1(self):
        self.data1[:-1] = self.data1[1:]
        self.data1[-1] = self.new_data1
        self.curve_1.setData(self.data1)

    def callback2(self):
        self.data2[:-1] = self.data2[1:]
        self.data2[-1] = self.new_data2
        self.curve_2.setData(self.data2)

    def main(self):
        while True:
            self.new_data1 = math.sin(2 * math.pi / 2.0 * time.time())
            self.new_data2 = math.cos(2 * math.pi / 2.0 * time.time())
            time.sleep(0.001)

    def update_data(self, new_data1, new_data2):
        self.new_data1 = new_data1
        self.new_data2 = new_data2


if __name__ == "__main__":
    plot = Plot()
    plot.main()
