import numpy as np
import matplotlib.pyplot as plt
import os

from AnalysisUnit import AnalysisUnit
from Fourier import ResonantFrequency
from Interface import asynchronous_pool_order
from ParsingUtils import ParsingUtils

from matplotlib.widgets import Button


class SpinWave(AnalysisUnit):
    def __init__(self, filename):
        super().__init__(filename)
        self.set_parameters(**self.startup_dict)
        if self.directory is None:
            raise ValueError("Invalid directory ")
        else:
            # prepare resulting directory
            self.result_directory = self.manage_directory(self.directory)
        self.required_extension = '.ovf'

    def plot_waves(self):
        self.file_names = list(map(lambda x: os.path.join(self.directory, x),
                                   sorted(filter(lambda x: x.endswith('.ovf'),
                                                 os.listdir(self.directory)))))
        self.total_its = len(self.file_names)
        self.i = 0
        header, vectors = ParsingUtils.binary_format_reader(
            filename=self.file_names[self.i])
        # reshape
        vectors = vectors.reshape((int(header['znodes']),
                                   int(header['ynodes']),
                                   int(header['xnodes']), 3))
        print(vectors.shape)

        fig = plt.figure()
        self.ax = plt.subplot2grid((5, 5), (0, 0), colspan=5,
                                   rowspan=3)  # axes_plot
        ax_bl = plt.subplot2grid(
            (5, 5), (4, 0), colspan=2, rowspan=1)  # axes_button_left
        ax_br = plt.subplot2grid((5, 5), (4, 3), colspan=2, rowspan=1)

        butt_l = Button(ax_bl, '\N{leftwards arrow}')  # or u'' on python 2
        butt_r = Button(ax_br, '\N{rightwards arrow}')

        butt_l.on_clicked(self.left_onclicked)
        butt_r.on_clicked(self.right_onclicked)

        mx = vectors[0, 0, :, 0]
        x_vals = [i*header['xstepsize'] for i in
                  range(int(header['xnodes']))]
        hpl = self.ax.plot(x_vals, mx)[0]
        self.ax.hpl = hpl
        self.ax.set_xlabel('x [nm]')
        self.ax.set_ylabel('Mx')
        self.ax.set_title(f"{self.i}/{self.total_its}")
        self.ax.axis([0, np.max(x_vals), -1, 1])
        plt.show()

    def replot_data(self):
        '''replot data after button push, assumes constant data shape'''
        header, vectors = ParsingUtils.binary_format_reader(
            filename=self.file_names[self.i])
        # reshape
        print(self.ax.hpl)
        vectors = vectors.reshape((int(header['znodes']),
                                   int(header['ynodes']),
                                   int(header['xnodes']), 3))
        mx = vectors[0, 0, :, 0]
        self.ax.hpl.set_ydata(mx)
        self.ax.set_title(f"{self.i}/{self.total_its}")
        self.ax.get_figure().canvas.draw()

    def left_onclicked(self, event):
        '''try to decrement data index, replot if success'''
        if self.i - 100 > 0:
            self.i -= 100
            self.replot_data()

    def right_onclicked(self, event):
        '''try to increment data index, replot if success'''
        if self.i + 100 < self.total_its:
            self.i += 100
            self.replot_data()


if __name__ == '__main__':
    spw = SpinWave('interface.json')
    spw.plot_waves()
