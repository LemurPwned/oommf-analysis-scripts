import numpy as np
import matplotlib.pyplot as plt
import os

from AnalysisUnit import AnalysisUnit
from Fourier import ResonantFrequency
from Interface import asynchronous_pool_order
from ParsingUtils import ParsingUtils

from matplotlib.widgets import Button
import matplotlib.animation as animation


class SpinWave(AnalysisUnit):
    def __init__(self, interface='interfaces/spin_wave.json',
                 defaults='interfaces/defaults/spin_defaults.json'):
        super().__init__(interface, defaults)
        self.set_parameters(**self.startup_dict)
        self.decode_args()
        if self.directory is None:
            raise ValueError("Invalid directory ")
        else:
            # prepare resulting directory
            self.result_directory = self.manage_directory(self.directory)
        self.required_extension = '.ovf'

    def decode_args(self):
        m_dict = {
            'mx': 0,
            'x': 0,
            'my': 1,
            'y': 1,
            'mz': 2,
            'z': 2
        }
        try:
            self.m_component = m_dict[self.component.lower()]
        except KeyError:
            raise ValueError("Invalid component.")
        if self.step < 1:
            raise ValueError("Argmuent step must be larger than 0")
        if self.write:
            Writer = animation.writers['ffmpeg']
            self.writer = Writer(
                fps=15, metadata=dict(artist='LemurPwned'), bitrate=1800)

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

        fig = plt.figure()
        if self.write:
            self.ax = fig.add_subplot(111)
        else:
            self.ax = plt.subplot2grid((5, 5), (0, 0), colspan=5,
                                       rowspan=3)  # axes_plot
            ax_bl = plt.subplot2grid(
                (5, 5), (4, 0), colspan=2, rowspan=1)  # axes_button_left
            ax_br = plt.subplot2grid((5, 5), (4, 3), colspan=2, rowspan=1)
            butt_l = Button(ax_bl, '\N{leftwards arrow}')  # or u'' on python 2
            butt_r = Button(ax_br, '\N{rightwards arrow}')

            butt_l.on_clicked(self.left_onclicked)
            butt_r.on_clicked(self.right_onclicked)
        m = vectors[0, 0, :, self.m_component]
        x_vals = [i*header['xstepsize'] for i in
                  range(int(header['xnodes']))]
        hpl = self.ax.plot(x_vals, m)[0]
        self.ax.hpl = hpl
        self.ax.set_xlabel('x [nm]')
        self.ax.set_ylabel(self.component)
        self.ax.set_title(
            f"{self.i}/{self.total_its}; POS z:{self.zshift} y:{self.yshift}")
        self.ax.axis([0, np.max(x_vals), -1.1, 1.1])

        if self.write:
            anim = animation.FuncAnimation(fig, self.replot_data, int(self.total_its/self.step),
                                           interval=50, blit=False)
            anim.save('spins.mp4', writer=self.writer)
            return

        plt.show()

    def replot_data(self, num=-1):
        '''replot data after button push, assumes constant data shape'''
        header, vectors = ParsingUtils.binary_format_reader(
            filename=self.file_names[self.i])
        # reshape
        vectors = vectors.reshape((int(header['znodes']),
                                   int(header['ynodes']),
                                   int(header['xnodes']), 3))
        m = vectors[self.zshift, self.yshift, :, self.m_component]
        self.ax.hpl.set_ydata(m)
        self.ax.set_title(
            f"{self.i}/{self.total_its}; POS z:{self.zshift} y:{self.yshift}")
        self.ax.get_figure().canvas.draw()
        if num != -1:
            if self.i + self.step < self.total_its:
                self.i += self.step
            return self.ax,

    def left_onclicked(self, event):
        '''try to decrement data index, replot if success'''
        if self.i - self.step > 0:
            self.i -= self.step
            self.replot_data()

    def right_onclicked(self, event):
        '''try to increment data index, replot if success'''
        if self.i + self.step < self.total_its:
            self.i += self.step
            self.replot_data()


if __name__ == '__main__':
    spw = SpinWave()
    spw.plot_waves()
