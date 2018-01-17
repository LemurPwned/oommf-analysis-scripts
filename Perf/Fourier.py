import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
import os

from multiprocessing import Pool
from Interface import asynchronous_pool_order

class ResonantFrequency:
    def __init__(self, directory):
        self.directory = directory
        self.start_time = 0
        self.stop_time = 0
        self.analysis_method = self.fourier_analysis
        self.time_step = 1e-11
        self.param_sweep = None
        self.dispersion = False
        self.ordered_param_set = []
        self.png = ".png"
        
    def set_parameters(self, **kwargs):
        """
        :param: **kwargs are the arguments to be passed to the main widget
        iterator
        """
        for k, v in kwargs.items():
            setattr(self, k, v)

    def search_directory_for_odt(self):
        """
        finds the possible .odt files in the specified directory
        :return: None
        """
        directory_roots = os.path.join(self.directory, '*/*.odt')
        filename_candidates = glob.glob(directory_roots, recursive=True)
        print("{} file candidates found...".format(len(filename_candidates)))
        if len(filename_candidates) == 0:
            quit()
        for filename in filename_candidates:
            print(filename)
        return filename_candidates

    def fourier_analysis(self, data_frame):
        """
        Analysis type: Discrete Fourier Transform
        :param data_frame: DataFrame object
        :return: set of resonant frequencies and their values
        """
        print("FOURIER ANALYSIS INITIATED")
        return self.find_max_frequency(df=data_frame, time_step=self.time_step)

    def initialize_analysis(self):
        """
        initializes analysis type specified in parameter file
        :return: None
        """
        # ask for all .odt files in the sub-roots of a specified directory
        file_names = self.search_directory_for_odt()
        self.set_parameters()
        global_mean_voltages = []
        R_pp = []

        output = asynchronous_pool_order(self.local_analysis, (), file_names)
        output = np.array(output)
        R_pp = output[:, 0]
        global_mean_voltages = output[:, 1]
        self.ordered_param_set = output[:, 2]
        self.ordered_param_set = np.array(self.ordered_param_set)
        R_pp = np.array(R_pp)

        fig = plt.figure()
        plt.plot(self.ordered_param_set, R_pp, 'o')
        fig.suptitle("R_pp(param)", fontsize=12)
        fig.savefig("Rpp " + str(self.start_time) + " " + str(self.stop_time)
                        +self.png)
        fig2 = plt.figure()
        plt.plot(self.ordered_param_set, global_mean_voltages, 'o')
        fig2.suptitle("Voltage(scale)", fontsize=12)
        fig2.savefig("Vol " + str(self.start_time) + " " + str(self.stop_time)
                    + self.png)

    def local_analysis(self, filename):
        param = self.extract_parameter_type(filename, self.param_name)
        # reads each .odt file and returns pandas DataFrame object
        df, stages = self.read_directory_as_df_file(filename)
        # performs specified data analysis
        shortened_df = self.cutout_sample(df, start_time=self.start_time, stop_time=self.stop_time)
        rmax = np.max(shortened_df['MF_Magnetoresistance::magnetoresistance'])
        rmin = np.min(shortened_df['MF_Magnetoresistance::magnetoresistance'])
        rdiff = rmax-rmin
        voltage, m_voltage = self.voltage_calculation(shortened_df)
        return rdiff, m_voltage, param

    def read_directory_as_df_file(self, filename):
        """
        Reads .odt file
        :param: filename is .odt file path
        :return: DataFrame and stages number
        """
        if filename is None:
            print("\nOdt file has not been found")
            return
        if not filename.endswith(".odt"):
            print("\nWrong file type passed, only .odt")
            return
        else:
            header_lines = 4
            header = []
            i = 0
            with open(filename, 'r') as f:
                while i < header_lines:
                    lines = f.readline()
                    header.append(lines)
                    i += 1
                lines = f.readlines()
            f.close()
            cols = header[-1]
            cols = cols.replace("} ", "")
            cols = cols.replace("{", "")
            cols = cols.replace("MF", "Oxs_MF")
            cols = cols.split("Oxs_")
            del cols[0]
            cols = [x.strip() for x in cols]
            cols = [x.replace("}", "") for x in cols]
            dataset = []
            lines = [x.strip() for x in lines]
            lines = [x.split(' ') for x in lines]
            for line in lines:
                temp_line = []
                for el in line:
                    try:
                        new_el = float(el)
                        temp_line.append(new_el)
                    except:
                        pass
                temp_line = np.array(temp_line, dtype=np.float32)
                if temp_line.shape[0] == 0:
                    continue
                dataset.append(temp_line)

            dataset = np.array(dataset[1:])
            df = pd.DataFrame.from_records(dataset, columns=cols)
            stages = len(lines) - 1
            return df, stages

    def cutout_sample(self, data, start_time=0.00, stop_time=100.00):
        """
        cuts out time interval from sample
        :param data: DataFrame object, must contain time column
        :param start_time: start time = here cutout begins
        :param stop_time:  stop time = here cutout ends
        :return: sliced DataFrame object
        """
        if start_time is None:
            return data
        if stop_time is None:
            return data.loc[(data['TimeDriver::Simulation time'] > start_time)]
        else:
            # print("Start time {}, stop time {}".format(start_time, stop_time))
            return data.loc[(data['TimeDriver::Simulation time'] >= start_time) &
                            (data['TimeDriver::Simulation time'] < stop_time)]

    def voltage_calculation(self, df_limited, frequency=15.31e9):
        avg_resistance = np.mean(df_limited['MF_Magnetoresistance::magnetoresistance'])
        power = 10e-6
        omega = 2 * np.pi * frequency
        phase = 0
        amplitude = np.sqrt(power / avg_resistance)
        current = amplitude * np.sin(omega * df_limited['TimeDriver::Simulation time'] + phase)
        voltage = df_limited['MF_Magnetoresistance::magnetoresistance'] * current
        mean_voltage = np.mean(voltage)
        return voltage, mean_voltage

    def subplot_fourier(self, fourier_data, time_step=1e-11, titles=None):
        """
        plots fourier data on stem graphs
        :param fourier_data: list of numpy arrays
        :param time_step: sampling step
        :param titles: used in graph legends
        :return:
        """
        frequency_step = np.fft.fftfreq(fourier_data[0].size, d=time_step)
        number = len(fourier_data) * 100 + 10
        if titles is None:
            titles = ["none" for x in fourier_data]
        for fourier_piece, title in zip(fourier_data, titles):
            number += 1
            plt.subplot(number)
            plt.stem(frequency_step, np.abs(fourier_piece))
            plt.title(title)
        plt.show()

    def find_max_frequency(self, df, time_step=1e-11, cols=('TimeDriver::mx',
                                                            'TimeDriver::my',
                                                            'TimeDriver::mz')):
        """
        Values given in columns must have common sampling frequency
        :param df: DataFrame object containing columns specified in cols
        :param time_step: sampling step
        :param cols: columns for which fourier transform is to be performed
        :return: numpy array of maximum frequencies and respective values
        """
        potential_fourier_data = []
        for col in cols:
            potential_fourier_data.append(np.fft.fft(df[col], axis=0))
        # fourier frequencies must be calculated first to know precise frequency
        frequency_steps = np.fft.fftfreq(potential_fourier_data[0].size, d=time_step)
        max_freq_set = []
        for freq_data in potential_fourier_data:
            freq_data = abs(freq_data)
            max_val = 0
            max_freq = 0
            for frequency, amp in zip(frequency_steps, freq_data):
                if np.abs(amp) > max_val and frequency > 0:
                    max_val = amp
                    max_freq = frequency
            print("MAX FREQ: {}, VALUE {}".format(max_freq / 1e9, max_val))
            max_freq_set.append([max_freq / 1e9, max_val])
        # display Fourier
        # self.subplot_fourier(potential_fourier_data, time_step=time_step, titles=cols)
        return np.array(max_freq_set, dtype=np.float64)

    def single_plot_columns(self, df, x_cols=('TimeDriver::Simulation time',
                                              'TimeDriver::Simulation time'),
                            y_cols=('TimeDriver::my',
                                    'TimeDriver::mz'), param="Unknown"):
        """
        plots a simple column set from a dataframe
        :param df: DataFrame object
        :param x_cols: x-columns from df to be plotted on x-axis
        :param y_cols: y-columns from df to be plotted on y=axis
        :return: None
        """
        handles = []
        df = self.cutout_sample(df, start_time=0, stop_time=5.0e9)
        for x_column, y_column in zip(x_cols, y_cols):
            ax, = plt.plot(df[x_column], df[y_column], label=y_column)
            handles.append(ax)
        plt.legend(handles=handles)
        plt.title("{} vs {} for param: {}".format(x_cols[0], y_cols[0], param))
        plt.show()

    def two_parameter_relation(self, parameter1, parameter2, xticks=None, title='Dispersion relation'):
        """
        Can build up a relation between a given parameter and parameter sweep
        specified in the parameter dict
        :param parameter: numpy array of size parameter sweep
        :param title: title of graph
        :return: None
        """
        plt.plot(parameter1, parameter2, '*')
        if xticks is not None:
            plt.xticks(xticks)
        plt.title(title)
        plt.show()

    def extract_parameter_type(self, filename, parameter_name, old_type=False):
        if old_type:
            base_param = filename.split("/")
            param_value = float(base_param[-2])
            return param_value
        else:
            base_param = filename.split(parameter_name + "_")
            param_value = float(base_param[-1].split("/")[0])
            print("ANALYSED PARAM: ", param_value)
            return param_value


if __name__ == "__main__":
    p_dir = r'/home/lemurpwned/Simulations/vsd_56_56_sweep_larger_saving'
    # p_dir = r"D:\Dokumenty\oommf-simulations\coupling_1em4"
    rf = ResonantFrequency(directory=p_dir)
    for param in [(3.1e-9, 59e-9), (6.1e-9, 59e-9), (9.1e-9, 49e-9),
                 (25.3e-9, 89e-9), (12.1e-9, 59e-9), (3.2e-9, 59e-9),
                 (3.2e-9, 58e-9), (3.2e-9, 53e-9)]:
        parameter_dict = {
            "time_step": 1e-11,
            "start_time": param[0],
            "stop_time": param[1],
            "dispersion": True,
            "param_name": 'scale'
        }
        rf.set_parameters(**parameter_dict)
        rf.initialize_analysis()
