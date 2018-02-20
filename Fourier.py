import numpy as np
import matplotlib.pyplot as plt

import os
import sys
import csv
import pickle

from Interface import asynchronous_pool_order
from AnalysisUnit import AnalysisUnit


class ResonantFrequency(AnalysisUnit):
    def __init__(self, filename):
        super().__init__(filename)
        self.set_parameters(**self.startup_dict)
        if self.directory is None:
            raise ValueError("Invalid directory ")
        else:
            # prepare resulting directory
            self.result_directory = self.manage_directory(self.directory)
        print("FINAL VALUES: {}".format(self.startup_dict))
        self.param_sweep = None
        self.ordered_param_set = []
        self.png = ".png"

        self.os_type = "windows" if "win" in sys.platform else "linux"
        self.delimiter = "/" if self.os_type == "linux" else "\\"
        self.initialize_analysis()

    def initialize_analysis(self):
        """
        initializes analysis type specified in parameter file
        :return: None
        """
        # ask for all .odt files in the sub-roots of a specified directory
        file_names = self.search_directory_for_odt()
        self.set_parameters()

        output = asynchronous_pool_order(self.local_analysis, (), file_names)
        output = np.array(output)
        self.R_pp = output[:, 0]
        self.global_mean_voltages = output[:, 1]
        self.ordered_param_set = output[:, 2]
        self.global_frequency_set = output[:, 3:]

        if self.dispersion:
            self.dispersion_module()
        else:
            self.resonance_peak_module()

    def dispersion_module(self):
        for i, vector_orientation in enumerate(['mx', 'my', 'mz']):
            fig = plt.figure()
            plt.plot(self.ordered_param_set, self.global_frequency_set[:, i], 'o')
            fig.suptitle(vector_orientation, fontsize=12)
            savename = vector_orientation + str(self.start_time) + " " + \
                        str(self.stop_time)
            savename = os.path.join(self.result_directory, savename)
            self.save_object(fig, savename)
        res_savepoint = os.path.join(self.result_directory, "resonant_frequencies.csv")
        with open(res_savepoint, 'w') as f:
            writer = csv.writer(f, delimiter=',')
            writer.writerows(zip(self.ordered_param_set,
                                 self.global_frequency_set[:, 0:]))

    def resonance_peak_module(self):
        fig = plt.figure()
        plt.plot(self.ordered_param_set, self.R_pp, 'o')
        fig.suptitle("R_pp(param)", fontsize=12)

        savename = "Rpp " + str(self.start_time) + " " + str(self.stop_time)
        savename = os.path.join(self.result_directory, savename)

        self.save_object(fig, savename)

        fig2 = plt.figure()
        plt.plot(self.ordered_param_set, self.global_mean_voltages, 'o')
        fig2.suptitle("Voltage(scale)", fontsize=12)

        savename = "Vol " + str(self.start_time) + " " + str(self.stop_time)
        savename = os.path.join(self.result_directory, savename)

        self.save_object(fig2, savename)

        res_savepoint = os.path.join(self.result_directory, "voltage_field_values.csv")
        with open(res_savepoint, 'w') as f:
            writer = csv.writer(f, delimiter=',')
            writer.writerows(zip(self.ordered_param_set,
                                 self.global_mean_voltages))

    def local_analysis(self, filename):
        param = self.extract_parameter_type(filename, self.param_name)
        # reads each .odt file and returns pandas DataFrame object
        pickle_path = os.path.join(os.path.dirname(filename),
                                   os.path.basename(filename).replace(".odt", "stages.pkl"))
        if self.clear or (not os.path.isfile(pickle_path)):
            df, stages = self.read_directory_as_df_file(filename)
        else:
            # if found, load pickle
            with open(pickle_path, 'rb') as f:
                df = pickle.load(f)
        # performs specified data analysis
        try:
            shortened_df = self.cutout_sample(df, start_time=self.start_time, stop_time=self.stop_time)
            r_max = np.max(shortened_df['MF_Magnetoresistance::magnetoresistance'])
            r_min = np.min(shortened_df['MF_Magnetoresistance::magnetoresistance'])
            r_diff = r_max-r_min
            voltage, m_voltage = self.voltage_calculation(shortened_df, self.resonant_frequency)
            svname = os.path.join(self.result_directory, str(param))
            frequency_set = self.find_max_frequency(shortened_df, self.time_step, param=svname)
            mx, my, mz = frequency_set[:, 0]
        except (ValueError) as e:
            print(e)
            return [0, 0, param, 0, 0, 0]
        return r_diff, m_voltage, param, mx, my, mz

    def multiple_parameter_analysis(self, filename):
        params = self.extract_parameter_type_dual_params(filename)
        return params

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

    def voltage_calculation(self, df_limited, frequency):
        avg_resistance = np.mean(df_limited['MF_Magnetoresistance::magnetoresistance'])
        power = 10e-6
        omega = 2 * np.pi * frequency
        phase = 0
        amplitude = np.sqrt(power / avg_resistance)
        current = amplitude * np.sin(omega * df_limited['TimeDriver::Simulation time'] + phase)
        voltage = df_limited['MF_Magnetoresistance::magnetoresistance'] * current
        mean_voltage = np.mean(voltage)
        return voltage, mean_voltage

    def subplot_fourier(self, fourier_data, time_step=1e-11, titles=None, savename=None):
        """
        plots fourier data on stem graphs
        :param fourier_data: list of numpy arrays
        :param time_step: sampling step
        :param titles: used in graph legends
        :return:
        """
        s = fourier_data[0].size
        fourier_data = [fd[int(0.1*s):int(0.9*s)] for fd in fourier_data]
        fourier_data = [fd[::2] for fd in fourier_data]
        time_step *= 2
        frequency_step = np.fft.fftfreq(fourier_data[0].size, d=time_step)
        if titles is None:
            titles = ["none" for x in fourier_data]
        for fourier_piece, title in zip(fourier_data, titles):
            fig2 = plt.figure()
            fig2.suptitle("Fourier for {}".format(title))
            plt.stem(frequency_step, np.abs(fourier_piece))
            self.save_object(fig2, savename + "_" + title)

    def find_max_frequency(self, df, time_step=1e-11, cols=('TimeDriver::mx',
                                                            'TimeDriver::my',
                                                            'TimeDriver::mz'), param=None):
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
        if self.dispersion:
            self.subplot_fourier(potential_fourier_data, titles=('mx', 'my', 'mz'), savename=param)
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
            # print("MAX FREQ: {}, VALUE {}".format(max_freq / 1e9, max_val))
            max_freq_set.append([max_freq / 1e9, max_val])
        # display Fourier
        return np.array(max_freq_set, dtype=np.float64)

    def single_plot_columns(self, df, x_cols=('TimeDriver::Simulation time',
                                              'TimeDriver::Simulation time'),
                            y_cols=('TimeDriver::my',
                                    'TimeDriver::mz'),
                            param="Unknown"):
        """
        plots a simple column set from a data frame
        :param df: DataFrame object
        :param x_cols: x-columns from df to be plotted on x-axis
        :param y_cols: y-columns from df to be plotted on y=axis
        :param param: is sweep-type
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

    def two_parameter_relation(self, parameter1, parameter2, xticks=None,
                               title='Dispersion relation'):
        """
        Can build up a relation between a given parameter and parameter sweep
        specified in the parameter dict
        :param parameter1: numpy array of size parameter sweep
        :param parameter2: second numpy array of size parameter sweep
        :param xticks: set ticks
        :param title: title of graph
        :return: None
        """
        plt.plot(parameter1, parameter2, '*')
        if xticks is not None:
            plt.xticks(xticks)
        plt.title(title)
        plt.show()

    def extract_parameter_type(self, filename, parameter_name):
        base_param = filename.split(parameter_name + "_")
        param_value = float(base_param[-1].split(self.delimiter)[0])
        return param_value



if __name__ == "__main__":
    rf = ResonantFrequency("interface.json")
