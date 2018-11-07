import numpy as np
import matplotlib.pyplot as plt

import os
import sys
import csv
import pickle

from Interface import asynchronous_pool_order
from AnalysisUnit import AnalysisUnit
from colorama import Fore, Style


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
        # sort using the first column, ie. params
        output = output[output[:, 0].argsort()]
        self.extracted_data_cols = [
            self.param_name, 'Rpp', 'Mvolt', 'Fx', 'Fy', 'Fz', 'mx', 'my', 'mz', 'ax', 'ay', 'az']
        self.p_dict = {
            col: output[:, i] for i, col in enumerate(self.extracted_data_cols)
        }
        if self.dispersion:
            self.dispersion_module()
        else:
            self.resonance_peak_module()

    def dispersion_module(self):
        for i, vector_orientation in enumerate(['x', 'y', 'z']):
            fig = plt.figure()
            plt.plot(self.p_dict[self.param_name],
                     self.p_dict['F' + vector_orientation], 'o')
            fig.suptitle(f"Frequency {vector_orientation}", fontsize=12)
            plt.xlabel(self.param_name)
            plt.ylabel("Frequency")
            savename = f"{vector_orientation}_{self.param_name}_frequency_{self.start_time}_{self.stop_time}"

            savename = os.path.join(self.result_directory, savename)
            self.save_object(fig, savename)

            fig = plt.figure()
            plt.plot(self.p_dict[self.param_name],
                     self.p_dict['m' + vector_orientation], 'o')
            fig.suptitle(f"Average_m{vector_orientation}", fontsize=12)
            plt.xlabel(self.param_name)
            plt.ylabel(f"Average_m{vector_orientation}")
            savename = f"Average_m{vector_orientation}_{self.param_name}_{self.start_time}_{self.stop_time}"
            savename = os.path.join(self.result_directory, savename)
            self.save_object(fig, savename)

        res_savepoint = os.path.join(
            self.result_directory, "resonant_frequencies.csv")
        with open(res_savepoint, 'w') as f:
            writer = csv.writer(f, delimiter=',')
            save_cols = [self.param_name, 'Fx', 'Fy', 'Fz', 'ax', 'ay', 'az']
            writer.writerow(save_cols)
            writer.writerows(zip(*map(lambda x: self.p_dict[x], save_cols)))

    def resonance_peak_module(self):
        fig = plt.figure()
        plt.plot(self.p_dict[self.param_name], self.p_dict['Rpp'], 'o')
        fig.suptitle("R_pp(param)", fontsize=12)

        savename = "Rpp " + str(self.start_time) + " " + str(self.stop_time)
        savename = os.path.join(self.result_directory, savename)

        self.save_object(fig, savename)

        fig2 = plt.figure()
        plt.plot(self.p_dict[self.param_name], self.p_dict['Mvolt'], 'o')
        fig2.suptitle("Voltage(scale)", fontsize=12)

        savename = "Vol " + str(self.start_time) + " " + str(self.stop_time)
        savename = os.path.join(self.result_directory, savename)

        self.save_object(fig2, savename)

        res_savepoint = os.path.join(
            self.result_directory, "voltage_field_values.csv")
        with open(res_savepoint, 'w') as f:
            writer = csv.writer(f, delimiter=',')
            writer.writerows(zip(self.p_dict[self.param_name],
                                 self.p_dict['Mvolt']))

    def local_analysis(self, filename):
        param = self.extract_parameter_type(filename, self.param_name)
        # reads each .odt file and returns pandas DataFrame object
        try:
            df = self.pickle_load_procedure(filename)
        except AssertionError as e:
            print(
                f"{Fore.RED}An error occurred of type: {e} in file: {filename}{Style.RESET_ALL}")
            return [0, 0, param, 0, 0, 0]
        # performs specified data analysis
        try:
            savename = os.path.join(self.result_directory, str(param))
            return (param, *self.standard_fourier_analysis(df, savename))
        except ValueError as e:
            print(
                f"{Fore.RED}PROBLEM ENCOUNTERED IN {filename} of {e}{Style.RESET_ALL}")
            return [0, 0, param, 0, 0, 0]
        return [0, 0, param, 0, 0, 0]

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
            return data.loc[(data['Oxs_TimeDriver::Simulation time'] > start_time)]
        else:
            return data.loc[(data['Oxs_TimeDriver::Simulation time'] >= start_time) &
                            (data['Oxs_TimeDriver::Simulation time'] < stop_time)]

    def voltage_calculation(self, df_limited, frequency):
        avg_resistance = np.mean(
            df_limited['MF_Magnetoresistance::magnetoresistance'])
        power = 10e-6
        omega = 2 * np.pi * frequency
        phase = 0
        amplitude = np.sqrt(power / avg_resistance)
        current = amplitude * np.sin(omega * df_limited['Oxs_TimeDriver::Simulation time']
                                     + phase)
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

    def find_max_frequency(self, df, time_step=1e-11, cols=('Oxs_TimeDriver::mx',
                                                            'Oxs_TimeDriver::my',
                                                            'Oxs_TimeDriver::mz')):
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
        frequency_steps = np.fft.fftfreq(
            potential_fourier_data[0].size, d=time_step)
        max_freq_set = []
        for freq_data in potential_fourier_data:
            freq_data = abs(freq_data)
            max_val = 0
            max_freq = 0
            for frequency, amp in zip(frequency_steps, freq_data):
                if np.abs(amp) > max_val and frequency > 0:
                    max_val = amp
                    max_freq = frequency
            max_freq_set.append([max_freq / 1e9, max_val])
        # display Fourier
        return np.array(max_freq_set, dtype=np.float64)

    def extract_parameter_type(self, filename, parameter_name):
        base_param = filename.split(parameter_name + "_")
        param_value = float(base_param[-1].split(self.delimiter)[0])
        if self.extract_frequency:
            self.set_resonant_frequency(param_value)
        return param_value


if __name__ == "__main__":
    rf = ResonantFrequency("interface.json")
    rf.initialize_analysis()
