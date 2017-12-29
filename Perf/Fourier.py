import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
import os

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
        directory_roots = os.path.join(self.directory, '*\*.odt')
        filename_candidates = glob.glob(directory_roots, recursive=True)
        print("{} file candidates found...".format(len(filename_candidates)))
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
        global_resonant_frequencies = []
        for filename in file_names:
            if self.dispersion: self.extract_parameter_type(filename, self.param_name)
            # reads each .odt file and returns pandas DataFrame object
            df, stages = self.read_directory_as_df_file(filename)
            # performs specified data analysis
            # self.single_plot_columns(df)
            shortened_df = self.cutout_sample(df, start_time=self.start_time, stop_time=self.stop_time)
            single_param_resonant_frequencies = self.fourier_analysis(data_frame=shortened_df)
            # voltage, m_voltage = self.voltage_calculation(shortened_df)
            global_resonant_frequencies.append(single_param_resonant_frequencies)
        global_resonant_frequencies = np.array(global_resonant_frequencies)
        self.ordered_param_set = np.array(self.ordered_param_set)
        print(global_resonant_frequencies.shape, self.ordered_param_set.shape)
        print(self.ordered_param_set)
        if self.dispersion:
            x_vals = self.ordered_param_set
            mx_relation = global_resonant_frequencies[:, 0, 0]  # takes resonant frequency
            my_relation = global_resonant_frequencies[:, 1, 0]
            mz_relation = global_resonant_frequencies[:, 2, 0]
            self.two_parameter_relation(x_vals, mx_relation, title='mx relation', xticks=x_vals[::3])
            self.two_parameter_relation(x_vals, my_relation, title='my relation', xticks=x_vals[::3])
            self.two_parameter_relation(x_vals, mz_relation, title='mz relation', xticks=x_vals[::3])
            mx_ampl_val = global_resonant_frequencies[:, 0, 1]  # takes resonant frequency amplitude value
            my_ampl_val = global_resonant_frequencies[:, 1, 1]
            mz_ampl_val = global_resonant_frequencies[:, 2, 1]
            self.two_parameter_relation(x_vals, mx_ampl_val, title='mx amplitude (coupl)', xticks=x_vals[::3])
            self.two_parameter_relation(x_vals, my_ampl_val, title='my amplitude (coupl)', xticks=x_vals[::3])
            self.two_parameter_relation(x_vals, mz_ampl_val, title='mz amplitude (coupl)', xticks=x_vals[::3])

            self.two_parameter_relation(mx_relation, mx_ampl_val, title='Resonance x')
            self.two_parameter_relation(my_relation, my_ampl_val, title='Resonance y')
            self.two_parameter_relation(mz_relation, mz_ampl_val, title='Resonance z')

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
            print("Start time {}, stop time {}".format(start_time, stop_time))
            return data.loc[(data['TimeDriver::Simulation time'] >= start_time) &
                            (data['TimeDriver::Simulation time'] < stop_time)]

    def voltage_calculation(self, df_limited):
        avg_resistance = np.mean(df_limited['MF_Magnetoresistance::magnetoresistance'])
        power = 10e-6
        frequency = 20e9
        omega = 2 * np.pi * frequency
        phase = 0
        amplitude = np.sqrt(power / avg_resistance)
        current = amplitude * np.sin(omega * df_limited['TimeDriver::Simulation time'] + phase)
        voltage = df_limited['MF_Magnetoresistance::magnetoresistance'] * current
        mean_voltage = np.mean(voltage)
        print("MEAN VOLTAGE {}".format(mean_voltage))
        plt.plot(df_limited['TimeDriver::Simulation time'], voltage)
        plt.plot(df_limited['TimeDriver::Simulation time'],
                 np.ones(df_limited['TimeDriver::Simulation time'].shape[0])*mean_voltage)
        plt.show()
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
                                                            'TimeDriv'
                                                            'er::mz')):
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
                if amp > max_val and frequency > 0:
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
                                    'TimeDriver::mz')):
        """
        plots a simple column set from a dataframe
        :param df: DataFrame object
        :param x_cols: x-columns from df to be plotted on x-axis
        :param y_cols: y-columns from df to be plotted on y=axis
        :return: None
        """
        handles = []
        df = self.cutout_sample(df, start_time=4.9e-9, stop_time=5.6e-9)
        for x_column, y_column in zip(x_cols, y_cols):
            ax, = plt.plot(df[x_column], df[y_column], label=y_column)
            handles.append(ax)
        plt.legend(handles=handles)
        plt.title("{} vs {}".format(x_cols[0], y_cols[0]))
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

    def extract_parameter_type(self, filename, parameter_name):
        base_param = filename.split(parameter_name + "_")
        param_value = float(base_param[1].split("\\")[0])
        print("ANALYSED PARAM: ", param_value)
        self.ordered_param_set.append(param_value)


if __name__ == "__main__":
    path = r"D:\Dokumenty\oommf-simulations\REZ\rez_minus1e3\Default\AFCoupFieldDomain.odt"
    p_dir = r'D:\Dokumenty\oommf-simulations\dispersion_nblc'
    # p_dir = r"D:\Dokumenty\oommf-simulations\voltage_sweep_50ns"
    rf = ResonantFrequency(directory=p_dir)
    param_sweep = np.array([-1e-4, -2e-4, -3e-4, 4e-4, -5e-4, -6e-3, -7e-4, -8e-4, -9e-4, -1e-3, 0,
                            2e-4, 3e-4, 4e-4, 5e-4, 6e-4, 7e-4,
                            8e-4, 9e-4, 1e-3])
    parameter_dict = {
        "time_step": 1e-11,
        "start_time": 5.2e-9,
        "stop_time": 10e-9,
        "param_sweep": param_sweep,
        "dispersion": True,
        "param_name": 'Amp'
    }
    rf.set_parameters(**parameter_dict)
    rf.initialize_analysis()
