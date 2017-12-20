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

    def set_parameters(self, **kwargs):
        """
        @param **kwargs are the arguments to be passed to the main widget
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
        print("FOURIER ANALYSIS INITIATED")
        return self.find_max_frequency(df=data_frame, time_step=self.time_step)

    def initialize_analysis(self):
        # ask for all .odt files in the sub-roots of a specified directory
        file_names = self.search_directory_for_odt()
        self.set_parameters()
        global_resonant_frequencies = []
        for filename in file_names:
            # reads each .odt file and returns pandas DataFrame object
            df, stages = self.read_directory_as_df_file(filename)
            # performs specified data analysis
            self.single_plot_columns(df)
            shortened_df = self.cutout_sample(df, start_time=self.start_time, stop_time=self.stop_time)
            single_param_resonant_frequencies = self.fourier_analysis(data_frame=shortened_df)
            global_resonant_frequencies.append(single_param_resonant_frequencies)
        global_resonant_frequencies = np.array(global_resonant_frequencies)
        mx_relation = global_resonant_frequencies[:, 0, 0]
        my_relation = global_resonant_frequencies[:, 1, 0]
        mz_relation = global_resonant_frequencies[:, 2, 0]
        self.two_parameter_relation(mx_relation, title='mx relation')
        self.two_parameter_relation(my_relation, title='my relation')
        self.two_parameter_relation(mz_relation, title='mz relation')

    def read_directory_as_df_file(self, filename):
        """
        Reads .odt file
        @param filename is .odt file path
        @return dataFrame and stages number
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

    def cutout_sample(self, data, start_time=None, stop_time=None):
        if start_time is None:
            return data
        if stop_time is None:
            return data.loc[(data['TimeDriver::Simulation time'] > start_time)]
        else:
            return data.loc[(data['TimeDriver::Simulation time'] > start_time) &
                            (data['TimeDriver::Simulation time']) < stop_time]

    def voltage_calculation(self, df, time_offset, time_stop=10e9):
        df_limited = df.loc[(df['TimeDriver::Simulation time'] > time_offset) &
                            (df['TimeDriver::Simulation time'] < time_stop)]
        avg_resistance = np.mean(df_limited['MF_Magnetoresistance::magnetoresistance'])
        power = 10e-6
        frequency = 1e8
        omega = 2*np.pi*frequency
        phase = 0
        amplitude = np.sqrt(power/avg_resistance)
        current = amplitude*np.sin(omega*df_limited['TimeDriver::Simulation time'] + phase)
        voltage = df_limited['MF_Magnetoresistance::magnetoresistance']*current
        mean_voltage = np.mean(voltage)
        print("MEAN VOLTAGE {}".format(mean_voltage))
        return voltage, mean_voltage

    def subplot_fourier(self, fourier_data, time_step=1e-11, titles=None):
        frequency_step = np.fft.fftfreq(fourier_data[0].size, d=time_step)
        number = len(fourier_data)*100 + 10
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
        :param df:
        :param time_step:
        :param cols:
        :return:
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
            max_freq_set.append([max_freq/1e9, max_val])
        # display Fourier
        self.subplot_fourier(potential_fourier_data, time_step=time_step, titles=cols)
        return np.array(max_freq_set, dtype=np.float64)

    def single_plot_columns(self, df, cols=('TimeDriver::mx',
                                            'TimeDriver::my',
                                            'TimeDriver::mz')):
        handles = []
        for column in cols:
            ax, = plt.plot(df[column], label=column)
            handles.append(ax)
        plt.legend(handles=handles)
        plt.show()

    def two_parameter_relation(self, parameter, title='Dispersion relation'):
        plt.plot(parameter, self.param_sweep, '*')
        plt.title(title)
        plt.show()


if __name__ == "__main__":
    path = r"D:\Dokumenty\oommf-simulations\REZ\rez_minus1e3\Default\AFCoupFieldDomain.odt"
    # directory = r'D:\Dokumenty\oommf-simulations\REZ\10nsResonance'
    p_dir = r"D:\Dokumenty\oommf-simulations\AFLC_dump\FCMPW_FieldSweep"
    rf = ResonantFrequency(directory=p_dir)
    parameter_dict = {
        "time_step": 1e-11,
        "start_time": 5.1e-9,
        "stop_time": 9.99e-9,
        "param_sweep": [-0.001, -0.0002, -0.0006, 0.0002, 0.00099999]
    }
    rf.set_parameters(**parameter_dict)
    rf.initialize_analysis()


