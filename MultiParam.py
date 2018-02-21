import numpy as np
import matplotlib.pyplot as plt

import os
import sys
import csv
import pickle
import pandas as pd

from Interface import asynchronous_pool_order
from Fourier import ResonantFrequency


class MultiParam(ResonantFrequency):
    def __init__(self, filename):
        super().__init__(filename)
        self.set_parameters(**self.startup_dict)
        if self.directory is None:
            raise ValueError("Invalid directory ")
        else:
            # prepare resulting directory
            self.result_directory = self.manage_directory(self.directory)
        self.base_param1 = 'P1'
        self.base_param2 = 'P2'

        self.merged_data = None
        self.png = ".png"

        self.os_type = "windows" if "win" in sys.platform else "linux"
        self.delimiter = "/" if self.os_type == "linux" else "\\"

    def initialize_analysis(self):
        file_names = self.search_directory_for_odt()
        self.set_parameters()
        # assign names to parameters
        self.extract_parameter_type_dual_params(file_names[0], setting=True)

        output = asynchronous_pool_order(self.multi_parameter_analysis, (), file_names)
        self.merged_data = pd.DataFrame(output, columns=[self.base_param1,
                                                         self.base_param2,
                                                         'Rpp_diff',
                                                         'M_volt', 'Fmx', 'Fmy', 'Fmz'])
        # in case two parameters could be reversed
        if self.reverse:
            self.base_param2, self.base_param1 = self.base_param1, self.base_param2
        savename = os.path.join(self.result_directory, "final_data_frame")
        self.save_object(self.merged_data, savename)
        print("FINISHED PARSING, NOW PLOTTING...")
        self.perform_plotting(self.dispersion)

    def perform_plotting(self, dispersion):
        """
        param 1 will be held constant while the other will be swept
        :return:
        """
        # keep param 1 constant
        if not dispersion:
            for param_value in self.merged_data[self.base_param1].unique():
                print(param_value)
                dir_name = self.manage_directory(self.result_directory, self.base_param1 +
                                                 "_" + str(param_value))
                constant_param1 = self.merged_data[self.merged_data[self.base_param1] == param_value]
                self.plot_saving("Rpp", dir_name, constant_param1[self.base_param2],
                                 constant_param1['R_diff'])
                self.plot_saving("Rpp", dir_name, constant_param1[self.base_param2],
                                 constant_param1['R_diff'])
        else:
            for param_value in self.merged_data[self.base_param1].unique():
                print(param_value)
                dir_name = self.manage_directory(self.result_directory, self.base_param1 +
                                                 "_" + str(param_value))
                constant_param1 = self.merged_data[self.merged_data[self.base_param1] == param_value]
                for vector_orientation in ['Fmx', 'Fmy', 'Fmz']:
                    self.plot_saving(vector_orientation, dir_name,
                                     constant_param1[self.base_param2],
                                     constant_param1[vector_orientation])
                res_savepoint = os.path.join(dir_name, "resonant_frequencies.csv")
                with open(res_savepoint, 'w') as f:
                    writer = csv.writer(f, delimiter=',')
                    writer.writerows(zip(constant_param1[self.base_param2].tolist(),
                                         constant_param1[['Fmx', 'Fmy', 'Fmz']].values))

    def plot_saving(self, name, dir_name, array1, array2=None):
        fig = plt.figure()
        if array2 is None:
            plt.plot(array1)
        else:
            plt.plot(array1, array2, 'o')
        fig.suptitle(name, fontsize=12)
        savename = name + str(self.start_time) + " " + str(self.stop_time)
        savename = os.path.join(dir_name, savename)
        self.save_object(fig, savename)
        plt.close(fig)

    def extract_parameter_type_dual_params(self, filename, setting=False):
        # extract top-level directory
        filename = os.path.dirname(filename).split(self.delimiter)[-1]
        # extract two param_set
        base_params = filename.split("_")
        base_param1 = (base_params[0], base_params[1])
        base_param2 = (base_params[2], base_params[3])
        if setting:
            self.base_param1 = base_params[0]
            self.base_param2 = base_params[2]
        return base_param1, base_param2

    def multi_parameter_analysis(self, filename):
        param1, param2 = self.extract_parameter_type_dual_params(filename)
        # reads each .odt file and returns pandas DataFrame object
        pickle_path = os.path.join(os.path.dirname(filename),
                                   os.path.basename(filename).replace(".odt",
                                                                      "stages.pkl"))
        if self.clear or (not os.path.isfile(pickle_path)):
            df, stages = self.read_directory_as_df_file(filename)
        else:
            # if found, load pickle
            with open(pickle_path, 'rb') as f:
                df = pickle.load(f)
        try:
            shortened_df = self.cutout_sample(df, start_time=self.start_time,
                                              stop_time=self.stop_time)
            r_max = np.max(shortened_df['MF_Magnetoresistance::magnetoresistance'])
            r_min = np.min(shortened_df['MF_Magnetoresistance::magnetoresistance'])
            r_diff = r_max-r_min
            voltage, m_voltage = self.voltage_calculation(shortened_df,
                                                          self.resonant_frequency)
            savename = os.path.join(self.result_directory, str(param1[1]) + "_" +
                                    str(param2[1]))
            frequency_set = self.find_max_frequency(shortened_df, self.time_step,
                                                    param=savename)
            mx, my, mz = frequency_set[:, 0]
        except ValueError as e:
            print(e)
            return [param1[1], param2[1], 0, 0, 0, 0, 0]
        return [param1[1], param2[1], r_diff, m_voltage, mx, my, mz]


if __name__ == "__main__":
    rf = MultiParam("interface.json")
    rf.initialize_analysis()