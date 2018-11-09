import numpy as np
import matplotlib.pyplot as plt

import os
import sys
import csv
import pickle
import pandas as pd
from Interface import asynchronous_pool_order
from ParsingUtils import AnalysisException
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
        self.zerout = [0 for i in range(len(self.base_data_cols))]

    def initialize_analysis(self):
        file_names = self.search_directory_for_odt()
        self.set_parameters()
        # assign names to parameters
        self.extract_parameter_type_dual_params(file_names[0], setting=True)

        output = asynchronous_pool_order(
            self.multi_parameter_analysis, (), file_names)
        self.extracted_data_cols = [self.base_param1,
                                    self.base_param2]
        self.extracted_data_cols.extend(self.base_data_cols)
        # ORDER is (r_diff, m_voltage, *frequency_set[:, 0], *avg_m, *angs)
        self.merged_data = pd.DataFrame(
            output, columns=self.extracted_data_cols)
        # in case two parameters could be reversed
        if self.reverse:
            self.base_param2, self.base_param1 = self.base_param1, self.base_param2
        savename = os.path.join(self.result_directory, "final_data_frame")
        self.save_object(self.merged_data, savename)
        self.merged_data.to_csv(os.path.join(self.result_directory,
                                             "CSV_DATA_VOLTAGES_RPP.csv"))
        print("FINISHED PARSING, NOW PLOTTING...")
        print(self.merged_data.columns)
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
                dir_name = os.path.join(self.result_directory, self.base_param1 +
                                        "_" + str(param_value))
                self.create_dir(dir_name)
                constant_param1 = self.merged_data[self.merged_data[self.base_param1] == param_value]
                self.plot_saving("Rpp", dir_name, constant_param1[self.base_param2],
                                 constant_param1['Rpp'])
                self.plot_saving("Mean voltage", dir_name, constant_param1[self.base_param2],
                                 constant_param1['Mvolt'])
                sname = f"{str(param_value).replace('-', 'm')}_Freq_result_{self.start_time}_{self.stop_time}.csv"
                res_savepoint = os.path.join(dir_name, sname)
                constant_param1[[self.base_param2, 'Rpp', 'Mvolt']].to_csv(
                    res_savepoint, index=False)
        else:
            for param_value in self.merged_data[self.base_param1].unique():
                print(param_value)
                dir_name = self.manage_directory(self.result_directory, self.base_param1 +
                                                 "_" + str(param_value))
                constant_param1 = self.merged_data[self.merged_data[self.base_param1] == param_value]
                for vector_orientation in ['x', 'y', 'z']:
                    self.plot_saving(f"F_m{vector_orientation}", dir_name,
                                     constant_param1[self.base_param2],
                                     constant_param1['F'+vector_orientation],
                                     self.base_param2, 'F' + vector_orientation)
                    self.plot_saving(f"Average_m{vector_orientation}", dir_name,
                                     constant_param1[self.base_param2],
                                     constant_param1['m'+vector_orientation],
                                     self.base_param2, 'Avg' + vector_orientation)
                sname = f"Resonant_frequencies_{str(param_value).replace('-', 'm')}.csv"
                res_savepoint = os.path.join(dir_name, sname)
                constant_param1[[self.base_param2, 'Fx',
                                 'Fy', 'Fz', 'ax', 'ay', 'az']].to_csv(res_savepoint, index=False)

    def plot_saving(self, name, dir_name, array1, array2=None, xlab=None, ylab=None):
        fig = plt.figure()
        if array2 is None:
            plt.plot(array1)
        else:
            plt.plot(array1, array2, 'o')
        fig.suptitle(name, fontsize=12)
        if xlab is not None:
            plt.xlabel(xlab)
        if ylab is not None:
            plt.ylabel(ylab)
        savename = f"{name}_{self.start_time}_{self.stop_time}"
        savename = os.path.join(dir_name, savename)
        self.save_object(fig, savename)
        plt.close(fig)

    def extract_parameter_type_dual_params(self, filename, setting=False):
        # extract top-level directory
        filename = os.path.dirname(filename).split(self.delimiter)[-1]
        # extract two param_set
        base_params = filename.split("_")
        base_param1 = (base_params[0], np.float64(base_params[1]))
        base_param2 = (base_params[2], np.float64(base_params[3]))
        if setting:
            self.base_param1 = base_params[0]
            self.base_param2 = base_params[2]

        if self.extract_frequency:
            if base_params[0] == self.frequency_name:
                self.set_resonant_frequency(base_params[1])
            elif base_params[2] == self.frequency_name:
                self.set_resonant_frequency(base_params[0])
        return base_param1, base_param2

    def multi_parameter_analysis(self, filename):
        param1, param2 = self.extract_parameter_type_dual_params(filename)
        # reads each .odt file and returns pandas DataFrame object
        try:
            df = self.pickle_load_procedure(filename)
        except AssertionError as e:
            raise AnalysisException(e, filename, null_val=[
                                    param1[1], param2[1], *self.zerout])
        try:
            savename = os.path.join(self.result_directory, str(param1[1]) + "_" +
                                    str(param2[1]))
            return [param1[1], param2[1], *self.standard_fourier_analysis(df, savename)]

        except ValueError as e:
            print(e)
            raise AnalysisException(e, filename, null_val=[
                                    param1[1], param2[1], *self.zerout])
        return [param1[1], param2[1], *self.zerout]


if __name__ == "__main__":
    rf = MultiParam("interface.json")
    rf.initialize_analysis()
