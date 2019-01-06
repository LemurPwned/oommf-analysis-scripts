import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import numpy as np
import json
import os
import glob
import pickle
import sys
import re
from Interface import Interface, ParsingStage
from ParsingUtils import ParsingUtils
from color_term import ColorCodes
from difflib import SequenceMatcher


class AnalysisUnit:
    def __init__(self, interface, defaults):
        self.defaults = defaults
        self.startup_dict = None
        self.directory = None
        self.clear = False
        self.time_step = None
        self.start_time = None
        self.stop_time = None
        self.R_pp = None
        self.global_mean_voltages = None
        self.global_frequency_set = None
        self.dispersion = None
        self.param_name = None
        self.resonant_frequency = None
        self.reverse = False
        self.frequency_name = 'freq'
        self.extract_frequency = False

        specification = self.extract_arguments_from_json(interface)
        self.set_inner_interface_specification(specification)
        self.base_data_cols = ['Rpp', 'Mvolt', 'Fx',
                               'Fy', 'Fz', 'mx', 'my', 'mz', 'ax', 'ay', 'az']

    def set_inner_interface_specification(self, specification):
        inner_interface = Interface(specification)
        ps = ParsingStage(inner_interface, self.defaults)
        self.startup_dict = ps.resultant_dict
        if self.startup_dict is None:
            raise ValueError("No arguments specified")

    def extract_arguments_from_json(self, filepath):
        with open(filepath, 'r') as f:
            spec = json.loads(f.read())
        return spec

    def set_parameters(self, **kwargs):
        """
        :param: **kwargs are the arguments to be passed to the main widget
        iterator
        """
        for k, v in kwargs.items():
            setattr(self, k, v)

    def save_object(self, object_type, savename):
        if type(object_type) == mpl.figure.Figure:
            object_type.savefig(f"{savename}.png")
            plt.close(object_type)
            return True
        elif type(object_type) == pd.DataFrame:
            object_type.to_pickle(f"{savename}.pkl")
            return True
        elif type(object_type) == pd.Series:
            object_type.to_pickle(
                f"{savename}_series_{object_type.columns}.pkl")
        return False

    def manage_directory(self, base_name, dir_name="Results"):
        base_dir_name = os.path.basename(base_name) if not base_name.endswith(
            '/') else os.path.basename(base_name[:-1])
        result_dir_name = dir_name + "_" + base_dir_name
        result_directory = os.path.join(base_name, result_dir_name)
        return self.create_dir(result_directory)

    def create_dir(self, directory):
        if os.path.isdir(directory):
            return directory
        else:
            os.mkdir(directory)
            return directory

    def search_directory_for_odt(self):
        """
        finds the possible .odt files in the specified directory
        :return: None
        """
        directory_roots = os.path.join(self.directory, '*/*.odt')
        filename_candidates = glob.glob(directory_roots, recursive=True)
        print(f"{ColorCodes.CYAN}{len(filename_candidates)}{ColorCodes.RESET_ALL} file candidates found...")
        print(
            f"{ColorCodes.MAGENTA}ROOT DIRECTORY{ColorCodes.RESET_ALL} {directory_roots}")
        if len(filename_candidates) == 0:
            print(
                f"{ColorCodes.RED}No files to analyze located ...{ColorCodes.RESET_ALL}")
            quit()
        return filename_candidates

    def pickle_load_procedure(self, filename):
        # reads each .odt file and returns pandas DataFrame object
        pickle_path = os.path.join(os.path.dirname(filename),
                                   os.path.basename(filename).replace(".odt",
                                                                      ".pkl"))
        if self.clear or (not os.path.isfile(pickle_path)):
            # print("\rPickle not found, parsing ... {}".format(pickle_path))
            df, _ = ParsingUtils.get_odt_file_data(filename)
            self.save_object(df, pickle_path.replace('.pkl', ''))
        else:
            # if found, load pickle
            with open(pickle_path, 'rb') as f:
                df = pickle.load(f)
        return df

    def standard_fourier_analysis(self, df, savename):
        # performs specified data analysis
        shortened_df = self.cutout_sample(df, start_time=self.start_time,
                                          stop_time=self.stop_time)
        r_max = np.max(shortened_df['MF_Magnetoresistance::magnetoresistance'])
        r_min = np.min(shortened_df['MF_Magnetoresistance::magnetoresistance'])
        r_diff = r_max-r_min
        _, m_voltage = self.voltage_calculation(shortened_df,
                                                self.resonant_frequency)
        frequency_set = self.find_max_frequency(shortened_df, self.time_step)
        avg_m = [np.mean(shortened_df[m]) for m in [
            'Oxs_TimeDriver::mx', 'Oxs_TimeDriver::my', 'Oxs_TimeDriver::mz']]
        angs = self.calculate_angles(shortened_df)
        return (r_diff, m_voltage, *frequency_set[:, 0], *avg_m, *angs)

    def calculate_angles(self, df):
        mag = np.mean(np.sqrt(np.sum(
            [np.power(df[m], 2)
             for m in ['Oxs_TimeDriver::mx', 'Oxs_TimeDriver::my', 'Oxs_TimeDriver::mz']])
        )
        )
        angs = [np.arccos(np.mean(df[m]/mag))*180/np.pi
                for m in ['Oxs_TimeDriver::mx', 'Oxs_TimeDriver::my', 'Oxs_TimeDriver::mz']]
        return angs

    def set_resonant_frequency(self, extracted_frequency):
        self.resonant_frequency = extracted_frequency

    def plot_magnetisation_trajectiories(self, df, cols, savedir):
        """
        plots the magnetisation trajectory using the 3d scatterplot
        """
        if (len(cols) != 3):
            raise ValueError("Invalid number of cuts to create trajectory")
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(df[cols[0]], df[cols[1]], df[cols[2]])
        ax.set_xlabel('Mx')
        ax.set_ylabel('My')
        ax.set_zlabel('Mz')
        savename = os.path.join(savedir, f"MTrajectory_{cols[0].strip()}")
        self.save_object(fig, savename)

    def extract_mag_cut_trajcetories(self, all_cols):
        """
        extracts the mag cut columns that are later used to plot
        the magnetisation trajectory from the cut
        """
        # find any magnetization
        patterns = ['(MF_X_MagCut:)([A-z]+:)(area mx)',
                    '(MF_Y_MagCut:)([A-z]+:)(area my)',
                    '(MF_Z_MagCut:)([A-z]+:)(area mz)']

        cuts = ['area mx', 'area my', 'area mz']
        mag_cuts = [[], [], []]
        for col in all_cols:
            for i, patt in enumerate(cuts):
                if patt in col:
                    mag_cuts[i].append(col)
        return mag_cuts

    def magcut_handler(self, df, savedir):
        """
        handles the presence of MF_*_MagCut columns and
        saves the magnetization directories
        """
        mag_cut_cols = self.extract_mag_cut_trajcetories(df.columns)
        mag_cut_pairs = []
        for i in range(len(mag_cut_cols[0])):
            ym = np.array(map(lambda x: SequenceMatcher(None, x,
                                                        mag_cut_cols[0][i]),
                              mag_cut_cols[1])).argmax(axis=0)
            zm = np.array(map(lambda x: SequenceMatcher(None, x,
                                                        mag_cut_cols[0][i]),
                              mag_cut_cols[2])).argmax(axis=0)
            mag_cut_pairs.append((mag_cut_cols[0][i], mag_cut_cols[1][ym],
                                  mag_cut_cols[2][zm]))
        for mag_cut_pair in mag_cut_pairs:
            self.plot_magnetisation_trajectiories(df, mag_cut_pair,
                                                  savedir)
