import matplotlib as mpl
import pandas as pd
import numpy as np
import json
import os
import glob
import pickle

from Interface import Interface, ParsingStage


class AnalysisUnit:
    def __init__(self, filename):
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

        specification = self.extract_arguments_from_json(filename)
        self.set_inner_interface_specification(specification)

    def set_inner_interface_specification(self, specification):
        inner_interface = Interface(specification)
        ps = ParsingStage(inner_interface)
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
            object_type.savefig(savename + '.png')
            return True
        elif type(object_type) == pd.DataFrame:
            object_type.to_pickle(savename + '.pkl')
            return True
        elif type(object_type) == pd.Series:
            object_type.to_pickle(savename + "_series" +
                                  object_type.columns + ".pkl")
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
        print("{} file candidates found...".format(len(filename_candidates)))
        if len(filename_candidates) == 0:
            quit()
        for filename in filename_candidates:
            print(filename)
        return filename_candidates

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
            cols = cols.replace("PBC", "Oxs_PBC")
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
            # save data frame
            stages = len(lines) - 1
            if not self.save_object(df, filename.replace(".odt", "stages")):
                print("Could not save {}".format(filename))
            return df, stages

    def pickle_load_procedure(self, filename):
        # reads each .odt file and returns pandas DataFrame object
        pickle_path = os.path.join(os.path.dirname(filename),
                                   os.path.basename(filename).replace(".odt",
                                                                      "stages.pkl"))
        if self.clear or (not os.path.isfile(pickle_path)):
            print("Pickle not found, parsing ... {}".format(filename))
            df, stages = self.read_directory_as_df_file(filename)
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
        mx, my, mz = frequency_set[:, 0]

        avg_mx = np.mean(shortened_df['TimeDriver::mx'])
        avg_my = np.mean(shortened_df['TimeDriver::my'])
        avg_mz = np.mean(shortened_df['TimeDriver::mz'])

        return r_diff, m_voltage, mx, my, mz, avg_mx, avg_my, avg_mz

    def set_resonant_frequency(self, extracted_frequency):
        self.resonant_frequency = extracted_frequency
