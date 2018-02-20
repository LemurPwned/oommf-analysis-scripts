import numpy as np
import matplotlib.pyplot as plt

import os
import sys
import csv
import pickle

from Interface import asynchronous_pool_order
from AnalysisUnit import AnalysisUnit


class MultiParam(AnalysisUnit):
    def __init__(self, filename):
        super().__init__(filename)
        self.set_parameters(**self.startup_dict)
        if self.directory is None:
            raise ValueError("Invalid directory ")
        else:
            # prepare resulting directory
            self.result_directory = self.manage_directory(self.directory)

    def extract_parameter_type_dual_params(self, filename):
        # extract top-level directory
        filename = os.path.dirname(filename).split(self.delimiter)[-1]

        # extract two param_set
        base_params = filename.split("_")
        base_param1 = (base_params[0], base_params[1])
        base_param2 = (base_params[2], base_params[3])
        return base_param1, base_param2

    def multiparameter_analysis(self, filename):
        param1, param2 = self.extract_parameter_type_dual_params(filename)
        # reads each .odt file and returns pandas DataFrame object
        pickle_path = os.path.join(os.path.dirname(filename),
                                   os.path.basename(filename).replace(".odt", "stages.pkl"))