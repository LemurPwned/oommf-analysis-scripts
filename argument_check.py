import os


class ArgumentCheck():
    def __init__(self, base_class):
        self.base_calling_class = base_class

    def does_directory_exist(self, directory):
        try:
            assert os.path.isdir(directory) == True
        except AssertionError:
            raise ValueError(
                f"Indicated directory: {directory} does not exist!")

    def param_name_not_compliant(self, param_name):
        try:
            pass
            # extract param name here
        except ValueError as ve:
            if ve.__cause__ == "could not convert string to float":
                raise ValueError("Parameter name was not compatible")

    def extract_frequency_but_no_frequency(self):
        raise AttributeError(
            "Asked to extract frequency but no frequency name given. Use as combination: -rf -fn")

    def time_warning(self, time):
        if time < 1e-2:
            raise Warning(f"Careful! {time} is suspiciously small!")
