import argparse
import json
from multiprocessing import Pool
from ParsingUtils import ParsingUtils, AnalysisException
from color_term import ColorCodes


class Interface:
    def __init__(self, specification):
        self.arg_list = specification['specification']
        self.parsed_args = self.define_input_parameters(
            specification["description"])
        self.defined_parameters = [x['name'] for x in self.arg_list]

    def define_input_parameters(self, desc):
        parser = argparse.ArgumentParser(description=desc)
        for argument in self.arg_list:
            if "action" in argument.keys():
                parser.add_argument("-" + argument['short'], "--" + argument['name'],
                                    help=argument['help'], action=argument['action'])
            else:
                parser.add_argument("-"+argument['short'], "--" + argument['name'],
                                    help=argument['help'],
                                    type=self.decode_type(argument['type']))
        return parser.parse_args()

    def decode_type(self, type_str):
        if type_str == "str":
            return str
        elif type_str == "int":
            return int
        elif type_str == "float":
            return float
        elif type_str == "bool":
            return bool


class ParsingStage:
    def __init__(self, interface, defaults):
        self.available_argument_list = interface.defined_parameters
        self.default_dict_path = defaults

        # immediately read the arguments
        self.resultant_dict = {}
        self.args = interface.parsed_args
        self.args_handler()

        self.read_json_dict_param(self.default_dict_path)

    def args_handler(self):
        for arg_name in self.available_argument_list:
            try:
                if getattr(self.args, arg_name) is not None:
                    self.set_dict_param(arg_name, getattr(self.args, arg_name))
            except TypeError:
                print(
                    f"{ColorCodes.RED}ASKED FOR NON-EXISTENT VALUE {arg_name}{ColorCodes.RESET_ALL}")

    def set_dict_param(self, param_name, param_val):
        self.resultant_dict[param_name] = param_val

    def read_json_dict_param(self, filepath):
        with open(filepath, 'r') as f:
            default_dict = json.loads(f.read())
            print(f"CORRECT DICT TYPE? {type(default_dict)}")
        print(f"DEFAULT DICTIONARY PARAMS DETECTED...\n{default_dict}")

        if (not isinstance(default_dict, dict)) or \
                (not isinstance(self.resultant_dict, dict)):
            msg = "Dictionary mismatch of types"
            raise TypeError(msg)
        elif type(default_dict) != dict:
            raise TypeError("Invalid type of entry")

        if self.resultant_dict["view"]:
            quit()
        # overwrite default dict with dict taken from arg parse
        # this line does it, mind the order!
        self.resultant_dict = {**default_dict, **self.resultant_dict}


def asynchronous_pool_order(func, args, object_list):
    pool = Pool()
    output_list = []
    mr_len = len(object_list)
    multiple_results = [pool.apply_async(func, (*args, object_type))
                        for object_type in object_list]
    for i, result in enumerate(multiple_results):
        try:
            value = result.get()
            output_list.append(value)
            ParsingUtils.flushed_loading_msg(
                f"Parsing...", i, mr_len)
        except AnalysisException as e:
            ParsingUtils.flushed_loading_msg(
                f"Parsing...", i, mr_len, err_msg=e.msg)
            output_list.append(e.null_val)
    return output_list
