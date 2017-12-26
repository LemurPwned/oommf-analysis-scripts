import argparse


class Interface:
    def __init__(self, arg_list):
        self.arg_list = arg_list

    def define_input_parameters(self, **kwargs):
        parser = argparse.ArgumentParser(description=kwargs['description'])
        for argument in self.arg_list:
            parser.add_argument(argument['name'], argument['short'], help=argument['help'], action=argument['action'])
        args = parser.parse_args()
