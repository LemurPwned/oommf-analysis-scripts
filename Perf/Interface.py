import argparse

class Interface:
    def __init__(self, **arglist):
        pass

    def define_input_parameters(self, **kwargs):
        parser = argparse.ArgumentParser(description=kwargs['description'])
        for argument in kwargs['arglist']:
            parser.add_argument(argument['name'], )
