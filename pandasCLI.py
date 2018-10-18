import pandas as pd
import os
import glob
import matplotlib.pyplot as plt


class PandasCLI:
    def __init__(self, d_directory):
        self.directory = d_directory
        self.df = None

    def run_in_folder(self):
        if self.directory is None:
            this_folder = os.path.curdir()
        else:
            this_folder = self.directory
        possible_pickles = glob.glob(os.path.join(this_folder, "*/*.pkl"))
        if possible_pickles is not None:
            try:
                selected = self.numbered_display(possible_pickles)
            except ValueError:
                print("Please select again")
                selected = self.numbered_display(possible_pickles)
            self.accept_pickle(selected)

    def query_loop(self):
        while True:
            selected_column = self.numbered_display(self.df.columns)
            self.plotting_module(selected_column)

    def numbered_display(self, array):
        max_i = len(array)
        for i, element in enumerate(array):
            cli_string = str(i+1) + ") " + element
            print(cli_string)
        print("q) QUIT")
        number = input("\nPlease select an option\n")
        if number == "q":
            quit()
        number = int(number)
        if number > max_i:
            raise ValueError("Invalid number")
        return array[number-1]

    def accept_pickle(self, filename):
        self.df = pd.read_pickle(filename)
        self.query_loop()

    def plotting_module(self, column):
        plt.plot(self.df[column])
        plt.title(column)
        plt.show()


if __name__ == "__main__":
    directory = r"D:\Dokumenty\Simulations\COUPLING_VBC_1500_900_42_AMP_5em5_SWEEP_1em5_FIXED_FREQ"
    pandas_cli = PandasCLI(d_directory=directory)
    pandas_cli.run_in_folder()
