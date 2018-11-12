import sys
import pandas as pd
import time
import numpy as np
from color_term import ColorCodes


class AnalysisException(Exception):
    def __init__(self, msg, filename, null_val=None):
        self.filename = filename
        self.msg = msg
        self.null_val = null_val

    def __str__(self):
        return f"Error: {self.msg} found in file: {self.filename} "


class ParsingUtils:

    @staticmethod
    def flushed_loading_msg(msg, progress, max_val, bar_length=50, err_msg=""):
        mult = int(progress*bar_length/max_val)+1
        if mult % 5 == 0:
            load_val_eq = "="*mult
            load_val_dot = "*"*(bar_length-mult)
            err_msg = str(err_msg).replace("\n", "")
            err_msg = f"{ColorCodes.RED}{err_msg}{ColorCodes.RESET_ALL}"
            if msg is not None:
                print(
                    f"\r{ColorCodes.BLUE}{msg}{ColorCodes.RESET_ALL} [{load_val_eq}{load_val_dot}] {ColorCodes.YELLOW}{int(progress*100/max_val)}%{ColorCodes.RESET_ALL} \t{err_msg}",
                    flush=True, end="")

            else:
                print(
                    f"\r[{load_val_eq}{load_val_dot}] {ColorCodes.GREEN}{int(progress*100/max_val)}%{ColorCodes.RESET_ALL}", flush=True, end="")

    @staticmethod
    def parse_odt_col(line):
        """
        This extracts a single column from a odt file
        OOMMF formatting support
        """
        cols = []
        line = line.replace('# Columns: ', '')
        while line != '':
            if line[0] == '{':
                patch = line[1:line.index('}')]
                if patch != '':
                    cols.append(patch)
                line = line[line.index('}')+1:]
            else:
                try:
                    patch = line[:line.index(' ')]
                    if patch != '':
                        cols.append(patch)
                    line = line[line.index(' ')+1:]
                except ValueError:
                    if line != "" and line != '\n':
                        # last trailing line
                        cols.append(patch)
                    line = ""
                    break
        return cols

    @staticmethod
    def get_odt_file_data(filename):
        """
        Reads .odt of .txt file
        @param: filename is .odt file path
        @return: dataFrame and stages number
        """
        if filename.endswith('.txt'):
            df = pd.read_table(filename)
            return df, len(df)
        elif filename.endswith('.odt'):
            header_lines = 4
            header = []
            i = 0
            with open(filename, 'r') as f:
                while i < header_lines:
                    lines = f.readline()
                    header.append(lines)
                    i += 1
                units = f.readline()
                lines = f.readlines()
            cols = header[-1]
            cols = ParsingUtils.parse_odt_col(cols)
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
                dataset.append(temp_line)
            dataset = dataset[:-1]
            df = pd.DataFrame.from_records(dataset, columns=cols)
            stages = len(lines) - 1
            return df, stages
        else:
            raise ValueError(f"Unsupported extension {filename}")


if __name__ == "__main__":
    m = 1000000
    for i in range(m):
        ParsingUtils.flushed_loading_msg("MESSAGE", i, m)
