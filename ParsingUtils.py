import sys
import pandas as pd
import time
import numpy as np
import struct
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

    @staticmethod
    def read_ovf(filename):
        with open(filename, 'r') as f:
            lines = f.readlines()
        vectors = [x.strip().split(' ') for x in lines if '#' not in x]
        vectors = [[float(row[0]), float(row[1]), float(row[2])]
                   for row in vectors]
        return np.array(vectors)

    @staticmethod
    def process_header(headers):
        """
        processes the header of each .omf file and return base_data dict
        @param: headers specifies header from .omf file
        """
        final_header = {}
        headers = headers.replace(' ', "")
        headers = headers.replace('\\n', "")
        headers = headers.replace('\'b\'', "")
        headers = headers.split('#')
        for header_ in headers:
            if ':' in header_:
                components = header_.split(':')
                try:
                    final_header[components[0]] = float(components[1])
                except:
                    final_header[components[0]] = components[1]
        return final_header

    @staticmethod
    def binary_format_reader(filename):
        """
        Reads binary formatted .omf or .ovf files
        """
        header_part = ""
        rawVectorData = None
        header = None
        with open(filename, 'rb') as f:
            x = f.readline()
            while x != b'# End: Header\n':
                header_part += str(x)
                x = f.readline()

            header = ParsingUtils.process_header(header_part)

            byte_type = f.readline()
            while byte_type == b'#\n':
                byte_type = f.readline()
            # compile struct byte type
            fmt, buff_size, val = ParsingUtils.decode_byte_size(byte_type)
            struct_object = struct.Struct(fmt)
            test_val = struct_object.unpack(f.read(buff_size))[0]
            if test_val != val:
                raise ValueError("Invalid file format with validation {} value, \
                                should be {}".format(test_val, val))

            k = int(header['xnodes']*header['ynodes']*header['znodes'])
            rawVectorData = ParsingUtils.standard_vertex_mode(
                f, k, struct_object, buff_size)
            f.close()
        assert rawVectorData is not None
        assert header is not None
        return header, rawVectorData

    @staticmethod
    def standard_vertex_mode(f, k, struct_object, buff):
        return np.array([(struct_object.unpack(f.read(buff))[0],
                          struct_object.unpack(f.read(buff))[0],
                          struct_object.unpack(f.read(buff))[0])
                         for i in range(int(k))])

    @staticmethod
    def decode_byte_size(byte_format_specification):
        """
        infers byte format based on IEEE header format specification 
        """
        if byte_format_specification == b'# Begin: Data Binary 4\n':
            # float precision - 4 bytes
            fmt = 'f'
            buffer_size = struct.calcsize(fmt)
            four_byte_validation = 1234567.0
            return fmt, buffer_size, four_byte_validation
        elif byte_format_specification == b'# Begin: Data Binary 8\n':
            # double precision - 8 bytes
            fmt = 'd'
            buffer_size = struct.calcsize(fmt)
            eight_byte_validation = 123456789012345.0
            return fmt, buffer_size, eight_byte_validation
        else:
            raise TypeError("Unknown byte specification {}".format(
                str(byte_format_specification)))


if __name__ == "__main__":
    m = 1000000
    for i in range(m):
        ParsingUtils.flushed_loading_msg("MESSAGE", i, m)
