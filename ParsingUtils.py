
class ParsingUtils:
    def __init__(self):
        pass

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
            stages = len(lines) - 1
            return df, stages
