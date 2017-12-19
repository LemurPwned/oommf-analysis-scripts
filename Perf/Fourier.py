import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
import os


class ResonantFrequency:
    def __init__(self, directory):
        self.directory = directory

    def read_directory_as_df_file(self, filename):
        """
        Reads .odt file
        @param filename is .odt file path
        @return dataFrame and stages number
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
            cols = cols.replace("MF", "Oxs_MF")
            cols = cols.replace("} ", "")
            cols = cols.replace("{", "")
            cols = cols.split("Oxs_")
            del cols[0]
            cols = [x.strip() for x in cols]
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

    def cutout_sample(self, data, start_time=None, stop_time=None):
        if start_time is None:
            return data
        if stop_time is None:
            return data['time'][(data['time'] > start_time)]
        else:
            return data['time'][(data['time'] > start_time) & (data['time']) < stop_time]

    def voltage_calculation(self, df, time_offset, time_stop=10e9):
        df_limited = df.loc[(df['TimeDriver::Simulation time'] > time_offset) &
                            (df['TimeDriver::Simulation time'] < time_stop)]
        avg_resistance = np.mean(df_limited['MF_Magnetoresistance::magnetoresistance'])
        power = 10e-6
        frequency = 1e8
        omega = 2*np.pi*frequency
        phase = 0
        amplitude = np.sqrt(power/avg_resistance)
        current = amplitude*np.sin(omega*df_limited['TimeDriver::Simulation time'] + phase)
        voltage = df_limited['MF_Magnetoresistance::magnetoresistance']*current
        mean_voltage = np.mean(voltage)
        return voltage, mean_voltage

    def fourier_analysis(self, sample):
        return np.fft.fft(sample, axis=0)

    def plot_fourier(self):
        pass


if __name__ == "__main__":
    # filename = glob.glob(os.path.join(self.directory, '*.odt'))[0]

    path = r"D:\Dokumenty\oommf-simulations\REZ\resonance_valid\Default\AFCoupFieldDomain.odt"
    directory = r'D:\Dokumenty\oommf-simulations\AFLC_dump\FCMPW_FieldSweep'
    # to_search = os.path.join(directory, '*.odt')
    # filepaths = glob.glob(to_search)
    # print(filepaths)
    f_p = ResonantFrequency(path)
    df, stages = f_p.read_directory_as_df_file(path)
    print(df.columns)
    time_offset = 30e-9
    time_stop = 100
    df_limited = df.loc[(df['TimeDriver::Simulation time'] > time_offset) &
                        (df['TimeDriver::Simulation time'] < time_stop)]
    mx = f_p.fourier_analysis(df_limited['TimeDriver::mx'])
    my = f_p.fourier_analysis(df_limited['TimeDriver::my'])
    mz = f_p.fourier_analysis(df_limited['TimeDriver::mz'])
    n = mx.size
    freq = np.fft.fftfreq(n, d=1e-11)
    for m in [mx, my, mz]:
        m = abs(m)
        max_val = 0
        max_freq = 0
        for frequency, amp in zip(freq, m):
            if amp > max_val and frequency > 0:
                max_val = amp
                max_freq = frequency
        print(max_freq/1e9, max_val)
    # print(max_freq/1e9, amp)
    #
    #plt.stem(freq, np.abs(mx), c='g')
    #plt.stem(freq, np.abs(my), c='r')
    # plt.stem(freq, np.abs(mz), c='o')
    # plt.show()
    plt.plot(df['TimeDriver::mx'], c='b')
    plt.plot(df['TimeDriver::my'], c='y')
    plt.plot(df['TimeDriver::mz'], c='g')
    plt.show()

    # voltage, mean_voltage = f_p.voltage_calculation(df, time_offset=26e-9, time_stop=36e-9)
    # print("Mean voltage: ", mean_voltage)
    # fourier = f_p.fourier_analysis(voltage)
    # n = fourier.size
        # freq = np.fft.fftfreq(n, d=1e-13)
    # max_val = 0
    # max_freq = 0
    # fourier = np.abs(fourier)
    # for frequency, amp in zip(freq, fourier):
    #     if amp > max_val and frequency >= 0:
    #         max_val = amp
    #         max_freq = frequency
