import pandas as pd 
import os 
import matplotlib.pyplot as plt
import numpy as np 
import csv 

def vsd_calculation(df_limited, frequency=10e9):
    avg_resistance = np.mean(df_limited['Resistance_[Ohm]'])
    power = 10e-6
    amplitude = np.sqrt(power / avg_resistance)
    current = amplitude*np.sin(2*np.pi*frequency*df_limited['Time_[ns]']*1e-9)
    voltage = current*df_limited['Resistance_[Ohm]']
    mean_voltage = np.mean(voltage)
    return voltage, mean_voltage, avg_resistance

def read_fft(df, cols=['M1x', 'M1y', 'M1z', 'M2x', 'M2y', 'M2z', 'Resistance_[Ohm]'], time_step=1e-11):
    """
    Values given in columns must have common sampling frequency
    :param df: DataFrame object containing columns specified in cols
    :param time_step: sampling step
    :param cols: columns for which fourier transform is to be performed
    :return: numpy array of maximum frequencies and respective values
    """
    potential_fourier_data = []
    for col in cols:
        potential_fourier_data.append(np.fft.fft(df[col], axis=0))
    # fourier frequencies must be calculated first to know precise frequency
    frequency_steps = np.fft.fftfreq(
        potential_fourier_data[0].size, d=time_step)
    max_freq_set = []
    max_val_set = []
    for freq_data in potential_fourier_data:
        freq_data = abs(freq_data)
        max_val = 0
        max_freq = 0
        for frequency, amp in zip(frequency_steps, freq_data):
            if np.abs(amp) > max_val and frequency > 0:
                max_val = amp
                max_freq = frequency
        max_freq_set.append(max_freq / 1e9)
        max_val_set.append(max_val)
    # display Fourier
    return max_freq_set, max_val_set

def read_h_scan(filename, time_skip=5, freq=5e9, savename='VSD.csv'):
    df = pd.read_csv(filename, delimiter='\t')
    max_time = df['Time_[ns]'].tail(1).item()
    tables = []
    cols = ['Field Hext', 'Resistance peak-peak', 'Voltage mix']
    units = ['mT', 'Ohm', 'V']
    for time in np.arange(0, max_time, time_skip):
        df_slice = df.loc[(df['Time_[ns]'] >= (time+10)) &
                            (df['Time_[ns]'] < (time+time_skip))]
        rpp = np.max(df_slice['Resistance_[Ohm]']) - np.min(df_slice['Resistance_[Ohm]'])
        _, vsd, _ = vsd_calculation(df_slice, freq)
        field = np.mean(df_slice['External_H'])/795.7747
        tables.append([field, rpp, vsd])

    with open(savename, 'w') as f:
        writer = csv.writer(f, delimiter=',', lineterminator='\n')
        writer.writerow(cols)
        writer.writerow(units)
        writer.writerows(tables)

def sweep_scan_fft(filename, time_skip, savename):
    df = pd.read_csv(filename, delimiter='\t')
    max_time = df['Time_[ns]'].tail(1).item()
    tables = []
    for time in np.arange(0, max_time, time_skip):
        df_slice = df.loc[(df['Time_[ns]'] >= (time+10)) &
                            (df['Time_[ns]'] < (time+10+9.210526))]
        max_frequencies, max_vals = read_fft(df_slice)
        J = np.mean(df_slice['J1'])
        print(J)
        tables.append([J, *max_frequencies, *max_vals])
    
    with open(savename, 'w') as f:
        writer = csv.writer(f, delimiter=',', lineterminator='\n')
        writer.writerows(tables)

def vcma_sweep_scan_fft(filename, time_skip):
    df = pd.read_csv(filename, delimiter='\t')
    max_time = df['Time_[ns]'].tail(1).item()
    tables = []
    for time in np.arange(0, max_time, time_skip):
        df_slice = df.loc[(df['Time_[ns]'] >= (time+10)) &
                            (df['Time_[ns]'] < (time+time_skip))]
        max_frequencies, _ = read_fft(df_slice)
        K = np.mean(df_slice['K1'])
        print(K)
        tables.append([K, *max_frequencies])
        
    
    with open('VCMA_DISP.csv', 'w') as f:
        writer = csv.writer(f, delimiter=',', lineterminator='\n')
        writer.writerows(tables)

if __name__ == "__main__":
    directory = r'D:\Pobrane\nSim\AllData.txt'
    # vcma_sweep_scan_fft(directory, 20)
    # sweep_scan_fft(directory, 20, savename="Perp_disp.csv")
    read_h_scan(directory, 20, 40e9, savename='40e9_8m5.csv')
