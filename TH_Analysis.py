import requests
import pandas as pd
import numpy as np
from pandas.io.json import json_normalize
import json
from flatten_json import flatten
import datetime
import time
import seaborn as sns

# Plotting settings
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (15,5)


class th_analysis():
    def __init__(self, filename):
        self.data = pd.read_csv(filename, sep=';')

    def get_th_dataframe(self):
        cols = self.data.columns
        self.data['Time'] = pd.to_datetime(self.data['Time'], utc=True)
        self.data.set_index('Time', inplace=True)
        self.data.index = self.data.index.tz_convert('America/Los_Angeles')
        self.data = self.data.sort_index()
        mask_t = self.data['Series'] == 'iot-devices.Temp1'
        mask_h = self.data['Series'] == 'iot-devices.Hum'
        return [self.data[mask_t], self.data[mask_h]]

    def daily_calculations(self, df):
        df_list = [group[1] for group in df.groupby(df.index.day)]  # split month into days
        day_time = {'morning': [datetime.time(6), datetime.time(12)],
                    'afternoon': [datetime.time(12), datetime.time(18)],
                    'evening': [datetime.time(18), datetime.time(0)], 'night': [datetime.time(0), datetime.time(6)]}
        output = {}
        for i in df_list:
            output[i.index.day[-1]] = [[i.between_time(day_time['morning'][0], day_time['morning'][1]).mean(),
                                        i.between_time(day_time['afternoon'][0], day_time['afternoon'][1]).mean(),
                                        i.between_time(day_time['evening'][0], day_time['evening'][1]).mean(),
                                        i.between_time(day_time['night'][0], day_time['night'][1]).mean()],
                                       [i.between_time(day_time['morning'][0], day_time['morning'][1]).std(),
                                        i.between_time(day_time['afternoon'][0], day_time['afternoon'][1]).std(),
                                        i.between_time(day_time['evening'][0], day_time['evening'][1]).std(),
                                        i.between_time(day_time['night'][0], day_time['night'][1]).std()]]
        return output

    def daily_calculations1(self, df, mode):
        df = df.resample('H').mean()
        if mode == 'Temp':
            df = (df-32)*5/9    # Convert to Celsius
        df_list = [group[1] for group in df.groupby(df.index.day)]  # split month into days
        output = {}
        for i in df_list:
            output[i.index.day[-1]] = i
        return output

    def daily_calculations2(self, df_t, df_h):
        df_t = df_t.resample('H').mean()
        df_t_c = (df_t['Value'] - 32) * 5 / 9  # Convert to Celsius
        df_h = df_h.resample('H').mean()
        df_thi = df_t_c - (0.55-0.0055*df_h['Value'])*(df_t_c-58.8)
        df_thi = df_thi.resample('H').mean()
        df_t_list = [group[1] for group in df_t.groupby(df_t.index.day)]  # split month into days
        output_t = {}
        for t in df_t_list:
            output_t[t.index.day[-1]] = t

        df_h_list = [group[1] for group in df_h.groupby(df_h.index.day)]  # split month into days
        output_h = {}
        for h in df_h_list:
            output_h[h.index.day[-1]] = h

        df_thi_list = [group[1] for group in df_thi.groupby(df_thi.index.day)]  # split month into days
        output_thi = {}
        for th in df_thi_list:
            output_thi[th.index.day[-1]] = th

        return [output_t, output_h, output_thi]

################### MAIN ###################

n_sensors = 16
sensors = np.array(range(0,n_sensors))
temp_data = []
hum_data = []
thi_data = []

# Inputs:
# Analysis
var = 'Temp'
# var = 'Hum'
# month = 'june'
# month = 'july'
month = 'august'

# INCLUDE THI CALCULATION
# Equation:
# THI = T - (0.55-0.0055*H)*(T-58.8)
for i in sensors:
    obj = th_analysis('lora/'+month+'/grafana_data_export ('+str(i)+').csv')
    data = obj.get_th_dataframe()
    df_t = data[0]
    df_h = data[1]
    # df_t.head()
    # print(df_t.index.is_unique)
    # df_t = (df_t - 32) * 5 / 9  # Convert to Celsius
    # df_thi = df_t - (0.55 - 0.0055 * df_h) * (df_t - 58.8)
    temp_data.append(obj.daily_calculations1(df_t,'Temp'))
    hum_data.append(obj.daily_calculations1(df_h, 'Hum'))
    thi_data.append(obj.daily_calculations2(df_t, df_h))

sensors_data = {'Temp': temp_data, 'Hum': hum_data}    # temp_data -> sensor_id -> days -> values (m,a,e,n)
sensors_data2 = {'Temp': thi_data[0], 'Hum': thi_data[1], 'THI': thi_data[2]}
sensor_list = []

for i in range(0,n_sensors):
    sensor_list.append('S'+str(i))

end_day = max(list(sensors_data[var][0]))
day = min(list(sensors_data[var][0]))
# day = day + 1
end_day = day + 1

while day < end_day + 1:
# day = 4     # day
    daily_data = []
    for s in sensors_data[var]:
        daily_data.append(s[day])


    time_index = daily_data[0].reset_index()
    time_index['Time'] = pd.to_datetime(time_index['Time'])
    time_index.set_index('Time', inplace=True)
    time_index.index = time_index.index.tz_convert('America/Los_Angeles')
    time_index = time_index.sort_index()
    for i,data in enumerate(daily_data):
        time_index[sensor_list[i]] = data
    time_index.reset_index(inplace=True)
    date = str(time_index['Time'][0].date())
    time_index.drop(columns=['Time','Value'], inplace=True)
    time_index.round(1)


    if var == 'Hum':
        sns.heatmap(time_index, cmap="RdBu_r",vmin=15, vmax=80, annot=True, fmt=".0f")
    else:
        sns.heatmap(time_index, cmap="RdBu_r", vmin=13, vmax=41, annot=True, fmt=".0f")
    plt.title(var+': '+date)
    plt.xlabel('Sensor')
    plt.ylabel('Hour of the day')
    # plt.savefig('lora/'+month+'/plots/'+var+'/'+var+'_'+date+'.png',bbox_inches='tight')
    plt.show()
    day = day + 1

