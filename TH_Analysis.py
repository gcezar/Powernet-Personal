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

    def daily_th_calc(self, df, mode):
        df = df.resample('H').mean()
        if mode == 'Temp':
            df = (df-32)*5/9    # Convert to Celsius
        df_list = [group[1] for group in df.groupby(df.index.day)]  # split month into days
        output = {}
        for i in df_list:
            output[i.index.day[-1]] = i
        return output

    def daily_thi_calc(self, df_t, df_h):
        # Temperature
        df_t = df_t.resample('H').mean()
        df_t = ((df_t['Value'] - 32) * 5 / 9).to_frame()  # Convert to Celsius
        # Humidity
        df_h = df_h.resample('H').mean()
        df_h = df_h['Value'].to_frame()
        # THI
        df_thi = (1.8*df_t+32) - ((0.55 - 0.0055 * df_h) * (1.8*df_t - 26.8))
        # df_thi = ser_thi.to_frame()

        df_thi_list = [group[1] for group in df_thi.groupby(df_thi.index.day)]  # split month into days
        output_thi = {}
        for th in df_thi_list:
            output_thi[th.index.day[-1]] = th

        return output_thi

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
# var = 'THI'
# month = 'june'
# Month = 'June'
# month = 'july'
# Month = 'July'
month = 'august'
Month = 'August'
# month = 'september'
# Month = 'September'
# month = 'october'
# Month = 'October'

for i in sensors:
    obj = th_analysis('lora/'+month+'/grafana_data_export ('+str(i)+').csv')
    data = obj.get_th_dataframe()
    df_t = data[0]
    df_h = data[1]

    temp_data.append(obj.daily_th_calc(df_t,'Temp'))
    hum_data.append(obj.daily_th_calc(df_h, 'Hum'))
    thi_data.append(obj.daily_thi_calc(df_t, df_h))

# sensors_data = {'Temp': temp_data, 'Hum': hum_data}    # temp_data -> sensor_id -> days -> values (m,a,e,n)
sensors_data = {'Temp': temp_data, 'Hum': hum_data, 'THI': thi_data}
sensor_list = []

for i in range(0,n_sensors):
    sensor_list.append('S'+str(i))
sensor_list[-1] = 'Sout'

end_day = max(list(sensors_data[var][0]))
day = min(list(sensors_data[var][0]))
# day = day + 1
# end_day = day + 1

# day = 15
index = 0
df_avg = pd.DataFrame(columns=['0'])
df_std = pd.DataFrame(columns=['0'])
while day < end_day + 1:
# while day < 18:
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
    elif var == 'Temp':
        sns.heatmap(time_index, cmap="RdBu_r", vmin=13, vmax=41, annot=True, fmt=".0f")
    else:
        sns.heatmap(time_index, cmap="RdBu_r", vmin=55, vmax=85, annot=True, fmt=".0f")
    plt.title(var+': '+date)
    plt.xlabel('Sensor')
    plt.ylabel('Hour of the day')
    plt.savefig('lora/'+month+'/plots/'+var+'/'+var+'_'+date+'.png',bbox_inches='tight')
    plt.show()
    day = day + 1
    avg = time_index.mean(axis=1).tolist()
    if len(avg) >= 24:
        delta = len(avg)-24
        if delta > 0:
            avg = avg[:24]
        df_avg[str(index)] = avg  # hourly avg for each day

    std = time_index.std(axis=1).tolist()
    if len(std) >= 24:
        delta = len(std) - 24
        if delta > 0:
            std = std[:24]
        df_std[str(index)] = std  # hourly std for each day

    index = index + 1

avg_thi = df_avg.mean(axis=1)
std_thi = df_std.mean(axis=1)
plt.errorbar(avg_thi.index, avg_thi, yerr=std_thi, marker="o")
plt.title('Average daily THI in ' + Month)
plt.xlabel('Hour of the day')
plt.ylabel('THI')
# plt.savefig('lora/monthly_thi/'+Month+'.png',bbox_inches='tight')
plt.show()