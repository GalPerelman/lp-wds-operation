import os
import datetime

import demands

T1 = datetime.datetime(2010, 1, 1, 0, 00)
T2 = datetime.datetime(2022, 1, 1, 0, 00)
history_dir = 'data/aj/Demands/48/History'
dem = demands.demands_from_DS_hisotry(history_dir, T1, T2)

zones_dem_dirs = [f.path for f in os.scandir('data/aj/Demands') if f.is_dir()]

for d in zones_dem_dirs:
    try:
        dem = demands.demands_from_DS_hisotry(d + '/History', T1, T2)
        z = (d).split(os.sep)[-1]
        dem['Demand'].to_csv('data/aj/Demands' + '/demands.' + str(z))
        print('===================')
    except FileNotFoundError:
        pass