import os
import pandas as pd
import numpy as np
import datetime
import time
import yaml

# local imports
from . import demands
from .sim_results import Results
from . import graphs
from . import utils
from . import opt

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

pd.set_option('display.max_columns', 999)
pd.set_option('display.width', 1000)
pd.set_option('display.float_format', lambda x: '%.3f' % x)


class Simulation:
    def __init__(self, data_folder, network, start_time, end_time, hr_step_size=1, dynamic_min_vol=False):
        self.start_time = time.time()
        self.t1 = start_time
        self.t2 = end_time
        self.data_folder = data_folder
        self.hr_step_size = hr_step_size
        self.dynamic_min_vol = dynamic_min_vol

        self.duration = self.get_duration()
        self.time_range = self.get_time_range()
        self.num_steps = self.get_num_steps()
        self.tariff_epsilon = 0.0002
        self.network = network

        self.date_tariff = utils.vectorized_tariff(self.data_folder, self.time_range)
        self.date_tariff = utils.split_range(self.date_tariff, self.hr_step_size)
        self.time_range = self.date_tariff.index

        self.lp_model = None
        self.results = None

        self.build()

    def get_time_range(self):
        if isinstance(self.t1, datetime.datetime) and isinstance(self.t2, datetime.datetime):
            tr = pd.date_range(start=self.t1, periods=self.duration, freq=str(self.hr_step_size) + "H")
        if isinstance(self.t1, int) and isinstance(self.t2, int):
            tr = pd.Series(np.arange(start=self.t1, stop=self.t2 + self.hr_step_size, step=self.hr_step_size))
        return tr

    def split_date_range(self, n):
        date_range = pd.date_range(start=self.t1, periods=self.duration, freq='60min')
        df = pd.DataFrame(index=date_range, columns=['tariff'])
        df.index.name = 'time'
        df['tariff'] = df.apply(lambda x: utils.get_tariff(x.name, 'high')[0], axis=1)
        df.reset_index(inplace=True)
        df['group'] = df['tariff'].ne(df['tariff'].shift()).cumsum()
        df = df.groupby('group')
        temp = pd.DataFrame()
        for group, data in df:
            data.reset_index(inplace=True)
            data = data.groupby(data.index // n).first()
            temp = pd.concat([temp, data])

        temp['step_size'] = temp['time'].diff()
        return temp

    def get_num_steps(self):
        return len(self.time_range)

    def get_duration(self):
        if isinstance(self.t1, datetime.datetime) and isinstance(self.t2, datetime.datetime):
            return utils.get_hours_between_timestamps(self.t1, self.t2)
        if isinstance(self.t1, int) and isinstance(self.t2, int):
            return self.t2 - self.t1

    def get_tariffs_vectors(self):
        low_voltage_tariff = self.time_range.apply(lambda x: utils.get_tariff(x, 'low')[1])
        high_voltage_tariff = self.time_range.apply(lambda x: utils.get_tariff(x, 'high')[1])
        up_voltage_tariff = self.time_range.apply(lambda x: utils.get_tariff(x, 'up')[1])
        df = pd.DataFrame({'low': low_voltage_tariff, 'high': high_voltage_tariff, 'up': up_voltage_tariff})
        return df

    def get_dynamic_min_vol(self, tank):
        qmax = sum([x.combs['flow'].max() for x in tank.inflows]) + sum([x.max_flow for x in tank.vsp_inflows])
        tank.vars['qmax'] = qmax
        tank.vars['v'] = tank.final_vol - tank.vars['demand'] - qmax

        dynamic_min = [tank.final_vol]
        for i, t in enumerate(self.time_range[::-1]):
            dynamic_min = [max(tank.min_vol, dynamic_min[0] + tank.vars.loc[t, 'demand'] - qmax)] + dynamic_min

        tank.vars['min_vol'] = dynamic_min[:-1]
        tank.min_vol = np.array(dynamic_min[:-1])
        tank.vars = tank.vars.drop(['qmax', 'v'], axis=1)

    def build(self):
        for s_name, s in self.network.pump_stations.items():
            s.vars = pd.concat([s.combs] * len(self.time_range))
            tariff_vector = np.repeat(self.date_tariff[s.voltage_type], len(s.combs)).values
            idx = pd.MultiIndex.from_arrays([np.repeat(self.time_range, len(s.combs)), s.vars.index])
            s.vars.set_index(idx, inplace=True)
            s.vars.index.set_names(['time', 'comb'], inplace=True)
            s.vars.loc[:, 'tariff'] = tariff_vector

            step_size = (s.vars.index.get_level_values(0).drop_duplicates().to_series().diff()).shift(-1).rename('dt')
            step_size[-1] = self.t2 - s.vars.index.get_level_values(0).max()
            step_size = step_size / np.timedelta64(1, 'h')
            s.vars = pd.merge(s.vars, step_size, left_index=True, right_index=True)
            s.vars = utils.separate_consecutive_tariffs(s.vars)
            s.vars['eps'] *= self.tariff_epsilon
            s.vars['cost'] = s.vars['flow'] * s.vars['se'] * s.vars['tariff'] * (1 + s.vars['eps'])

        for vsp_name, vsp in self.network.vsp.items():
            vsp.vars = pd.DataFrame(index=self.time_range)
            vsp.vars.index.set_names('time', inplace=True)
            vsp.vars['tariff'] = self.date_tariff[vsp.voltage_type]
            vsp.vars['cost'] = vsp.power * vsp.vars['tariff']

        for w_name, w in self.network.wells.items():
            w.vars = pd.concat([w.combs] * len(self.time_range))
            tariff_vector = np.repeat(self.date_tariff[w.voltage_type], len(w.combs)).values
            idx = pd.MultiIndex.from_arrays([np.repeat(self.time_range, len(w.combs)), w.vars.index])
            w.vars.set_index(idx, inplace=True)
            w.vars.index.set_names(['time', 'comb'], inplace=True)

            w.vars.loc[:, 'tariff'] = tariff_vector
            step_size = (w.vars.index.get_level_values(0).drop_duplicates().to_series().diff())
            step_size = step_size.shift(-1).rename('step_size')
            step_size[-1] = self.t2 - w.vars.index.get_level_values(0).max()
            step_size = step_size / np.timedelta64(1, 'h')
            w.vars = pd.merge(w.vars, step_size, left_index=True, right_index=True)
            w.vars['cost'] = w.vars['flow'] * w.vars['se'] * w.vars['tariff']

        for cv_name, cv in self.network.control_valves.items():
            cv.vars = pd.concat([cv.combs] * len(self.time_range))
            idx = pd.MultiIndex.from_arrays([np.repeat(self.time_range, len(cv.combs)), cv.vars.index])
            cv.vars.set_index(idx, inplace=True)
            cv.vars.index.set_names(['time', 'comb'], inplace=True)
            cv.vars['cost'] = 0

        for v_name, v in self.network.valves.items():
            v.vars = pd.DataFrame(index=self.time_range)
            v.vars.index.set_names('time', inplace=True)
            v.vars['cost'] = 0

        for t_name, t in self.network.tanks.items():
            t.vars = pd.DataFrame(index=self.time_range)
            t.vars['cost'] = 0
            demands_file = os.path.join(self.data_folder, 'Demands', 'demands.' + str(int(t.zone)))
            t.vars['demand'] = demands.load_from_csv(demands_file, self.t1, self.t2)
            t.vars = demands.demand_factorize(t.zone, t.vars, self.data_folder)
            if self.dynamic_min_vol:
                self.get_dynamic_min_vol(t)
            else:
                t.vars['min_vol'] = np.full(shape=(len(t.vars), 1), fill_value=t.min_vol)
                t.min_vol = np.full(shape=(len(t.vars), 1), fill_value=t.min_vol)

    def lp_formulate(self):
        self.lp_model = opt.LP(self)
        self.lp_model.declare_vars()
        self.lp_model.one_comb_only()
        self.lp_model.mass_balance()
        self.lp_model.entities_dependency()
        self.lp_model.units_availability(self.data_folder + '/availability.csv')
        self.lp_model.station_volume()
        self.lp_model.multi_stations_volume()
        self.lp_model.vsp_volume()
        self.lp_model.vsp_changes()
        self.lp_model.max_power()
        self.lp_model.objective_func()
        return self.lp_model

    def lp_run(self, integer=False):
        if integer:
            return self.lp_model.solve_integer()
        else:
            return self.lp_model.solve_primal()

    def get_results(self):
        self.results = Results(self)
