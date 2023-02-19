import os
import pandas as pd
import numpy as np
import datetime
import time

# local imports
from . import utils
from . import demands
from .sim_results import Results
from . import electricity
from . import opt

pd.set_option('display.max_columns', 999)
pd.set_option('display.width', 1000)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.options.mode.chained_assignment = 'raise'

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
        self.tariff_epsilon = 0.0002
        self.network = network
        self.vars = pd.DataFrame()

        self.date_tariff = electricity.vectorized_tariff(self.data_folder, self.time_range)
        self.date_tariff = self.split_range(self.date_tariff, self.hr_step_size)
        self.time_range = self.date_tariff.index
        self.num_steps = self.get_num_steps()

        self.demand_tanks = {}  # collection of tanks with actual demand values - for uncertainty analysis
        self.lp_model = None
        self.results = None

        self.build()

    def get_time_range(self):
        if isinstance(self.t1, datetime.datetime) and isinstance(self.t2, datetime.datetime):
            tr = pd.date_range(start=self.t1, periods=self.duration, freq="H")
        elif isinstance(self.t1, int) and isinstance(self.t2, int):
            tr = pd.Series(np.arange(start=self.t1, stop=self.t2 + 1, step=1))
        return tr

    def split_range(self, df, n):
        df.index.name = 'time'
        df['group'] = (df['name'].ne(df['name'].shift()) | (df['day'].ne(df['day'].shift()))).cumsum()
        df = df.groupby('group')
        temp = pd.DataFrame()
        for group, data in df:
            data.reset_index(inplace=True)
            data = data.groupby(data.index // n).first()
            temp = pd.concat([temp, data])
        temp['step_size'] = temp['time'].diff().shift(-1) / np.timedelta64(1, 'h')
        temp.loc[temp.index[-1], 'step_size'] = (self.t2 - temp['time'].iloc[-1]) / np.timedelta64(1, 'h')
        temp.index = temp['time']
        return temp

    def match_to_sim_timeindex(self, df):
        df.index = self.get_time_range()
        temp = pd.DataFrame(index=self.date_tariff.index)
        temp['group'] = range(len(temp))
        df = pd.merge(temp, df, left_index=True, right_index=True, how='right').ffill()
        df = df.groupby('group').sum()
        df.index = temp.index
        (df)
        return df

    def get_num_steps(self):
        return len(self.time_range)

    def get_duration(self):
        if isinstance(self.t1, datetime.datetime) and isinstance(self.t2, datetime.datetime):
            return utils.get_hours_between_timestamps(self.t1, self.t2)
        if isinstance(self.t1, int) and isinstance(self.t2, int):
            return self.t2 - self.t1

    def get_tariffs_vectors(self):
        low_voltage_tariff = self.time_range.apply(lambda x: electricity.get_tariff(x, 'low')[1])
        high_voltage_tariff = self.time_range.apply(lambda x: electricity.get_tariff(x, 'high')[1])
        up_voltage_tariff = self.time_range.apply(lambda x: electricity.get_tariff(x, 'up')[1])
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
            temp = pd.concat([s.combs] * len(self.time_range))
            tariff_vector = np.repeat(self.date_tariff[s.voltage_type], len(s.combs)).values
            idx = pd.MultiIndex.from_arrays([np.repeat(self.time_range, len(s.combs)), temp.index])
            temp.set_index(idx, inplace=True)
            temp.index.set_names(['time', 'comb'], inplace=True)
            temp.loc[:, 'tariff'] = tariff_vector

            step_size = (temp.index.get_level_values(0).drop_duplicates().to_series().diff()).shift(-1).rename('dt')
            step_size[-1] = self.t2 - temp.index.get_level_values(0).max()
            step_size = step_size / np.timedelta64(1, 'h')
            temp = pd.merge(temp, step_size, left_index=True, right_index=True)
            temp = separate_consecutive_tariffs(temp)
            temp['eps'] *= self.tariff_epsilon
            temp['cost'] = temp['flow'] * temp['se'] * temp['tariff'] * (1 + temp['eps'])

            temp.loc[:, 'network_element'] = s
            temp.loc[:, 'name'] = s_name
            temp.loc[:, 'element_type'] = 'station'
            self.vars = pd.concat([self.vars, temp])
            s.vars = temp

        for vsp_name, vsp in self.network.vsp.items():
            temp = pd.DataFrame(index=self.time_range)
            temp.index.set_names('time', inplace=True)
            temp['tariff'] = self.date_tariff[vsp.voltage_type]
            temp['cost'] = vsp.power * temp['tariff']

            temp.loc[:, 'network_element'] = vsp
            temp.loc[:, 'name'] = vsp_name
            temp.loc[:, 'element_type'] = 'vsp'
            self.vars = pd.concat([self.vars, temp])
            vsp.vars = temp

        for w_name, w in self.network.wells.items():
            temp = pd.concat([w.combs] * len(self.time_range))
            tariff_vector = np.repeat(self.date_tariff[w.voltage_type], len(w.combs)).values
            idx = pd.MultiIndex.from_arrays([np.repeat(self.time_range, len(w.combs)), temp.index])
            temp.set_index(idx, inplace=True)
            temp.index.set_names(['time', 'comb'], inplace=True)

            temp.loc[:, 'tariff'] = tariff_vector
            step_size = (temp.index.get_level_values(0).drop_duplicates().to_series().diff())
            step_size = step_size.shift(-1).rename('step_size')
            step_size[-1] = self.t2 - temp.index.get_level_values(0).max()
            step_size = step_size / np.timedelta64(1, 'h')
            temp = pd.merge(temp, step_size, left_index=True, right_index=True)
            temp['cost'] = temp['flow'] * temp['se'] * temp['tariff']

            temp.loc[:, 'network_element'] = w
            temp.loc[:, 'name'] = w_name
            temp.loc[:, 'element_type'] = 'well'
            self.vars = pd.concat([self.vars, temp])
            w.vars = temp

        for cv_name, cv in self.network.control_valves.items():
            temp = pd.concat([cv.combs] * len(self.time_range))
            idx = pd.MultiIndex.from_arrays([np.repeat(self.time_range, len(cv.combs)), temp.index])
            temp.set_index(idx, inplace=True)
            temp.index.set_names(['time', 'comb'], inplace=True)
            temp['cost'] = 0

            temp.loc[:, 'network_element'] = cv
            temp.loc[:, 'name'] = cv_name
            temp.loc[:, 'element_type'] = 'cv'
            self.vars = pd.concat([self.vars, temp])
            cv.vars = temp

        for v_name, v in self.network.valves.items():
            temp = pd.DataFrame(index=self.time_range)
            temp.index.set_names('time', inplace=True)
            temp['cost'] = 0

            temp.loc[:, 'network_element'] = v
            temp.loc[:, 'name'] = v_name
            temp.loc[:, 'element_type'] = 'valve'
            self.vars = pd.concat([self.vars, temp])
            v.vars = temp

        for t_name, t in self.network.tanks.items():
            t.vars = pd.DataFrame(index=self.time_range)
            t.vars['cost'] = 0
            demands_file = os.path.join(self.data_folder, 'Demands', 'demands.' + str(int(t.zone)))
            tank_demands = self.match_to_sim_timeindex(demands.load_from_csv(demands_file, self.t1, self.t2))
            t.vars['demand'] = tank_demands
            t.vars = demands.demand_factorize(t.zone, t.vars, self.data_folder)
            if self.dynamic_min_vol:
                self.get_dynamic_min_vol(t)
            else:
                t.vars['min_vol'] = np.full(shape=(len(t.vars), 1), fill_value=t.min_vol)
                t.min_vol = np.full(shape=(len(t.vars), 1), fill_value=t.min_vol)

            if t.vars['demand'].all():
                self.demand_tanks[t_name] = t

        self.vars.reset_index(inplace=True, drop=True)

    def lp_formulate(self):
        self.lp_model = opt.LP(self)
        self.lp_model.declare_vars()
        self.lp_model.one_comb_only()
        self.lp_model.mass_balance()
        self.lp_model.entities_dependency()
        self.lp_model.units_availability(self.data_folder + '/availability.csv')
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


def separate_consecutive_tariffs(df):
    df['group'] = df['tariff'].ne(df['tariff'].shift()).cumsum()
    df['auxiliary'] = df['group'].diff().replace({0: np.nan})
    df.iloc[[0, 0], df.columns.get_loc('auxiliary')] = 1
    df['auxiliary'] = df['auxiliary'].ffill() + df.groupby(df['auxiliary'].notnull().cumsum()).cumcount()
    df['eps'] = ((df['auxiliary'] - 1) / len(df.index.get_level_values('comb').drop_duplicates())).astype(int)
    df = df.drop('auxiliary', axis=1)
    return df


