import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from matplotlib import dates as mdates
from matplotlib.ticker import FormatStrFormatter
from matplotlib import ticker as mtick

from . import utils


class SimGraphs:
    def __init__(self, sim, x_ticks='hours', n_rows=2):
        self.sim = sim
        self.x_ticks = x_ticks
        self.n_rows = n_rows

        if self.x_ticks == 'hours':
            self.dt = self.sim.hr_step_size
        else:
            self.dt = pd.to_timedelta(self.sim.hr_step_size, unit='h')

    def tariff_background(self, ax):
        alpha = 0.2

        df = pd.DataFrame({'Datetime': self.sim.time_range})
        df = pd.merge(df, self.sim.date_tariff['name'], left_on='Datetime', right_on='time')
        for (TARIFF, _), group in df.groupby(['name', df['name'].ne('ON').cumsum()]):
            if TARIFF == 'ON':
                start = group.index.min()
                end = group.index.max() + self.dt
                ax.axvspan(start, end, facecolor='none', edgecolor='red', alpha=alpha, hatch='/////', linewidth=1)

        for (TARIFF, _), group in df.groupby(['name', df['name'].ne('OFF').cumsum()]):
            if TARIFF == 'OFF':
                start = group.index.min()
                end = group.index.max() + self.dt
                ax.axvspan(start, end, facecolor='none', edgecolor='green', alpha=alpha, hatch='/////', linewidth=1)

        return ax

    def tank(self, tank, ax=None, linestyle='solid', level=False, color='k', ylabel=False, label=False, background=True):
        if ax is None:
            fig, ax = plt.subplots()

        x = [self.sim.t1] + tank.vars.index.to_list()
        y = [tank.initial_vol] + tank.vars['value'].to_list()
        x0 = self.sim.t1
        y0 = tank.initial_vol

        if level:
            y = tank.vol_to_level(np.array(y))
            y0 = tank.initial_level

        if self.x_ticks == 'hours':
            x = range(len(x))
            x0 = 0

        if not label:
            label = tank.name
        ax.plot(x0, y0, 'r', marker='o', markersize=4)
        ax.plot(x, y, marker='o',color=color,  markersize=4, markerfacecolor='none', linestyle=linestyle, label=label)

        # ax.plot(tank.vars.index, tank.vars['demand'], marker='o', markersize=4, markerfacecolor='none')
        ax.plot(x, tank.vars['min_vol'].to_list() + [tank.final_vol], linewidth=1, c='k')
        ax.grid()
        if ylabel:
            ax.set_ylabel(ylabel)

        if self.x_ticks == 'datetime':
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%d/%m %H:%M'))
            ax.tick_params(axis='x', rotation=30)

        if background:
            ax = self.tariff_background(ax)
        return ax

    def all_tanks(self, level=False, sharey=False):
        n_cols = int(len(self.sim.network.tanks)/2 + 1)
        fig, axes = plt.subplots(nrows=self.n_rows, ncols=n_cols, sharex=True, sharey=sharey, figsize=(12, 6))
        axes = axes.ravel()

        for i, (tank_name, tank) in enumerate(self.sim.network.tanks.items()):
            axes[i] = self.tank(tank, level=level, ax=axes[i])
            axes[i].set_title(tank_name)

        plt.subplots_adjust(left=0.08, right=0.92, wspace=0.4, hspace=0.3)

    def facility_flow(self, facility, facility_type, ax=None):
        if ax is None:
            fig, ax = plt.subplots()

        df = facility.vars.copy()
        if facility_type in ['PumpStation', 'Well']:
            df['value'] = np.where(df['value'] > 0.999, 1, df['value'])
            df['value'] = np.where(df['value'] < 0.0001, 0, df['value'])
            df = df[['flow', 'value']]
            df['result_flow'] = df['flow'] * df['value']

        elif facility_type in ['VSP', 'Valve']:
            df['result_flow'] = df['value']

        df = df.groupby(level='time').sum()

        if self.x_ticks == 'hours':
            df.index = range(len(df))

        ax.step(df.index, df['result_flow'], 'o-', color='k', markerfacecolor='none', markersize=4, where='post',
                label=facility.name)

        ax.set_ylabel('flow ($m^3$/hr)')
        ax.ticklabel_format(useOffset=False, style='plain', axis='y')

        ax = self.tariff_background(ax)
        ax.grid()
        return ax

    def facilities_flow(self, facilities):
        fig, axes = plt.subplots(nrows=2, ncols=math.ceil(len(facilities) / 2),
                                 sharex=True, figsize=(12, 6))
        axes = axes.ravel()
        for i, (f_name, f) in enumerate(facilities.items()):
            axes[i] = self.facility_flow(f, f.__class__.__name__, axes[i])
            axes[i].set_title(f_name)

        plt.subplots_adjust(left=0.08, right=0.92, wspace=0.4, hspace=0.3)

    def power_station(self, power_station, ax=None):
        if ax is None:
            fig, ax = plt.subplots()

        df = pd.DataFrame(index=self.sim.time_range)
        for e in power_station.elements:
            temp = e.vars.copy()
            temp['value'] = np.where(temp['value'] > 0.001, 1, 0)
            temp['result_power'] = temp['value'] * temp['power']
            temp = temp.groupby(level='time').max()
            df = pd.merge(df, temp['result_power'].rename(e.name), left_index=True, right_index=True)

        df['result_power'] = df.sum(axis=1)

        if self.x_ticks == 'hours':
            df.index = range(len(df))

        ax.step(df.index, df['result_power'], 'o-', color='k', markerfacecolor='none', markersize=4, where='post',
                label=power_station.name)

        ax.set_ylabel('Power (kWh)')
        ax = self.tariff_background(ax)
        ax.grid()
        return ax

    def all_power_stations(self, power_stations: dict):
        fig, axes = plt.subplots(nrows=2, ncols=math.ceil(len(power_stations) / 2),
                                 sharex=True, figsize=(12, 6))
        axes = axes.ravel()
        for i, (ps_name, ps) in enumerate(power_stations.items()):
            axes[i] = self.power_station(ps, ax=axes[i])
            axes[i].set_title(ps_name)

        plt.subplots_adjust(left=0.08, right=0.92, wspace=0.4, hspace=0.3)
        plt.suptitle('Max power consumption')

    def system_demand(self):
        total_demand = 0
        for t_name, t in self.sim.network.tanks.items():
            total_demand += t.vars['demand']

        fig, ax = plt.subplots()
        ax.plot(total_demand)

    def all_valves(self):
        fig, axes = plt.subplots(nrows=2, ncols=int(len(self.sim.network.valves) / 2 + 1),
                                 sharex=True, figsize=(12, 6))
        axes = axes.ravel()

        for i, (v_name, v) in enumerate(self.sim.network.valves.items()):
            axes[i] = self.valve_flow(v, axes[i])
            axes[i].set_title(v_name)

        plt.subplots_adjust(left=0.08, right=0.92, wspace=0.4, hspace=0.3)

    def pressure_zone(self, tank, ax=None):
        if ax is None:
            fig, ax = plt.subplots()

        for x in tank.inflows:
            ax = self.facility_flow(x, ax)
        for x in tank.v_inflows:
            ax = self.valve_flow(x, ax)

        for x in tank.outflows:
            ax = self.facility_flow(x, ax)
        for x in tank.v_outflows:
            ax = self.valve_flow(x, ax)

        return ax

    def all_pressure_zones(self):
        fig, axes = plt.subplots(nrows=2, ncols=int(len(self.sim.network.tanks)/2), sharex=True, figsize=(12, 6))
        axes = axes.ravel()

        for i, (t_name, t) in enumerate(self.sim.network.tanks.items()):
            self.pressure_zone(t, axes[i])
            axes[i].set_title(t_name)
            axes[i].legend()

        plt.subplots_adjust(left=0.08, right=0.92, wspace=0.4, hspace=0.3)

    def plot_multivariate_sample(self, demands, sample):
        fig, axes = plt.subplots(nrows=2, ncols=math.ceil(demands.shape[0] / 2), sharex=True, figsize=(12, 6))
        axes = axes.ravel()

        for i in range(demands.shape[0]):
            axes[i].plot(sample[:, :, i].T, c='C0', alpha=0.6)
            axes[i].plot(demands[i, :], c='k')
            axes[i].grid()

        plt.tight_layout()
        plt.show()