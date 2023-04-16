import numpy as np
import pandas as pd
import math
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import dates as mdates
from matplotlib.colors import TwoSlopeNorm
from matplotlib.ticker import FormatStrFormatter
from matplotlib import ticker as mtick
from matplotlib.lines import Line2D
import matplotlib.patheffects as PathEffects


class Colors:
    BlwtRd = ["#023047", "#126782", "#219ebc", "#ffffff", "#DC7F2E", "#CE6550", "#AF4831"]
    WhtBlRd = ["#ffffff", "#8ecae6", "#046B9F", "#fdf4b0", "#ffbf1f", "#b33005"]


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

    def tank(self, tank, ax=None, linestyle='solid', level=False, color='k', ylabel=False, label=False,
             background=True, min_vol=True, demand=False):
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
        ax.plot(x, y, marker='o', color=color, markersize=4, markerfacecolor='none', linestyle=linestyle, label=label)
        if min_vol:
            ax.plot(x, [tank.min_vol[0]] + tank.min_vol.tolist(), linewidth=1, c='k')
        if demand:
            ax.plot(tank.vars.index, tank.vars['demand'], marker='o', markersize=4, markerfacecolor='none')
        
        if ylabel:
            ax.set_ylabel(ylabel)

        if self.x_ticks == 'datetime':
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%d/%m %H:%M'))
            ax.tick_params(axis='x', rotation=30)

        if background:
            ax = self.tariff_background(ax)

            # Legend
            marksize = 8
            font_size = 9
            legend_elements = [Line2D([0], [0], marker='o', markerfacecolor=(255 / 255, 0, 0, 0.2), label='High',
                                      markeredgecolor=(0, 0, 0), linewidth=0, markeredgewidth=0.4, markersize=marksize),
                               Line2D([2], [0], marker='o', markerfacecolor=(0, 128 / 255, 0, 0.2), label='Low',
                                      markeredgecolor=(0, 0, 0), linewidth=0, markeredgewidth=0.4, markersize=marksize)]

            leg = ax.legend(handles=legend_elements, title="Electricity tariff", fontsize=font_size)
            leg.set_title('Electricity tariff', prop={'size': font_size})
            leg._legend_box.align = "left"

        ax.grid()
        return ax

    def all_tanks(self, level=False, sharey=False):
        n_cols = int(len(self.sim.network.tanks) / 2 + 1)
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

        df = df[['result_flow']].groupby(level='time').sum()
        df = df.round(3)
        if self.x_ticks == 'hours':
            df.index = range(len(df))

        ax.step(df.index, df['result_flow'], 'o-', color='k', markerfacecolor='none', markersize=4, where='post',
                label=facility.name)

        ax.set_ylabel('flow ($m^3$/hr)')
        ax.ticklabel_format(useOffset=False, style='plain', axis='y')

        ax = self.tariff_background(ax)
        ax.grid()
        return ax

    def facilities_flow(self, facilities, nrows=2):
        fig, axes = plt.subplots(nrows=nrows, ncols=math.ceil(len(facilities) / nrows),
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

        for x in tank.pumps_inflows:
            ax = self.facility_flow(x, ax)
        for x in tank.v_inflows:
            ax = self.valve_flow(x, ax)

        for x in tank.pumps_outflows:
            ax = self.facility_flow(x, ax)
        for x in tank.v_outflows:
            ax = self.valve_flow(x, ax)

        return ax

    def all_pressure_zones(self):
        fig, axes = plt.subplots(nrows=2, ncols=int(len(self.sim.network.tanks) / 2), sharex=True, figsize=(12, 6))
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

    def station_gantt(self, station, ax):
        if ax is None:
            fig, ax = plt.subplots()

        df = station.vars.copy()
        df.index = df.index.droplevel(1)
        # df = df[df['value'] != 0]

        cols_to_drop = ['flow', 'se', 'power', 'efficiency', 'head', 'num_units', 'tariff', 'step_size',
                        'group', 'eps', 'cost', 'availability', 'dt']
        cols_to_drop = list(set(cols_to_drop) & set(df.columns))
        df.drop(cols_to_drop, axis=1, inplace=True)
        df = df.replace({'ON': 1, 'OFF': 0})

        units_cols = [col for col in df.columns if col != 'value']
        df = df.loc[:, units_cols].multiply(df['value'], axis='index')
        df = df.groupby(level=0).sum()
        df = df.reindex(sorted(df.columns), axis=1)

        num_units = len(units_cols)
        pe = [PathEffects.Stroke(linewidth=6, foreground='black'),
              PathEffects.Stroke(foreground='black'),
              PathEffects.Normal()]

        for i, unit in enumerate(units_cols):
            temp = df[[unit]].copy()
            temp.index = range(len(temp))
            temp.loc[:, 'start'] = temp.index
            temp.loc[:, 'end'] = temp['start'] + pd.Series(temp[unit])
            ax.hlines([i], temp.index.min(), temp.index.max() + 1, linewidth=5, color='w', path_effects=pe)
            ax.hlines(np.repeat(i, len(temp)), temp['end'], temp['start'], linewidth=5, colors='black')

        temp = self.sim.network.wells['W1'].vars.copy()
        temp.index = range(len(temp))
        temp.loc[:, 'start'] = temp.index
        temp.loc[:, 'end'] = temp['start'] + pd.Series(temp['value'])
        ax.hlines([i + 1], temp.index.min(), temp.index.max() + 1, linewidth=5, color='w', path_effects=pe)
        ax.hlines(np.repeat(i + 1, len(temp)), temp['end'], temp['start'], linewidth=5, colors='black')

        ax.xaxis.grid(True)
        ax.set_yticks([i for i in range(len(df.columns) + 1)])
        ax.set_yticklabels(['Pump 1', 'Pump 2', 'Well'])
        return ax


def correlation_matrix(mat, major_ticks=False, norm=False, hex_colors=Colors.WhtBlRd):
    if norm:
        mat = (mat - mat.min()) / (mat.max() - mat.min())

    cmap = get_continuous_cmap(hex_colors)
    mat_norm = max(abs(mat.min()), abs(mat.max()))
    im = plt.imshow(mat, cmap=cmap, vmin=0, vmax=mat_norm)
    ax = plt.gca()

    ax.tick_params(which='minor', bottom=False, left=False)
    cbar = plt.colorbar(im, ticks=mtick.AutoLocator())

    # Major ticks
    if major_ticks:
        ax.set_xticks(np.arange(-0.5, mat.shape[0], major_ticks))
        ax.set_yticks(np.arange(-0.5, mat.shape[0], major_ticks))
        ax.set_xticklabels(np.arange(0, mat.shape[0] + major_ticks, major_ticks))
        ax.set_yticklabels(np.arange(0, mat.shape[0] + major_ticks, major_ticks))
        ax.grid(which='major', color='k', linestyle='-', linewidth=1)

    # Grid lines based on minor ticks
    ax.grid(which='minor', color='k', linestyle='-', linewidth=0.5, alpha=0.4)
    ax.set_xticks(np.arange(-0.5, mat.shape[0], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, mat.shape[0], 1), minor=True)

    plt.subplots_adjust(top=0.9, bottom=0.13, left=0.055, right=0.9, hspace=0.2, wspace=0.2)


def hex_to_rgb(value):
    """
    Converts hex to rgb colours
    value: string of 6 characters representing a hex colour.
    Returns: list length 3 of RGB values
    """
    value = value.strip("#")  # removes hash symbol if present
    lv = len(value)
    return tuple(int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))


def rgb_to_dec(value):
    """
    Converts rgb to decimal colours (i.e. divides each value by 256)
    value: list (length 3) of RGB values
    Returns: list (length 3) of decimal values
    """
    return [v / 256 for v in value]


def get_continuous_cmap(hex_list, float_list=None):
    """
        creates and returns a color map that can be used in heat map figures.
        If float_list is not provided, colour map graduates linearly between each color in hex_list.
        If float_list is provided, each color in hex_list is mapped to the respective location in float_list.

        Parameters
        ----------
        hex_list: list of hex code strings
        float_list: list of floats between 0 and 1, same length as hex_list. Must start with 0 and end with 1.

        Returns color map
    """
    rgb_list = [rgb_to_dec(hex_to_rgb(i)) for i in hex_list]
    if float_list:
        pass
    else:
        float_list = list(np.linspace(0, 1, len(rgb_list)))

    cdict = dict()
    for num, col in enumerate(['red', 'green', 'blue']):
        col_list = [[float_list[i], rgb_list[i][num], rgb_list[i][num]] for i in range(len(float_list))]
        cdict[col] = col_list
    cmp = mpl.colors.LinearSegmentedColormap('my_cmp', segmentdata=cdict, N=256)
    return cmp
