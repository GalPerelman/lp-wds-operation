import os
import pandas as pd
import numpy as np

pd.set_option('display.max_columns', 999)
pd.set_option('display.width', 1000)


class Network:
    def __init__(self, data_folder, name=None):
        self.data_folder = data_folder
        self.pump_stations = {}
        self.vsp = {}
        self.wells = {}
        self.tanks = {}
        self.control_valves = {}
        self.valves = {}
        self.power_stations = {}
        self.zones = {}

        self.build()
        self.comb_elements = {**self.pump_stations, **self.wells, **self.control_valves}
        self.flow_elements = {**self.pump_stations, **self.wells, **self.control_valves, **self.valves, **self.vsp}
        self.cost_elements = {**self.pump_stations, **self.wells, **self.vsp}

    def __getitem__(self, item):
        elements = {**self.pump_stations, **self.wells, **self.control_valves, **self.valves, **self.vsp, **self.tanks}
        return elements[item]

    def declare_stations(self):
        df = pd.read_csv(os.path.join(self.data_folder, 'stations.csv'))
        for index, row in df.iterrows():
            ps = PumpStation(**dict(row), data_folder=self.data_folder, reduced_combs=False)
            self.pump_stations[row['name']] = ps

    def declare_vsp(self):
        df = pd.read_csv(self.data_folder + '/vsp.csv')
        for index, row in df.iterrows():
            vsp = VSP(**dict(row))
            self.vsp[row['name']] = vsp

    def declare_wells(self):
        df = pd.read_csv(self.data_folder + '/wells.csv')
        df = df[df['include'] == 1]
        for index, row in df.iterrows():
            w = Well(**dict(row))
            self.wells[row['name']] = w

    def declare_tanks(self):
        df = pd.read_csv(self.data_folder + '/tanks.csv')
        for index, row in df.iterrows():
            t = Tank(**dict(row))
            self.tanks[row['name']] = t

    def declare_control_valves(self):
        df = pd.read_csv(self.data_folder + '/control_valves.csv')
        if not df.empty:
            for index, row in df.iterrows():
                cv = CV(**dict(row))
                self.control_valves[row['name']] = cv

    def declare_valves(self):
        df = pd.read_csv(self.data_folder + '/valves.csv')
        if not df.empty:
            for index, row in df.iterrows():
                v = Valve(**dict(row))
                self.valves[row['name']] = v

    def declare_zones(self):
        df = pd.read_csv(self.data_folder + '/zones.csv')
        for index, row in df.iterrows():
            z = Zone(**dict(row))
            self.zones[row['name']] = z

    def declare_power_stations(self):
        df = pd.read_csv(self.data_folder + '/power_stations.csv')
        for index, row in df.iterrows():
            ps = PowerStation(**dict(row))
            self.power_stations[row['name']] = ps

    def set_tanks_links(self):
        for t_name, t in self.tanks.items():
            t.inflows = [s for s in self.pump_stations.values() if s.to_zone_id == t.zone] \
                        + [w for w in self.wells.values() if w.to_zone_id == t.zone]

            t.outflows = [s for s in self.pump_stations.values() if s.from_zone_id == t.zone]

            t.vsp_inflows = [vsp for vsp in self.vsp.values() if vsp.to_zone_id == t.zone]
            t.vsp_outflows = [vsp for vsp in self.vsp.values() if vsp.from_zone_id == t.zone]

            t.cv_inflows = [cv for cv in self.control_valves.values() if cv.to_zone_id == t.zone]
            t.cv_outflows = [cv for cv in self.control_valves.values() if cv.from_zone_id == t.zone]

            t.v_inflows = [v for v in self.valves.values() if v.to_zone_id == t.zone]
            t.v_outflows = [v for v in self.valves.values() if v.from_zone_id == t.zone]

    def set_power_stations(self):
        for ps_name, ps in self.power_stations.items():
            ps.elements = [s for s in self.pump_stations.values() if s.power_station == ps_name]

    def build(self):
        self.declare_stations()
        self.declare_vsp()
        self.declare_wells()
        self.declare_control_valves()
        self.declare_valves()
        self.declare_tanks()
        self.declare_zones()
        self.declare_power_stations()

        self.set_tanks_links()
        self.set_power_stations()


class PumpStation:
    def __init__(self, data_folder, name, from_zone_id, to_zone_id, voltage_type, power_station, reduced_combs=False,
                 **kwargs):
        self.data_folder = data_folder
        self.name = name
        self.voltage_type = voltage_type
        self.from_zone_id = from_zone_id
        self.to_zone_id = to_zone_id
        self.power_station = power_station
        self.__dict__.update(kwargs)

        if reduced_combs:
            self.combs = self.reduce_combs()
        else:
            self.combs = self.load_combs()

    def load_combs(self):
        return pd.read_csv(self.data_folder + '/Combs/' + self.name + '.csv')

    def reduce_combs(self):
        self.combs = self.load_combs()
        return self.combs.loc[self.combs.groupby('num_units')['se'].idxmin()].reset_index().drop('index', axis=1)


class VSP:
    def __init__(self, name, from_zone_id, to_zone_id, power, voltage_type, min_flow, max_flow, init_flow, **kwargs):
        self.name = name
        self.from_zone_id = from_zone_id
        self.to_zone_id = to_zone_id
        self.power = power
        self.voltage_type = voltage_type
        self.min_flow = min_flow
        self.max_flow = max_flow
        self.init_flow = init_flow
        self.__dict__.update(kwargs)


class Well:
    def __init__(self, name, to_zone_id, flow, se, voltage_type, **kwargs):
        self.name = name
        self.to_zone_id = to_zone_id
        self.combs = pd.DataFrame({'flow': flow, 'se': se}, index=[0])
        self.voltage_type = voltage_type
        self.__dict__.update(kwargs)


class CV:
    def __init__(self, data_folder, name, from_zone_id, to_zone_id, **kwargs):
        self.data_folder = data_folder
        self.name = name
        self.from_zone_id = from_zone_id
        self.to_zone_id = to_zone_id
        self.__dict__.update(kwargs)

        self.combs = self.load_combs()

    def load_combs(self):
        return pd.read_csv(self.data_folder + '/Combs/' + self.name + '.csv')


class Valve:
    def __init__(self, name, from_zone_id, to_zone_id, max_flow, min_flow, **kwargs):
        self.name = name
        self.from_zone_id = from_zone_id
        self.to_zone_id = to_zone_id
        self.max_flow = max_flow
        self.min_flow = min_flow
        self.__dict__.update(kwargs)


class Tank:
    def __init__(self, name, zone, diameter, max_level, min_level, initial_level, final_level, pattern_zone, **kwargs):
        self.name = name
        self.zone = zone
        self.diameter = diameter
        self.max_level = max_level
        self.min_level = min_level
        self.initial_level = initial_level
        self.max_vol = self.level_to_vol(self.max_level)
        self.min_vol = self.level_to_vol(self.min_level)
        self.initial_vol = self.level_to_vol(self.initial_level)
        self.pattern_zone = pattern_zone
        self.__dict__.update(kwargs)

        self.final_level = self.get_final_level(final_level)
        self.final_vol = self.level_to_vol(self.final_level)

        self.inflows = None
        self.outflows = None
        self.cv_inflows = None
        self.cv_outflows = None
        self.v_inflows = None
        self.v_outflows = None
        self.vsp_inflows = None
        self.vsp_outflows = None        

    def get_final_level(self, x):
        if np.isnan(x):
            return self.initial_level
        else:
            return x

    def level_to_vol(self, level):
        return level * np.pi * self.diameter ** 2 / 4

    def vol_to_level(self, vol):
        return np.sqrt((4 * vol) / (np.pi * self.diameter ** 2))


class Zone:
    def __init__(self, name, zone_id, tank, **kwargs):
        self.name = str(name)
        self.zone_id = zone_id
        self.tank = tank
        self.__dict__.update(kwargs)


class PowerStation:
    def __init__(self, name, max_power_off, max_power_on, **kwargs):
        self.name = name
        self.max_power = max_power_off
        self.max_power_on = max_power_on
        self.__dict__.update(kwargs)

        self.elements = None
