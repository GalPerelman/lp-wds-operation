import pandas as pd
import numpy as np

pd.set_option('display.max_columns', 999)
pd.set_option('display.width', 1000)


class Results:
    def __init__(self, sim):
        self.sim = sim

        self.summary = pd.DataFrame()
        self.detailed_summary = pd.DataFrame()

        self.total_demand = self.get_summary()

    def get_summary(self):
        total_demand = 0

        for s_name, s in self.sim.network.pump_stations.items():
            vol = sum(s.vars['flow'] * s.vars['value'])
            energy = sum(s.vars['flow'] * s.vars['se'] * s.vars['value'])
            cost = sum(s.vars['cost'] * s.vars['value'])
            s_res = pd.DataFrame({'name': s.name, 'vol': vol, 'energy': energy, 'cost': cost}, index=[0])
            self.summary = pd.concat([self.summary, s_res])

            s.vars['name'] = s_name
            self.detailed_summary = pd.concat([self.detailed_summary, s.vars])

        for vsp_name, vsp in self.sim.network.vsp.items():
            vol = sum(vsp.vars['value'])
            energy = sum(vsp.power * vsp.vars['value'])
            cost = sum(vsp.vars['cost'] * vsp.vars['value'])
            vsp_res = pd.DataFrame({'name': vsp.name, 'vol': vol, 'energy': energy, 'cost': cost}, index=[0])
            self.summary = pd.concat([self.summary, vsp_res])

            vsp.vars['name'] = vsp_name
            self.detailed_summary = pd.concat([self.detailed_summary, vsp.vars])

        for w_name, w in self.sim.network.wells.items():
            vol = sum(w.vars['flow'] * w.vars['value'])
            energy = sum(w.vars['flow'] * w.vars['se'] * w.vars['value'])
            cost = sum(w.vars['cost'] * w.vars['value'])
            w_res = pd.DataFrame({'name': w.name, 'vol': vol, 'energy': energy, 'cost': cost}, index=[0])
            self.summary = pd.concat([self.summary, w_res])

            w.vars['name'] = w_name
            self.detailed_summary = pd.concat([self.detailed_summary, w.vars])

        for cv_name, cv in self.sim.network.control_valves.items():
            vol = sum(cv.vars['flow'] * cv.vars['value'])
            cv_res = pd.DataFrame({'name': cv.name, 'vol': vol, 'energy': 0, 'cost': 0}, index=[0])
            self.summary = pd.concat([self.summary, cv_res])

            cv.vars['name'] = cv_name
            self.detailed_summary = pd.concat([self.detailed_summary, cv.vars])

        for v_name, v in self.sim.network.valves.items():
            vol = sum(v.vars['value'])
            v_res = pd.DataFrame({'name': v.name, 'vol': vol, 'energy': 0, 'cost': 0}, index=[0])
            self.summary = pd.concat([self.summary, v_res])

            v.vars['name'] = v_name
            self.detailed_summary = pd.concat([self.detailed_summary, v.vars])

        for tank_name, tank in self.sim.network.tanks.items():
            total_demand += tank.vars['demand'].sum()

            # set tanks index as all other elements. future dev: restandartization all tables
            df = tank.vars.copy()
            df['name'] = tank_name
            df.index = pd.MultiIndex.from_arrays([df.index, np.full(df.index.shape, 0)])
            self.detailed_summary = pd.concat([self.detailed_summary, df])

        self.summary = self.summary[self.summary['vol'] != 0]
        self.summary.reset_index(inplace=True)
        self.summary['se'] = self.summary['energy'] / self.summary['vol']
        self.summary['sc'] = self.summary['cost'] / self.summary['vol']
        return total_demand
