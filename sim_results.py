import pandas as pd


class Results:
    def __init__(self, sim):
        self.sim = sim
        self.summary, self.total_demand = self.get_summary()

    def get_summary(self):
        df = pd.DataFrame(columns=['name', 'vol', 'energy', 'cost'])
        all = pd.DataFrame()
        total_demand = 0

        for s_name, s in self.sim.network.pump_stations.items():
            vol = sum(s.vars['flow'] * s.vars['value'])
            energy = sum(s.vars['flow'] * s.vars['se'] * s.vars['value'])
            cost = sum(s.vars['cost'] * s.vars['value'])
            s_res = pd.DataFrame({'name': s.name, 'vol': vol, 'energy': energy, 'cost': cost}, index=[len(df)])
            df = pd.concat([df, s_res])
            all = pd.concat([all, s.vars])

        for vsp_name, vsp in self.sim.network.vsp.items():
            vol = sum(vsp.vars['value'])
            energy = sum(vsp.power * vsp.vars['value'])
            cost = sum(vsp.vars['cost'] * vsp.vars['value'])
            vsp_res = pd.DataFrame({'name': vsp.name, 'vol': vol, 'energy': energy, 'cost': cost}, index=[len(df)])
            df = pd.concat([df, vsp_res])
            all = pd.concat([all, vsp.vars])

        for w_name, w in self.sim.network.wells.items():
            vol = sum(w.vars['flow'] * w.vars['value'])
            energy = sum(w.vars['flow'] * w.vars['se'] * w.vars['value'])
            cost = sum(w.vars['cost'] * w.vars['value'])
            w_res = pd.DataFrame({'name': w.name, 'vol': vol, 'energy': energy, 'cost': cost}, index=[len(df)])
            df = pd.concat([df, w_res])
            all = pd.concat([all, w.vars])

        for cv_name, cv in self.sim.network.control_valves.items():
            vol = sum(cv.vars['flow'] * cv.vars['value'])
            cv_res = pd.DataFrame({'name': cv.name, 'vol': vol, 'energy': 0, 'cost': 0}, index=[len(df)])
            df = pd.concat([df, cv_res])
            all = pd.concat([all, cv.vars])

        for v_name, v in self.sim.network.valves.items():
            vol = sum(v.vars['value'])
            v_res = pd.DataFrame({'name': v.name, 'vol': vol, 'energy': 0, 'cost': 0}, index=[len(df)])
            df = pd.concat([df, v_res])
            all = pd.concat([all, v.vars])

        for tank_name, tank in self.sim.network.tanks.items():
            total_demand += tank.vars['demand'].sum()
            all = pd.concat([all, tank.vars])

        self.total_demand = total_demand

        df = df[df['vol'] != 0]
        df['se'] = df['energy'] / df['vol']
        df['sc'] = df['cost'] / df['vol']

        return df, total_demand





    # def units_hours(self):
#     summary = pd.DataFrame()
#     for s in network.pump_station.all:
#         df = s.vars.copy()
#         df = df.drop(['step_size', 'group', 'eps'], axis=1)
#         df = df[df['value'] > 0].replace({'ON': 1, 'OFF': 0})
#         if df.empty:
#             continue
#
#         df.index = df.index.droplevel(1)
#         units_cols = [col for col in df.columns if col[:4] == 'unit']
#         df = df.loc[:, units_cols].multiply(df['value'], axis='index')
#         df = df.groupby(level=0).sum()
#         df = df.reindex(sorted(df.columns), axis=1)
#         df = pd.DataFrame(df.sum(axis=0)).reset_index()
#         df.columns = ['unit', 'hours']
#         df['station'] = s.name
#         summary = pd.concat([summary, df])
#     return summary


    # def combs_summary(self):
    #     summary = pd.DataFrame()
    #     for s in network.pump_station.all:
    #         df = s.vars.copy()
    #         df = df.drop(['step_size','group','eps'], axis = 1)
    #         df = df[df['value'] > 0].replace({'ON':1,'OFF':0})
    #         if df.empty:
    #             continue
    #
    #         mean_cols = [x for x in df.columns if (x != "cost") & (x != 'value')]
    #         agg_dict = {col:'mean' for col in mean_cols}
    #         agg_dict['value'] = 'sum'
    #         df = df.groupby('comb').agg(agg_dict)
    #
    #         unit_cols = [col for col in df.columns if col[:4] == 'unit']
    #         for u in unit_cols:
    #             temp = df.loc[df[u] == 1].copy()
    #             temp = temp.drop(unit_cols, axis=1)
    #             temp['unit'] = u[4:]
    #             temp['station'] = s.name
    #             summary = pd.concat([summary, temp])
    #     return summary