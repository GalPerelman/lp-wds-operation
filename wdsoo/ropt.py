import os
import pandas as pd
import numpy as np
import operator

from rsome import ro
import rsome as rso
from rsome import clp_solver as clp
from rsome import lpg_solver as lpg
from rsome import grb_solver as grb

# local imports
from . import uncertainty_utils

# solvers info
GRB_STATUS = {1: 'LOADED', 2: 'OPTIMAL', 3: 'INFEASIBLE', 4: 'INF_OR_UNBD', 5: 'UNBOUNDED'}
CLP_STATUS = {-1: 'unknown', 0: 'OPTIMAL', 1: 'primal infeasible', 2: 'dual infeasible',
              3: 'stopped on iterations or time', 4: 'stopped due to errors', 5: 'stopped by event handler'}


class RO:
    def __init__(self, sim, gamma, dem_corr, cost_corr, uncertainty_set, vars_type='C'):
        self.sim = sim
        self.gamma = gamma
        self.dem_corr = dem_corr
        self.cost_corr = cost_corr
        self.uncertainty_set = uncertainty_set
        self.vars_type = vars_type

        self.M = 999999
        self.dem_map = self.demand_uncertainty_map()
        self.u_norms = {'Ellipsoid': 2, 'Box': np.inf}
        # self.u_data = pd.read_csv(os.path.join(self.sim.data_folder, 'uncertainty.csv'))

        self.vars_df = pd.DataFrame(columns=['network_entity', 'name', 'entity_type', 'var', 'cost_vec'])
        self.rand_vars_df = pd.DataFrame(columns=['name', 'set', 'gamma', 'delta', 'var', 'const'])
        self.model = ro.Model()

    def build(self):
        self.declare_vars()
        self.declare_random_vars()

        self.objective_func()

        self.one_comb_only()
        self.mass_balance()
        self.vsp_volume()
        self.vsp_changes()
        self.max_power()

    def declare_vars(self):
        for s_name, s in self.sim.network.pump_stations.items():
            x = self.model.dvar(shape=self.sim.num_steps * len(s.combs), vtype=self.vars_type, name=s.name + '_x')
            self.vars_df = self.vars_df.append({'network_entity': s, 'name': s.name,
                                                'entity_type': 'station', 'var': x,
                                                'cost_vec': s.vars['cost'].to_numpy().astype(float)},
                                               ignore_index=True)
            self.model.st(x >= 0)
            self.model.st(x <= 1)

        for vsp_name, vsp in self.sim.network.vsp.items():
            x = self.model.dvar(shape=self.sim.num_steps, vtype='C', name=vsp.name + '_x')
            self.vars_df = self.vars_df.append({'network_entity': vsp, 'name': vsp.name, 'entity_type': 'vsp', 'var': x,
                                                'cost_vec': vsp.vars['cost'].to_numpy().astype(float)},
                                               ignore_index=True)

            self.model.st(x >= vsp.min_flow)
            self.model.st(x <= vsp.max_flow)
            if not np.isnan(vsp.init_flow):
                self.model.st(x[0] == vsp.init_flow)

        for well_name, well in self.sim.network.wells.items():
            x = self.model.dvar(shape=self.sim.num_steps, vtype=self.vars_type, name=well.name + '_x')
            self.vars_df = self.vars_df.append({'network_entity': well, 'name': well.name, 'entity_type': 'well',
                                                'var': x, 'cost_vec': well.vars['cost'].to_numpy().astype(float)},
                                               ignore_index=True)
            self.model.st(x >= 0)
            self.model.st(x <= 1)

        for cv_name, cv in self.sim.network.control_valves.items():
            x = self.model.dvar(shape=self.sim.num_steps * len(cv.combs), vtype=self.vars_type, name=cv.name + '_x')
            self.vars_df = self.vars_df.append({'network_entity': cv, 'name': cv.name, 'entity_type': 'cv', 'var': x,
                                                'cost_vec': cv.vars['cost'].to_numpy().astype(float)},
                                               ignore_index=True)

            self.model.st(x >= 0)
            self.model.st(x <= 1)

        for v_name, v in self.sim.network.valves.items():
            x = self.model.dvar(shape=self.sim.num_steps, vtype='C', name=v.name + '_x')
            self.vars_df = self.vars_df.append({'network_entity': v, 'name': v.name, 'entity_type': 'cv', 'var': x,
                                                'cost_vec': v.vars['cost'].to_numpy().astype(float)},
                                               ignore_index=True)

            self.model.st(x >= v.min_flow)
            self.model.st(x <= v.max_flow)

    def declare_random_vars(self):
        for i, row in self.u_data.iterrows():
            name = row['name']
            norm = 2                #self.u_norms[row['u']]
            gamma = self.gamma      #row['gamma']
            delta = row['delta']

            var = self.vars_df.loc[self.vars_df['name'] == name]
            z = self.model.rvar(len(var['cost_vec'].values[0]), name='z_' + name)
            z_set = (rso.norm(z, norm) <= gamma)
            temp = pd.DataFrame({'name': name, 'norm': norm, 'gamma': gamma, 'delta': delta, 'var': z, 'const': z_set},
                                index=[len(self.rand_vars_df)])
            self.rand_vars_df = pd.concat([self.rand_vars_df, temp])

    def demand_uncertainty_map(self):
        demands = np.array([t.vars['demand'].values for t in self.sim.network.tanks.values()])
        demands = demands.T
        delta = uncertainty_utils.get_constant_correlation_set(demands, self.dem_corr)
        return delta

    def cost_uncertainty_map(self):
        cost = np.hstack([x.vars['cost'].values.T for x in self.sim.network.cost_elements.values()])
        cost = cost.T
        print(cost)
        delta = uncertainty_utils.get_constant_correlation_set(cost, self.dem_corr)
        # return delta

    def comb_matrix(self, x, inout, param=1):
        """ return a matrix for elements with discrete hydraulic states (pumps combinations) """
        flow_direction = {'in': 1, 'out': -1}
        matrix = np.zeros([self.sim.num_steps, self.sim.num_steps * len(x.combs)])
        rows = np.hstack([np.repeat(i, len(x.combs)) for i in range(self.sim.num_steps)])
        cols = np.arange(len(x.combs) * self.sim.num_steps)
        matrix[rows, cols] = flow_direction[inout]
        if param != 1:
            matrix = matrix * x.vars[param].to_numpy()
        return matrix

    def not_comb_matrix(self, inout):
        """ return a matrix for elements with continuous variables (valves flow, vsp) """
        flow_direction = {'in': 1, 'out': -1}
        v_matrix = flow_direction[inout] * np.diag(np.ones(self.sim.num_steps))
        return v_matrix

    def one_comb_only(self):
        for station_name, station in self.sim.network.comb_elements.items():
            x = self.vars_df.loc[self.vars_df['name'] == station.name, 'var'].values[0]
            matrix = np.zeros([self.sim.num_steps, self.sim.num_steps * len(station.combs)])
            rows = np.hstack([np.repeat(i, len(station.combs)) for i in range(self.sim.num_steps)])
            cols = np.arange(len(station.combs) * self.sim.num_steps)
            matrix[rows, cols] = 1
            self.model.st(matrix @ x <= (np.ones((self.sim.num_steps, 1))))

    def mass_balance(self):
        z = self.model.rvar(self.sim.num_steps)
        z_set = (rso.norm(z, self.uncertainty_set) <= self.gamma)
        for tank_name, tank in self.sim.network.tanks.items():
            demand = tank.vars['demand'].to_numpy() + self.dem_corr @ z
            for t in range(self.sim.num_steps):
                LHS = tank.initial_vol
                for s in tank.inflows:
                    temp_matrix = self.comb_matrix(s, 'in', 'flow')[:t + 1, :(t + 1) * len(s.combs)]
                    x = self.vars_df.loc[self.vars_df['name'] == s.name, 'var'].values[0]
                    LHS = LHS + (temp_matrix * x[:(t + 1) * len(s.combs)]).sum()
                for vsp in tank.vsp_inflows:
                    temp_matrix = self.not_comb_matrix('in')[:t + 1, :t + 1]
                    x = self.vars_df.loc[self.vars_df['name'] == vsp.name, 'var'].values[0]
                    LHS = LHS + (temp_matrix * x[:t + 1]).sum()

                for s in tank.outflows:
                    temp_matrix = self.comb_matrix(s, 'out', 'flow')[:t + 1, :(t + 1) * len(s.combs)]
                    x = self.vars_df.loc[self.vars_df['name'] == s.name, 'var'].values[0]
                    LHS = LHS + (temp_matrix * x[:(t + 1) * len(s.combs)]).sum()
                for vsp in tank.vsp_outflows:
                    temp_matrix = self.not_comb_matrix('out')[:t + 1, :t + 1]
                    x = self.vars_df.loc[self.vars_df['name'] == vsp.name, 'var'].values[0]
                    LHS = LHS + (temp_matrix * x[:t + 1]).sum()

                self.model.st((LHS - demand[:t + 1].sum() >= tank.min_vol).forall(z_set))
                self.model.st((LHS - demand[:t + 1].sum() <= tank.max_vol).forall(z_set))

            # Final volume constraint - last LHS is for t = T
            self.model.st((LHS - demand.sum() >= tank.final_vol).forall(z_set))

    def vsp_volume(self):
        operators = {'le': operator.le, 'ge': operator.ge, 'eq': operator.eq}
        df = pd.read_csv(self.sim.data_folder + '/vsp_volume.csv')
        for i, row in df.iterrows():
            start = pd.to_datetime(row['start'], dayfirst=True)
            end = pd.to_datetime(row['end'], dayfirst=True)
            vsp_name = row['vsp']
            volume = row['vol']
            operator_type = row['constraint_type']

            vsp = self.vars_df.loc[self.vars_df['name'] == vsp_name, 'network_entity'].values[0]
            vsp_x = self.vars_df.loc[self.vars_df['name'] == vsp.name, 'var'].values[0]

            vsp.vars['aux'] = 0
            mask = ((vsp.vars.index >= start) & (vsp.vars.index <= end))
            vsp.vars.loc[mask, 'aux'] = 1

            matrix = self.not_comb_matrix('in')
            matrix = np.multiply(matrix, vsp.vars['aux'].to_numpy()[:, np.newaxis])  # row-wise multiplication
            lhs = matrix.sum(axis=0) @ vsp_x  # sum of matrix rows to get the total flow for period
            self.model.st(operators[operator_type](lhs, volume))
            vsp.vars = vsp.vars.drop('aux', axis=1)

    def vsp_changes(self):
        df = pd.read_csv(self.sim.data_folder + '/vsp_changes.csv')
        for i, row in df.iterrows():
            start = pd.to_datetime(row['start'], dayfirst=True)
            end = pd.to_datetime(row['end'], dayfirst=True)
            vsp_name = row['vsp']

            vsp = self.vars_df.loc[self.vars_df['name'] == vsp_name, 'network_entity'].values[0]
            vsp_x = self.vars_df.loc[self.vars_df['name'] == vsp.name, 'var'].values[0]

            vsp.vars['aux'] = 0
            mask = ((vsp.vars.index > start) & (vsp.vars.index < end))
            vsp.vars.loc[mask, 'aux'] = 1

            matrix = np.diag(np.ones(self.sim.num_steps))
            rows, cols = np.indices((self.sim.num_steps, self.sim.num_steps))
            row_vals = np.diag(rows, k=-1)
            col_vals = np.diag(cols, k=-1)
            matrix[row_vals, col_vals] = -1
            matrix[0, 0] = 0

            matrix = np.multiply(matrix, vsp.vars['aux'].to_numpy()[:, np.newaxis])  # row-wise multiplication
            self.model.st(matrix @ vsp_x == 0)

    def max_power(self):
        df = pd.read_csv(self.sim.data_folder + '/power_stations.csv')
        for i, row in df.iterrows():
            ps_name = row['name']
            ps = self.sim.network.power_stations[ps_name]

            lhs = 0
            for s in ps.elements:
                matrix = self.comb_matrix(s, 'in', 'power')
                x = self.vars_df.loc[self.vars_df['name'] == s.name, 'var'].values[0]
                b = self.model.dvar(shape=x.shape, vtype='B', name=s.name + '_b')
                self.model.st(x - b <= 0)
                lhs = lhs + matrix @ b

            rhs = self.sim.date_tariff['name']
            rhs = np.where(rhs == 'ON', row['max_power_on'], row['max_power_off'])
            self.model.st(lhs <= rhs)

    def get_uncertain_params(self, name):
        params = self.u_data.loc[self.u_data['name'] == name]
        norm = self.u_norms[params['u'].values[0]]
        gamma = params['gamma'].values[0]
        delta = params['delta'].values[0]

        return norm, gamma, delta

    def objective_func(self):
        obj = 0
        usets = []

        # z1 = self.model.rvar(72, name='z_ps')
        # z1_set = (rso.norm(z1, 2) <= self.gamma)
        # z2 = self.model.rvar(24, name='z_w')
        # z2_set = (rso.norm(z2, 2) <= self.gamma)
        # usets = [z1_set, z2_set]

        e1 = self.vars_df.loc[self.vars_df['name'] == 'P1', 'network_entity'].values[0]
        e2 = self.vars_df.loc[self.vars_df['name'] == 'W1', 'network_entity'].values[0]

        x1 = self.vars_df.loc[self.vars_df['name'] == 'P1', 'var'].values[0]
        x2 = self.vars_df.loc[self.vars_df['name'] == 'W1', 'var'].values[0]

        c1 = self.vars_df.loc[self.vars_df['name'] == 'P1', 'cost_vec'].values[0]
        c2 = self.vars_df.loc[self.vars_df['name'] == 'W1', 'cost_vec'].values[0]

        # d1 = self.cost_delta['P1']
        # d2 = self.cost_delta['W1']

        m1 = self.comb_matrix(e1, 'in')
        m2 = self.not_comb_matrix('in')

        z = self.model.rvar(shape=(2, 24))
        # z_set1 = (rso.norm(z.reshape(-1), 2) <= self.gamma)
        z_set1 = (rso.norm(z[0, :], 2) <= self.gamma)
        z_set2 = (rso.norm(z[1, :], 2) <= self.gamma)
        map = self.cost_corr @ z

        cc1 = (map[0, :] @ m1 @ x1).sum()
        cc2 = (map[1, :] @ m2 @ x2).sum()
        obj = (c1 @ x1).sum() + (c2 @ x2).sum() + cc1 + cc2

        self.model.minmax(obj, [z_set1, z_set2])

        # for i, row in self.vars_df.iterrows():
        #     x = row['var']
        #     c = row['cost_vec']
        #     if row['name'] in self.u_data['name'].to_list():
        #         # z = self.rand_vars_df.loc[self.rand_vars_df['name'] == row['name'], 'var'].values[0]
        #         # z_set = self.rand_vars_df.loc[self.rand_vars_df['name'] == row['name'], 'const'].values[0]
        #         # d = self.rand_vars_df.loc[self.rand_vars_df['name'] == row['name'], 'delta'].values[0]
        #         # c = c * (1 + d * z)
        #         d = self.cost_delta[row['name']]
        #         c = c + d @ z
        #         usets.append(z_set)
        #
        #     obj += (c @ x).sum()
        # self.model.minmax(obj, usets)

        # x1 = self.vars_df.loc[self.vars_df['name'] == 'P1', 'var'].values[0]
        # c1 = self.vars_df.loc[self.vars_df['name'] == 'P1', 'cost_vec'].values[0]
        # x2 = self.vars_df.loc[self.vars_df['name'] == 'W1', 'var'].values[0]
        # c2 = self.vars_df.loc[self.vars_df['name'] == 'W1', 'cost_vec'].values[0]
        #
        # z1 = self.model.rvar(len(c1))
        # z1_set = (rso.norm(z1, 2) <= 1)
        # z2 = self.model.rvar(len(c2))
        # z2_set = (rso.norm(z2, 2) <= 1)
        # c1 = c1 * (1 + 0.2 * z1)
        # c2 = c2 * (1 + 0.05 * z2)
        # obj = (c1 @ x1).sum() + (c2 @ x2).sum()
        # self.model.minmax(obj, [z1_set, z2_set])

    def solve(self):
        self.model.solve(grb, display=False, params={'DualReductions': 0})
        obj, x, status = self.model.solution.objval, self.model.solution.x, self.model.solution.status
        status = GRB_STATUS[status]
        self.get_results()
        return obj, x, status

    def get_results(self):
        for i, row in self.vars_df.iterrows():
            x = row['var']
            element = row['network_entity']
            element.vars['value'] = x.get()

        self.tanks_balance()

    def tanks_balance(self):
        for t_name, t in self.sim.network.tanks.items():
            df = pd.DataFrame(index=self.sim.time_range, data={'df': 0})
            for x in t.inflows + t.cv_inflows:
                qin = x.vars['flow'] * x.vars['value']
                qin = qin.groupby(level='time').sum()
                df = pd.merge(df, qin.rename(x.name), left_index=True, right_index=True)

            for x in t.vsp_inflows + t.v_inflows:
                qin = x.vars['value']
                df = pd.merge(df, qin.rename(x.name), left_index=True, right_index=True)

            for x in t.outflows + t.cv_outflows:
                qout = x.vars['flow'] * x.vars['value']
                qout = -1 * qout.groupby(level='time').sum()
                df = pd.merge(df, qout.rename(x.name), left_index=True, right_index=True)

            for x in t.vsp_outflows + t.v_outflows:
                qout = -1 * x.vars['value']
                df = pd.merge(df, qout.rename(x.name), left_index=True, right_index=True)

            df['inflow'] = df.sum(axis=1)
            df['demand'] = -t.vars['demand']
            df['volume'] = t.initial_vol + (df['inflow'] + df['demand']).cumsum()
            t.vars['value'] = df['volume']
            t.vars['inflow'] = df['inflow']
