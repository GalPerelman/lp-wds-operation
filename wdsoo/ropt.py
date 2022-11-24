import os
import pandas as pd
import numpy as np
import operator
import json

from rsome import ro
import rsome as rso
from rsome import clp_solver as clp
from rsome import lpg_solver as lpg
from rsome import grb_solver as grb

# local imports
from . import uncertainty_utils as uutils

# solvers info
GRB_STATUS = {1: 'LOADED', 2: 'OPTIMAL', 3: 'INFEASIBLE', 4: 'INF_OR_UNBD', 5: 'UNBOUNDED'}
CLP_STATUS = {-1: 'unknown', 0: 'OPTIMAL', 1: 'primal infeasible', 2: 'dual infeasible',
              3: 'stopped on iterations or time', 4: 'stopped due to errors', 5: 'stopped by event handler'}


class RO:
    def __init__(self, sim, gamma, uset_type, vars_type='C'):
        self.sim = sim
        self.gamma = gamma
        self.uset_type = uset_type
        self.vars_type = vars_type

        self.ucategories = self.init_uncertainty()

        self.vars_df = pd.DataFrame(columns=['network_entity', 'name', 'entity_type', 'var', 'cost_vec'])
        self.model = ro.Model()

    def build(self):
        self.declare_vars()
        self.one_comb_only()
        self.mass_balance()
        self.vsp_volume()
        self.vsp_changes()
        self.max_power()
        self.objective_func()

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

    def init_uncertainty(self):
        ucategories = {}
        with open(os.path.join(self.sim.data_folder, 'uncertainty.json')) as f:
            udata = json.load(f)

            for ucat, ucat_data in udata.items():
                cat_elements = {}
                for idx, (e_name, e_data) in enumerate(ucat_data['elements'].items()):
                    ue = UElement(ucat, e_name, idx, **e_data)
                    cat_elements[e_name] = ue

                uc = UCategory(ucat, cat_elements, ucat_data['elements_correlation'])
                ucategories[ucat] = uc

        return ucategories

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
        """ return a matrix for elements with continuous variables (valves, vsp, tanks) """
        flow_direction = {'in': 1, 'out': -1}
        matrix = flow_direction[inout] * np.diag(np.ones(self.sim.num_steps))
        return matrix

    def one_comb_only(self):
        for station_name, station in self.sim.network.comb_elements.items():
            x = self.vars_df.loc[self.vars_df['name'] == station.name, 'var'].values[0]
            matrix = np.zeros([self.sim.num_steps, self.sim.num_steps * len(station.combs)])
            rows = np.hstack([np.repeat(i, len(station.combs)) for i in range(self.sim.num_steps)])
            cols = np.arange(len(station.combs) * self.sim.num_steps)
            matrix[rows, cols] = 1
            self.model.st(matrix @ x <= (np.ones((self.sim.num_steps, 1))))

    def mass_balance_lhs(self, tank, t):
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

        return LHS

    def mass_balance(self):
        """
        Δ:      affine map - a matrix defines the uncertainty set correlations. computed from cov matrix Σ = ΔΔᵀ
        p:      a multiplication of affine map with random variable p = Δ @ z

        In operation problems we have two mapping actions:
            map correlation between time steps
            map correlation between different elements

        """
        if len(self.ucategories['demand'].elements) > 0:
            z_dem = self.model.rvar((self.sim.num_steps, len(self.ucategories['demand'].elements)), name='z_dem')
            tanks_delta = self.ucategories['demand'].affine_map
            if tanks_delta is not None:
                p = tanks_delta @ z_dem.T

        for tank_name, tank in self.sim.network.tanks.items():
            demand = tank.vars['demand'].to_numpy()
            if tank_name in self.ucategories['demand'].elements.keys():
                uelement = self.ucategories['demand'].elements[tank_name]
                z_set = rso.norm(z_dem[:, uelement.idx], self.uset_type) <= self.gamma
                time_delta = uelement.time_affine_map

                if time_delta is None and tanks_delta is None:
                    demand += z_dem[:, uelement.idx] * 0
                    pass

                elif time_delta is not None and tanks_delta is None:
                    demand += time_delta @ z_dem[:, uelement.idx]

                elif time_delta is None and tanks_delta is not None:
                    demand += p[uelement.idx, :]

                elif time_delta is not None and tanks_delta is not None:
                    demand += p[uelement.idx, :] @ time_delta

                for t in range(self.sim.num_steps):
                    lhs = self.mass_balance_lhs(tank, t)
                    self.model.st((lhs - demand[:t + 1].sum() >= tank.min_vol[t]).forall(z_set))
                    self.model.st((lhs - demand[:t + 1].sum() <= tank.max_vol).forall(z_set))

                # Final volume constraint - last LHS is for t = T
                lhs = self.mass_balance_lhs(tank, self.sim.num_steps)
                self.model.st((lhs - demand.sum() >= tank.final_vol).forall(z_set))

            else:
                for t in range(self.sim.num_steps):
                    lhs = self.mass_balance_lhs(tank, t)
                    self.model.st(lhs - demand[:t + 1].sum() >= tank.min_vol[t])
                    self.model.st(lhs - demand[:t + 1].sum() <= tank.max_vol)
                # Final volume constraint - last LHS is for t = T
                self.model.st(lhs - demand.sum() >= tank.final_vol)

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

    def objective_func(self):
        """
        Δ:      affine map - a matrix defines the uncertainty set correlations. computed from cov matrix Σ = ΔΔᵀ
        p:      a multiplication of affine map with random variable p = Δ @ z

        In operation problems we have two mapping actions:
            map correlation between time steps
            map correlation between different elements

        """
        obj = 0
        u_sets = []

        if len(self.ucategories['cost'].elements) > 0:
            z_cost = self.model.rvar((self.sim.num_steps, len(self.ucategories['cost'].elements)), name='z_cost')
            elements_delta = self.ucategories['cost'].affine_map
            if elements_delta is not None:
                p = elements_delta @ z_cost.T  # (n_elements x n_elements) @ (n_elements x T) = (n_elements x T)

        for i, row in self.vars_df.iterrows():
            x = row['var']
            c = row['cost_vec']
            name = row['name']
            element = row['network_entity']

            if name in self.ucategories['cost'].elements.keys():
                if hasattr(element, 'combs'):
                    matrix = self.comb_matrix(element, 'in')
                else:
                    matrix = self.not_comb_matrix('in')

                uelement = self.ucategories['cost'].elements[name]
                z_set = rso.norm(z_cost[:, uelement.idx], self.uset_type) <= self.gamma
                u_sets.append(z_set)
                time_delta = uelement.time_affine_map

                if time_delta is None and elements_delta is None:
                    ucost = 0

                elif time_delta is not None and elements_delta is None:
                    ucost = time_delta @ z_cost[:, uelement.idx] @ matrix @ x

                elif time_delta is None and elements_delta is not None:
                    ucost = p[uelement.idx, :] @ matrix @ x

                elif time_delta is not None and elements_delta is not None:
                    ucost = p[uelement.idx, :] @ time_delta @ matrix @ x

                obj += ucost

            obj += (c @ x).sum()

        self.model.minmax(obj, u_sets)

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


class UElement:
    def __init__(self, ucat, name, idx, element_type, time_std, std, time_corr):
        self.ucat = ucat
        self.name = name
        self.idx = idx
        self.element_type = element_type
        self.time_std = time_std
        self.std = std
        self.time_corr = time_corr

        self.time_affine_map = self.get_time_affine_map()

    def get_time_affine_map(self):
        if self.time_std is not None and self.time_corr is not None:
            return uutils.uset_from_std(self.time_std, self.time_corr)
        else:
            return


class UCategory:
    def __init__(self, name, elements: dict, elements_correlation):
        self.name = name
        self.elements = elements
        self.elements_correlation = elements_correlation

        self.affine_map = self.get_elements_affine_map()

    def get_elements_affine_map(self):
        std = [e.std for ename, e in self.elements.items()]
        if self.elements_correlation is not None:
            return uutils.uset_from_std(std, self.elements_correlation)
        else:
            return

