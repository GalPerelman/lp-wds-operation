import sys
import pandas as pd
import time
import numpy as np
import datetime
import matplotlib.pyplot as plt
import operator

from scipy import sparse
from cylp.cy import CyClpSimplex
from cylp.py.modeling.CyLPModel import CyLPModel, CyLPArray
from cylp.py.utils.sparseUtil import csr_matrixPlus, csc_matrixPlus


class LP:
    def __init__(self, sim):
        self.sim = sim
        self.model = CyLPModel()

        self.M = 999999
        self.solution = None

        self.vars_df = pd.DataFrame(columns=['network_entity', 'name', 'entity_type', 'var', 'cost_vec'])

    def declare_vars(self):
        for station_name, station in self.sim.network.pump_stations.items():
            x = self.model.addVariable(station.name + '_x', dim=self.sim.num_steps * len(station.combs), isInt=True)
            self.vars_df = self.vars_df.append({'network_entity': station,
                                                'name': station.name,
                                                'entity_type': 'station',
                                                'var': x,
                                                'cost_vec': station.vars['cost'].to_numpy().astype(float)},
                                               ignore_index=True)
            self.model.addConstraint(x >= 0)
            self.model.addConstraint(x <= 1)

        for vsp_name, vsp in self.sim.network.vsp.items():
            x = self.model.addVariable(vsp.name + '_x', dim=self.sim.num_steps, isInt=False)
            self.vars_df = self.vars_df.append({'network_entity': vsp,
                                                'name': vsp.name,
                                                'entity_type': 'vsp',
                                                'var': x,
                                                'cost_vec': vsp.vars['cost'].to_numpy().astype(float)},
                                               ignore_index=True)

            self.model.addConstraint(x >= vsp.min_flow)
            self.model.addConstraint(x <= vsp.max_flow)
            if not np.isnan(vsp.init_flow):
                self.model.addConstraint(x[0] == vsp.init_flow)

        for well_name, well in self.sim.network.wells.items():
            x = self.model.addVariable(well.name + '_x', dim=self.sim.num_steps, isInt=True)
            self.vars_df = self.vars_df.append({'network_entity': well,
                                                'name': well.name,
                                                'entity_type': 'well',
                                                'var': x,
                                                'cost_vec': well.vars['cost'].to_numpy().astype(float)},
                                               ignore_index=True)
            self.model.addConstraint(x >= 0)
            self.model.addConstraint(x <= 1)

        for cv_name, cv in self.sim.network.control_valves.items():
            x = self.model.addVariable(cv.name + '_x', dim=self.sim.num_steps * len(cv.combs), isInt=True)
            self.vars_df = self.vars_df.append({'network_entity': cv,
                                                'name': cv.name,
                                                'entity_type': 'cv',
                                                'var': x,
                                                'cost_vec': cv.vars['cost'].to_numpy().astype(float)},
                                               ignore_index=True)

            self.model.addConstraint(x >= 0)
            self.model.addConstraint(x <= 1)

        for v_name, v in self.sim.network.valves.items():
            x = self.model.addVariable(v.name + '_x', dim=self.sim.num_steps, isInt=False)
            self.vars_df = self.vars_df.append({'network_entity': v,
                                                'name': v.name,
                                                'entity_type': 'cv',
                                                'var': x,
                                                'cost_vec': v.vars['cost'].to_numpy().astype(float)},
                                               ignore_index=True)

            self.model.addConstraint(x >= v.min_flow)
            self.model.addConstraint(x <= v.max_flow)

        for tank_name, tank in self.sim.network.tanks.items():
            x = self.model.addVariable(tank.name + '_x', dim=self.sim.num_steps, isInt=False)
            self.vars_df = self.vars_df.append({'network_entity': tank,
                                                'name': tank.name,
                                                'entity_type': 'tank',
                                                'var': x,
                                                'cost_vec': tank.vars['cost'].to_numpy().astype(float)},
                                               ignore_index=True)

            gap_x = self.model.addVariable(tank.name + '_gap_x', dim=self.sim.num_steps, isInt=False)
            self.vars_df = self.vars_df.append({'network_entity': tank,
                                                'name': tank.name + '_gap_x',
                                                'entity_type': 'tank',
                                                'var': gap_x,
                                                'cost_vec': np.full(shape=(1, self.sim.num_steps), fill_value=self.M)},
                                               ignore_index=True)

            self.model.addConstraint(x >= 0)
            self.model.addConstraint(gap_x >= 0)

    def one_comb_only(self):
        for station_name, station in self.sim.network.comb_elements.items():
            x = self.vars_df.loc[self.vars_df['name'] == station.name, 'var'].values[0]
            matrix = np.zeros([self.sim.num_steps, self.sim.num_steps * len(station.combs)])
            rows = np.hstack([np.repeat(i, len(station.combs)) for i in range(self.sim.num_steps)])
            cols = np.arange(len(station.combs) * self.sim.num_steps)
            matrix[rows, cols] = 1
            self.model.addConstraint(sparse.csr_matrix(matrix) * x <= CyLPArray(np.ones((self.sim.num_steps, 1))))

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

    def tank_matrix(self):
        return np.diag(np.ones(self.sim.num_steps))

    def mass_balance(self):
        for tank_name, tank in self.sim.network.tanks.items():
            LHS = 0
            x = self.vars_df.loc[self.vars_df['name'] == tank.name + '_gap_x', 'var'].values[0]
            LHS = LHS + sparse.csr_matrix(self.not_comb_matrix('in')) * x

            for s in tank.inflows:
                temp_matrix = self.comb_matrix(s, 'in', 'flow')
                x = self.vars_df.loc[self.vars_df['name'] == s.name, 'var'].values[0]
                LHS = LHS + sparse.csr_matrix(temp_matrix) * x
            for vsp in tank.vsp_inflows:
                x = self.vars_df.loc[self.vars_df['name'] == vsp.name, 'var'].values[0]
                LHS = LHS + sparse.csr_matrix(self.not_comb_matrix('in')) * x
            for cv in tank.cv_inflows:
                temp_matrix = self.comb_matrix(cv, 'in', 'flow')
                x = self.vars_df.loc[self.vars_df['name'] == cv.name, 'var'].values[0]
                LHS = LHS + sparse.csr_matrix(temp_matrix) * x
            for v in tank.v_inflows:
                x = self.vars_df.loc[self.vars_df['name'] == v.name, 'var'].values[0]
                LHS = LHS + sparse.csr_matrix(self.not_comb_matrix('in')) * x

            for s in tank.outflows:
                temp_matrix = self.comb_matrix(s, 'out', 'flow')
                x = self.vars_df.loc[self.vars_df['name'] == s.name, 'var'].values[0]
                LHS = LHS + sparse.csr_matrix(temp_matrix) * x
            for vsp in tank.vsp_outflows:
                x = self.vars_df.loc[self.vars_df['name'] == vsp.name, 'var'].values[0]
                LHS = LHS + sparse.csr_matrix(self.not_comb_matrix('out')) * x
            for cv in tank.cv_outflows:
                temp_matrix = self.comb_matrix(cv, 'out', 'flow')
                x = self.vars_df.loc[self.vars_df['name'] == cv.name, 'var'].values[0]
                LHS = LHS + sparse.csr_matrix(temp_matrix) * x
            for v in tank.v_outflows:
                x = self.vars_df.loc[self.vars_df['name'] == v.name, 'var'].values[0]
                LHS = LHS + sparse.csr_matrix(self.not_comb_matrix('out')) * x

            tank_matrix = np.diag(-np.ones(self.sim.num_steps))
            rows, cols = np.indices((self.sim.num_steps, self.sim.num_steps))
            row_vals = np.diag(rows, k=-1)
            col_vals = np.diag(cols, k=-1)
            tank_matrix[row_vals, col_vals] = 1
            tank_x = self.vars_df.loc[self.vars_df['name'] == tank.name, 'var'].values[0]
            LHS = LHS + sparse.csr_matrix(tank_matrix) * tank_x
            Beq = tank.vars['demand'].to_numpy().reshape(self.sim.num_steps, 1).copy()
            Beq[0] = Beq[0] - tank.initial_vol
            self.model.addConstraint(LHS == CyLPArray(Beq))

            # Tanks min and max vol constraints
            self.model.addConstraint(tank_x >= tank.min_vol)
            self.model.addConstraint(tank_x <= tank.max_vol)

            # Tanks final vol constraint
            self.model.addConstraint(tank_x[self.sim.num_steps - 1] >= tank.final_vol)

    def entities_dependency(self):  # e1,coef1,e2,coef2):
        """
            Add flow dependency constraint between two entities
            sum of e * coef <= 0

            e    (str)    entity name
            coef (int)    entity coef - mostly for determine flow direction
        """
        df = pd.read_csv(self.sim.data_folder + '/entities_dependency.csv')
        for i, row in df.iterrows():
            e1, coef1, e2, coef2 = row
            network_e1 = self.vars_df.loc[self.vars_df['name'] == e1, 'network_entity'].values[0]
            x_e1 = self.vars_df.loc[self.vars_df['name'] == e1, 'var'].values[0]
            type_e1 = self.vars_df.loc[self.vars_df['name'] == e1, 'entity_type'].values[0]
            network_e2 = self.vars_df.loc[self.vars_df['name'] == e2, 'network_entity'].values[0]
            x_e2 = self.vars_df.loc[self.vars_df['name'] == e2, 'var'].values[0]
            type_e2 = self.vars_df.loc[self.vars_df['name'] == e2, 'entity_type'].values[0]

            matrix_e1, matrix_e2 = 1, 1
            if type_e1 == 'station':
                matrix_e1 = sparse.csr_matrix(self.comb_matrix(network_e1, 'in', 'flow'))
            if type_e2 == 'station':
                matrix_e2 = sparse.csr_matrix(self.comb_matrix(network_e2, 'in', 'flow'))
            self.model.addConstraint(coef1 * matrix_e1 * x_e1 + coef2 * matrix_e2 * x_e2 <= 0)

    def units_availability(self, file):
        df = pd.read_csv(file)
        for i, row in df.iterrows():
            name = row['station_name']
            unit = str((row['unit'])).replace('.0', '')
            start = pd.to_datetime(row['start'], dayfirst=True)
            end = pd.to_datetime(row['end'], dayfirst=True)
            try:
                station = self.vars_df.loc[self.vars_df['name'] == name, 'network_entity'].values[0]
            except IndexError:
                continue

            station.vars['availability'] = 0
            mask = (station.vars.index.get_level_values(0) >= start) & (station.vars.index.get_level_values(0) <= end) \
                   & (station.vars['unit' + str(unit)] == 'ON')
            station.vars.loc[mask, 'availability'] = 1

            matrix = station.vars['availability'].to_numpy() * self.comb_matrix(station, 'in')
            matrix = sparse.csr_matrix(matrix)
            x = self.vars_df.loc[self.vars_df['name'] == name, 'var'].values[0]
            self.model.addConstraint(matrix * x == 0)
            station.vars = station.vars.drop('availability', axis=1)

    def tanks_constraints_from_file(self, file):
        """ Constraints types:
            1: <= (smaller or equal)
            2: => (larger or equal)
        """
        operators_dict = {1: operator.le, 2: operator.ge, 3: operator.eq}
        df = pd.read_csv(file)
        for i, row in df.iterrows():
            name = row['tank_name']
            start = pd.to_datetime(row['start'], dayfirst=True)
            end = pd.to_datetime(row['end'], dayfirst=True)
            const_type = row['type']
            value = row['value']

            try:
                tank = self.vars_df.loc[self.vars_df['name'] == 'tank_' + name, 'network_entity'].values[0]
                tank_x = self.vars_df.loc[self.vars_df['name'] == tank.name, 'var'].values[0]
            except IndexError:
                continue

            tank.vars['constraint'] = 0
            mask = (tank.vars.index >= start) & (tank.vars.index <= end)
            tank.vars.loc[mask, 'constraint'] = 1
            matrix = np.diag(np.ones(self.sim.num_steps)) * tank.vars['constraint'].to_numpy()
            matrix = sparse.csr_matrix(matrix)

            # if const_type == 1:
            #     self.model.addConstraint(matrix * tank_x <= value)
            # if const_type == 2:
            #     self.model.addConstraint(matrix * tank_x >= value)
            self.model.addConstraint(operators_dict[const_type](matrix * tank_x, RHS))
            tank.vars = tank.vars.drop('constraint', axis=1)

    def tanks_constraints_from_vector(self, tank_name, const_type, vector):
        """ Constraints types:
            1: <= (smaller or equal)
            2: => (larger or equal)
        """
        try:
            tank = self.vars_df.loc[self.vars_df['name'] == tank_name, 'network_entity'].values[0]
            tank_x = self.vars_df.loc[self.vars_df['name'] == tank.name, 'var'].values[0]
        except IndexError:
            return

        matrix = np.diag(np.ones(self.sim.num_steps))
        matrix = sparse.csr_matrix(matrix)
        if const_type == 1:
            self.model.addConstraint(matrix * tank_x <= vector)
        if const_type == 2:
            self.model.addConstraint(matrix * tank_x >= vector)

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
            lhs = sparse.csr_matrix(matrix.sum(axis=0)) * vsp_x  # sum of matrix rows to get the total flow for period
            self.model.addConstraint(operators[operator_type](lhs, volume))
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
            self.model.addConstraint(sparse.csr_matrix(matrix) * vsp_x == 0)

    def max_power(self):
        df = pd.read_csv(self.sim.data_folder + '/power_stations.csv')
        for i, row in df.iterrows():
            ps_name = row['name']
            ps = self.sim.network.power_stations[ps_name]

            lhs = 0
            for s in ps.elements:
                matrix = self.comb_matrix(s, 'in', 'power')
                x = self.vars_df.loc[self.vars_df['name'] == s.name, 'var'].values[0]
                lhs = lhs + sparse.csr_matrix(matrix) * x

            rhs = self.sim.date_tariff['name']
            rhs = np.where(rhs == 'ON', row['max_power_on'], row['max_power_off'])
            self.model.addConstraint(lhs <= CyLPArray(rhs))

    def station_volume(self):
        operators = {1: operator.le, 2: operator.ge, 3: operator.eq}
        df = pd.read_csv(self.sim.data_folder + '/stations_vol_const.csv')
        for i, row in df.iterrows():
            station_name = row['station']
            volume = row['vol']
            operator_type = row['constraint_type']

            try:
                station = self.vars_df.loc[self.vars_df['name'] == station_name, 'network_entity'].values[0]
                station_x = self.vars_df.loc[self.vars_df['name'] == station.name, 'var'].values[0]

                matrix = self.comb_matrix(station, 'in', 'flow')
                # LHS = sum([sparse.csr_matrix(matrix[i, :]) * station_x for i in range(matrix.shape[0])])
                lhs = sparse.csr_matrix(matrix.sum(axis=0))*station_x
                self.model.addConstraint(operators[operator_type](lhs, volume))
            except IndexError:
                continue

    def multi_stations_volume(self):
        operators = {1: operator.le, 2: operator.ge, 3: operator.eq}
        df = pd.read_csv(self.sim.data_folder + '/multi_stations_vol.csv')

        for i in df['id'].unique():
            station_names = df.loc[df['id'] == i, 'station_name'].to_list()
            total_vol = df.loc[df['id'] == i, 'total_vol'].mean()
            operator_type = df.loc[df['id'] == i, 'constraint_type'].mean()
            lhs = 0
            for station_name in station_names:
                station = self.vars_df.loc[self.vars_df['name'] == station_name, 'network_entity'].values[0]
                station_x = self.vars_df.loc[self.vars_df['name'] == station.name, 'var'].values[0]

                matrix = self.comb_matrix(station, 'in', 'flow')
                lhs += sparse.csr_matrix(matrix.sum(axis=0)) * station_x
            self.model.addConstraint(operators[operator_type](lhs, total_vol))

    def objective_func(self):
        obj = 0
        for i, row in self.vars_df.iterrows():
            x = row['var']
            c = CyLPArray(row['cost_vec'])
            obj += c * x

        self.model.objective = obj

    def solve_primal(self):
        self.solution = CyClpSimplex(self.model)
        self.solution.logLevel = 0
        status = self.solution.primal()
        objective = self.solution.objectiveValue
        self.get_results()
        print(self.solution.getStatusString())
        return status, objective, time.time() - self.sim.start_time

    def solve_integer(self):
        cbcModel = CyClpSimplex(self.model).getCbcModel()
        # cbcModel.maximumSeconds = 500
        cbcModel.integerTolerance = 0.15
        cbcModel.logLevel = 1
        status = cbcModel.solve()
        objective = cbcModel.objectiveValue
        return status, objective, time.time() - self.sim.start_time

    def get_results(self):
        df = pd.DataFrame(columns=['name', 'vol', 'energy', 'cost'])
        total_demand = 0

        for s_name, s in self.sim.network.pump_stations.items():
            s.vars['value'] = self.sim.lp_model.solution.primalVariableSolution[s.name + '_x']
            vol = sum(s.vars['flow'] * s.vars['value'])
            energy = sum(s.vars['flow'] * s.vars['se'] * s.vars['value'])
            cost = sum(s.vars['cost'] * s.vars['value'])
            s_res = pd.DataFrame({'name': s.name, 'vol': vol, 'energy': energy, 'cost': cost}, index=[len(df)])
            df = pd.concat([df, s_res])

        for vsp_name, vsp in self.sim.network.vsp.items():
            vsp.vars['value'] = self.sim.lp_model.solution.primalVariableSolution[vsp.name + '_x']
            vol = sum(vsp.vars['value'])
            energy = sum(vsp.power * vsp.vars['value'])
            cost = sum(vsp.vars['cost'] * vsp.vars['value'])
            vsp_res = pd.DataFrame({'name': vsp.name, 'vol': vol, 'energy': energy, 'cost': cost}, index=[len(df)])
            df = pd.concat([df, vsp_res])

        for w_name, w in self.sim.network.wells.items():
            w.vars['value'] = self.sim.lp_model.solution.primalVariableSolution[w.name + '_x']
            vol = sum(w.vars['flow'] * w.vars['value'])
            energy = sum(w.vars['flow'] * w.vars['se'] * w.vars['value'])
            cost = sum(w.vars['cost'] * w.vars['value'])
            w_res = pd.DataFrame({'name': w.name, 'vol': vol, 'energy': energy, 'cost': cost}, index=[len(sim_results)])
            sim_results = pd.concat([sim_results, w_res])

        for cv_name, cv in self.sim.network.control_valves.items():
            cv.vars['value'] = self.sim.lp_model.solution.primalVariableSolution[cv.name + '_x']
        for v_name, v in self.sim.network.valves.items():
            v.vars['value'] = self.sim.lp_model.solution.primalVariableSolution[v.name + '_x']
        for tank_name, tank in self.sim.network.tanks.items():
            tank.vars['value'] = self.sim.lp_model.solution.primalVariableSolution[tank.name + '_x']

            total_demand += tank.vars['demand'].sum()

        self.total_demand = total_demand

        df = df[df['vol'] != 0]
        df['se'] = df['energy'] / df['vol']
        df['sc'] = df['cost'] / df['vol']
        return df

    def get_gap_vars(self):
        gap_vars = pd.DataFrame(index=self.sim.time_range)
        for tank_name, tank in self.sim.network.tanks.items():
            # try:
            x = self.sim.lp_model.solution.primalVariableSolution[tank.name + '_gap_x']
            gap_vars[tank_name] = x

            # except Exception as e:
            #     print(e)

        if gap_vars.sum().sum() > 0:
            print(gap_vars)
        return gap_vars





# DRAFTS

# def station_vol_by_periods():
    # df = pd.read_csv(file)
    # for i, row in df.iterrows():
    #     name = row['station_name']
    #     start = pd.to_datetime(row['start'], dayfirst=True)
    #     end = pd.to_datetime(row['end'], dayfirst=True)
    #     value = row['value']
    #
    #     try:
    #         station = self.vars_df.loc[self.vars_df['name'] == name, 'network_entity'].values[0]
    #         station_x = self.vars_df.loc[self.vars_df['name'] == station.name, 'var'].values[0]
    #     except IndexError:
    #         continue
    #
    #     station.vars['constraint'] = 0
    #     mask = (station.vars.index.get_level_values(0) >= start) & (station.vars.index.get_level_values(0) <= end)
    #     station.vars.loc[mask, 'constraint'] = 1
    #     matrix = self.entity_matrix(station, 'in', 'flow')
    #     print(matrix.shape)
    #     print(station.vars['constraint'].to_numpy().shape)
    #     r = station.vars['constraint'].to_numpy()
    #     r = r.reshape(-1, 1)
    #     matrix = matrix
    #     matrix = sparse.csr_matrix(matrix)
    #     self.model.addConstraint((matrix * station_x @ r) <= value)
    #     station.vars = station.vars.drop('constraint', axis=1)