import os
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt

# local imports
from . import uncertainty_utils as uutils

np.set_printoptions(threshold=np.inf, linewidth=500)
np.set_printoptions(suppress=True)


class MC:
    def __init__(self, sim, n_sim=1000):
        self.sim = sim
        self.n_sim = n_sim

        self.t = self.sim.num_steps
        self.epsilon = 0.0001
        self.udata = uutils.init_uncertainty(os.path.join(self.sim.data_folder, 'uncertainty', 'uncertainty.json'))
        self.num_tanks = len(self.sim.network.tanks)

    def get_volume_violations(self, sample, demands, plot=False, **kwargs):
        sim_violations = []
        all_samples = pd.DataFrame()
        for i, (tank_name, tank) in enumerate(self.sim.network.tanks.items()):
            tank_inflow = self.sim.network.tanks[tank_name].vars['inflow'].values.reshape(-1, 1)

            sample_d = (tank.initial_vol + (tank_inflow - sample[i * self.t: (i + 1) * self.t, :]).cumsum(axis=0))
            all_samples = pd.concat([all_samples, pd.DataFrame(sample)], axis=0)
            tank_violations = self.get_tanks_violations(tank, sample_d)
            sim_violations.append(tank_violations)

        sim_violations = np.vstack(sim_violations)
        sim_violations = np.count_nonzero(sim_violations == 1, axis=0)  # counts number of violations in each sample
        sim_violations = np.array([np.ones(sim_violations.shape), sim_violations]).min(axis=0)
        if plot:
            self.plot(sample, sum(sim_violations) / self.n_sim, kwargs['header_str'])
        return sum(sim_violations) / self.n_sim, sim_violations

    def plot_tanks_vol(self, sample, vr, header_str):
        fig, axes = plt.subplots(nrows=2, ncols=1 + math.floor(len(self.sim.network.tanks) / 2),
                                 sharex=True, figsize=(12, 6))
        axes = axes.ravel()

        for i, (tank_name, tank) in enumerate(self.sim.network.tanks.items()):
            tank_inflow = self.sim.network.tanks[tank_name].vars['inflow'].values.reshape(-1, 1)
            res = (tank.initial_vol + (tank_inflow - sample[i * self.t: (i + 1) * self.t, :]).cumsum(axis=0))

            #  Visualizing the sample
            res = np.vstack([np.ones((1, self.n_sim)) * tank.initial_vol, res])
            axes[i].set_title(tank_name)
            axes[i].plot(res, c='C0', alpha=0.4)
            axes[i].plot(range(len(tank.vars) + 1), list(tank.min_vol) + [tank.final_vol],
                         c='r', alpha=0.6, linestyle='--', zorder=5)
            axes[i].hlines(tank.max_vol, xmin=0, xmax=24, color='grey', zorder=5, alpha=0.8)
            axes[i].plot(range(len(tank.vars) + 1), [tank.initial_vol] + tank.vars['value'].to_list(), c='k')
            axes[i].grid()

        plt.suptitle(f'Monte Carlo simulations - Violations rate: {vr * 100:.0f}%\n{header_str}')
        plt.tight_layout()

    def plot_sample(self, sample, demands):
        fig, axes = plt.subplots(nrows=2, ncols=int(len(self.sim.network.tanks) / 2 + 1), sharex=True, figsize=(12, 6))
        axes = axes.ravel()
        t = self.sim.num_steps
        for i, (tank_name, tank) in enumerate(self.sim.network.tanks.items()):
            axes[i].set_title(tank_name)
            axes[i].plot(sample[i * t: (i + 1) * t, :], c='C0', alpha=0.1)
            axes[i].plot(demands.T[i * t: (i + 1) * t], c='k')
            axes[i].grid()

    def get_tanks_violations(self, tank, res):
        z_max_vol = np.where(res - tank.max_vol <= self.epsilon, 0, 1)  # 1 for constraint violation, 0 otherwise
        n_max_vol = np.count_nonzero(z_max_vol == 1, axis=0)  # counts number of violations in each sample
        n_max_vol = np.array([np.ones(n_max_vol.shape), n_max_vol]).min(axis=0)  # 1 if any violation in the sample

        z_min_vol = np.where(- res + tank.min_vol.reshape(-1, 1) <= self.epsilon, 0, 1)
        n_min_vol = np.count_nonzero(z_min_vol == 1, axis=0)
        n_min_vol = np.array([np.ones(n_min_vol.shape), n_min_vol]).min(axis=0)

        z_final = np.where(- res[-1, :] + tank.final_vol <= self.epsilon, 0, 1)  # 1 for violation, 0 otherwise

        violations = np.vstack([n_max_vol, n_min_vol])  # stack all tanks violations
        n_violations = np.count_nonzero(violations == 1, axis=0)  # counts number of violations in each sample
        n_violations = np.array([np.ones(n_violations.shape), n_violations]).min(axis=0)  # 1 if any violation
        return n_violations

    def get_cost_violations(self, sample_cost, nominal_data, deterministic_obj):
        calculated_sample = (nominal_data['value'].values.reshape(-1, 1) * sample_cost).sum(axis=0)
        # 1 for constraint violation, 0 otherwise
        n_violations = np.where(deterministic_obj - calculated_sample <= 0, 1, 0)
        return sum(n_violations)/len(n_violations)

    def uniform_sample(self):
        demands = np.array([t.vars['demand'].values for t in self.sim.network.tanks.values()])
        demands = demands.T
        t, n_tanks = demands.shape
        non_zero_idx = np.argwhere(demands.any(axis=0)).flatten()

        nominal = demands.T.reshape(-1, 1)
        lb = nominal * 0.75
        ub = nominal * 1.25

        z = np.random.uniform(low=-1, high=1, size=(nominal.shape[0], self.n_sim))
        z = z * (ub - lb)
        sample = np.zeros((t * n_tanks, self.n_sim))
        for i, idx in enumerate(list(non_zero_idx)):
            sample[idx * t: (idx + 1) * t] = nominal[idx * t: (idx + 1) * t] + z[idx * t: (idx + 1) * t]

        return sample, demands.T

    def get_nominal(self, u_category):
        if u_category == 'demand':
            nominal = np.hstack([t.vars['demand'].values for t in self.sim.network.tanks.values()])
        if u_category == 'cost':
            nominal = np.hstack([x.vars['cost'].values for x in self.sim.network.cost_elements.values()])
        return nominal

    def multivariate_normal_cost(self, u_category):
        rng = np.random.default_rng()
        t = self.sim.num_steps
        n = len(self.sim.network.cost_elements)

        nominal_data = pd.DataFrame()
        for xname, x in self.sim.network.cost_elements.items():
            temp = x.vars.copy()
            temp['name'] = xname
            temp.reset_index(inplace=True)
            nominal_data = pd.concat([nominal_data, temp])

        nominal_data.reset_index(inplace=True)
        uelements = {ename: e for ename, e in self.sim.network.cost_elements.items()
                     if ename in self.udata['cost'].elements}

        sigma = self.udata[u_category].sigma
        z = rng.multivariate_normal(np.zeros(sigma.shape[0]), sigma, size=self.n_sim, method='cholesky').T

        sample = np.zeros((len(nominal_data), self.n_sim)) + nominal_data['cost'].values.reshape(-1, 1)
        for i, (ename, e) in enumerate(uelements.items()):
            idx = nominal_data.loc[nominal_data['name'] == ename].index
            random = np.repeat(z[i * t: (i + 1) * t], len(idx) / t, axis=0)
            sample[idx] = sample[idx] + random

        return sample

    def multivariate_normal(self, u_category):
        rng = np.random.default_rng()

        nominal = self.get_nominal(u_category).reshape(-1, 1)
        nominal = nominal.T
        t, n = nominal.shape

        non_zero_idx = np.argwhere(nominal.any(axis=0)).flatten()
        sigma = self.udata[u_category].sigma
        z = rng.multivariate_normal(np.zeros(sigma.shape[0]), sigma, size=self.n_sim, method='cholesky').T

        nominal = nominal.T.reshape(-1, 1)
        sample = np.zeros((t * n, self.n_sim))

        for i, idx in enumerate(list(non_zero_idx)):
            sample[idx * t: (idx + 1) * t] = nominal[idx * t: (idx + 1) * t] + z[i * t: (i + 1) * t]

        if u_category == 'demand':
            sample[sample < 0] = 0

        return sample, nominal.T

    def multivariate_in_ellipse(self):
        sample, nominal = self.multivariate_normal()
        sample = np.apply_along_axis(lambda x: x / np.sqrt(np.sum(x ** 2)), axis=0, arr=sample)
        return sample, nominal
