import os
import numpy as np
import matplotlib.pyplot as plt

# local imports
from . import uncertainty_utils as uutils

np.set_printoptions(threshold=np.inf)


class MC:
    def __init__(self, sim, n_sim, gamma, rho):
        self.sim = sim
        self.n_sim = n_sim
        self.gamma = gamma
        self.rho = rho

        self.t = self.sim.num_steps
        self.epsilon = 0.0001
        self.udata = uutils.init_uncertainty(os.path.join(self.sim.data_folder, 'uncertainty', 'uncertainty_backup.json'))
        self.num_tanks = len(self.sim.network.tanks)

    def run(self, sample_function, plot=False):
        sample, demands = sample_function(self)

        sim_violations = []
        for i, (tank_name, tank) in enumerate(self.sim.network.tanks.items()):
            tank_inflow = self.sim.network.tanks[tank_name].vars['inflow'].values.reshape(-1, 1)

            res = (tank.initial_vol + (tank_inflow - sample[i * self.t: (i + 1) * self.t, :]).cumsum(axis=0))
            tank_violations = self.get_violations_for_tank(tank, res)
            sim_violations.append(tank_violations)

        sim_violations = np.vstack(sim_violations)
        sim_violations = np.count_nonzero(sim_violations == 1, axis=0)  # counts number of violations in each sample
        sim_violations = np.array([np.ones(sim_violations.shape), sim_violations]).min(axis=0)
        if plot:
            self.plot(sample, sum(sim_violations)/self.n_sim)
        return sum(sim_violations)/self.n_sim

    def plot(self, sample, vr):
        fig, axes = plt.subplots(nrows=2, ncols=int(len(self.sim.network.tanks) / 2 + 1), sharex=True, figsize=(12, 6))
        axes = axes.ravel()

        for i, (tank_name, tank) in enumerate(self.sim.network.tanks.items()):
            tank_inflow = self.sim.network.tanks[tank_name].vars['inflow'].values.reshape(-1, 1)
            res = (tank.initial_vol + (tank_inflow - sample[i * self.t: (i + 1) * self.t, :]).cumsum(axis=0))

            #  Visualizing the sample
            res = np.vstack([np.ones((1, self.n_sim)) * tank.initial_vol, res])
            axes[i].set_title(tank_name)
            axes[i].plot(res, c='C0', alpha=0.4)
            axes[i].plot(range(len(tank.vars) + 1), list(tank.min_vol) + [tank.final_vol], c='grey', alpha=0.8)
            axes[i].hlines(tank.max_vol, xmin=0, xmax=24, color='grey', zorder=5, alpha=0.8)
            axes[i].plot(range(len(tank.vars) + 1), [tank.initial_vol] + tank.vars['value'].to_list(), c='k')
            axes[i].grid()

        plt.suptitle(f'Monte Carlo simulations - Violations rate: {vr*100:.0f}')
        plt.tight_layout()

    def plot_sample(self, sample, demands):
        fig, axes = plt.subplots(nrows=2, ncols=int(len(self.sim.network.tanks) / 2 + 1), sharex=True, figsize=(12, 6))
        axes = axes.ravel()
        t = demands.shape[1]

        for i, (tank_name, tank) in enumerate(self.sim.network.tanks.items()):
            axes[i].set_title(tank_name)
            axes[i].plot(sample[i * t: (i + 1) * t, :], c='C0', alpha=0.4)
            axes[i].plot(demands[i, :], c='k')
            axes[i].grid()

    def get_violations_for_tank(self, tank, res):
        z_max_vol = np.where(res - tank.max_vol <= self.epsilon, 0, 1)  # 1 for constraint violation, 0 otherwise
        n_max_vol = np.count_nonzero(z_max_vol == 1, axis=0)  # counts number of violations in each sample
        n_max_vol = np.array([np.ones(n_max_vol.shape), n_max_vol]).min(axis=0)  # 1 if any violation in the sample

        z_min_vol = np.where(- res + tank.min_vol.reshape(-1, 1) <= self.epsilon, 0, 1)
        n_min_vol = np.count_nonzero(z_min_vol == 1, axis=0)
        n_min_vol = np.array([np.ones(n_min_vol.shape), n_min_vol]).min(axis=0)

        z_final = np.where(- res[-1, :] + tank.final_vol <= self.epsilon, 0, 1)  # 1 for constraint violation, 0 otherwise

        violations = np.vstack([n_max_vol, n_min_vol, z_final])  # stack all tanks violations
        n_violations = np.count_nonzero(violations == 1, axis=0)  # counts number of violations in each sample
        n_violations = np.array([np.ones(n_violations.shape), n_violations]).min(axis=0)  # 1 if any violation in the sample
        return n_violations

    def normal(self):
        # To clean in the future
        demands = np.array([t.vars['demand'].values for t in self.sim.network.tanks.values()])
        demands = demands.T
        time_steps, n_tanks = demands.shape
        non_zero_idx = np.argwhere(demands.any(axis=0)).flatten()
        demands = demands[:, demands.all(axis=0)]
        dem_tanks = demands.shape[1]

        zeta = np.random.normal(0, 1, size=(demands.shape[0], demands.shape[1], self.n_sim))
        cov = uutils.observations_cov(demands, self.rho)

        d = np.linalg.cholesky(cov)
        sample = d @ zeta
        sample = sample.reshape(self.n_sim, demands.shape[0], demands.shape[1])
        sample = demands + sample

        A = np.zeros(shape=(self.n_sim, time_steps, n_tanks))
        B = np.zeros(shape=(time_steps, n_tanks))
        for i, idx in enumerate(list(non_zero_idx)):
            A[:, :, idx] = sample[:, :, i]
            B[:, idx] = demands[:, i]

        sample = A
        demands = B
        return sample, demands.T

    def uniform_sample(self):
        rng = np.random.default_rng()
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
            sample[idx * t: (idx + 1) * t] = nominal[idx * t: (idx + 1) * t] + z[i * t: (i + 1) * t]

        return sample, demands.T

    def multivariate_sample(self):
        rng = np.random.default_rng()
        demands = np.array([t.vars['demand'].values for t in self.sim.network.tanks.values()])
        demands = demands.T
        t, n_tanks = demands.shape
        non_zero_idx = np.argwhere(demands.any(axis=0)).flatten()

        sigma = self.udata['demand'].sigma
        z = rng.multivariate_normal(np.zeros(sigma.shape[0]), sigma, size=self.n_sim, method='cholesky').T

        z = np.random.normal(0, 1, size=((sigma.shape[0], self.n_sim)))
        z = self.udata['demand'].delta @ z

        nominal = demands.T.reshape(-1, 1)
        sample = np.zeros((t * n_tanks, self.n_sim))

        for i, idx in enumerate(list(non_zero_idx)):
            sample[idx * t: (idx + 1) * t] = nominal[idx * t: (idx + 1) * t] + z[i * t: (i + 1) * t]

        return sample, demands.T

    def multivariate_in_ellipse(self):
        rng = np.random.default_rng()
        demands = np.array([t.vars['demand'].values for t in self.sim.network.tanks.values()])
        demands = demands.T
        t, n_tanks = demands.shape
        non_zero_idx = np.argwhere(demands.any(axis=0)).flatten()

        z = np.random.normal(0, 1, size=((sigma.shape[0], self.n_sim)))
        z = np.apply_along_axis(lambda x: x / np.sqrt(np.sum(x ** 2)), axis=0, arr=z)
        z = self.udata['demand'].delta @ z

        nominal = demands.T.reshape(-1, 1)
        sample = np.zeros((t * n_tanks, self.n_sim))

        for i, idx in enumerate(list(non_zero_idx)):
            sample[idx * t: (idx + 1) * t] = nominal[idx * t: (idx + 1) * t] + z[i * t: (i + 1) * t]

        return sample, demands.T


if __name__ == "__main__":
    pass
    # res = pd.read_csv('Tank 7.csv', sep=',', header=None).values
    # max_vol, min_vol = 1179, 620.7  # tank 8
    # max_vol, min_vol = 500, 200  # tank 1
    #
    # print(res)
    # z_max_vol = np.where(res - max_vol < 0, 0, 1)  # 1 for constraint violation, 0 otherwise
    # print(z_max_vol)
    # n_max_vol = np.count_nonzero(z_max_vol == 1, axis=0)  # counts number of violations in each sample
    # print(n_max_vol)
    # n_max_vol = np.array([np.ones(n_max_vol.shape), n_max_vol]).min(axis=0)  # 1 if any violation in the sample
    # print(n_max_vol)
