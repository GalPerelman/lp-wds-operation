import numpy as np
from numpy.random import default_rng
import pandas as pd
import matplotlib.pyplot as plt

np.set_printoptions(linewidth=np.inf)


class MC:
    def __init__(self, sim, n_sim, gamma, delta, rho):
        self.sim = sim
        self.n_sim = n_sim
        self.gamma = gamma
        self.delta = delta
        self.rho = rho

        self.epsilon = 0.0001
        self.num_tanks = len(self.sim.network.tanks)

    def run(self, distribution, plot=False):
        cases = {'multivariate': self.multivariate_sample(),
                 'uniform': self.uniform_sample(),
                 'normal': self.normal()}

        sample, demands = cases[distribution]
        # sample, demands = self.multivariate_sample()
        # sample, demands = self.uniform_sample()
        # sample, demands = self.normal()

        sim_violations = []
        for i, (tank_name, tank) in enumerate(self.sim.network.tanks.items()):
            tank_inflow = self.sim.network.tanks[tank_name].vars['inflow'].values.reshape(-1, 1)
            res = (tank.initial_vol + (tank_inflow - sample[:, :, i].T).cumsum(axis=0))
            tank_violations = self.get_violations_for_tank(tank, res)
            sim_violations.append(tank_violations)

        sim_violations = np.vstack(sim_violations)
        sim_violations = np.count_nonzero(sim_violations == 1, axis=0)  # counts number of violations in each sample
        sim_violations = np.array([np.ones(sim_violations.shape), sim_violations]).min(axis=0)
        if plot:
            self.plot(sample)
        return sum(sim_violations)/self.n_sim

    def plot(self, sample):
        fig, axes = plt.subplots(nrows=2, ncols=int(len(self.sim.network.tanks) / 2 + 1), sharex=True, figsize=(12, 6))
        axes = axes.ravel()

        for i, (tank_name, tank) in enumerate(self.sim.network.tanks.items()):
            tank_inflow = self.sim.network.tanks[tank_name].vars['inflow'].values.reshape(-1, 1)
            res = (tank.initial_vol + (tank_inflow - sample[:, :, i].T).cumsum(axis=0))

            #  Visualizing the sample
            res = np.vstack([np.ones((1, self.n_sim)) * tank.initial_vol, res])
            axes[i].set_title(tank_name)
            axes[i].plot(res, c='C0', alpha=0.4)
            axes[i].plot(range(len(tank.vars) + 1), list(tank.min_vol) + [tank.final_vol], c='grey', alpha=0.8)
            axes[i].hlines(tank.max_vol, xmin=0, xmax=24, color='grey', zorder=5, alpha=0.8)
            axes[i].plot(range(len(tank.vars) + 1), [tank.initial_vol] + tank.vars['value'].to_list(), c='k')
            axes[i].grid()

        # plt.suptitle(f'Monte Carlo simulation\nViolations rate: {(sum(sim_violations) / self.n_sim):.2f}')
        plt.tight_layout()

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

    @staticmethod
    def constant_correlation_mat(size, rho):
        mat = np.ones((size, size)) * rho
        diag = np.diag_indices(size)
        mat[diag] = 1.
        return mat

    def normal(self):
        demands = np.array([t.vars['demand'].values for t in self.sim.network.tanks.values()])
        demands = demands.T
        zeros_idx = np.where(~demands.all(axis=0))

        zeta = np.random.normal(0, 1, size=(demands.shape[0], demands.shape[1], self.n_sim))
        cov = get_cov_by_correlation(demands, self.rho, norm=False)

        d = np.linalg.cholesky(cov)
        sample = d @ zeta
        sample = sample.reshape(self.n_sim, demands.shape[0], demands.shape[1])
        sample = demands + sample
        return sample, demands.T

    def uniform_sample(self):
        demands = np.array([t.vars['demand'].values for t in self.sim.network.tanks.values()])
        demands = demands.T
        zeros_idx = np.where(~demands.all(axis=0))

        zeta = np.random.normal(0, 1, size=(demands.shape[0], demands.shape[1], self.n_sim))
        y = np.apply_along_axis(lambda x: x / np.sqrt(np.sum(x ** 2)), axis=0, arr=zeta) * self.gamma

        cov = get_cov_by_correlation(demands, self.rho)
        d = np.linalg.cholesky(cov)

        sample = d @ y
        sample = sample.reshape(self.n_sim, demands.shape[0], demands.shape[1])
        sample = demands + sample
        sample[:, :, zeros_idx] = 0
        return sample, demands.T

    def multivariate_sample(self):
        demands = np.array([t.vars['demand'].values for t in self.sim.network.tanks.values()])
        demands = demands.T
        zeros_idx = np.where(~demands.all(axis=0))

        cov = get_cov_by_correlation(demands, self.rho).T
        rng = np.random.default_rng()
        sample = rng.multivariate_normal(np.zeros(demands.shape[1]), cov, size=(demands.shape[0], self.n_sim),
                                         method='cholesky')

        sample = sample.reshape(self.n_sim, demands.shape[0], demands.shape[1])
        sample = demands * (1 + sample)
        # sample[sample < 0] = 0
        sample[:, :, zeros_idx] = 0
        return sample, demands.T

    def multivariate_in_ellipse(self):
        demands = np.array([t.vars['demand'].values for t in self.sim.network.tanks.values()])
        demands = demands.T
        zeros_idx = np.where(~demands.all(axis=0))

        Z = np.random.normal(0, 1, size=(self.n_sim, demands.shape[0], demands.shape[1]))
        Z = np.apply_along_axis(lambda x: x / np.sqrt(np.sum(x ** 2)), axis=0, arr=Z)
        z = (Z[:, :, 5].T)
        D = np.linalg.cholesky(cov)
        sample = Z @ D.T


def get_poitive_defined_cov(demands):
    """ observation (hours) in rows, features (demands zones) in columns - rowvar=False"""
    cov = np.cov(demands, rowvar=False)
    return cov



def constant_correlation_mat(size, rho):
    mat = np.ones((size, size)) * rho
    diag = np.diag_indices(size)
    mat[diag] = 1.
    return mat


def get_cov_by_correlation(demands, rho, norm=True):
    if demands.shape[1] == 1:
        std = demands.std(axis=0)
    else:
        std = demands.std(axis=1)

    if norm:
        std = std / np.linalg.norm(std)
    sigma = np.zeros((demands.shape[1], demands.shape[1]))
    np.fill_diagonal(sigma, std)
    corr = constant_correlation_mat(demands.shape[1], rho)
    cov = sigma @ corr @ sigma
    return cov





def plot_multivariate_sample(demands, sample):
    import math
    fig, axes = plt.subplots(nrows=2, ncols=math.ceil(demands.shape[1] / 2), sharex=True, figsize=(12, 6), sharey=True)
    axes = axes.ravel()

    for i in range(demands.shape[1]):
        axes[i].plot(sample[:, :, i].T, c='C0', alpha=0.3)
        axes[i].plot(demands[:, i], c='k')
        axes[i].grid()
    plt.tight_layout()


if __name__ == "__main__":
    res = pd.read_csv('Tank 7.csv', sep=',', header=None).values
    max_vol, min_vol = 1179, 620.7  # tank 8
    max_vol, min_vol = 500, 200  # tank 1

    print(res)
    z_max_vol = np.where(res - max_vol < 0, 0, 1)  # 1 for constraint violation, 0 otherwise
    print(z_max_vol)
    n_max_vol = np.count_nonzero(z_max_vol == 1, axis=0)  # counts number of violations in each sample
    print(n_max_vol)
    n_max_vol = np.array([np.ones(n_max_vol.shape), n_max_vol]).min(axis=0)  # 1 if any violation in the sample
    print(n_max_vol)










# #     demands = np.loadtxt('demands.csv', delimiter=',').T
# #     cov = get_cov_by_correlation(demands, 0.8).T
# #     N = 50
# #
# #     Z = np.random.normal(0, 1, size=(N, 24, 5))
# #     # Z = np.apply_along_axis(lambda x: x / np.sqrt(np.sum(x ** 2)), axis=0, arr=Z)
# #     print(np.mean(Z))
# #     print(np.std(Z))
# #     D =np.linalg.cholesky(cov)
# #     A = Z@D.T
# #     print(np.mean(A), np.std(A))
# #     A = demands + A
# #     A[A < 0] = 0
# #     plot_multivariate_sample(demands, A)
# #
# #     rng = np.random.default_rng()
# #     sample = rng.multivariate_normal(np.zeros(demands.shape[1]), cov, size=(demands.shape[0], N), method='cholesky')
# #     sample = sample.reshape(N, demands.shape[0], demands.shape[1])
# #     sample = demands + sample
# #     sample[sample < 0] = 0
# #     total_dem = (sample.sum(axis=1).sum(axis=1))
# #     plot_multivariate_sample(demands, sample)
# #
# #     plt.show()


###############################
    # std = demands.std(axis=1)
    # sigma = np.zeros((num_tanks, num_tanks))
    # # np.fill_diagonal(sigma, np.full(shape=(num_tanks, 1), fill_value=self.delta))
    # np.fill_diagonal(sigma, std)
    # corr = self.constant_correlation_mat(num_tanks, 0.8)
    # cov = sigma @ corr @ sigma

    # zeta = np.random.normal(loc=0, scale=1, size=(self.sim.num_steps, self.n_sim, demands.shape[0]))
    # y = (zeta / sum(np.sqrt(zeta ** 2))) * self.gamma
    # D = np.linalg.cholesky(cov)
    #
    # sample = demands.mean(axis=1) + y @ D
    # sample.reshape((self.n_sim, demands.shape[0], self.sim.num_steps))