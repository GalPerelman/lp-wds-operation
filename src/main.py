import os
import sys
import datetime
import matplotlib.pyplot as plt

from network import Network
from simulation import Simulation
import graphs

data_folder = 'data/Sopron'
net = Network(data_folder)

t1 = datetime.datetime(2020, 8, 2, 0, 00)
t2 = datetime.datetime(2020, 8, 3, 0, 00)
sim = Simulation(data_folder, net, t1, t2)


sim.lp_formulate()
obj = sim.lp_run(integer=False)
sim.get_results()
print(obj)

print(sim.results.summary)
#
gr = graphs.SimGraphs(sim)
gr.all_tanks(level=False)
gr.facilities_flow(sim.network.pump_stations)
gr.facilities_flow(sim.network.vsp)
gr.all_power_stations(sim.network.power_stations)

# gr.all_valves()
# gr.all_pressure_zones()

# gr.system_demand()
# print(sim.network.pump_stations['B4p'].vars)
# print(sim.network.valves['v_guy'].vars)

plt.show()