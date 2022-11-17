import datetime
import matplotlib.pyplot as plt

from .network import Network
from .simulation import Simulation
from . import graphs

data_folder = 'data/sopron'
net = Network(data_folder)

t1 = datetime.datetime(2020, 8, 2, 0, 00)
t2 = datetime.datetime(2020, 8, 3, 0, 00)
sim = Simulation(data_folder, net, t1, t2)

sim.lp_formulate()
obj = sim.lp_run(integer=False)
sim.get_results()
print(obj)
print(sim.results.summary)

gr = graphs.SimGraphs(sim)
gr.all_tanks(level=False)
gr.facilities_flow(sim.network.pump_stations)
gr.facilities_flow(sim.network.vsp)
gr.all_power_stations(sim.network.power_stations)

gr.all_valves()
gr.all_pressure_zones()
gr.system_demand()
plt.show()