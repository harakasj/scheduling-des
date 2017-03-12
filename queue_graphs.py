"""
load the binary, make some histograms.
"""


# Need to import class to read the instances 
from queue_sim import Job
import pickle
import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt

# load the instances
arrivals = pickle.load(open( "process_data.p","rb") )

wait_t = []
trnd_t = []

for a in arrivals:
    wait_t.append(a.wait_t)
    trnd_t.append(a.trnd_t)

fig, (ax1, ax2) = plt.subplots( 2,1)
ax1.hist(wait_t,normed=1,bins=50, facecolor='blue')
ax1.set_title('Wait Time (norm)')
ax2.hist(trnd_t,normed=1, bins=50, facecolor='blue')
ax2.set_title('Turnaround Time (norm)')
fig.subplots_adjust(hspace=0.5)
plt.show()