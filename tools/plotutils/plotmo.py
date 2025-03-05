#!/bin/python3

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('agg')
import matplotlib.legend
import numpy as np
import sys

#plot_sum = True

conditions = [
    {
        'filename' : 'mo_level.0.dat',
        'linestyle': '-',
        'linewidth': 0.5,
    },
]

plt.rcParams["figure.dpi"] = 600
#plt.rcParams["font.size"] = 10
#plt.rcParams["figure.figsize"] = [ 3.0, 3.0 ]

fig, ax = plt.subplots()

for condition in conditions:
    
    xdata = []
    ydata = []
    
    with open( condition['filename'], 'r') as datafile:

        for iline, line in enumerate(datafile):

            ll = line.split()
            xdata.append( float(ll[3]) )
            y_step = [ float(word) for word in ll[5:] ]

            n_mo = len(y_step)

            ydata.append(y_step)

    xdata = np.array(xdata)
    ydata = np.array(ydata)

    for i_mo in range(n_mo):

        ax.plot(
            xdata,
            ydata[:,i_mo],
            linewidth = 0.5,
        )

ax.set_xlabel('Time [fs]')
ax.set_ylabel('MO level')

ax.set_ylim( (-1, 1) )
#ax.set_xlim( (0, 5) )

ax.legend(frameon = False)

plt.savefig(
    'mo.png',
    bbox_inches = 'tight',
    pad_inches = 0,
)
