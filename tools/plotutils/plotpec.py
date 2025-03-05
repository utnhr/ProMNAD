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
        'filename' : 'pec.0.dat',
        'linestyle': '-',
#        'linewidth': 1.0,
    },
]

plt.rcParams["figure.dpi"] = 600
plt.rcParams["font.size"] = 10
plt.rcParams["figure.figsize"] = [ 4.0, 3.0 ]

fig, ax = plt.subplots()

for condition in conditions:
    
    xdata = []
    ydata = []
    
    with open( condition['filename'], 'r') as datafile:

        for iline, line in enumerate(datafile):

            ll = line.split()
            xdata.append( float(ll[3]) )
            y_step = [ float(word) for word in ll[5:] ]

            n_pec = len(y_step)

            ydata.append(y_step)

    xdata = np.array(xdata)
    ydata = np.array(ydata) * 27.2114 # a.u. -> eV

    for i_pec in range(n_pec):

        if i_pec == 5:
            linewidth = 2.0
            color = 'red'
        elif i_pec == 6:
            linewidth = 2.0
            color = 'blue'
        else:
            linewidth = 0.7
            color = None

        ax.plot(
            xdata,
            ydata[:,i_pec],
            linewidth = linewidth,
            color = color,
            label = "%d" % i_pec,
        )

ax.set_xlabel('Time [fs]')
ax.set_ylabel('Potential [eV]')

#ax.set_ylim( (-1, 1) )
ax.set_xlim( (0, 50) )

#ax.legend(frameon = False)

plt.savefig(
    'pec.png',
    bbox_inches = 'tight',
    pad_inches = 0,
)
