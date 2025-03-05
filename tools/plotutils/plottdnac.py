#!/bin/python3

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('agg')
import matplotlib.legend
import numpy as np
import sys

#plot_sum = True

conditions = [
#    {
#        'filename' : 'cmo_tdnac.dat',
#        'linestyle': '-',
#        'linewidth': 1.0,
#        'plot_range': [i for i in range(27, 39) ],
#        'add_label': True,
#        'n_occ': 33,
#    },
    {
        'filename' : 'mo_tdnac.dat',
        'linestyle': '-',
        'linewidth': 1.0,
        'plot_range': [i for i in range(27, 39) ],
        'add_label': True,
        'n_occ': 33,
    },
]

plt.rcParams["figure.dpi"] = 300
#plt.rcParams["font.size"] = 10
#plt.rcParams["figure.figsize"] = [ 3.0, 3.0 ]

fig, ax = plt.subplots()

for condition in conditions:
    
    xdata = []
    ydata = []
    
    with open( condition['filename'], 'r') as datafile:

        while True:

            ll = datafile.readline().split()
            
            try:
                xdata.append( float(ll[3]) )
            except:
                break

            n_read_mo = 0

            step_ydata = []

            while True:

                n_read_mo += 1

                ll = datafile.readline().split()

                n_ao = len(ll) // 2

                tdnac_vec = []

                for iword, word in enumerate(ll):

                    if iword % 2 == 0:
                        real_part = float(word.rstrip('+'))
                    else:
                        imag_part = float(word.rstrip('j,'))
                        c = real_part + (0.0+1.0j)*imag_part
                        tdnac_vec.append(c)

                tdnac_vec = np.array(tdnac_vec)

                step_ydata.append( np.linalg.norm(tdnac_vec) )

                n_mo = n_ao # assuming no basis-set truncation

                if n_read_mo == n_mo:
                   break
            
            step_ydata = np.array(step_ydata)

            ydata.append(step_ydata)
            
    xdata = np.array(xdata)
    ydata = np.array(ydata)

    for i_mo in reversed(condition['plot_range']):

        if condition['add_label']:
            if i_mo + 1 < condition['n_occ']:
                shift = condition['n_occ'] - (i_mo + 1)
                label = "HOMO-%d" % shift
            elif i_mo + 1 == condition['n_occ']:
                label = "HOMO"
            elif i_mo + 1 == condition['n_occ'] + 1:
                label = "LUMO"
            elif i_mo + 1 > condition['n_occ'] + 1:
                shift = i_mo + 1 - (condition['n_occ'] + 1)
                label = "LUMO+%d" % shift
        else:
            label = None

        ax.plot(
            xdata,
            ydata[:,i_mo],
            linewidth = condition['linewidth'],
            linestyle = condition['linestyle'],
            label = label,
        )

ax.set_xlabel('Time [fs]')
ax.set_ylabel('MO TDNAC')

ax.set_ylim( (0.0, 0.5) )
ax.set_xlim( (0, 60) )

ax.legend(frameon = False)

plt.savefig(
    'tdnac.png',
    bbox_inches = 'tight',
    pad_inches = 0,
)
