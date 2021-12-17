# Izhikevich Spiking Neurons - Regular Spiking (RS), Fast SPiking (FS) and Chattering (CH) neurons 
# Author - BORIS KUNDU
# Usage - python Izhikevich_Spiking_Neurons.py

#Import packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Placeholders For mean spike rate
meanSpikeRateRS = []
meanSpikeRateFS = []

#Display plots for Membrane Potential vs Time
def display(tspan, V, figNum, title,curr,plot):
    plt.subplot(1,5,plot)
    x_label = 'Time Step'
    y_label = 'Membrane Potential (mV)'
    plt.xlabel(x_label,fontsize=8)
    plt.ylabel(y_label,fontsize=8)
    plt.xticks(fontsize = 8)
    plt.yticks(fontsize = 8)
    figTitle = "Figure " + str(figNum) + " " + title + " (I=" + str(curr) +")"
    plt.title(figTitle,fontsize=8)
    plt.plot(tspan,V)
    
#Function for generating RS neuron
def compute(a,b,c,d,title,curr,figNum,plot):
    # Steps
    steps = 1000

    # Step size in ms
    tau = 0.25
    
    #Threshold
    spikeT = 30
    
    #Count spikes
    spikeCount = 0
    
    #tspan is the time interval
    tspan = np.arange(0, steps+tau, tau)
    
    # Time-series of membrane potentials
    V = np.zeros(len(tspan))
    #Initialized to resting potential
    V[0] = -64
    
    # Time-series of resting potential
    u = np.zeros(len(tspan))
    u[0] = b*V[0]
    
    #Input current
    I = np.zeros(len(tspan))
    I[200:] = curr
    
    for i in range(1, len(tspan)):
        # Check if this is a spike
        if V[i-1] < spikeT:
            V[i]    = V[i-1] + tau * (0.04 * (V[i-1]**2) + 5 * V[i-1] + 140 - u[i-1] + I[i-1])
            u[i]    = u[i-1] + tau * a * (b * V[i-1] - u[i-1])
        else:
            spikeCount = spikeCount +1
            # Put spike value
            V[i-1] = spikeT    
            # Reset membrane potential
            V[i] = c
            # Reset recovery potential
            u[i] = u[i-1] + d       

    #Display neuron plot
    if curr in[1,10,20,30,40] and figNum < 13:
        display(tspan, V, figNum, title, curr,plot)
    elif (figNum >= 13):
        display(tspan, V, figNum, title, curr,plot)
    #Retuen Mean Spike Rate
    return (spikeCount/800)

#Function for generating RS neuron
def regularSpiking(a = 0.02, b = 0.2, c = -65, d = 8):
    sr = 0
    fignum = 0
    plt.figure(num = 'Regular Spiking (RS) Neuron', figsize = (100,10))
    for curr in range (0,41): 
        #Call compute
        if curr in [1,10,20,30,40]:
            fignum = fignum + 1
        sr = compute(a,b,c,d,'RS',curr,fignum,fignum)
        meanSpikeRateRS.append(sr)
    plt.subplots_adjust(left = 0.04, right = 1, bottom=0.08, wspace=0.3, top = 0.97)
    plt.show()

#Function for generating FS neuron
def fastSpiking(a = 0.1, b = 0.2, c = -65, d = 2):
    sr = 0
    fignum = 6
    plot = 0
    plt.figure(num = 'Fast Spiking (FS) Neuron', figsize = (100,10))
    for curr in range (0,41): 
        #Call compute
        if curr in [1,10,20,30,40]:
            fignum = fignum + 1
            plot = plot + 1
        sr = compute(a,b,c,d,'FS',curr,fignum,plot)
        meanSpikeRateFS.append(sr)
    plt.subplots_adjust(left = 0.04, right = 1, bottom=0.08, wspace=0.3, top = 0.97)
    plt.show()

#Function for generating CH neuron
def chattering(a = 0.02, b = 0.2, c = -50, d = 2):
    sr = 0
    fignum = 12
    plot = 0
    plt.figure(num = 'Chattering (CH) Neuron', figsize = (100,10))
    for curr in [1,5,10,15,20]: 
        #Call compute
        fignum = fignum + 1
        plot = plot + 1
        sr = compute(a,b,c,d,'CH',curr,fignum,plot)
    plt.subplots_adjust(left = 0.04, right = 1, bottom=0.08, wspace=0.3, top = 0.97)
    plt.show()

#Generate Regular Spiking (RS) neurons
regularSpiking()

#Plot Mean Spike Rate for RS
I = [x for x in range(0,41)]
x_label = 'I'
y_label = 'R'
plt.figure(num = 'Mean Spike Rate (R) vs Input (I) for Regular Spiking (RS) Neuron')
plt.xlabel(x_label)
plt.ylabel(y_label)
#Plot I vs R for RS
plt.title("Figure 6 R vs I for RS")
plt.plot(I,meanSpikeRateRS, color='blue',linewidth=2,marker='o',markersize='5',markerfacecolor='yellow',markeredgewidth=2,markeredgecolor='green')
plt.show()

#Generate Fast Spiking (FS) neurons
fastSpiking()

#Plot Mean Spike Rate for RS and FS together
fig,axes = plt.subplots(figsize=(6,3),num='Mean Spike Rate (R) vs Input (I) for Regular Spiking (RS) and Fast Spiking (FS) Neuron')

axes.set_title('Figure 12 R vs I for RS and FS')
axes.set_xlabel(x_label)
axes.set_ylabel(y_label)

axes.plot(I,meanSpikeRateRS,label = 'Regular Spiking (RS) Neuron')
axes.plot(I,meanSpikeRateFS,label = 'Fast Spiking (FS) Neuron')
axes.legend()
fig.show()

#Generate Chattering (CH) neurons
chattering()