import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


### PLOTS OF STEPS AGAINST TIME


steps=[5,10,20,30,40,50,60,70,80,90,100,200,300,400,500,600,700,800,900,1000]
t_l=[0.312,0.632,0.1259,1.875,2.439,3.03,3.46,4.01,4.54,5.58,6.12,11.87,17.19,24.27,31.55,38.35,43.627,49.85,56.11,62.35]
t_omp=[0.556,1.13,2.25,3.38,4.519,5.63,6.77,7.95,9.08,10.24,11.23,22.6,33.87,45.12,56.7,67.55,79.3,89.7,101.5,112.37]
t_mpi=[0.606,0.8977,1.784,2.68,3.566,4.465,5.36,6.25,7.123,8.13,8.9,17.8,26.6,37.4,46.4,53.2,62.3,71.4,80.5,90.4]


threads=[2,4,8,12]
time=[11.39,8.04,5.788,5.036]
for i in range(4):
    time[i]=1/time[i]

for i in range(len(t_omp)):
    t_omp[i]=t_omp[i]/2.1

plt.xlabel('Steps')
plt.ylabel('Time (s)')
plt.plot(steps,t_l,marker='o',label='linear',markersize=4,linewidth=1.5)
plt.plot(steps,t_omp,marker='o',label='omp 16',markersize=4,linewidth=1.5)
plt.plot(steps,t_mpi,marker='o',label='mpi 16 thread',markersize=4,linewidth=1.5)
plt.legend(loc='best')
plt.show()


plt.xlabel('Threads')
plt.ylabel('Time (s^-1)')
#plt.plot(threads,time)
plt.plot(threads,time)
plt.show()
