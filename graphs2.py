import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


#### GRAPHS AND FITTING FUNCTIONS FOR DETERMING THE FRACTION OF SERIAL CODE ######

threads=np.arange(15+1) + 1
times_1250=[3.543,1.774,1.189,0.8977,0.7331,0.6129,0.528,0.462,0.422,0.381,0.347,0.32,0.303,0.283,0.264,0.2488]
times_3500=[27.697,13.86,9.31,6.98,5.711,4.77,4.13,3.65,3.28,2.96,2.69,2.49,2.35,2.22,2.03,1.91]
times_2000=[9.06,4.54,3.038,2.3,1.87,1.566,1.3447,1.186,1.078,0.9716,0.8846,0.819,0.773,0.719,0.672,0.63]
times_750=[1.278,0.642,0.43,0.322,0.265,0.22,0.192,0.1679,0.1537,0.139,0.127,0.116,0.111,0.103,0.097,0.091]
times_500=[0.566,0.284,0.1911,0.143,0.117,0.099,0.0865,0.0756,0.0688,0.0624,0.0576,0.0528,0.0503,0.0473,0.0447,0.0415]

T_1=times_500[0]
T_2=times_750[0]
T_3=times_1250[0]
T_4=times_2000[0]
T_5=times_3500[0]


def func1(threads,Fs):
    return (T_1*(Fs + (1-Fs)/threads))
def func2(threads,Fs):
    return (T_2*(Fs + (1-Fs)/threads))
def func3(threads,Fs):
    return (T_3*(Fs + (1-Fs)/threads))
def func4(threads,Fs):
    return (T_4*(Fs + (1-Fs)/threads))
def func5(threads,Fs):
    return (T_5*(Fs + (1-Fs)/threads))

T1popt,a=curve_fit(func1,threads,times_500)
T2popt,a=curve_fit(func2,threads,times_750)
T3popt,a=curve_fit(func3,threads,times_1250)
T4popt,a=curve_fit(func4,threads,times_2000)
T5popt,a=curve_fit(func5,threads,times_3500)

print(T1popt)
print(T2popt)
print(T3popt)
print(T4popt)
print(T5popt)

##t1popt is the fraction serial for each particle number

plt.xlabel('Number of compute cores')
plt.ylabel('Time (s)')
plt.plot(threads,times_500,marker='o',label='500 part')
plt.plot(threads,times_750,marker='o',label='750 part')
plt.plot(threads,times_1250,marker='o',label='1250 part')
plt.plot(threads,times_2000,marker='o',label='2000 part')
plt.plot(threads,times_3500,marker='o',label='3500 part')
plt.legend(loc='best')
plt.show()

plt.xlabel('Particle number')
plt.ylabel('Fraction serial')
parts=[500,750,1250,2000,3500]
Fsa=[T1popt,T2popt,T3popt,T4popt,T5popt]
plt.plot(parts,Fsa,marker='o')
plt.show()
