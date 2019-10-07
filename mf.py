#Read antenna signal and apply noise
import numpy as np
import scipy.signal
import scipy.fftpack
import math
import random
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
import os

i=raw_input("Antenna number: ")
index = 'raw_antena' + str(i)

#function to convert CGS to SI of the electrin field
def to_SI(e):
	e_si = e*29979199999.34
	return e_si

#to nanosecond
def to_ns(t):
	t_ns = t*100000000
	return t_ns    

#signal
filename_dat = index + '.dat' 
t, x, y, z = np.loadtxt(filename_dat, delimiter ='\t', usecols =(0, 1, 2, 3), unpack = True)

#template
i=raw_input("Antenna number for template: ")
index = 'raw_antena' + str(i) + '.dat'
tt, xt, yt, zt = np.loadtxt(index, delimiter ='\t', usecols =(0, 1, 2, 3), unpack = True)

# creating a noise with the same dimension as the dataset
mu, sigma = 0, 15
noiseX = np.random.normal(mu, sigma, x.size) 
noiseY = np.random.normal(mu, sigma, y.size) 
noiseZ = np.random.normal(mu, sigma, z.size) 

signalX = to_SI(x) + noiseX
signalY = to_SI(y) + noiseY
signalZ = to_SI(z) + noiseZ

#FFT BANDPASS
def filter_frequency(s,t,f_min,f_max):
	y_fft = scipy.fftpack.fft(s)
	W = scipy.fftpack.fftfreq(s.size, d=t[1]-t[0])

	# If our original signal time was in seconds, this is now in Hz    
	cut_f_signal = y_fft.copy()

	cut_f_signal[(W<f_min*1000000)] = 0
	cut_f_signal[(W>f_max*1000000)] = 0
	cut_signal = scipy.fftpack.ifft(cut_f_signal) 
	
	return cut_signal 

x_inp_i = filter_frequency(signalX,t,50,200)
template = filter_frequency(to_SI(xt),tt,50,200)
template = template[0:500]

#matched filter
reverse_tem = template[::-1]
a = np.convolve(x_inp_i,reverse_tem)
a = np.absolute(a)

plt.subplot(3, 1, 1)
plt.plot(range(0,len(template)),template,'o')
plt.title('Template')
plt.ylabel('Ex [uV/m]')

plt.subplot(3, 1, 2)
plt.plot(range(0,len(x_inp_i)),np.absolute(x_inp_i),'o')
plt.title('Absolute signal of all antennas')
plt.xlabel('time (ns)')
plt.ylabel('Ex [uV/m]')

plt.subplot(3, 1, 3)
plt.plot(range(0,len(a)),a,'o')
plt.title('Apllying matched filter')
plt.xlabel('time (ns)')
plt.ylabel('Ex [uV/m]')

plt.show()