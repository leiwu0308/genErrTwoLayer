import numpy as np 
import matplotlib 
matplotlib.use('Agg')
import matplotlib.pyplot as plt 

factors = [0.01,0.1,1,2,5,10,20,30,50,100]
gd = [70.35,71.71,63.96,45.56,24.63,15.02,12.74,10.04,12.35,10.5]
random_guess = [10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.,10.]

phn_teacc = [62.54,62.09,63.47,65.85,65.87,66.05,63.71,66.76,64.82]
phn_pnorm = [31,31,31,33,42,62,76.1,41,62]

wd_teacc = [68.，68, 69.06，67.75]
wd_pnorm = [65，66,120,230]

plt.semilogx(factors,gd,'-ob',markerfacecolor='w',
	markersize=10,label='GD solution')
plt.semilogx(factors,random_guess,'--k',label='random guess')
plt.semilogx(factors[0:7],phn_teacc,'-or',markerfacecolor='w',label='GD+path norm')

plt.xlabel('Variance of initialization', fontsize=20)
plt.ylabel('Test accuracy (%)',fontsize=20)
plt.ylim([0,73])
plt.legend(fontsize=15)
# plt.text(0.1,13,'Random guess',color='blue')

plt.savefig('figures/teacc_vs_initialization.png',bbox_inches='tight')


