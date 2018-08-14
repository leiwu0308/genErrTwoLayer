import numpy as np 
import matplotlib 
matplotlib.use('Agg')
# matplotlib.rcParams['text.usetex'] = True
# matplotlib.rcParams['text.latex.unicode'] = True
import matplotlib.pyplot as plt 

# factors = [0.01,0.1,1,2,5,10,20,30,50,100]
# gd = [70.35,71.71,63.96,45.56,24.63,15.02,12.74,10.04,12.35,10.5]
# random_guess = [10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.,10.]

# phn_teacc = [62.54,62.09,63.47,65.85,65.87,66.05,63.71,66.76,64.82]
# phn_pnorm = [31,31,31,33,42,62,76.1,41,62]

# wd_teacc = [68.，68, 69.06，67.75]
# wd_pnorm = [65，66,120,230]

# plt.semilogx(factors,gd,'-ob',markerfacecolor='w',
# 	markersize=10,label='GD solution')
# plt.semilogx(factors,random_guess,'--k',label='random guess')
# plt.semilogx(factors[0:7],phn_teacc,'-or',markerfacecolor='w',label='GD+path norm')

# plt.xlabel('Variance of initialization', fontsize=20)
# plt.ylabel('Test accuracy (%)',fontsize=20)
# plt.ylim([0,73])
# plt.legend(fontsize=15)
# # plt.text(0.1,13,'Random guess',color='blue')

# plt.savefig('figures/teacc_vs_initialization.png',bbox_inches='tight')


class AsympRate:
	def __init__(self,x0,bias,slope=0.5):
		self.x0 = x0
		self.b = bias
		self.slope = slope

	def __call__(self,x):
		return - self.slope * (x-self.x0) + self.b

# data 1
nsamples = [10, 50, 100, 500, 1000, 5000, 10000,50000]
tot_loss = [2.39e-2,1.61e-2,1.22e-2,7.91e-3,6.27e-3,3.31e-3,2.81e-3,1.61e-3]
tr_loss = [2.2e-3,1.3e-3,1.3e-3,1.4e-3,1.1e-3,5.6e-4,5.6e-4,3.7e-4]
te_loss = [9.6e-2,6.9e-2,5.2e-2,3.1e-2,2.6e-2,1.4e-2,1.0e-2,5.9e-3]

# data 2
data =[ 
	[10,2.39e-2,2.2e-3,9.6e-2,6.8e0],
	[30,2.01e-2,2.0e-3,7.8e-2,1.7e1],
	[60,1.49e-2,1.3e-3,6.4e-2,2.6e1],
	[90,1.22e-2,1.2e-3,5.9e-2,3.1e1],
	[120,9.99e-3,8.3e-4,5.3e-2,3.4e1],
	[150,9.70e-3,9.8e-4,5.2e-2,4.1e1],
	[180,9.12e-3,9.2e-4,4.9e-2,4.6e1]
]

# data 3 
data = [
	[10,4.34e-2,8.7e-3,9.7e-2,5.4],
	[40,3.23e-2,5.4e-3,7.2e-2,1.7e1],
	[80,2.29e-2,3.5e-3,5.8e-2,2.4e1],
	[120,1.83e-2,2.7e-3,5.2e-2,2.9e1]
]

# data 4
data = [
	[5,4.94e-2,2.1e-3,6.9e-2,1.9],
	[50,1.02e-2,9.1e-4,3.1e-2,3.6],
	[100,7.34e-3,8.3e-4,1.4e-2,5.1],
	[500,2.58e-3,4.1e-4,4.7e-3,8.5],
	[1000,1.65e-3,2.7e-4,2.8e-3,11]
]

data = np.log10(np.asarray(data))
xR= data[:,0]
func_teacc = AsympRate(xR[0],data[0,3],slope=0.5)
func_pnorm = AsympRate(xR[0],data[0,4],slope=-0.5)
print(func_pnorm.slope)


te_asym = func_teacc(xR)
pnorm_asym = func_pnorm(xR)
te_real = data[:,3]
pnorm_real = data[:,4]



plt.plot(xR,te_real,'-o',label=r'test loss')
plt.plot(xR,te_asym,'--k',label=r'$1/\sqrt{n}$')
plt.legend(fontsize=15)
plt.xlabel(r'$\log(n)$',fontsize=15)
plt.ylabel(r'$\log(error)$',fontsize=15)
plt.savefig('./figures/teloss_vs_nsample.png',bbox_inches='tight')


plt.figure()
plt.plot(xR,pnorm_real,'-*',label=r'path norm')
plt.plot(xR,pnorm_asym,'--k',label=r'$n^{1/2}$')
plt.legend(fontsize=15)
plt.xlabel(r'$\log(n)$',fontsize=15)
plt.ylabel(r'$\log(error)$',fontsize=15)
plt.savefig('./figures/pnorm_vs_nsample.png',bbox_inches='tight')
