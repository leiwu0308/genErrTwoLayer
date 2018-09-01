import os
import pickle
import argparse
import json
import math
from collections import defaultdict, OrderedDict
import numpy as np
import matplotlib as mpl 

mpl.use('Agg')
mpl.rc('text',usetex=True)
mpl.rcParams['legend.fontsize']=20
mpl.rcParams['xtick.labelsize'] = 20
mpl.rcParams['ytick.labelsize'] = 20
mpl.rcParams['axes.labelsize'] = 25
mpl.rcParams['axes.titlesize'] = 20
mpl.rcParams['lines.linewidth'] = 3
mpl.rcParams['lines.markersize'] = 15
import matplotlib.pyplot as plt

class AsympRate:
	def __init__(self,x0,bias,slope = -0.5):
		self.x0 = x0
		self.b = bias
		self.slope = slope

	def __call__(self,x):
		y = self.slope * (x-self.x0) + self.b
		return y



def read_data(dirname,x_key,y_key):
	res = []
	filenames = os.listdir(dirname)
	for filename in filenames:
		filename = os.path.join(dirname,filename)
		with open(filename,'rb') as f:
			data = pickle.load(f)
		res.append((data[x_key],data[y_key]))

	res = sorted(res,key=lambda t:t[0])
	x,y = zip(*res)
	return x,y


def plot_init_vs_test_accuracy(save_file='tmp.png'):
	plt.xscale('log')
	xR,yR = read_data('checkpoints/mnist_w10000_n100_lmbd1/','init_fac','test_accuracy')
	yR = np.asarray(yR)
	yR_mean, yR_std = yR.mean(axis=1), yR.std(axis=1)
	plt.errorbar(xR,yR_mean,yerr=yR_std,color='k',linestyle='-',marker='o',markersize=10,label=r'$\lambda=1$')

	xR,yR = read_data('checkpoints/mnist_w10000_n100_lmbd0/','init_fac','test_accuracy')
	yR = np.asarray(yR)
	yR_mean, yR_std = yR.mean(axis=1), yR.std(axis=1)
	plt.errorbar(xR,yR_mean,yerr=yR_std,color='r',linestyle='-',marker='*',label=r'$\lambda=0$')

	plt.xlabel(r'initialization: $\kappa$')
	plt.ylabel(r'Test accuracy(\%)')
	plt.legend(loc=3)
	plt.savefig(save_file,bbox_inches='tight')



def plot_n_vs_test_error(dirname,save_file='tmp.pdf'):
	xR,yR = read_data(dirname,'nsamples','test_error')
	xR, yR = np.log10(xR), np.asarray(yR)
	yR_mean, yR_std = yR.mean(axis=1), yR.std(axis=1)

	asym_rate_func = AsympRate(xR[-1],yR_mean[-1])
	yR_bound = asym_rate_func(xR)

	plt.errorbar(xR,yR_mean,yerr=yR_std,color='r',linestyle='-',marker='o',markersize=10,label='Test Error')
	# plt.plot(xR,yR_bound,linestyle='--',color='k',label=r'$n^{-1/2}$')
	plt.xlabel(r'$\log_{10}(n)$')
	plt.ylabel(r'$\log_{10}$(MSE)')
	plt.legend()
	plt.savefig(save_file,bbox_inches='tight')

def plot_n_vs_path_norm(dirname,save_file='tmp.pdf'):
	xR,yR = read_data(dirname,'nsamples','path_norm')
	xR, yR = np.log10(xR), np.log10(np.asarray(yR))
	yR_mean, yR_std = yR.mean(axis=1), yR.std(axis=1)

	asym_rate_func = AsympRate(xR[-1],yR_mean[-1],slope=0.5)
	yR_bound = asym_rate_func(xR)

	plt.errorbar(xR,yR_mean,yerr=yR_std,color='r',linestyle='-',marker='o',markersize=10,label='Path Norm')
	plt.plot(xR,yR_bound,color='k',linestyle='--',label=r'$n^{-1/2}$')
	plt.xlabel(r'$\log_{10}(n)$')
	plt.ylabel(r'$\log_{10} \|\theta\|_{\mathcal{P}}$')
	plt.legend()
	plt.savefig(save_file,bbox_inches='tight')

def plot_w_vs_test_error(save_file):
	xR,yR = read_data('checkpoints/mnist_n100_lambd1_w/','width','test_accuracy')
	yR = np.asarray(yR)
	yR_mean, yR_std = yR.mean(axis=1), yR.std(axis=1)
	plt.errorbar(np.log10(xR),yR_mean,yerr=yR_std,color='k',linestyle='-',marker='o',markersize=10,label=r'$\lambda=1$')

	xR,yR = read_data('checkpoints/mnist_n100_lambd0_w/','width','test_accuracy')
	yR = np.asarray(yR)
	yR_mean, yR_std = yR.mean(axis=1), yR.std(axis=1)
	plt.errorbar(np.log10(xR),yR_mean,yerr=yR_std,color='r',linestyle='-',marker='*',label=r'$\lambda=0$')

	plt.xlabel(r'$\log_{10}(m)$')
	plt.ylabel(r'Test accuracy(\%)')
	plt.legend(loc=4)
	plt.savefig(save_file,bbox_inches='tight')

def plot_w_vs_path_norm(save_file):
	plt.xscale('log')
	plt.yscale('log')
	xR,yR = read_data('checkpoints/mnist_n100_lambd1_w/','width','path_norm')
	yR = np.asarray(yR)
	yR_mean, yR_std = yR.mean(axis=1), yR.std(axis=1)
	plt.errorbar(xR,yR_mean,yerr=yR_std,color='k',linestyle='-',marker='o',markersize=10,label=r'$\lambda=1$')

	xR,yR = read_data('checkpoints/mnist_n100_lambd0_w/','width','path_norm')
	yR = np.asarray(yR)
	yR_mean, yR_std = yR.mean(axis=1), yR.std(axis=1)
	plt.errorbar(xR,yR_mean,yerr=yR_std,color='r',linestyle='-',marker='*',label=r'$\lambda=0$')

	plt.xlabel(r'network width: $m$')
	plt.ylabel(r'path norm: $\|\theta\|_{\mathcal{P}}$')
	# plt.ylim(ymin=0)
	plt.legend(loc=2)
	plt.savefig(save_file,bbox_inches='tight')

def main_n():
	plot_init_vs_test_error('checkpoints/mnist_w10000_n100/','figures/mnist_init_vs_test_error.pdf')
	plot_n_vs_test_error('checkpoints/cifar10_w10000_init1_lmbd0.5','figures/cifar10_n_vs_test_error.png')


if __name__ == '__main__':
	# plot_n_vs_path_norm('checkpoints/mnist_w10000_init1/','figures/mnist_n_vs_path_norm.pdf')
	plot_init_vs_test_accuracy('figures/mnist_init_vs_test_error.pdf')
	# plot_n_vs_test_error('checkpoints/mnist_w10000_init1/','figures/mnist_n_vs_test_error.pdf')
	# plot_n_vs_path_norm('checkpoints/cifar10_w10000_init1_lmbd0.5','figures/cifar10_n_vs_path_norm.pdf')
	# plot_n_vs_test_error('checkpoints/cifar10_w10000_init1_lmbd0.5','figures/cifar10_n_vs_test_error.pdf')
	# plot_w_vs_path_norm('figures/mnist_w_vs_path_norm.pdf')
	# plot_w_vs_test_error('figures/mnist_w_vs_test_error.pdf')


