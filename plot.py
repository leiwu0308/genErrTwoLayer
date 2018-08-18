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

def plot_n_vs_test_error():
	xR,yR = read_data('checkpoints/cifa','nsamples','test_error')
	xR, yR = np.log10(xR), np.log10(np.asarray(yR))
	yR_mean, yR_std = yR.mean(axis=1), yR.std(axis=1)

	asym_rate_func = AsympRate(xR[-1],yR_mean[-1])
	yR_bound = asym_rate_func(xR)

	plt.errorbar(xR,yR_mean,yerr=yR_std,linestyle='-',marker='o',markersize=10,label='Test Error')
	plt.plot(xR,yR_bound,linestyle='-',marker='*',label=r'$n^{-1/2}$')
	plt.xlabel(r'$\log_{10}(n)$')
	plt.ylabel(r'$\log_{10}$(MSE)')
	plt.legend()
	plt.savefig('figures/mnist_n_vs_test_error.pdf',bbox_inches='tight')

def plot_n_vs_path_norm():
	xR,yR = read_data('checkpoints/mnist_w10000_init1','nsamples','path_norm')
	xR, yR = np.log10(xR), np.log10(np.asarray(yR))
	yR_mean, yR_std = yR.mean(axis=1), yR.std(axis=1)

	asym_rate_func = AsympRate(xR[-1],yR_mean[-1],slope=0.5)
	yR_bound = asym_rate_func(xR)

	plt.errorbar(xR,yR_mean,yerr=yR_std,linestyle='-',marker='o',markersize=10,label='Path Norm')
	plt.plot(xR,yR_bound,linestyle='-',marker='*',label=r'$n^{-1/2}$')
	plt.xlabel(r'$\log_{10}(n)$')
	plt.ylabel(r'$\|\theta\|_{\mathcal{P}}$')
	plt.legend()
	plt.savefig('figures/mnist_n_vs_path_norm.pdf',bbox_inches='tight')


def main():
	argparser = argparse.ArgumentParser()
	argparser.add_argument('--data_dir',default='')
	argparser.add_argument('--plot',default='')
	args = argparser.parse_args()

	plots = {
		'nsamples': plot_n_vs,
		'width':0
	}
	plot = plots[args.plot]
	plot(args)

if __name__ == '__main__':
	# plot_n_vs_test_error()
	plot_n_vs_path_norm()
