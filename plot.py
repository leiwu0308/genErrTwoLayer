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
mpl.rcParams['legend.fontsize']=25
mpl.rcParams['xtick.labelsize'] = 25
mpl.rcParams['ytick.labelsize'] = 25
mpl.rcParams['axes.labelsize'] = 30
mpl.rcParams['axes.titlesize'] = 20
import matplotlib.pyplot as plt


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
	return xR,yR

def plot_n_vs(dirname):
	xR,yR = read_data(dirname,'nsamples','path_norm')
	plt.loglog(xR,yR,'-o',label='Path norm')
	plt.savefig('figures/mnist_n_vs_pathnorm.png',bbox_inches='tight')


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
	x,y=read_data('checkpoints','nsamples','path_norm')
	print(x,y)
