import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)

import matplotlib.pyplot as plt
import pickle, copy, sys, os, csv
import numpy as np
import scipy.optimize, joblib
from matplotlib.colors import LogNorm

def get_train_test_data(foldername, max_size,num_of_features=44):
	series = []
	sizes = []
	feature_mins = [np.inf]*num_of_features
	feature_maxs = [-1*np.inf]*num_of_features
	for filename in os.listdir("./" + foldername + "/"):
		try:
		        f = open("./" + foldername + "/" + filename)
		except:
		        continue
		reader = csv.reader(f)

		lines = []
		for line in reader:
		        lines.append(line)

		temp_arr = np.array(lines[1:min(max_size + 1,len(lines))]).astype(float)

		series.append(temp_arr) 
		sizes.append(temp_arr.shape[0])
                

	feature_maxs = np.load("feature_maxs.npy")
        feature_mins = np.load("feature_mins.npy")

	for ind in range(len(series)):
		#min-max normalization of series
		series[ind] = (series[ind] - feature_mins)/np.where((feature_maxs - feature_mins) != 0.0, (feature_maxs - feature_mins), 1.0)
		#series[ind] = np.lib.pad(series[ind], ((0,max(0,max_size - series[ind].shape[0])),(0,0)), 'constant')
	series = np.array(series)
	#print series	
	print series.shape
	return series
    
headers_file = open("headers.csv","r").read()
headers = headers_file.split(",")


num_features = 17
num_hidden = 15
output_size = num_features


test_set = np.array(get_train_test_data(sys.argv[1], max_size=3000, num_of_features=num_features))	

lstm = torch.load("lstm_trained_model")
output_layer = torch.load("output_layer_trained_model")
timesteps = 0
for index in range(0,len(test_set)):
    hidden = (autograd.Variable(torch.zeros(1, 1, num_hidden).type(torch.cuda.FloatTensor), requires_grad=True), autograd.Variable(torch.zeros(1, 1, num_hidden).type(torch.cuda.FloatTensor), requires_grad=True))
    input_data = copy.deepcopy(test_set[index])
    input_torchvar = autograd.Variable(torch.cuda.FloatTensor(input_data), requires_grad=True)
    output, hidden = lstm(input_torchvar.view(len(input_data), 1, -1), hidden)
    output_2 = output_layer(output)

    grad_matrix = []
    for output_i in range(input_data.shape[1]):
        output_index = (100,0,output_i)
        grad_mask = torch.zeros(output_2.size()).type(torch.cuda.FloatTensor)
        grad_mask[output_index] = 1.0
            
        first_grads = torch.autograd.grad(output_2, input_torchvar, grad_outputs=grad_mask, retain_graph=True, create_graph=True, only_inputs=True, allow_unused=False)

        grad_matrix.append(first_grads[0].data[:101].cpu().numpy())
    grad_matrix = np.array(grad_matrix)

    for timestep in range(100,0,-1):
         determinant = np.linalg.det(grad_matrix[:,timestep,:].astype(np.float64))
         #print timestep, determinant
         if abs(determinant) <= 1e-150:
            timesteps  += 100 - timestep
            break
    #grad_matrix[output_index] = first_grads[0].data
    #print output_2.shape

print timesteps/float(len(test_set))


