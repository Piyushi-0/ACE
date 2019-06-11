
from __future__ import print_function
import numpy as np
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import sys,os, csv, math
import sklearn.model_selection, sklearn.preprocessing 
import operator
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt

torch.manual_seed(1)

def mse_step(a , b):
        diff = a - b
        d = diff.cpu().data.numpy()
        d = np.absolute(d)**2
        n = d.shape[2]
        d = d.sum(axis = 2)/float(n)
        
        return d

def mse_feature(a, b):
	diff = a - b
	d = diff.cpu().data.numpy()
	d = np.absolute(d)**2
	d = d.sum(axis = 0)
	total_error = d.sum(axis = 1)
	d = (d/total_error)*100
	
	return d

def get_train_test_set(foldername, num_features):
    series = []
    sizes = []
    feature_mins = [np.inf]*num_features
    feature_maxs = [-np.inf]*num_features
    for filename in os.listdir("./" + foldername + "/"):
         try:
                        f = open("./" + foldername + "/" + filename)
         except:
                        continue
   	 reader = csv.reader(f)
   	 lines = []
   	 for line in reader:
   	     lines.append(line)
   	 temp_arr = np.array(lines[1:min(3001, len(lines))]).astype(float)
    
   	 series.append(temp_arr)
     	 sizes.append(temp_arr.shape[0])
     
    feature_maxs = np.load("feature_maxs.npy")
    feature_mins = np.load("feature_mins.npy")
    
    for i in range(len(series)):
        series[i] = (series[i] - feature_mins)/np.where((feature_maxs - feature_mins) != 0.0, (feature_maxs - feature_mins), 1.0)
    series = np.array(series) 
    return series


headers_file = open("headers.csv","r").read()
features = headers_file.split(",")

num_features = 17
num_hidden = 15
output_size = num_features

# sys.argv[1] = foldername
test_set = get_train_test_set(sys.argv[1],num_features)
print(test_set.shape)

lstm = nn.LSTM(num_features, num_hidden).cuda()
output_layer = nn.Linear(num_hidden, output_size).cuda()
loss_function = nn.MSELoss()
#optimizer = optim.Adam([{'params': lstm.parameters()}, {'params': output_layer.parameters()}], lr = 0.0003)

lstm = torch.load("lstm_trained_model")
output_layer = torch.load("output_layer_trained_model")

loss_function = nn.MSELoss()
loss_val = 0
mse = []
for input_data in test_set:
    lstm.zero_grad()
    output_layer.zero_grad()
    hidden = (autograd.Variable(torch.zeros(1, 1, num_hidden).type(torch.cuda.FloatTensor), requires_grad= True), autograd.Variable(torch.zeros(1, 1, num_hidden).type(torch.cuda.FloatTensor), requires_grad= True))
    input_torchvar = autograd.Variable(torch.cuda.FloatTensor(input_data[:-1]), requires_grad= True)
    output, hidden = lstm(input_torchvar.view(len(input_data)-1, 1, -1), hidden)
    output_2 = output_layer(output)
    loss = loss_function(output_2, autograd.Variable(torch.cuda.FloatTensor(input_data[1:]).view(len(input_data)-1, 1, -1)))
    loss_val += loss
    
    mse_per_feature = mse_feature(output_2, autograd.Variable(torch.cuda.FloatTensor(input_data[1:])).view(len(input_data) - 1, 1, -1))
    #mse.extend(mse_per_step)
    break

mse = mse_per_feature[0]
features = [x for _,x in sorted(zip(mse, features), reverse = True)]
mse = [y for y,_ in sorted(zip(mse, features), reverse = True)]
for i in range(len(features)):
    print(features[i],":", mse[i])



