import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)
import os,csv,math, sys,joblib
import matplotlib.pyplot as plt
import numpy as np
import sklearn.model_selection, sklearn.preprocessing, copy
from scipy.special import factorial
from sklearn.preprocessing import MinMaxScaler

lstm = torch.load("lstm_trained_model").cpu()
output_layer = torch.load("output_layer_trained_model").cpu()

num_features = 1
num_hidden = 1
output_size = 1

#causal_strength
mean_vector = [0.0]*15
expected_values = []
baseline_expectation_do_x = []

cov_data = np.zeros((15,15)) + np.eye(15)*0.2

baseline_expectation_do_x  =[0.5076135909386664, 0.2711588819908657, 5.629422495855416, 0.022658583459495278, 0.011492128996702376, 0.011273759561358017, 0.011215446889473235, 0.01119493013212923, 0.011193315653370518, 0.011205433818904565, 0.011231432442822552, 0.011275079370316233, 0.011344116069245506, 0.011452266314474399, 0.011623865903464321]

for useless in range(5):
	coin_flip = np.random.binomial(1,0.5,1)
	T = np.random.randint(10,15)
	input_data = np.random.normal(0.0,0.2,T)
	if coin_flip == 1:
		    input_data[:3] = 1.0 + np.random.normal(0.0,0.2,3)
		    y_label = 1.0
	else:
		    input_data[:3] = -1.0 + np.random.normal(0.0,0.2,3)
		    y_label = 0.0


	print input_data[:3]
	#integrated gradients
	hidden = autograd.Variable(torch.zeros(1, 1, num_hidden).type(torch.FloatTensor), requires_grad=True)
	input_torchvar = autograd.Variable(torch.FloatTensor(input_data), requires_grad=True)
	output, hidden = lstm(input_torchvar.view(len(input_data), 1, -1), hidden)
	output_2 = F.sigmoid(output_layer(output[-1]))
	temp1 = output_2.data.view(1).cpu().numpy()[0]

	hidden = autograd.Variable(torch.zeros(1, 1, num_hidden).type(torch.FloatTensor), requires_grad=True)
	input_torchvar = autograd.Variable(torch.FloatTensor(np.zeros(input_data.shape[0])), requires_grad=True)
	output, hidden = lstm(input_torchvar.view(len(input_data), 1, -1), hidden)
	output_2 = F.sigmoid(output_layer(output[-1]))
	temp2 = output_2.data.view(1).cpu().numpy()[0]
	
	integrated_gradients = np.zeros(input_data.shape[0])
	path_vector = np.zeros(input_data.shape[0])
	final_vector = input_data	
	for calls_to_gradients in range(5000):
		hidden = autograd.Variable(torch.zeros(1, 1, num_hidden).type(torch.FloatTensor), requires_grad=True)
		input_torchvar = autograd.Variable(torch.FloatTensor(path_vector), requires_grad=True)
		output, hidden = lstm(input_torchvar.view(len(path_vector), 1, -1), hidden)
		output_2 = F.sigmoid(output_layer(output[-1]))

		first_grads = torch.autograd.grad(output_2, input_torchvar, retain_graph=True, create_graph=True, only_inputs=True, allow_unused=False)
		integrated_gradients += first_grads[0].data.cpu().numpy()
		
		path_vector += (1.0/5000)*(final_vector - np.zeros(input_data.shape[0]))
	myMatrix = (integrated_gradients/5000.0)*input_data.reshape(1,-1)
	print myMatrix
	#myMatrix =np.ma.masked_where((np.abs(myMatrix) < 1e-4), myMatrix)
	myMatrix = np.ma.masked_where(myMatrix < 0.0, myMatrix)
	plt.imshow(myMatrix, cmap='coolwarm', interpolation='nearest')
	plt.colorbar()
	plt.savefig("Integrated_gradients_" + str(useless) + ".png", dpi=1000)

	plt.clf()
   	
	#causal analysis
	input_data = np.array(input_data)
	average_causal_effects = []
	for t in range(len(input_data)):
		expected_value = 0.0
		expectation_do_x = []
		inp = copy.deepcopy(mean_vector)
	
		inp[t] = input_data[t]
		hidden = autograd.Variable(torch.zeros(1, 1, num_hidden).type(torch.FloatTensor), requires_grad=True)
		input_torchvar = autograd.Variable(torch.FloatTensor(inp), requires_grad=True)
		output, hidden = lstm(input_torchvar.view(len(inp), 1, -1), hidden)
		output_2 = F.sigmoid(output_layer(output[-1]))

		val = output_2.data.view(1).cpu().numpy()[0]
		
		first_grads = torch.autograd.grad(output_2, input_torchvar, retain_graph=True, create_graph=True, only_inputs=True, allow_unused=False) #as only one output

		#print causal_effect
		#calculating the hessian (selective-terms-only)
		first_grad_shape = first_grads[0].data.size()
		lower_order_grads = first_grads
                    
		for dimension in range(len(mean_vector)):
		    if dimension == t:
			continue
		    grad_mask = torch.zeros(first_grad_shape)
		    grad_mask[dimension] = 1.0
		    
		    higher_order_grads = torch.autograd.grad(lower_order_grads, input_torchvar, grad_outputs=grad_mask, retain_graph=True, create_graph=True, only_inputs=True, allow_unused=False) 
		    higher_order_grads_array = np.array(higher_order_grads[0].data)
	
		    
		    temp_cov = copy.deepcopy(cov_data)
		    temp_cov[dimension][t] = 0.0
		    val += 0.5*np.sum(higher_order_grads_array*temp_cov[dimension])


		average_causal_effects.append(val)

	
	average_causal_effects = np.array(average_causal_effects) - np.array(baseline_expectation_do_x)[:len(average_causal_effects)]
	
	myMatrix = average_causal_effects.reshape(1,-1)
	print myMatrix
	myMatrix = np.ma.masked_where(myMatrix < 0.0, myMatrix)
	plt.imshow(myMatrix, cmap='coolwarm', interpolation='nearest')
	plt.colorbar()
	plt.savefig("Causal_analysis_" + str(useless) + ".png", dpi=1000)

	plt.clf()

	#gradient sensitivity
	hidden = autograd.Variable(torch.zeros(1, 1, num_hidden).type(torch.FloatTensor), requires_grad=True)
	input_torchvar = autograd.Variable(torch.FloatTensor(inp), requires_grad=True)
	output, hidden = lstm(input_torchvar.view(len(inp), 1, -1), hidden)
	output_2 = F.sigmoid(output_layer(output[-1]))
	
	first_grads = torch.autograd.grad(output_2, input_torchvar, retain_graph=True, create_graph=True, only_inputs=True, allow_unused=False)

	myMatrix = first_grads[0].data.cpu().numpy().reshape(1,-1)
	print myMatrix
	myMatrix = np.ma.masked_where(myMatrix < 0.0, myMatrix)
	#myMatrix =np.ma.masked_where((np.abs(myMatrix) < 1e-4), myMatrix)
	plt.imshow(myMatrix, cmap='coolwarm', interpolation='nearest')
	plt.colorbar()
	plt.savefig("Gradient_sensitivity_" + str(useless) + ".png", dpi=1000)

	plt.clf()


	print
	

