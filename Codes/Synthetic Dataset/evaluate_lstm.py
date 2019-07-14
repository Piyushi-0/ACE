'''Code for comparing change in loss and change in prediction on intervening'''
import joblib
import numpy as np
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import sys
torch.manual_seed(1)
np.random.seed(1)

num_features = 1
num_hidden = 1
output_size = 1

#defining the model
lstm = torch.load("lstm_trained_model")
output_layer = torch.load("output_layer_trained_model")

loss_function = nn.MSELoss()

loss_val = 0
loss_val0=0
loss_val1=0
loss_val2=0

counter = 1.0

pred_changed0=0
pred_changed1=0
pred_changed2=0

while(1):
        coin_flip = np.random.binomial(1,0.5,1)

        T = np.random.randint(10,15)
        input_data = np.random.normal(0.0,0.2,T)
        #Without imputing any dimension
        if coin_flip == 1:
                    input_data[1] = 1.0 + np.random.normal(0.0,0.2,1)
                    input_data[2] = 1.0 + np.random.normal(0.0,0.2,1)
                    input_data[0] = 1.0 + np.random.normal(0.0,0.2,1)
                    
                    y_label = 1.0
        else:
                    input_data[1] = -1.0 + np.random.normal(0.0,0.2,1)
                    input_data[2] = -1.0 + np.random.normal(0.0,0.2,1)
                    input_data[0] = -1.0 + np.random.normal(0.0,0.2,1)
                    
                    y_label = 0.0

	hidden = autograd.Variable(torch.zeros(1, 1, num_hidden).type(torch.cuda.FloatTensor), requires_grad=True)

	input_torchvar = autograd.Variable(torch.cuda.FloatTensor(input_data), requires_grad=True)
	output, hidden = lstm(input_torchvar.view(len(input_data), 1, -1), hidden)
	output_2 = F.sigmoid(output_layer(output[-1]))

	pred = output_2.item()#Prediction without imputing
	
	#Imputing 0
        if coin_flip == 1:
                    input_data[0] = np.random.normal(0.0,0.2,1)
                    input_data[1] = 1.0+np.random.normal(0.0,0.2,1)
                    input_data[2] = 1.0+np.random.normal(0.0,0.2,1)                                        
                                                       
                    y_label = 1.0
        else:
                    input_data[0] = np.random.normal(0.0,0.2,1)
                    input_data[1] = -1.0 + np.random.normal(0.0,0.2,1)
                    input_data[2] = -1.0+np.random.normal(0.0,0.2,1)

                    y_label = 0.0

	hidden = autograd.Variable(torch.zeros(1, 1, num_hidden).type(torch.cuda.FloatTensor), requires_grad=True)

	input_torchvar = autograd.Variable(torch.cuda.FloatTensor(input_data), requires_grad=True)
	output, hidden = lstm(input_torchvar.view(len(input_data), 1, -1), hidden)
	output_2 = F.sigmoid(output_layer(output[-1]))

	pred_0= output_2.item()#Prediction on imputing 0		
	
	if (pred_0-0.5)*(pred-0.5)<0:#If prediction changed
		pred_changed0+=1
	#Imputing 1
        if coin_flip == 1:
                    input_data[0] = 1.0+np.random.normal(0.0,0.2,1)
                    input_data[1] = np.random.normal(0.0,0.2,1)
                    input_data[2] = 1.0+np.random.normal(0.0,0.2,1)                                        
                                                       
                    y_label = 1.0
        else:
                    input_data[0] = -1.0+np.random.normal(0.0,0.2,1)
                    input_data[1] = np.random.normal(0.0,0.2,1)
                    input_data[2] = -1.0+np.random.normal(0.0,0.2,1)

                    y_label = 0.0

	hidden = autograd.Variable(torch.zeros(1, 1, num_hidden).type(torch.cuda.FloatTensor), requires_grad=True)

	input_torchvar = autograd.Variable(torch.cuda.FloatTensor(input_data), requires_grad=True)
	output, hidden = lstm(input_torchvar.view(len(input_data), 1, -1), hidden)
	output_2 = F.sigmoid(output_layer(output[-1]))

	pred_1= output_2.item()#Prediction on imputing 0		
	
	if (pred_1-0.5)*(pred-0.5)<0:#If prediction changed
		pred_changed1+=1
	#Imputing 2
        if coin_flip == 1:
                    input_data[0] = 1.0+np.random.normal(0.0,0.2,1)
                    input_data[1] = 1.0+np.random.normal(0.0,0.2,1)
                    input_data[2] = np.random.normal(0.0,0.2,1)                                        
                                                       
                    y_label = 1.0
        else:
                    input_data[0] = -1.0+np.random.normal(0.0,0.2,1)
                    input_data[1] = -1.0 + np.random.normal(0.0,0.2,1)
                    input_data[2] = np.random.normal(0.0,0.2,1)

                    y_label = 0.0

	hidden = autograd.Variable(torch.zeros(1, 1, num_hidden).type(torch.cuda.FloatTensor), requires_grad=True)

	input_torchvar = autograd.Variable(torch.cuda.FloatTensor(input_data), requires_grad=True)
	output, hidden = lstm(input_torchvar.view(len(input_data), 1, -1), hidden)
	
	output_2 = F.sigmoid(output_layer(output[-1]))

	pred_2= output_2.item()#Prediction on imputing 0		
	
	if (pred_2-0.5)*(pred-0.5)<0:#If prediction changed
		pred_changed2+=1
	
	loss_val += abs(y_label - pred)
	loss_val0+=abs(y_label-pred_0)	
	loss_val1+=abs(y_label-pred_1)
	loss_val2+=abs(y_label-pred_2)		

	if counter == 10000:
		print('Avg Loss without imputing')
		print (loss_val/counter*1.0)
		print('Avg loss on imputing 0:')
		print(loss_val0/counter*1.0)
		print('Avg loss on imputing 1:')
		print(loss_val1/counter*1.0)
		print('Avg loss on imputing 2:')
		print(loss_val2/counter*1.0)		
				
		print('Avg change in prediction on imputing 0:')			
		print (pred_changed0/counter*1.0)
		print('Avg change in prediction on imputing 1:')		
		print (pred_changed1/counter*1.0)
		print('Avg change in prediction on imputing 2:')		
		print (pred_changed2/counter*1.0)				
		sys.exit()
	counter += 1.0

    
    	

