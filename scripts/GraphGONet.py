# Useful packages

import torch
import torch.nn as nn
import torch_geometric
import networkx as nx
from torchvision import transforms
from base_model import Net
import torch.nn.functional as F
from captum.attr import LayerGradientXActivation

import matplotlib.pyplot as plt
import seaborn
import numpy as np
import pandas as pd
from math import *

from utils import *

import os
import pickle
import time
import sys
import argparse
import json
import warnings
from tqdm import tqdm
import pdb

class LogitNormLoss(nn.Module):

    def __init__(self, device, t=1.0):
        super(LogitNormLoss, self).__init__()
        self.device = device
        self.t = t

    def forward(self, x, target):
        norms = torch.norm(x, p=2, dim=-1, keepdim=True) + 1e-7
        logit_norm = torch.div(x, norms) / self.t
        return F.cross_entropy(logit_norm, target)


def train(args):

	warnings.filterwarnings("ignore")

	# Load the data
	print("Loading the input gene data...")
	start = time.time()

	loaded = np.load(os.path.join(args.dir_files,"genes_annotated.npz"))
	adj_mat_fc1 = loaded["mask"]

	trainset = GeneExpressionDataset(file_name=os.path.join(args.dir_data,"X_train.npz"),n_samples=args.n_samples,mask_features = adj_mat_fc1,transform=transforms.Compose([ToTensor()]),n_classes=args.n_classes, class_weights= args.class_weight)
	if args.n_samples:
		print("#Train samples: {}".format(args.n_samples))
	else:
		print("#Train samples: {}".format(trainset.X.shape[0]))
	validset = GeneExpressionDataset(file_name=os.path.join(args.dir_data,"X_val.npz"),mask_features = adj_mat_fc1,transform=transforms.Compose([ToTensor()]),n_classes=args.n_classes)
	if args.processing=="train_and_evaluate":
		testset = GeneExpressionDataset(file_name=os.path.join(args.dir_data,"X_test.npz"),mask_features = adj_mat_fc1,transform=transforms.Compose([ToTensor()]),n_classes=args.n_classes)
        
	trainloaderGE = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,shuffle=True, num_workers=0)
	validloaderGE = torch.utils.data.DataLoader(validset, batch_size=args.batch_size,shuffle=True, num_workers=0)
	if args.processing=="train_and_evaluate":
		testloaderGE = torch.utils.data.DataLoader(testset, batch_size=args.batch_size,shuffle=False, num_workers=0)

	end = time.time()
	elapsed = end - start
	print("Total time: {}h {}min {}sec".format(time.gmtime(elapsed).tm_hour,
	time.gmtime(elapsed).tm_min,
	time.gmtime(elapsed).tm_sec))

	# Load the useful files to build the architecture
	print("Processing the GO layers...")
	start = time.time()

	connection_matrix = pd.read_csv(os.path.abspath(os.path.join(args.dir_files,"matrix_connection_{}.csv".format(args.type_graph))),index_col=0)
	graph = nx.read_gpickle(os.path.join(args.dir_files,"gobp-{}-converted".format(args.type_graph))) #read the GO graph wich will be converted into the hidden layers of the network
	graph = from_networkx(graph, dim_inital_node_embedding=args.dim_init,label=args.n_classes)

	n_samples = trainset.X.shape[0]
	data_list = [graph.clone() for i in np.arange(n_samples)] #the same network architecture is used accross the patients from the same dataset
	trainloaderGraph = DataLoader(data_list, batch_size=args.batch_size,shuffle=False)

	n_samples = validset.X.shape[0]
	data_list = [graph.clone() for i in np.arange(n_samples)]
	validloaderGraph = DataLoader(data_list, batch_size=args.batch_size,shuffle=False)

	if args.processing=="train_and_evaluate":
		n_samples = testset.X.shape[0]
		data_list = [graph.clone() for i in np.arange(n_samples)]
		testloaderGraph = DataLoader(data_list, batch_size=args.batch_size,shuffle=False)

	end = time.time()
	elapsed=end - start
	print("Total time: {}h {}min {}sec".format(time.gmtime(elapsed).tm_hour,
	time.gmtime(elapsed).tm_min,
	time.gmtime(elapsed).tm_sec))

	# Launch the model
	print("Launching the learning")
	device = torch.device(args.device)
	model = Net(n_genes=args.n_inputs,n_nodes=args.n_nodes,n_nodes_annot=args.n_nodes_annotated,n_nodes_emb=args.dim_init,n_classes=args.n_classes,
               n_prop1=args.n_prop1,adj_mat_fc1=connection_matrix.values,selection=args.selection_op,ratio=args.selection_ratio).to(device)
	print(model)
#	print("(model mem allocation) - Memory available : {:.2e}".format(torch.cuda.memory_reserved(0)-torch.cuda.memory_allocated(0)))

	if args.optimizer=="adam":#specify the optimizer
		optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
	elif args.optimizer=="rmsprop":
		optimizer = torch.optim.RMSprop(model.parameters(), lr=args.lr)
	elif args.optimizer=="momentum":
		optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)

	if args.n_classes<2:
		output_fn = torch.nn.Sigmoid()
		if args.class_weight:
			loss_fn = LogitNormLoss(device, t=1.0).to(device)
		else:
			loss_fn = LogitNormLoss()
	else:
		output_fn = torch.nn.Softmax(dim=1)
		if args.class_weight:
			loss_fn = LogitNormLoss(device, t=1.0).to(device)
		else:
			loss_fn = LogitNormLoss()
        
	if args.es:
		early_stopping = EarlyStopping(patience=args.patience, verbose=True,delta=args.delta)

	acc_valid , loss_valid, acc_train, loss_train = [],[],[],[]

	start = time.time()

	for epoch in tqdm(range(args.n_epochs)):

		# Training
		model.train()
		idx=0
		for batch,batchGraph in zip(trainloaderGE,trainloaderGraph):
			batchGE,labels=batch
			batchGE=batchGE.to(device)
			labels=labels.to(device)
			batchGraph=batchGraph.to(device)
#			if (epoch==0) & (idx==0):
#				print("(data mem allocation) - Memory available : {:.2e}".format(torch.cuda.memory_reserved(0)-torch.cuda.memory_allocated(0)))
			optimizer.zero_grad()
			out = model(transcriptomic_data=batchGE,graph_data=batchGraph)
			labels = labels.type(torch.LongTensor)
			loss = loss_fn(out, labels)
			t_loss = loss.item()
			loss.backward()
			optimizer.step()
#			if (epoch==0) & (idx==0): 
#				print("(loss mem allocation) - Memory available : {:.2e}".format(torch.cuda.memory_reserved(0)-torch.cuda.memory_allocated(0)))
			with torch.no_grad():
				model.fc1.weight.grad.mul_(model.adj_mat_fc1)
			optimizer.step()

			if not(args.display_step):
				if idx==0:
					t_loss = loss.view(1).item()
					t_out = output_fn(out).detach().cpu().numpy()
					if args.n_classes>=2:
						t_out=t_out.argmax(axis=1)
					ground_truth = labels.detach().cpu().numpy()
				else:
					t_loss = np.hstack((t_loss,loss.item()))
					if args.n_classes>=2:                        
						t_out = np.hstack((t_out,output_fn(out).argmax(axis=1).detach().cpu().numpy()))
					else:                        
						t_out = np.hstack((t_out,output_fn(out).detach().cpu().numpy()))         
					ground_truth = np.hstack((ground_truth,labels.detach().cpu().numpy()))

			idx+=1

		if not(args.display_step):
			acc_train.append(get_accuracy(ground_truth,t_out,n_classes=args.n_classes))
			loss_train.append(np.mean(t_loss))

		# Compute loss and accuracy after an epoch on the train and valid set
		model.eval()
		with torch.no_grad():
			idx = 0
			for batch,batchGraph in zip(validloaderGE,validloaderGraph):
				batchGE,labels=batch
				batchGE=batchGE.to(device)
				labels=labels.to(device)
				batchGraph=batchGraph.to(device)
				if idx==0:
					t_out = model(transcriptomic_data=batchGE,graph_data=batchGraph)
					labels = labels.type(torch.LongTensor)
					t_loss = loss_fn(t_out, torch.LongTensor(labels)).view(1).item()
					t_out = output_fn(t_out).detach().cpu().numpy()
					if args.n_classes>=2:
						t_out=t_out.argmax(axis=1)                        
					ground_truth = labels.detach().cpu().numpy()
				else:
					out = model(transcriptomic_data=batchGE,graph_data=batchGraph)
					t_loss = np.hstack((t_loss,loss_fn(out, labels.long()).item())) 
					if args.n_classes>=2:                        
						t_out = np.hstack((t_out,output_fn(out).argmax(axis=1).detach().cpu().numpy()))
					else:                        
						t_out = np.hstack((t_out,output_fn(out).detach().cpu().numpy()))        
					ground_truth = np.hstack((ground_truth,labels.detach().cpu().numpy()))
				idx+=1

			acc_valid.append(get_accuracy(ground_truth,t_out,n_classes=args.n_classes))
			loss_valid.append(np.mean(t_loss))

		if (args.display_step and (((epoch+1) % args.display_step == 0) or (epoch==0))) :
			with torch.no_grad():
				idx = 0
				for batch,batchGraph in zip(trainloaderGE,trainloaderGraph):
					batchGE,labels=batch
					batchGE=batchGE.to(device)
					labels=labels.to(device)
					batchGraph=batchGraph.to(device)
					if idx==0:
						t_out = model(transcriptomic_data=batchGE,graph_data=batchGraph)
						t_loss = loss_fn(t_out, torch.LongTensor(labels)).view(1).item()
						t_out = output_fn(t_out).detach().cpu().numpy()
						if args.n_classes>=2:
							t_out=t_out.argmax(axis=1)
						ground_truth = labels.detach().cpu().numpy()
					else:
						out = model(transcriptomic_data=batchGE,graph_data=batchGraph)
						t_loss = np.hstack((t_loss,loss_fn(out, labels).item()))
						if args.n_classes>=2:                        
							t_out = np.hstack((t_out,output_fn(out).argmax(axis=1).detach().cpu().numpy()))
						else:                        
							t_out = np.hstack((t_out,output_fn(out).detach().cpu().numpy()))     
						ground_truth = np.hstack((ground_truth,labels.detach().cpu().numpy()))
					idx+=1

			acc_train.append(get_accuracy(ground_truth,t_out,n_classes=args.n_classes))
			loss_train.append(np.mean(t_loss))

			print('| Epoch: {}/{} | Train: Loss {:.4f} Accuracy : {:.4f} '\
					'| Test: Loss {:.4f} Accuracy : {:.4f}\n'.format(epoch+1,args.n_epochs,loss_train[epoch],acc_train[epoch],loss_valid[epoch],acc_valid[epoch]))

		if args.es: 
			early_stopping(loss_valid[epoch], model)
			if early_stopping.early_stop:
				print("Early stopping")
				print('| Epoch: {}/{} | Train: Loss {:.4f} Accuracy : {:.4f} '\
					'| Test: Loss {:.4f} Accuracy : {:.4f}\n'.format(epoch+1,args.n_epochs,loss_train[epoch],acc_train[epoch],loss_valid[epoch],acc_valid[epoch]))
				args.n_epochs = epoch + 1
				break

	if args.processing=="train_and_evaluate":
		model.eval()
		with torch.no_grad():
			idx = 0
			for batch,batchGraph in zip(testloaderGE,testloaderGraph):
				batchGE,labels=batch
				batchGE=batchGE.to(device)
				labels=labels.to(device)
				batchGraph=batchGraph.to(device)
				if idx==0:
					t_out = model(transcriptomic_data=batchGE,graph_data=batchGraph)
					t_loss = loss_fn(t_out, labels.long()).view(1).item()
					t_out = output_fn(t_out).detach().cpu().numpy()
					if args.n_classes>=2:
						t_out=t_out.argmax(axis=1)
					ground_truth = labels.detach().cpu().numpy()
				else:
					out = model(transcriptomic_data=batchGE,graph_data=batchGraph)
					labels = labels.type(torch.LongTensor)
					t_loss = np.hstack((t_loss,loss_fn(out, labels).item()))
					if args.n_classes>=2:                        
						t_out = np.hstack((t_out,output_fn(out).argmax(axis=1).detach().cpu().numpy()))
					else:                        
						t_out = np.hstack((t_out,output_fn(out).detach().cpu().numpy()))     
					ground_truth = np.hstack((ground_truth,labels.detach().cpu().numpy()))
				idx+=1
			acc_test = get_accuracy(ground_truth,t_out,n_classes=args.n_classes)
			loss_test = np.mean(t_loss)  
            
		performances = {
				'loss_train':loss_train,'loss_valid':loss_valid,'loss_test':loss_test,
				'acc_train':acc_train,'acc_valid':acc_valid,'acc_test':acc_test   
				}
        
	else:
		performances = {
				'loss_train':loss_train,'loss_valid':loss_valid,
				'acc_train':acc_train,'acc_valid':acc_valid
				}

	if args.save: torch.save(model.state_dict(), os.path.join(args.dir_save,args.checkpoint))

	end = time.time()
	elapsed=end - start
	print("Total time: {}h {}min {}sec ".format(time.gmtime(elapsed).tm_hour,
	time.gmtime(elapsed).tm_min,
	time.gmtime(elapsed).tm_sec))
    
	args.learning_time = elapsed

	return performances

def main():

	# Experiment setting
	parser = argparse.ArgumentParser()

	# -- Configuration of the environnement --
	parser.add_argument('--dir_log', type=str, default="log", help="dir_log")
	parser.add_argument('--dir_files', type=str, default='files', help='repository for all the files needed for the training and the evaluation')
	parser.add_argument('--dir_data', type=str, default='data', help='repository of the dataset')
	parser.add_argument('--file_extension', type=int, default=None, help="option to save different models with the same setting")    
	parser.add_argument('--save', action='store_true', help="Do you need to save the model?")
	parser.add_argument('--restore', action='store_true', help="Do you want to restore a previous model?")
	parser.add_argument('--processing', type=str, default="train_and_evaluate", help="What to do with the model? {train,train_and_evaluate,evaluate,predict}")

	# -- Architecture of the neural network --
	parser.add_argument('--type_graph', type=str, default="truncated", help='type of GO graph considered (truncated,entire)')
	parser.add_argument('--n_samples', type=int, default=None, help="number of samples to use")
	parser.add_argument('--n_inputs', type=int, default=36834, help="number of features")
	parser.add_argument('--n_nodes', type=int, default=10663, help="number of nodes of GO graph")
	parser.add_argument('--n_nodes_annotated', type=int, default=8249, help="number of nodes annotated with the genes")
	parser.add_argument('--dim_init', type=int, default=1, help="initial dimension")
	parser.add_argument('--n_prop1', type=int, default=1, help="dimension after propagation")
	parser.add_argument('--n_classes', type=int, default=1, help="number of classes")

	# -- Learning and Hyperparameters --
	parser.add_argument('--selection_op', type=str, default=None, help='type of selection (random,top)')
	parser.add_argument('--selection_ratio', type=float, default=0.5, help='selection ratio')
	parser.add_argument('--optimizer', type=str, default='adam', help="optimizer {adam, momentum, adagrad, rmsprop}")
	parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
	parser.add_argument('--es', action='store_true', help=' set earlystopping')
	parser.add_argument('--patience', type=int, default=10, help='patience for earlystopping')
	parser.add_argument('--delta', type=float, default=0.001, help='delta for earlystopping')
	parser.add_argument('--batch_size', type=int, default=64, help="the number of examples in a batch")
	parser.add_argument('--n_epochs', type=int, default=50, help='maximum number of epochs')
	parser.add_argument('--display_step', type=int, default=None, help="when to print the performances")
	parser.add_argument('--device', type=str, default='cpu', help="GPU device (cpu,cuda)")
	parser.add_argument('--class_weight', action='store_true', help="balance imbalance data?")

	args = parser.parse_args()
    
	args.checkpoint = "model.pt"
    
	if not(os.path.isdir(args.dir_log)):
		os.mkdir(args.dir_log)
    
	if args.selection_op:
		args.dir_save=os.path.join(args.dir_log,'GraphGONet_SELECTOP={}_SELECTRATIO={}'.format(args.selection_op,args.selection_ratio))
	else:
		args.dir_save=os.path.join(args.dir_log,'GraphGONet_SELECTOP={}'.format(args.selection_op))

	if args.n_samples:
		args.dir_save+="_N_SAMPLES={}".format(args.n_samples)

	if args.file_extension:
		args.dir_save+="_{}".format(args.file_extension)      

	if args.processing=="train" or args.processing=="train_and_evaluate":

		start_full = time.time()
	
		if not(os.path.isdir(args.dir_save)):
			os.mkdir(args.dir_save)

		performances = train(args)

		with open(os.path.join(args.dir_save,'model_args.txt'), 'w') as f:
			json.dump(args.__dict__, f, indent=2)   

		with open(os.path.join(args.dir_save,"histories.txt"), "wb") as fp:
			#Pickling
			pickle.dump(performances, fp)

		end = time.time()
		elapsed =end - start_full
		print("Total time full process: {}h {}min {}sec".format(time.gmtime(elapsed).tm_hour,
		time.gmtime(elapsed).tm_min,
		time.gmtime(elapsed).tm_sec))
        
	# elif FLAGS.processing=="evaluate":
	    
	#     evaluate(dir_save=dir_save)
	    
	# elif FLAGS.processing=="predict":
	    
	#     np.savez_compressed(os.path.join(dir_save,'y_test_hat'),y_hat=predict(dir_save=dir_save))

if __name__ == "__main__":
	main()