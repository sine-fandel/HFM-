import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error, f1_score, accuracy_score, precision_score, recall_score, roc_auc_score
from RBMF.RBMF import HFMFodel, RestrictedBoltzmannMachineModel



class MovieLens20MDataset(torch.utils.data.Dataset):
	"""
	MovieLens 20M Dataset

	Data preparation
		treat samples with a rating less than 3 as negative samples

	:param dataset_path: MovieLens dataset path

	Reference:
		https://grouplens.org/datasets/movielens
	"""

	def __init__(self, dataset_path, sep=',', engine='c', header='infer'):
		data = pd.read_csv (dataset_path, sep=sep, engine=engine, header=header).to_numpy()[:, :3]
		self.items = data[:, :2].astype(np.int) - 1  # -1 because ID begins from 1
		self.targets = self.__preprocess_target(data[:, 2]).astype(np.float32)
		self.field_dims = np.max (self.items, axis=0) + 1
		self.user_field_idx = np.array((0, ), dtype=np.long)
		self.item_field_idx = np.array((1,), dtype=np.long)

	def __len__(self):
		return self.targets.shape[0]

	def __getitem__(self, index):
		return self.items[index], self.targets[index]

	def __preprocess_target(self, target):
		target[target <= 3] = 0
		target[target > 3] = 1
		return target


class MovieLens1MDataset(MovieLens20MDataset):
	"""
	MovieLens 1M Dataset

	Data preparation
		treat samples with a rating less than 3 as negative samples

	:param dataset_path: MovieLens dataset path

	Reference:
		https://grouplens.org/datasets/movielens
	"""

	def __init__(self, dataset_path):
		super().__init__(dataset_path, sep='::', engine='python', header=None)

class MovieLens100kDataset (MovieLens20MDataset) :
	"""
	Movielen 100k Dataset
	"""

	def __init__ (self, dataset_path) :
		super ().__init__ (dataset_path, sep='\t', engine="python", header=None)


def train_rbm (rbm, training_set, train_target_result, batch_size, epoch, reconerr, nb_users, learning_rate, metric) :
	train_recon_error = 0  # RMSE reconstruction error initialized to 0 at the beginning of training
	s = 0.  # a counter (float type)
	result_list = []

	for id_user in range(0, nb_users - batch_size, batch_size):

		# At the beginning, v0 = vk. Then we update vk
		vk = training_set[id_user : id_user + batch_size]
		v0 = training_set[id_user : id_user + batch_size]

		# vk = vk.cuda ()
		# v0 = v0.cuda ()
		
		ph0, _ = rbm.sample_h (v0)
		# ph0 = ph0.cuda ()
		for k in range(10):
			_, hk = rbm.sample_h(vk)
			_, vk = rbm.sample_v(hk)

			# hk = hk.cuda ()
			# vk = vk.cuda ()

			vk[v0 < 0] = v0[v0 < 0]

		phk, _ = rbm.sample_h(vk)
		# phk = phk.cuda ()

		# Calculate the loss using contrastive divergence
		rbm.train (v0, vk, ph0, phk, learning_rate)

		# Compare vk updated after the training to v0 (the target)
		# vk = vk.cpu ()
		# v0 = v0.cpu ()
		train_recon_error += torch.sqrt (torch.mean ((v0[v0 >= 0] - vk[v0 >= 0]) ** 2))
		s += 1.
		result_list.append (vk)

	# Update RMSE reconstruction error
	reconerr.append (train_recon_error / s)

	train_score = 0
	if metric == 0 :
		print (training_set)
		print (result_list)
		train_score = roc_auc_score (training_set, result_list)
		# train_score = roc_auc_score (target_list, result_list)

	elif metric == 1 :
		train_score = accuracy_score (training_set, result_list)
		# train_score = accuracy_score (target_list, result_list)

	elif metric == 2 :
		train_score = precision_score (training_set, result_list)
		# train_score = precision_score (target_list, result_list)

	elif metric == 3 :
		train_score = recall_score (training_set, result_list)
		# train_score = recall_score (target_list, result_list)

	print ('Epoch: ' + str (epoch) + '- RMSE Reconstruction Error: ' + str (train_recon_error.data.numpy () / s))
	return train_recon_error.data.numpy () / s, train_score

def test_rbm (rbm, test_set, test_target_result, training_set, nb_users, metric=0) :
	test_loss = 0
	s = 0.
	testing_result = []
	target = test_set[test_set >= 0].tolist ()
	# test_target_list = torch.tensor (test_target_list, dtype=torch.long)
	# print (test_target_list)
	for id_user in range (nb_users) :
		v = test_set[id_user : id_user + 1]
		vt = test_set[id_user : id_user + 1]
		# print (vt)

		# v = v.cuda ()
		# vt = vt.cuda ()
		# test_target_result = torch.from_numpy (test_target_result).view (1, -1).float ()
		# tar = test_target_result[test_target_result >= 0].tolist ()
		if len (vt[vt >= 0]) > 0:
			_, h = rbm.sample_h (v)
			_, v = rbm.sample_v (h)
			
			v[v > 0.5] = 1.
			v[v <= 0.5] = 0.
			
			v = v[vt >= 0].tolist ()
			
			
			testing_result += v
			# target += res

			s += 1.

	if metric == 0 :
		# print (target)
		result = roc_auc_score (target, testing_result)
		# result = roc_auc_score (test_target_list, testing_result)
		print('roc_auc_score: '+ str (result))

	elif metric == 1 :
		# print (test_set.shape)
		# print (testing_result.shape)
		result = accuracy_score (target, testing_result)
		# result = accuracy_score (test_target_list, testing_result)

		print('accuracy_score: '+ str (result))

	elif metric == 2 :
		result = precision_score (target, testing_result)
		# result = accuracy_score (test_target_list, testing_result)

		print ("precision_score: " + str (result))

	elif metric == 3 :
		result = recall_score (target, testing_result)
		# result = recall_score (test_target_list, testing_result)

		print ("recall_score: " + str (result))

	return result

def train (model, optimizer, data_loader, criterion, device, batch_size, field_dims, log_interval=100) :
	model.train ()
	total_loss = 0
	training_result = np.full (field_dims, -1.)
	target_result = np.full (field_dims, -1)
	target_list = []
	for i, (fields, target) in enumerate (data_loader) :
		# fields, target = fields.to (device), target.to (device)
		y = model (fields)
		for j in range (fields.shape[0]) :
			training_result[fields[j, 0]][fields[j, 1]] = y[j]
			target_result[fields[j, 0], fields[j, 1]] = target[j]

		# target_list.append (target.float ())
		loss = criterion (y, target.float ())
		model.zero_grad ()
		loss.backward ()
		optimizer.step ()
		total_loss += loss.item ()
		
	return total_loss, training_result, target_result

def test (model, data_loader,  field_dims, device) :
	model.eval ()
	targets, predicts = list(), list()
	test_result = np.full (field_dims, -1.)
	target_result = np.full (field_dims, -1)
	for i, (fields, target) in enumerate (data_loader) :
		# fields, target = fields.to (device), target.to (device)
		y = model (fields)
		for j in range (fields.shape[0]) :
			test_result[fields[j, 0]][fields[j, 1]] = y[j]
			target_result[fields[j, 0], fields[j, 1]] = target[j]
		targets.extend (target.tolist ())
		predicts.extend (y.tolist ())
		# print (targets)
		# test_target_list.append (target)
	
	# print (test_result)

	return mean_squared_error (targets, predicts), test_result, target_result


def show_hfmf (dataset, num_hidden=100, latent_dim=8, batch_size=1024, learning_rate=0.003, epoch=50, ds=0, metric=0, dataname=None, metrics=None) :
	batch_size = batch_size
	weight_decay = 1e-6
	learning_rate  = learning_rate
	epoch = epoch
	print ("____________HFMF____________")

	train_length = int (len (dataset) * 0.8)
	test_length = len (dataset) - train_length
	train_dataset, test_data = random_split (dataset, (train_length, test_length))

	train_data_loader = DataLoader (train_dataset, batch_size=batch_size)
	test_data_loader = DataLoader (test_data, batch_size=batch_size)

	field_dims = dataset.field_dims
	print (field_dims)
	hfmf_model = HFMFodel (field_dims[0], field_dims[1], latent_dim, field_dims)
	criterion = nn.BCELoss ()
	optimizer = torch.optim.Adam (params=hfmf_model.parameters (), lr=learning_rate, weight_decay=weight_decay)

	n_vis = field_dims[1]
	n_hid = num_hidden

	rbm = RestrictedBoltzmannMachineModel (n_vis, n_hid)
	nb_users = field_dims[0]
	reconerr = []
	testing_loss = []

	train_score = []
	test_score = []
	epoch_list = []

	
	for i in range (epoch) :
		tarining_loss, hfmf_result, train_target_result = train (hfmf_model, optimizer, train_data_loader, criterion, batch_size=batch_size, field_dims=field_dims, device=None)

		hfmf_result = torch.from_numpy (hfmf_result)

		# if i == epoch - 1 :
		# 	for j in range (50) :
		temp = hfmf_result
		zero = torch.zeros_like (hfmf_result)
		one = torch.ones_like (hfmf_result)
		hfmf_result = torch.where (hfmf_result > 0.5, one, zero)
		hfmf_result[temp == -1] = -1

		train_hfmf_result, temp_result = train_rbm (rbm, hfmf_result, train_target_result, 100, i, reconerr, nb_users, learning_rate * 2, metric=metrics)

		_, test_hfmf_result, test_target_result = test (hfmf_model, test_data_loader, field_dims, device=None)
		test_hfmf_result = torch.from_numpy (test_hfmf_result)
		temp = test_hfmf_result
		zero = torch.zeros_like (test_hfmf_result)
		one = torch.ones_like (test_hfmf_result)
		test_hfmf_result = torch.where (test_hfmf_result > 0.5, one, zero)
		test_hfmf_result[temp == -1] = -1

		test_loss = test_rbm (rbm, test_hfmf_result, test_target_result, hfmf_result, nb_users, metric=metric)
		
		testing_loss.append (test_loss)

		train_score.append (temp_result)
		epoch_list.append (i)

	# x = range (0, epoch)

	# plt.figure (1)
	# plt.plot (x, train_score, label="train")
	# plt.plot (x, testing_loss, label="test")
	# # plt.legend()
	# plt.xlabel ('epoch')
	# plt.ylabel (metrics)
	# plt.title (metrics + " vs epoch (dataset : " + dataname + ")")
	# plt.legend(loc='best')
	# plt.savefig ('./learning_curve/result_' + dataname + '_' + metrics + '_lr:' + str (learning_rate) + '.jpg')

	return testing_loss, train_score



# show_hfmf ()

# print (show_hfmf ())

	# auc = test (mf_model, test_data_loader, device=None)
	# print ('epoch : ', i, '  BCE loss : ', tarining_loss, '  testing_mse : ', auc)