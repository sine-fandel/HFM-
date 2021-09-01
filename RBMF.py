import torch
import torch.nn as nn
import numpy as np
import torch.fft as fft

class RestrictedBoltzmannMachineModel():
	"""
	Implementation of RBM model
	"""

	def __init__ (self, nv, nh):
		self.W = torch.randn(nv, nh)
		self.a = torch.randn(nv)
		self.b = torch.randn(nh)
		self.W = self.W.double ()
		self.a = self.a.double ()
		self.b = self.b.double ()

		# self.W = self.W.cuda ()
        # self.a = self.a.cuda ()
        # self.b = self.b.cuda ()

	def sample_h (self, x):
		activation = torch.matmul(x, self.W) + self.b
		p_h_given_v = torch.sigmoid(activation)

		return p_h_given_v, p_h_given_v

	def sample_v (self, y):
		activation = torch.matmul(y, self.W.t()) + self.a
		p_v_given_h = torch.sigmoid(activation)

		return p_v_given_h, p_v_given_h

	def train (self, v0, vk, ph0, phk, lr=0.01):
		self.W += lr * torch.matmul(v0.t(), ph0) - torch.matmul(vk.t(), phk)
		self.a += lr * torch.sum((v0 - vk), 0)
		self.b += lr * torch.sum((ph0 - phk), 0)


class FeaturesLinear (torch.nn.Module) :

	def __init__ (self, field_dims, output_dim) :
		super ().__init__ ()
		self.fc = nn.Embedding (sum (field_dims), output_dim)
		self.bias = nn.Parameter (torch.zeros (output_dim, ))
		self.offsets = np.array ((0, *np.cumsum (field_dims)[:-1]), dtype=np.long)
	
	def forward (self, x) :
		x = x + x.new_tensor (self.offsets).unsqueeze (0)

		return torch.sum (self.fc (x), dim=1) + self.bias

class FeaturesEmbedding(torch.nn.Module):

	def __init__(self, field_dims, embed_dim):
		super().__init__()
		self.embedding = torch.nn.Embedding(sum(field_dims), embed_dim)
		self.offsets = np.array((0, *np.cumsum(field_dims)[:-1]), dtype=np.long)
		torch.nn.init.xavier_uniform_(self.embedding.weight.data)

	def forward(self, x):
		"""
		:param x: Long tensor of size ``(batch_size, num_fields)``
		"""
		x = x + x.new_tensor(self.offsets).unsqueeze(0)
		return self.embedding(x)

class CcovNetwork (torch.nn.Module) :

	def ccov (self, x1, x2) :
		"""
		CCOV operation
		"""

		return fft.ifft (fft.fft (x1) * fft.fft (x2))

	def forward (self, x) :
		num_fields = x.shape[1]
		row, col = list (), list ()
		for i in range (num_fields - 1) :
			for j in range (i + 1, num_fields) :
				row.append (i)
				col.append (j)

		return np.real (torch.sum (self.ccov (x[ : , row], x[ :, col]), dim=2))

class HFMFodel (nn.Module) :
	""" A combination model of HFM and MF
	"""

	def __init__ (self, user_num, item_num, factor_num, field_dims, embed_dim=8, dropout=0, num_layers=3) :
		super ().__init__ ()

		'''
		Define the GMF model
		'''
		self.embed_user_layer = nn.Embedding (user_num, factor_num)
		self.embed_item_layer = nn.Embedding (item_num, factor_num)
		self.GMF_predict_layer = nn.Linear (factor_num, 1)
		self._init_weight_ ()

		'''
		Define the HFM model
		'''
		num_fields = len (field_dims)
		self.embedding = FeaturesEmbedding (field_dims, factor_num)
		self.linear = FeaturesLinear (field_dims, 1)
		self.ccov = CcovNetwork ()

		'''
		Define the HFMF model
		'''
		self.embedding_layer = nn.Embedding ((item_num + user_num) * 2, embed_dim)

		MLP_modules = []
		for i in range (num_layers) :
			MLP_modules.append (nn.Dropout (p=dropout))
			MLP_modules.append (nn.Linear (2, 2))
			MLP_modules.append (nn.ReLU ())

		self.MLP_layers = nn.Sequential (*MLP_modules)

		self.HFMF_predict_layer = torch.nn.Linear (2, 1)

	def _init_weight_ (self) :
		nn.init.normal_ (self.embed_item_layer.weight, std=0.01)
		nn.init.normal_ (self.embed_user_layer.weight, std=0.01)

	def forward (self, x) :
		'''
		feed data to HFM
		'''
		x1 = self.linear (x) + self.ccov (self.embedding (x))

		'''
		feed data to GMF
		'''
		embed_user = self.embed_user_layer (x[ : , 0])
		embed_item = self.embed_item_layer (x[ : , 1])

		x2 = self.GMF_predict_layer (embed_item * embed_user)

		'''
		Combine x1 and x2
		'''
		x = torch.cat ((x1, x2), dim=1)
		# zero = torch.zeros_like (x, dtype=int)
		# one = torch.ones_like (x, dtype=int)
		# x = torch.where (x > 0.5, one, zero)
		# x = self.embedding_layer (x)
		
		# x = self.MLP_layers (x)

		return torch.sigmoid (self.HFMF_predict_layer (x).squeeze (dim=1))

 