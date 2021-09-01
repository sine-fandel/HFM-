import torch
import torch.nn as nn

# Creating the architecture of the Neural Network
class RestrictedBoltzmannMachineModel():
	"""
	Implementation of RBM model
	"""

	def __init__ (self, nv, nh):
		self.W = torch.randn(nv, nh)
		self.a = torch.randn(nv)
		self.b = torch.randn(nh)

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


class GeneralMatrixFactorizationModel (torch.nn.Module) :
	"""
	Implementation of GMF model
	"""

	def __init__ (self, user_num, item_num, factor_num) :
		super ().__init__ ()
		self.embed_user_layer = torch.nn.Embedding (user_num, factor_num)
		self.embed_item_layer = torch.nn.Embedding (item_num, factor_num)
		self.predict_layer = torch.nn.Linear (factor_num, 1)
		self._init_weight_ ()

	def _init_weight_(self):
		torch.nn.init.normal_ (self.embed_user_layer.weight, std=0.01)
		torch.nn.init.normal_ (self.embed_item_layer.weight, std=0.01)

	def forward (self, user, item) :
		embed_user = self.embed_user_layer (user)
		embed_item = self.embed_item_layer (item)
		
		prediction = self.predict_layer (embed_user * embed_item)
		# print (torch.sigmoid (prediction.view (-1)))
		# return torch.sigmoid (prediction.view (-1))
		return torch.sigmoid (prediction.view (-1))

class RBMF (torch.nn.Module) :
	"""
	Combine the RBM with GMF
	"""
	def __init__ (self, user_num, item_num, embed_dim=32, dropout=0.2, num_layers=3) :
		super ().__init__ ()
		self.embeding_layer1 = torch.nn.Embedding (item_num + user_num, embed_dim)
		self.embeding_layer2 = torch.nn.Embedding (item_num + user_num, embed_dim)

		MLP_modules = []
		for i in range (num_layers) :
			MLP_modules.append (nn.Dropout (p=dropout))
			MLP_modules.append (nn.Linear (embed_dim * 2, embed_dim * 2))
			MLP_modules.append (nn.ReLU ())

		self.MLP_layers = nn.Sequential (*MLP_modules)

		self.predict_layer = torch.nn.Linear (embed_dim * 2, 1)

	def forward (self, x1, x2) :

		x1 = self.embeding_layer1 (x1)
		x2 = self.embeding_layer2 (x2)
		# print (self.predict_layer (torch.cat ((x1, x2), dim=2)).squeeze(dim=2).shape)
		# x1 = x1.to (torch.float32)
		# x2 = x2.to (torch.float32)
		x = torch.cat ((x1, x2), dim=2)

		x = self.MLP_layers (x)

		return torch.sigmoid (self.predict_layer (x).squeeze (dim=2))

