import numpy  as np 
import pandas as pd
import torch.utils.data as Data


from data    import *
from networks import *





class Framework():
	def __init__(self, args):
		self.args = args
		self.net  = Snail(N, K, in_channel, batch_size)
		self.loss_fn   = nn.CrossEntropyLoss()
		self.optimizer = torch.optim.Adam(self.net.parameters(), lr=args.lr)
		
		self.train_data = None
		self.test_data  = None
		self.val_data   = None

	def train(self):
		print("------------- Start Training ---------------")
		for epoch in range(self.args.epoch):
			for x, y in self.train_data:
				x , y , target = batch_for_few_shot(x, y, 10)
				
				out  = self.net(x, y)
				loss = self.loss_fn(out, target)
				loss.backward()
				self.optimizer.step()
				self.optimizer.zero_grad() 

	def test(self):
		pass

	def validation(self):
		pass

	def get_data(self, mode = None, date = None):
		#  mode = train / val/ test
		#  data = [start, end].
		path = "./data/" + self.args.dataset + ".csv"
		data = pd.read_csv(path, index_col = "Ntime")
		data.index = pd.to_datetime(data.index, format='%Y%m%d')
		feature = data.loc[date[0]:date[1], :'Federal Fund Rate'].values
		label   = data.loc[date[0]:date[1], 'label'].values
		feature, label = to_window(feature, label, self.args.time_step, self.args.image_type)
		

		if(mode == 'train'):
			torch_dataset = Data.TensorDataset(feature, label)
			data_loader   = Data.DataLoader(
				dataset    = torch_dataset,
				batch_size =  100, 
				shuffle    =  False)
			self.train_data = data_loader

		elif(mode == 'val'):
			torch_dataset = Data.TensorDataset(feature, label)
			data_loader   = Data.DataLoader(
				dataset    = torch_dataset,
				batch_size =  100, 
				shuffle    =  False)
			self.val_data = data_loader

		elif(mode == 'test'):
			torch_dataset = Data.TensorDataset(feature, label)
			data_loader   = Data.DataLoader(
				dataset    = torch_dataset,
				batch_size =  100, 
				shuffle    =  False)
			self.test_data = data_loader

