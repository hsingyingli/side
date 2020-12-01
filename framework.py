import numpy  as np 
import pandas as pd
import torch.utils.data as Data


from data    import *
from networks import *





class Framework():
	def __init__(self, args):
		self.args = args
		#,N, K, in_channels, out_channels, , key_size, value_size
		self.net  = Snail(N = args.N, K = args.K, in_channels = args.in_channels,\
							 out_channels = args.out_channels,  key_size= args.key_size, value_size = args.value_size)
		self.loss_fn   = nn.CrossEntropyLoss()
		self.optimizer = torch.optim.Adam(self.net.parameters(), lr=args.lr)
		
		self.data = {}
		self.rise = None  # dict  , feature, label, date
		self.fall = None  # dict  , feature, label, date
		

	def train(self):
		print("------------- Start Training ---------------")
		
		for epoch in range(self.args.epoch):
			x, y, target = batch_for_few_shot(self.data, self.rise, self.fall, self.args.batch, self.args.train_start, self.args.train_end)

			pred = self.net(x, y)
			loss = self.loss_fn(pred, target)
			loss.backward()
			self.optimizer.step()
			self.optimizer.zero_grad()

			pred = torch.max(pred, 1)[1].data.numpy()
			accuracy = float((pred == target.detach().numpy()).astype(int).sum()) / float(target.detach().numpy().size)


			test_loss , test_acc = self.test()
			print("|| EPOCH: %3d  || Train Loss: %.6f || Train Accuracy: %.6f || Test Loss: %.6f || Test Accuracy: %.6f "\
										%(epoch, loss.detach().numpy(), accuracy, test_loss, test_acc))



	def test(self):
		print("------------- Start testing ---------------")
		self.net.eval()
		x, y, target = batch_for_few_shot(self.data, self.rise, self.fall, self.args.batch, self.args.test_start, self.args.test_end)
		pred = self.net(x, y)
		loss = self.loss_fn(pred, target)
		pred = torch.max(pred, 1)[1].data.numpy()
		accuracy = float((pred == target.detach().numpy()).astype(int).sum()) / float(target.detach().numpy().size)
		self.net.train()
		return loss.detach().numpy(),  accuracy 
		

	# def validation(self):
	# 	self.net.eval()
	# 	print("------------- Start validating ---------------")
	# 	x, y, target = batch_for_few_shot(self.data, self.rise, self.fall, self.args.batch, self.args.val_start, self.args.val_end)
	# 	pred = self.net(x, y)
	# 	loss = self.loss_fn(pred, target)
		

	def get_data(self, date = None):
		#  mode = train / val/ test
		#  data = [start, end].
		path = "./data/" + self.args.dataset + ".csv"
		data = pd.read_csv(path, index_col = "Ntime")

		data.index = pd.to_datetime(data.index, format='%Y%m%d')
		feature = data.loc[date[0]:date[1], :'Federal Fund Rate'].values
		label   = data.loc[date[0]:date[1], 'label'].values
		date    = data.loc[date[0]:date[1]].index.values
		feature, label, date = to_window(feature, label, date, self.args.time_step, self.args.image_type)
		self.data["feature"], self.data["label"], self.data["date"] = feature, label, date
		self.rise, self.fall = get_rise_fall(self.data)


		
		