
import argparse
from framework import *



def main(args):
	model = Framework(args)
	
	print('start transform time series data to image ')
	model.get_data(mode = "train", date = [args.train_start, args.train_end])   # data : list [start end] 
	model.get_data(mode = "val"  , date = [args.val_start, args.val_end])
	model.get_data(mode = "test" , date = [args.test_start, args.test_end])
	print('transform finish')
	print('--------------------Start Training------------------')
	model.train()
	print('--------------------Start Testing------------------')
	model.test()

	print('Experiment Finish!!')



if __name__ == "__main__":
	parser = argparse.ArgumentParser() 
	parser.add_argument('--dataset'           	   , type = str   , default = "S&P500")
	parser.add_argument('--image_type'             , type = str   , default = "GADF")
	parser.add_argument('--train_start'            , type = str   , default = "2009-06")
	parser.add_argument('--train_end'              , type = str   , default = "2015-05")
	parser.add_argument('--val_start'              , type = str   , default = "2015-06")
	parser.add_argument('--val_end'                , type = str   , default = "2015-12")
	parser.add_argument('--test_start'             , type = str   , default = "2016-01")
	parser.add_argument('--test_end'               , type = str   , default = "2016-09")
	parser.add_argument('--time_step'              , type = int   , default = 22)
	parser.add_argument('--epoch'              , type = int   , default = 100)
	args = parser.parse_args()
	print(args)
	print("-"*100)

	main(args)





# 2008-06 ~ 2016-09

# train : N =  K