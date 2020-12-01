
import argparse
from framework import *



def main(args):
	model = Framework(args)
	
	print('|| Start Loading Data ......')
	model.get_data(date = [args.data_start, args.data_end])   # data : list [start end] 
	print('|| Loading Finish')
	model.train()
	# model.test()

	print('Experiment Finish!!')



if __name__ == "__main__":
	parser = argparse.ArgumentParser() 
	parser.add_argument('--dataset'           	   , type = str   , default = "S&P500")
	parser.add_argument('--image_type'             , type = str   , default = "GADF")
	parser.add_argument('--data_start'			   , type = str   , default = "2008-07")
	parser.add_argument('--data_end'			   , type = str   , default = "2010-09")
	parser.add_argument('--train_start'            , type = str   , default = "2009-06-01")
	parser.add_argument('--train_end'              , type = str   , default = "2010-05-31")
	# parser.add_argument('--val_start'              , type = str   , default = "2015-06-01")
	# parser.add_argument('--val_end'                , type = str   , default = "2015-12-31")
	parser.add_argument('--test_start'             , type = str   , default = "2010-06-01")
	parser.add_argument('--test_end'               , type = str   , default = "2010-09-30")
	parser.add_argument('--time_step'              , type = int   , default = 22)
	parser.add_argument('--epoch'                  , type = int   , default = 100)
	parser.add_argument('--K'                  	   , type = int   , default = 10)
	parser.add_argument('--N'                 	   , type = int   , default = 2)
	parser.add_argument('--batch'  		           , type = int   , default = 21)  # rise : 10 shot , fall: 10  shot, target: 1
	parser.add_argument('--in_channels'  		   , type = int   , default = 19)
	parser.add_argument('--out_channels'  		   , type = int   , default = 64)
	parser.add_argument('--key_size'  	     	   , type = int   , default = 64)
	parser.add_argument('--value_size'  		   , type = int   , default = 32)
	parser.add_argument('--lr'  	        	   , type = float , default = 0.001)
	args = parser.parse_args()
	print(args)
	print("-"*100)

	main(args)





# 2008-06 ~ 2016-09

# train : N =  K