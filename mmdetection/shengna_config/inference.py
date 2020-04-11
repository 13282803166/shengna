import argparse
import os
import cv2
import torch
from mmdet.apis import inference_detector, init_detector, show_result
import numpy as np
from mmdet.ops.nms.nms_wrapper import nms
from torch.utils.data import Dataset, DataLoader
import time
import csv

CLASSES=('target',)

class MyDataset(Dataset):
	def __init__(self, data_dir):
		data_list = os.listdir(data_dir)
		self.data_list = []
		for data in data_list:
			if not '.jpg' in data:
				continue
			self.data_list.append(data)
	
		self.data_dir = data_dir

	def __getitem__(self, index):
		filename = self.data_list[index]
		image_path = os.path.join(self.data_dir, filename)
		im = cv2.imread(image_path)
		return im, filename

	def __len__(self):
		return len(self.data_list)

def parse_args():
	#
	parser = argparse.ArgumentParser(description='inference')
	parser.add_argument('--config', default='shengna_config/cascade_r50_dcn.py', type=str, help='test config file path')
	parser.add_argument('--model', default='shengna_model/r50/epoch_12.pth', type=str, help='model file path')
	parser.add_argument('--gpu', type=str, default='0', help='gpu id')
	parser.add_argument('--data_dir', type=str, default='../../shengna/a-test-image/test', help='data dir')
	args = parser.parse_args()
	return args

def main():
	args = parse_args()
	os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
	model = init_detector(args.config, args.model, device=torch.device('cuda', 0))

	start = time.time()

	dataset = MyDataset(args.data_dir)
	file_count = len(dataset)
	dataloader = DataLoader(dataset=dataset, batch_size=1, shuffle=False, num_workers=2)

	results = []
	category_counts = {}
	for f_idx, data in enumerate(dataloader):
		img, filename = data[0][0].numpy(), data[1][0]
		image_id = filename.split('.')[0]
		dets = inference_detector(model, img)
		for index, bboxes in enumerate(dets):
			category = CLASSES[index]
			if not category in category_counts:
				category_counts[category] = 0

			for bbox in bboxes:
				category_counts[category] += 1
				results.append((category, image_id+'.xml', str(bbox[4]), str(bbox[0]), str(bbox[1]), str(bbox[2]), str(bbox[3])))
			
		if f_idx % 50 == 0:
			print("{}/{}".format(f_idx, file_count))

	csvfile = open('../shengna_results.csv', 'w')
	writer = csv.writer(csvfile)
	writer.writerow(['name', 'image_id', 'confidence', 'xmin', 'ymin', 'xmax', 'ymax'])
	writer.writerows(results)
	csvfile.close()
	
	print(category_counts)
	print("use time = {}".format(time.time()-start))

if __name__ == '__main__':
	main()


