import argparse
import os
import cv2
import torch
from mmdet.apis import inference_detector, init_detector, show_result
import json
import numpy as np
from mmdet.ops.nms.nms_wrapper import nms, soft_nms
from torch.utils.data import Dataset, DataLoader
import time
import mmcv
from mmdet.datasets import build_dataloader, build_dataset
import csv

CLASSES = ('target',)

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
	parser = argparse.ArgumentParser(description='inference')
	parser.add_argument('--gpu', type=str, default='0', help='gpu id')
	parser.add_argument('--data_dir', type=str, default='../../shengna/b-test-image/test', help='val data dir')
	parser.add_argument('--config', type=str, default='shengna_config/best_config.json')
	args = parser.parse_args()
	return args

def get_config_weights(config, sizes):
        weights = []
        for size in sizes:
                weight = config['{}_weight'.format(size)]
                weights.append(weight)
        return weights

def get_weight(weights, sizes, area):
        for index, size in enumerate(sizes):
                if area < size ** 2:
                        break

        return weights[index]

def main():
	args = parse_args()
	os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
	config_file = open(args.config)
	configs = json.load(config_file)
	config_file.close()

	infer_results = []
	dataset = MyDataset(args.data_dir)
	file_count = len(dataset)
	dataloader = DataLoader(dataset=dataset, batch_size=1, shuffle=False, num_workers=1)

	models = []
	model_configs = []
	for config in configs:
		if not config['use']:
			continue
		print('config = {}, model = {}'.format(config['config'], config['model']))
		model = init_detector(config['config'], config['model'], device=torch.device('cuda', 0))
		models.append(model)
		model_configs.append(config)

	start = time.time()
	for image_index, data in enumerate(dataloader):
		im, filename = data[0][0].numpy(), data[1][0]
		image_id = filename.split('.')[0]
		all_dets = []
		for config, model in zip(model_configs, models):
			dets = inference_detector(model, im)
			for index, det in enumerate(dets):
				dets[index] = det[det[:,4]>=config[CLASSES[index]]]
				dets[index][:,4] *= config["{}_weight".format(CLASSES[index])]

			if len(all_dets) == 0:
				all_dets = dets
			else:
				for det_index, (all_det, det) in enumerate(zip(all_dets, dets)):
					all_dets[det_index] = np.concatenate((all_det, det), axis=0)

		for index, all_det in enumerate(all_dets):
			all_dets[index] = soft_nms(all_dets[index], 0.5, min_score=0.001)[0][0:100]

		for index, bboxes in enumerate(all_dets):
			category = CLASSES[index]
			for bbox in bboxes:
				infer_results.append((category, image_id+'.xml', str(bbox[4]), str(bbox[0]), str(bbox[1]), str(bbox[2]), str(bbox[3])))

		end = time.time()
		print('\r', '{}/{}, use time:{}'.format(image_index, file_count, end -start), end='', flush=True)
		start = end

	csvfile = open('../shengna_results.csv', 'w')
	writer = csv.writer(csvfile)
	writer.writerow(['name', 'image_id', 'confidence', 'xmin', 'ymin', 'xmax', 'ymax'])
	writer.writerows(infer_results)
	csvfile.close()

if __name__ == '__main__':
	os.system('rm -rf result_vis/*')
	main()
