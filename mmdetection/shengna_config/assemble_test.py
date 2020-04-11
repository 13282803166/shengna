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

CLASSES = ('target',)
class MyDataset(Dataset):
        def __init__(self, val_path, data_dir):
                val_f = open(val_path)
                lines = val_f.read().splitlines()
                self.data_list = []
                for data in lines:
                        self.data_list.append(data)

                self.data_dir = data_dir

        def __getitem__(self, index):
                image_file = self.data_list[index]
                image_path = os.path.join(self.data_dir, image_file + '.jpg')
                im = cv2.imread(image_path)
                return im, image_file

        def __len__(self):
                return len(self.data_list)

class MultipleKVAction(argparse.Action):
    """
    argparse action to split an argument into KEY=VALUE form
    on the first = and append to a dictionary.
    """

    def _is_int(self, val):
        try:
            _ = int(val)
            return True
        except Exception:
            return False

    def _is_float(self, val):
        try:
            _ = float(val)
            return True
        except Exception:
            return False

    def _is_bool(self, val):
        return val.lower() in ['true', 'false']

    def __call__(self, parser, namespace, values, option_string=None):
        options = {}
        for val in values:
            parts = val.split('=')
            key = parts[0].strip()
            if len(parts) > 2:
                val = '='.join(parts[1:])
            else:
                val = parts[1].strip()
            # try parsing val to bool/int/float first
            if self._is_bool(val):
                import json
                val = json.loads(val.lower())
            elif self._is_int(val):
                val = int(val)
            elif self._is_float(val):
                val = float(val)
            options[key] = val
        setattr(namespace, self.dest, options)

def parse_args():
	parser = argparse.ArgumentParser(description='inference')
	parser.add_argument('--gpu', type=str, default='0', help='gpu id')
	parser.add_argument('--file_path', type=str, default='../../shengna/train/ImageSets/Main/val.txt', help='val file path')
	parser.add_argument('--data_dir', type=str, default='../../shengna/train/', help='val data dir')
	parser.add_argument('--config', type=str, default='shengna_config/config.json')
	parser.add_argument('--result', type=str, default='shengna_config/assemble_results.json')
	parser.add_argument('--options', nargs='+', action=MultipleKVAction, help='custom options')
	args = parser.parse_args()
	return args

def main():
	args = parse_args()
	os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
	config_file = open(args.config)
	configs = json.load(config_file)
	config_file.close()

	results = []
	json_results = []
	image_dir = os.path.join(args.data_dir, 'JPEGImages')
	anno_dir = os.path.join(args.data_dir, 'Annotations')
	dataset = MyDataset(args.file_path, image_dir)
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
		g_config = config['config']

	print(g_config)
	cfg = mmcv.Config.fromfile(g_config)
	coco_dataset = build_dataset(cfg.data.test)

	category_counts = {}
	start = time.time()
	for image_index, data in enumerate(dataloader):
		im, filename = data[0][0].numpy(), data[1][0]
		all_dets = []
		all_config_dets = []
		for config, model in zip(model_configs, models):
			dets = inference_detector(model, im)
			json_dets = []
			for det in dets:
				json_dets.append(det.tolist())
			all_config_dets.append(json_dets)

			for index, det in enumerate(dets):
				dets[index] = det[det[:,4]>=config[CLASSES[index]]]
				dets[index][:,4] *= config["{}_weight".format(CLASSES[index])]

			if len(all_dets) == 0:
				all_dets = dets
			else:
				for det_index, (all_det, det) in enumerate(zip(all_dets, dets)):
					all_dets[det_index] = np.concatenate((all_det, det), axis=0)

		json_results.append(all_config_dets)

		for index, all_det in enumerate(all_dets):
			all_dets[index] = soft_nms(all_dets[index], 0.5, min_score=0.001)[0][0:100]

		results.append(all_dets)
		end = time.time()
		print('\r', '{}/{}, use time:{}'.format(image_index, file_count, end -start), end='', flush=True)
		start = end

	#mmcv.dump(results, 'eval/assemble_result.pkl')
	kwargs = {} if args.options is None else args.options
	coco_dataset.evaluate(results, 'bbox', **kwargs)
	with open(args.result, 'w') as fp:
		json.dump(json_results, fp, indent=4, separators=(',', ': '))

if __name__ == '__main__':
	os.system('rm -rf result_vis/*')
	main()
