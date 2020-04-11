import json
import mmcv
import numpy as np
import argparse
from mmdet.ops.nms.nms_wrapper import nms
from mmdet.ops.nms.nms_wrapper import soft_nms
from mmdet.datasets import build_dataloader, build_dataset

CLASSES = ('target', )
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
	parser.add_argument('--config', type=str, default='shengna_config/config.json')
	parser.add_argument('--options', nargs='+', action=MultipleKVAction, help='custom options')
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
	config_file = open(args.config)
	configs = json.load(config_file)
	config_file.close()

	model_configs = []
	for config in configs:
		if config['use']:
			print('config = {}, model = {}'.format(config['config'], config['model']))
		model_configs.append(config)
		g_config = config['config']

	cfg = mmcv.Config.fromfile(g_config)
	coco_dataset = build_dataset(cfg.data.test)

	with open('shengna_config/assemble_results.json') as fp:
		results = []
		json_dets = json.load(fp)
		for file_index, file_dets in enumerate(json_dets):
			all_dets = []
			for config_index, config_dets in enumerate(file_dets):
				config = model_configs[config_index]
				if config['use'] == False:
					continue

				dets = []
				for index, det in enumerate(config_dets):
					if len(det) == 0:
						det = np.empty((0, 5))
					else:
						det = np.array(det)

					det = det[det[:,4]>=config[CLASSES[index]]]
					det[:,4] *= config["{}_weight".format(CLASSES[index])]
					dets.append(det)

				if len(all_dets) == 0:
					all_dets = dets
				else:
					for det_index, (all_det, det) in enumerate(zip(all_dets, dets)):
						all_dets[det_index] = np.concatenate((all_det, det), axis=0)

			for index, all_det in enumerate(all_dets):
				#all_dets[index] = soft_nms(all_dets[index], 0.5, min_score=0.001)[0]
				all_dets[index] = soft_nms(all_dets[index], 0.5, min_score=0.001)[0][0:100]
			results.append(all_dets)

		kwargs = {} if args.options is None else args.options
		coco_dataset.evaluate(results, 'bbox', **kwargs)

if __name__ == "__main__":
        main()
