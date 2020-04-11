import json
import mmcv
import numpy as np
import argparse
from mmdet.ops.nms.nms_wrapper import nms
from mmdet.ops.nms.nms_wrapper import soft_nms
from mmdet.datasets import build_dataloader, build_dataset

CLASSES = ('target',)

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

def get_results(config_weights, json_dets, model_configs):
	print("\n\n\n\n current weights = {}".format(config_weights))
	results = []
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

				weight = config_weights[config_index]
				if weight == 0:
					det = np.empty((0, 5))
				else:
					det = det[det[:,4]>=config[CLASSES[index]]]

				det[:,4] *= weight
				dets.append(det)

			if len(all_dets) == 0:
				all_dets = dets
			else:
				for det_index, (all_det, det) in enumerate(zip(all_dets, dets)):
					all_dets[det_index] = np.concatenate((all_det, det), axis=0)

		for index, all_det in enumerate(all_dets):
			if len(all_det) == 0:
				continue
			all_dets[index] = soft_nms(all_dets[index], 0.5)[0][0:100]

		results.append(all_dets)

	return results

def get_best_weights(config_weights, best_weights, aps, max_aps, score):
	for index, (ap, max_ap) in enumerate(zip(aps, max_aps)):
		if ap > max_ap:
			max_aps[index] = ap
			best_weights[index] = config_weights

	print("---->get_best_weights: old score = {}, score = {}<-----".format(score, np.mean(max_aps)))
	return best_weights, max_aps

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
	#weights = [0, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
	#weights = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
	weights = [0.8, 0.85, 0.9, 0.95, 1.0]
	fp = open('shengna_config/assemble_results.json')
	kwargs = {} if args.options is None else args.options
	json_dets = json.load(fp)
	fp.close()

	max_aps = [0 for i in range(len(CLASSES))]
	best_weights = [[0 for j in range(len(model_configs))] for i in range(len(CLASSES))]
	for weight1 in weights:
		for weight2 in weights:
			for weight3 in weights:
				config_weights = [weight1, 1.0, weight2, weight3]
				results = get_results(config_weights, json_dets, model_configs)
				_, stats, aps = coco_dataset.evaluate(results, 'bbox', **kwargs)
				score = stats[0]
				best_weights, max_aps = get_best_weights(config_weights, best_weights, aps, max_aps, score)

	'''
	for weight1 in weights:
		for weight2 in weights:
			for weight3 in weights:
				config_weights = [weight1, 1.0, weight2, weight3]
				results = get_results(config_weights, json_dets, model_configs)
				_, stats, aps = coco_dataset.evaluate(results, 'bbox', **kwargs)
				score = stats[0]
				best_weights, max_aps = get_best_weights(config_weights, best_weights, aps, max_aps, score)


	for weight1 in weights:
		for weight2 in weights:
			for weight3 in weights:
				config_weights = [weight1, weight2, 1.0, weight3]
				results = get_results(config_weights, json_dets, model_configs)
				_, stats, aps = coco_dataset.evaluate(results, 'bbox', **kwargs)
				score = stats[0]
				best_weights, max_aps = get_best_weights(config_weights, best_weights, aps, max_aps, score)


	for weight1 in weights:
		for weight2 in weights:
			for weight3 in weights:
				config_weights = [weight1, weight2, weight3, 1.0]
				results = get_results(config_weights, json_dets, model_configs)
				_, stats, aps = coco_dataset.evaluate(results, 'bbox', **kwargs)
				score = stats[0]
				best_weights, max_aps = get_best_weights(config_weights, best_weights, aps, max_aps, score)

	'''
	for cls_index, config_weights in enumerate(best_weights):
		for model_index, weight in enumerate(config_weights):
			if weight == 0:
				model_configs[model_index]["{}".format(CLASSES[cls_index])] = 1.0
			model_configs[model_index]["{}_weight".format(CLASSES[cls_index])] = weight
			print("model_configs[{}][\"{}\"_weight] = {}".format(model_index, CLASSES[cls_index], weight))
	with open('shengna_config/best_config.json', 'w', encoding='utf-8') as fp:
		json.dump(configs, fp, ensure_ascii=False, indent=4, separators=(',', ': '))


if __name__ == "__main__":
        main()
