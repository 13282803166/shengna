import json
import mmcv
import numpy as np
import argparse
from mmdet.ops.nms.nms_wrapper import nms
from mmdet.ops.nms.nms_wrapper import soft_nms
from mmdet.datasets import build_dataloader, build_dataset

CLASSES=('target',)
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
        parser.add_argument('--config', type=str, default='configs/config.json')
        parser.add_argument('--options', nargs='+', action=MultipleKVAction, help='custom options')
        args = parser.parse_args()
        return args


def get_det_score(score, json_dets, dataset, model_configs, config_idx, **kwargs):
        results = []
        for file_index, file_dets in enumerate(json_dets):
                all_dets = []
                config_dets = file_dets[config_idx]
                config = model_configs[config_idx]
                dets = []
                for index, det in enumerate(config_dets):
                        if len(det) == 0:
                                det = np.empty((0, 5))
                        else:
                                det = np.array(det)

                        det = det[det[:,4]>=score]
                        det[:,4] *= config["{}_weight".format(CLASSES[index])]
                        dets.append(det)

                for index, det in enumerate(dets):
                        if len(dets[index]) == 0:
                                continue
                        dets[index] = soft_nms(dets[index], 0.5)[0]
                results.append(dets)

        _, stats, aps = dataset.evaluate(results, 'bbox', **kwargs)
        return stats[0], aps

def get_best_score(best_scores, score, aps, max_aps):
        print_list = []
        for index, (ap, max_ap) in enumerate(zip(aps, max_aps)):
                src_score = best_scores[index]
                if ap >= max_ap:
                        max_aps[index] = ap
                        best_scores[index] = score


                ####print####
                if ap > max_ap:
                        content = "index:{}, score: {}->{}, ap: {}->{}".format(index, src_score, score, max_ap, ap)
                        print_list.append(content)

        for line in print_list:
                print(line)

        return best_scores

def get_best_scores(scores, json_dets, dataset, model_configs, **kwargs):
        for i in range(len(model_configs)):
                best_score = scores[0]
                max_score = 0
                best_scores = [0 for k in range(len(CLASSES))]
                max_aps = [0 for k in range(len(CLASSES))]
                for score in scores:
                        print('\n ------->model: {}, score: {}<------'.format(i, score))
                        det_score, aps = get_det_score(score, json_dets, dataset, model_configs, i, **kwargs)
                        best_scores = get_best_score(best_scores, score, aps, max_aps)

                for det_index in range(len(CLASSES)):
                        model_configs[i]["{}".format(CLASSES[det_index])] = best_scores[det_index]
                print("\n---------------->cls: {}, best_score: {}, max_score: {}<----------------".format(det_index, best_score, max_score))
        with open('shengna_config/best_score_config.json', 'w', encoding='utf-8') as fp:
                json.dump(model_configs, fp, ensure_ascii=False, indent=4, separators=(',', ': '))


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
        scores = [0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        fp = open('shengna_config/assemble_results.json')
        kwargs = {} if args.options is None else args.options
        json_dets = json.load(fp)
        fp.close()

        get_best_scores(scores, json_dets, coco_dataset, model_configs, **kwargs)


if __name__ == "__main__":
        main()
