import json

CLASSES = ('target', )

configs = []
config1 = {}
config1['config'] = 'best_model/shengna/ohem/cascade_r50_focal_ohem.py'
config1['model'] = 'best_model/shengna/ohem/epoch_12.pth'
config1['use'] = True
for cls in CLASSES:
	config1[cls] = 0.001
for cls in CLASSES:
	config1["{}_weight".format(cls)] = 1.0
configs.append(config1)

config2 = {}
config2['config'] = 'best_model/shengna/1_1.5/cascade_r50_c100.py'
config2['model'] = 'best_model/shengna/1_1.5/epoch_12.pth'
config2['use'] = True
for cls in CLASSES:
	config2[cls] = 0.001
for cls in CLASSES:
	config2["{}_weight".format(cls)] = 1.0
configs.append(config2)

config3 = {}
config3['config'] = 'best_model/shengna/1.25_1.75/cascade_r50_c125.py'
config3['model'] = 'best_model/shengna/1.25_1.75/epoch_12.pth'
config3['use'] = True
for cls in CLASSES:
	config3[cls] = 0.001
for cls in CLASSES:
	config3["{}_weight".format(cls)] = 1.0
configs.append(config3)


config4 = {}
config4['config'] = 'best_model/shengna/r101/cascade_r101_dcn_gcb.py'
config4['model'] = 'best_model/shengna/r101/epoch_12.pth'
config4['use'] = True
for cls in CLASSES:
	config4[cls] = 0.001
for cls in CLASSES:
	config4["{}_weight".format(cls)] = 1.0
configs.append(config4)


with open('shengna_config/config.json', 'w', encoding='utf-8') as fp:
	json.dump(configs, fp, ensure_ascii=False, indent=4, separators=(',', ': '))

