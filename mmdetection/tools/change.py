import os
import torch

def main():
	number_stage = 3
	anchor_scale_count = 3
	num_classes = 5

	#src_model, save_model='pretrained/cascade_r50_dcn_gcb_gn.pth','pretrained/r50_dcn_gcb_c{}_c{}.pth'.format(num_classes, number_stage)
	#src_model, save_model='pretrained/cascade_r101_dcn_gcb.pth','pretrained/r101_dcn_gcb_c{}_c{}.pth'.format(num_classes, number_stage)
	#src_model, save_model='pretrained/cascade_rcnn_se_resnext50_32x4d_fpn_1x_20191123_map435.pth', 'pretrained/se_x50_c{}_c{}.pth'.format(num_classes, number_stage)
	#src_model, save_model='pretrained/cascade_rcnn_se_resnext101_32x4d_fpn_1x_20191123_map454.pth', 'pretrained/se_x101_c{}_c{}.pth'.format(num_classes, number_stage)
	#src_model, save_model='pretrained/cascade_rcnn_dconv_c3-c5_r101_fpn_1x_20190125-aaa877cc.pth','pretrained/r101_dcn_c{}.pth'.format(num_classes)
	#src_model, save_model='pretrained/cascade_rcnn_dconv_c3-c5_r50_fpn_1x_20190125-dfa53166.pth','pretrained/r50_dcn_c{}_c{}.pth'.format(num_classes, number_stage)
	#src_model, save_model='pretrained/cascade_rcnn_r50_fpn_20e_20181123-db483a09.pth','pretrained/r50_c{}.pth'.format(num_classes)
	#src_model, save_model='pretrained/cascade_rcnn_hrnetv2p_w48_20e_20190810-f40ed8e1.pth','pretrained/hr_c{}.pth'.format(num_classes)
	src_model,save_model='pretrained/htc_dconv_c3-c5_mstrain_400_1400_x101_64x4d_fpn_20e_20190408-0e50669c.pth','pretrained/x101_64_dcn_c{}.pth'.format(num_classes)
	#src_model,save_model='pretrained/cascade_rcnn_r50_fpn_20e_20181123-db483a09.pth','pretrained/r50.pth'
	#src_model,save_model='pretrained/cascade_rcnn_x101_64x4d_fpn_2x_20181218-5add321e.pth', 'pretrained/x101_64.pth'
	#src_model,save_model='pretrained/cascade_rcnn_x101_32x4d_fpn_2x_20181218-28f73c4c.pth','pretrained/x101_32_c{}.pth'.format(num_classes)
	#src_model,save_model='pretrained/cascade_rcnn_hrnetv2p_w48_20e_20190810-f40ed8e1.pth', 'pretrained/hrnet.pth'
	#src_model,save_model='pretrained/cascade_mask_rcnn_r4_gcb_dconv_c3-c5_x101_32x4d_fpn_syncbn_1x_20190602-b4164f6b.pth','pretrained/x101_32_gcb_dcn_syncbn.pth'
	#src_model,save_model=
	new_model = {}
	source_classes = 81

	model_coco = torch.load(src_model)
	new_model["meta"] = model_coco["meta"]
	print(new_model["meta"]["config"])
	new_model["state_dict"] = {}
	for state_dict in model_coco["state_dict"]:
		if "mask_head" in state_dict or "semantic_head" in state_dict:
			continue

		new_model["state_dict"][state_dict] = model_coco["state_dict"][state_dict]
	
	# weight
	new_model["state_dict"]["bbox_head.0.fc_cls.weight"].resize_(num_classes,1024)
	new_model["state_dict"]["bbox_head.1.fc_cls.weight"].resize_(num_classes,1024)
	new_model["state_dict"]["bbox_head.2.fc_cls.weight"].resize_(num_classes,1024)
	# bias	
	new_model["state_dict"]["bbox_head.0.fc_cls.bias"].resize_(num_classes)
	new_model["state_dict"]["bbox_head.1.fc_cls.bias"].resize_(num_classes)
	new_model["state_dict"]["bbox_head.2.fc_cls.bias"].resize_(num_classes)
	if num_classes < source_classes:
		pass

	loop = num_classes // source_classes + 1
	for i in range(loop):
		start = i * source_classes
		if (i+1) * source_classes < num_classes:
			end = (i + 1) * source_classes
			new_model["state_dict"]["bbox_head.0.fc_cls.weight"][start:end] = new_model["state_dict"]["bbox_head.0.fc_cls.weight"][:source_classes]
			new_model["state_dict"]["bbox_head.1.fc_cls.weight"][start:end] = new_model["state_dict"]["bbox_head.1.fc_cls.weight"][:source_classes]
			new_model["state_dict"]["bbox_head.2.fc_cls.weight"][start:end] = new_model["state_dict"]["bbox_head.2.fc_cls.weight"][:source_classes]
			new_model["state_dict"]["bbox_head.0.fc_cls.bias"][start:end] = new_model["state_dict"]["bbox_head.0.fc_cls.bias"][:source_classes]
			new_model["state_dict"]["bbox_head.1.fc_cls.bias"][start:end] = new_model["state_dict"]["bbox_head.1.fc_cls.bias"][:source_classes]
			new_model["state_dict"]["bbox_head.2.fc_cls.bias"][start:end] = new_model["state_dict"]["bbox_head.2.fc_cls.bias"][:source_classes]

		else:
			end = num_classes - start
			new_model["state_dict"]["bbox_head.0.fc_cls.weight"][start:] = new_model["state_dict"]["bbox_head.0.fc_cls.weight"][0:end]
			new_model["state_dict"]["bbox_head.1.fc_cls.weight"][start:] = new_model["state_dict"]["bbox_head.1.fc_cls.weight"][0:end]
			new_model["state_dict"]["bbox_head.2.fc_cls.weight"][start:] = new_model["state_dict"]["bbox_head.2.fc_cls.weight"][0:end]
			new_model["state_dict"]["bbox_head.0.fc_cls.bias"][start:] = new_model["state_dict"]["bbox_head.0.fc_cls.bias"][0:end]
			new_model["state_dict"]["bbox_head.1.fc_cls.bias"][start:] = new_model["state_dict"]["bbox_head.1.fc_cls.bias"][0:end]
			new_model["state_dict"]["bbox_head.2.fc_cls.bias"][start:] = new_model["state_dict"]["bbox_head.2.fc_cls.bias"][0:end]


	if number_stage == 4:
		new_model["state_dict"]["bbox_head.3.fc_cls.weight"] = new_model["state_dict"]["bbox_head.2.fc_cls.weight"]
		new_model["state_dict"]["bbox_head.3.fc_cls.bias"] = new_model["state_dict"]["bbox_head.2.fc_cls.bias"]
		new_model["state_dict"]["bbox_head.3.fc_reg.weight"] = new_model["state_dict"]["bbox_head.2.fc_reg.weight"]
		new_model["state_dict"]["bbox_head.3.fc_reg.bias"] = new_model["state_dict"]["bbox_head.2.fc_reg.bias"]
		new_model["state_dict"]["bbox_head.3.fc_cls.weight"] = new_model["state_dict"]["bbox_head.2.fc_cls.weight"]
		new_model["state_dict"]["bbox_head.3.shared_fcs.0.weight"] = new_model["state_dict"]["bbox_head.2.shared_fcs.0.weight"]
		new_model["state_dict"]["bbox_head.3.shared_fcs.0.bias"] = new_model["state_dict"]["bbox_head.2.shared_fcs.0.bias"]
		new_model["state_dict"]["bbox_head.3.shared_fcs.1.weight"] = new_model["state_dict"]["bbox_head.2.shared_fcs.1.weight"]
		new_model["state_dict"]["bbox_head.3.shared_fcs.1.bias"] = new_model["state_dict"]["bbox_head.2.shared_fcs.1.bias"]

	if anchor_scale_count == 5:
		new_model["state_dict"]["rpn_head.rpn_cls.weight"].resize_(5, 256, 1, 1)
		new_model["state_dict"]["rpn_head.rpn_cls.weight"][3] = new_model["state_dict"]["rpn_head.rpn_cls.weight"][0]
		new_model["state_dict"]["rpn_head.rpn_cls.weight"][4] = new_model["state_dict"]["rpn_head.rpn_cls.weight"][1]

		new_model["state_dict"]["rpn_head.rpn_cls.bias"].resize_(5)
		new_model["state_dict"]["rpn_head.rpn_cls.bias"][3] = new_model["state_dict"]["rpn_head.rpn_cls.bias"][0]
		new_model["state_dict"]["rpn_head.rpn_cls.bias"][4] = new_model["state_dict"]["rpn_head.rpn_cls.bias"][1]

		new_model["state_dict"]["rpn_head.rpn_reg.weight"].resize_(20, 256, 1, 1)
		new_model["state_dict"]["rpn_head.rpn_reg.bias"].resize_(20)
		new_model["state_dict"]["rpn_head.rpn_reg.weight"][12:] = new_model["state_dict"]["rpn_head.rpn_reg.weight"][:8]
		new_model["state_dict"]["rpn_head.rpn_reg.bias"][12:] = new_model["state_dict"]["rpn_head.rpn_reg.bias"][:8]


	#save new model
	torch.save(new_model, save_model)

if __name__ == "__main__":
	os.environ['CUDA_VISIBLE_DEVICES'] = '3'
	main()

