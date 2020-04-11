import argparse
import os
import cv2
import torch
from mmdet.apis import inference_detector, init_detector, show_result
import json
import numpy as np
from mmdet.ops.nms.nms_wrapper import nms
from torch.utils.data import Dataset, DataLoader
import time
from PIL import Image, ImageDraw, ImageFont
import xml.etree.ElementTree as ET

#CLASSES=('尼龙制品', '塑料衣架', '榴莲壳', '粉笔', '口服液瓶', '打火机', '废弃针管', '矿泉水瓶', '金属工具', '易拉罐', '胶带', '风干食品', '绿植', '塑料泡沫', '牙刷', '报纸', '电线', '尿片', '创可贴', '玉米棒', '食品塑料盒', '饼干', '指甲油瓶', '花生壳', '蟹壳', '瓜子壳', '木制品', '中药渣', '中药材', '药片', '核桃', '水银温度计', '胶囊', '纽扣电池', '烟盒', '烟头', '薯片', '豆类', '油漆桶', '食用油桶', '枕头', '勺子', '粉扑', '插座', '塑料玩具', '玻璃制品', 'X光片', '塑料包装', '睫毛膏', '窗玻璃', '药品包装', '保温杯', '节能灯', '一次性塑料餐盒餐具', '药瓶', '金属瓶罐', '输液管', '日历', '鞋', '眼药水瓶', '金属制品', '丝绸手绢', '优盘', '动物内脏', '竹制品', '水彩笔', '锂电池', '甘蔗皮', '大骨棒', '棉签', '糖果', '染发剂壳', '棉被', '电路板', '书包', '床上用品', '旧毛巾', '镜子', '金属筷子', '树叶', '杂草', '信封', '狗尿布', '食品及日用品玻璃瓶罐', '一次性口罩', '农药瓶', '碎玻璃片', '口红', '笔', '西瓜子', '橡皮泥', '糕饼', '防碎气泡膜', '鸡骨头', '坚果壳', '香蕉皮', '硬塑料', '西瓜皮', '一次性手套', '利乐包', '篮球', '皮带', '蓄电池', '快递纸袋', '食品外包装盒', '广告单', '灯管灯泡', '菌菇类', '面食', '粽子', '打印纸', '面包', '干果', '吸铁石', '菜刀', '眼影', '牙膏皮', '足球', '菜叶', '枣核', '手机电池', '数据线', '消毒液瓶', '叉子', '罐头盒', '图书期刊', '毛发', '荔枝壳', '卫生纸', '杀虫剂及罐', '胶卷', '湿巾', '排骨-小肋排', '充电宝', '螺蛳', '贝类去硬壳', '苹果核', '话梅核', '纸巾', '墨盒', '粉底液') #142

CLASSES=('尼龙制品', '塑料衣架', '榴莲壳', '粉笔', '口服液瓶', '打火机', '废弃针管', '矿泉水瓶', '金属工具', '易拉罐', '胶带', '风干食品', '绿植', '塑料泡沫', '牙刷', '报纸', '电线', '尿片', '创可贴', '玉米棒', '食品塑料盒', '饼干', '指甲油瓶', '花生壳', '蟹壳', '瓜子壳', '木制品', '中药渣', '中药材', '药片', '核桃', '水银温度计', '胶囊', '纽扣电池', '烟盒', '烟头', '薯片', '豆类', '油漆桶', '食用油桶', '枕头', '勺子', '粉扑', '插座', '塑料玩具', '玻璃制品', 'X光片', '塑料包装', '睫毛膏', '窗玻璃', '药品包装', '保温杯', '节能灯', '一次性塑料餐盒餐具', '药瓶', '金属瓶罐', '输液管', '日历', '鞋', '眼药水瓶', '金属制品', '丝绸手绢', '优盘', '动物内脏', '竹制品', '水彩笔', '锂电池', '甘蔗皮', '大骨棒', '棉签', '糖果', '染发剂壳', '棉被', '电路板', '书包', '床上用品', '旧毛巾', '镜子', '金属筷子', '树叶', '杂草', '信封', '狗尿布', '食品及日用品玻璃瓶罐', '一次性口罩', '农药瓶', '碎玻璃片', '口红', '笔', '西瓜子', '橡皮泥', '糕饼', '防碎气泡膜', '鸡骨头', '坚果壳', '香蕉皮', '硬塑料', '西瓜皮', '一次性手套', '利乐包', '篮球', '皮带', '蓄电池', '快递纸袋', '食品外包装盒', '广告单', '灯管灯泡', '菌菇类', '面食', '粽子', '打印纸', '面包', '干果', '吸铁石', '菜刀', '眼影', '牙膏皮', '足球', '菜叶', '枣核', '手机电池', '数据线', '消毒液瓶', '叉子', '罐头盒', '图书期刊', '毛发', '荔枝壳', '卫生纸', '杀虫剂及罐', '排骨-小肋排', '充电宝', '螺蛳', '贝类去硬壳', '苹果核', '纸巾', '墨盒', '粉底液') #139

#CLASSES = ('瓜子壳', '核桃', '花生壳', '毛豆壳', '西瓜子', '枣核', '话梅核', '苹果皮', '柿子皮', '西瓜皮', '香蕉皮', '柚子皮', '荔枝壳', '芒果皮', '苹果核', '干果', '桔子皮', '饼干', '面包', '糖果', '宠物饲料', '风干食品', '蜜饯', '肉干', '冲泡饮料粉', '奶酪', '罐头', '糕饼', '薯片', '树叶', '杂草', '绿植', '鲜花', '豆类', '动物内脏', '绿豆饭', '谷类及加工物', '贝类去硬壳', '虾', '面食', '肉类', '五谷杂粮', '排骨-小肋排', '鸡', '鸡骨头', '螺蛳', '鸭', '鱼', '菜根', '菜叶', '菌菇类', '鱼鳞', '调料', '茶叶渣', '咖啡渣', '粽子', '动物蹄', '小龙虾', '蟹壳', '酱料', '鱼骨头', '蛋壳', '中药材', '中药渣', '镜子', '玻璃制品', '窗玻璃', '碎玻璃片', '化妆品玻璃瓶', '食品及日用品玻璃瓶罐', '保温杯', '玻璃杯', '图书期刊', '报纸', '食品外包装盒', '鞋盒', '利乐包', '广告单', '打印纸', '购物纸袋', '日历', '快递纸袋', '信封', '烟盒', '易拉罐', '金属制品', '吸铁石', '铝制品', '金属瓶罐', '金属工具', '罐头盒', '勺子', '菜刀', '叉子', '锅', '金属筷子', '数据线', '塑料玩具', '矿泉水瓶', '塑料泡沫', '塑料包装', '硬塑料', '一次性塑料餐盒餐具', '电线', '塑料衣架', '密胺餐具', '亚克力板', 'PVC管', '插座', '化妆品塑料瓶', '篮球', '足球', 'KT板', '食品塑料盒', '食用油桶', '塑料杯', '塑料盆', '一次性餐盒', '废弃衣服', '鞋', '碎布', '书包', '床上用品', '棉被', '丝绸手绢', '枕头', '毛绒玩具', '皮带', '电路板', '充电宝', '木制品', '优盘', '灯管灯泡', '节能灯', '二极管', '纽扣电池', '手机电池', '镍镉电池', '锂电池', '蓄电池', '胶卷', '照片', '指甲油瓶', 'X光片', '农药瓶', '杀虫剂及罐', '蜡烛', '墨盒', '染发剂壳', '消毒液瓶', '油漆桶', '药品包装', '药瓶', '废弃针管', '输液管', '口服液瓶', '眼药水瓶', '水银温度计', '水银血压计', '胶囊', '药片', '固体杀虫剂', '甘蔗皮', '坚果壳', '橡皮泥', '毛发', '棉签', '创可贴', '口红', '笔', '纸巾', '胶带', '湿巾', '水彩笔', '打火机', '防碎气泡膜', '榴莲壳', '睫毛膏', '眼影', '仓鼠浴沙', '大骨棒', '旧毛巾', '竹制品', '粉笔', '一次性口罩', '一次性手套', '粉底液', '灰土', '尼龙制品', '尿片', '雨伞', '带胶制品', '牙膏皮', '狗尿布', '椰子壳', '粉扑', '破碗碟', '陶瓷', '卫生纸', '烟头', '假睫毛', '猫砂', '牙刷', '玉米棒')

category2id = {'瓜子壳': 1, '核桃': 2, '花生壳': 3, '毛豆壳': 4, '西瓜子': 5, '枣核': 6, '话梅核': 7, '苹果皮': 8, '柿子皮': 9, '西瓜皮': 10, '香蕉皮': 11, '柚子皮': 12, '荔枝壳': 13, '芒果皮': 14, '苹果核': 15, '干果': 16, '桔子皮': 17, '饼干': 18, '面包': 19, '糖果': 20, '宠物饲料': 21, '风干食品': 22, '蜜饯': 23, '肉干': 24, '冲泡饮料粉': 25, '奶酪': 26, '罐头': 27, '糕饼': 28, '薯片': 29, '树叶': 30, '杂草': 31, '绿植': 32, '鲜花': 33, '豆类': 34, '动物内脏': 35, '绿豆饭': 36, '谷类及加工物': 37, '贝类去硬壳': 38, '虾': 39, '面食': 40, '肉类': 41, '五谷杂粮': 42, '排骨-小肋排': 43, '鸡': 44, '鸡骨头': 45, '螺蛳': 46, '鸭': 47, '鱼': 48, '菜根': 49, '菜叶': 50, '菌菇类': 51, '鱼鳞': 52, '调料': 53, '茶叶渣': 54, '咖啡渣': 55, '粽子': 56, '动物蹄': 57, '小龙虾': 58, '蟹壳': 59, '酱料': 60, '鱼骨头': 61, '蛋壳': 62, '中药材': 63, '中药渣': 64, '镜子': 65, '玻璃制品': 66, '窗玻璃': 67, '碎玻璃片': 68, '化妆品玻璃瓶': 69, '食品及日用品玻璃瓶罐': 70, '保温杯': 71, '玻璃杯': 72, '图书期刊': 73, '报纸': 74, '食品外包装盒': 75, '鞋盒': 76, '利乐包': 77, '广告单': 78, '打印纸': 79, '购物纸袋': 80, '日历': 81, '快递纸袋': 82, '信封': 83, '烟盒': 84, '易拉罐': 85, '金属制品': 86, '吸铁石': 87, '铝制品': 88, '金属瓶罐': 89, '金属工具': 90, '罐头盒': 91, '勺子': 92, '菜刀': 93, '叉子': 94, '锅': 95, '金属筷子': 96, '数据线': 97, '塑料玩具': 98, '矿泉水瓶': 99, '塑料泡沫': 100, '塑料包装': 101, '硬塑料': 102, '一次性塑料餐盒餐具': 103, '电线': 104, '塑料衣架': 105, '密胺餐具': 106, '亚克力板': 107, 'PVC管': 108, '插座': 109, '化妆品塑料瓶': 110, '篮球': 111, '足球': 112, 'KT板': 113, '食品塑料盒': 114, '食用油桶': 115, '塑料杯': 116, '塑料盆': 117, '一次性餐盒': 118, '废弃衣服': 119, '鞋': 120, '碎布': 121, '书包': 122, '床上用品': 123, '棉被': 124, '丝绸手绢': 125, '枕头': 126, '毛绒玩具': 127, '皮带': 128, '电路板': 129, '充电宝': 130, '木制品': 131, '优盘': 132, '灯管灯泡': 133, '节能灯': 134, '二极管': 135, '纽扣电池': 136, '手机电池': 137, '镍镉电池': 138, '锂电池': 139, '蓄电池': 140, '胶卷': 141, '照片': 142, '指甲油瓶': 143, 'X光片': 144, '农药瓶': 145, '杀虫剂及罐': 146, '蜡烛': 147, '墨盒': 148, '染发剂壳': 149, '消毒液瓶': 150, '油漆桶': 151, '药品包装': 152, '药瓶': 153, '废弃针管': 154, '输液管': 155, '口服液瓶': 156, '眼药水瓶': 157, '水银温度计': 158, '水银血压计': 159, '胶囊': 160, '药片': 161, '固体杀虫剂': 162, '甘蔗皮': 163, '坚果壳': 164, '橡皮泥': 165, '毛发': 166, '棉签': 167, '创可贴': 168, '口红': 169, '笔': 170, '纸巾': 171, '胶带': 172, '湿巾': 173, '水彩笔': 174, '打火机': 175, '防碎气泡膜': 176, '榴莲壳': 177, '睫毛膏': 178, '眼影': 179, '仓鼠浴沙': 180, '大骨棒': 181, '旧毛巾': 182, '竹制品': 183, '粉笔': 184, '一次性口罩': 185, '一次性手套': 186, '粉底液': 187, '灰土': 188, '尼龙制品': 189, '尿片': 190, '雨伞': 191, '带胶制品': 192, '牙膏皮': 193, '狗尿布': 194, '椰子壳': 195, '粉扑': 196, '破碗碟': 197, '陶瓷': 198, '卫生纸': 199, '烟头': 200, '假睫毛': 201, '猫砂': 202, '牙刷': 203, '玉米棒': 204}

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

def indent(elem, level=0):
        i = "\n" + level*"\t"
        if len(elem):
                if not elem.text or not elem.text.strip():
                        elem.text = i + "\t"
                if not elem.tail or not elem.tail.strip():
                        elem.tail = i
                for elem in elem:
                        indent(elem, level+1)
                if not elem.tail or not elem.tail.strip():
                        elem.tail = i
        else:   
                if level and (not elem.tail or not elem.tail.strip()):
                        elem.tail = i

def create_xml(anno_dir, image_dir, filename, dets, width, height):
	name_prefix = filename.split('.')[0]
	xml_name = name_prefix + ".xml"
	xml_path = os.path.join(anno_dir, xml_name)
	root = ET.Element('annotation')
	xml_folder = ET.SubElement(root, 'folder')
	xml_folder.text = 'vis'
	xml_jpg_name = ET.SubElement(root, 'filename')
	xml_jpg_name.text = filename
	xml_image_path = ET.SubElement(root, 'path')
	xml_image_path.text = os.path.join(image_dir, filename)
	xml_source = ET.SubElement(root, 'source')
	xml_database = ET.SubElement(xml_source, 'database')
	xml_database.text = 'Unknown'
	xml_size = ET.SubElement(root, 'size')
	xml_width = ET.SubElement(xml_size, 'width')
	xml_width.text = str(width)
	xml_height = ET.SubElement(xml_size, 'height')
	xml_height.text = str(height)
	xml_depth = ET.SubElement(xml_size, 'depth')
	xml_depth.text = '3'
	xml_segmented = ET.SubElement(root, 'segmented')
	xml_segmented.text = '0'
	for index, bboxes in enumerate(dets):
		category = CLASSES[index]
		for bbox in bboxes:
			score = bbox[4]
			if score < 0.3:
				continue

			xml_object = ET.SubElement(root, 'object')
			xml_object_diff = ET.SubElement(xml_object, 'difficult')
			xml_object_diff.text = '0'
			xml_object_name = ET.SubElement(xml_object, 'name')
			xml_object_name.text = category
			xml_object_bndbox = ET.SubElement(xml_object, 'bndbox')
			xml_bndbox_xmin = ET.SubElement(xml_object_bndbox, 'xmin')
			xml_bndbox_xmin.text = str(int(bbox[0]))
			xml_bndbox_ymin = ET.SubElement(xml_object_bndbox, 'ymin')
			xml_bndbox_ymin.text = str(int(bbox[1]))
			xml_bndbox_xmax = ET.SubElement(xml_object_bndbox, 'xmax')
			xml_bndbox_xmax.text = str(int(bbox[2]))
			xml_bndbox_ymax = ET.SubElement(xml_object_bndbox, 'ymax')
			xml_bndbox_ymax.text = str(int(bbox[3]))
	
	indent(root)
	tree = ET.ElementTree(root)
	tree.write(xml_path, encoding="utf-8", xml_declaration=True)

def parse_args():
	#
	parser = argparse.ArgumentParser(description='inference')
	parser.add_argument('--config', type=str, help='test config file path')
	parser.add_argument('--model', type=str, help='model file path')
	parser.add_argument('--gpu', type=str, default='0', help='gpu id')
	parser.add_argument('--data_dir', type=str, default='../data/test/JPEGImages', help='data dir')
	parser.add_argument('--show_result', type=str, default='False')
	args = parser.parse_args()
	return args

def change_cv2_draw(image, content, point):
    cv2img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pilimg = Image.fromarray(cv2img)
    draw = ImageDraw.Draw(pilimg)
    font = ImageFont.truetype("font/simhei.ttf", 30, encoding="utf-8")
    draw.text(point, content, (0, 0, 0), font=font)
    image = cv2.cvtColor(np.array(pilimg), cv2.COLOR_RGB2BGR)
    return image

def main():
	args = parse_args()
	os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
	model = init_detector(args.config, args.model, device=torch.device('cuda', 0))

	start = time.time()

	dataset = MyDataset(args.data_dir)
	file_count = len(dataset)
	dataloader = DataLoader(dataset=dataset, batch_size=1, shuffle=False, num_workers=1)

	#val_f = open('../data/test/val_open.json')
	#val_json = json.load(val_f)
	#images = val_json['images']
	#file_count = len(images)
	results = []
	show_scale = 2
	category_counts = {}
	less_category = ['苹果核', '日历', '贝类去硬壳', '湿巾', '充电宝', '粉底液', '面包', '墨盒', '排骨-小肋排', '鞋', '眼影']
	#for f_idx, image in enumerate(images):
		#image_id = image['image_id']
		#file_path = os.path.join(args.data_dir, image_id + '.jpg')
		#img = cv2.imread(file_path)

	anno_dir = 'annotations'
	if not os.path.exists(anno_dir):
		os.mkdir(anno_dir)

	image_dir = 'D:/天池比赛/垃圾分类/test/vis/'
	for f_idx, data in enumerate(dataloader):
		img, filename = data[0][0].numpy(), data[1][0]
		image_id = filename.split('.')[0]
		dets = inference_detector(model, img)
		show =False
		create_xml(anno_dir, image_dir, filename, dets, img.shape[1], img.shape[0])
		for index, bboxes in enumerate(dets):
			category = CLASSES[index]
			if not category in category_counts:
				category_counts[category] = 0

			if category == 'no':
				continue

			category_id = category2id[category]
			for bbox in bboxes:
				if category in less_category:
					pass
					#print(category, image_id)
				category_counts[category] += 1
				score = bbox[4].item()
				x = int(round(bbox[0].item(),0))
				y = int(round(bbox[1].item(),0))
				w = int(round(bbox[2].item() - bbox[0].item(), 0))
				h = int(round(bbox[3].item() - bbox[1].item(), 0))
				results.append({"image_id":image_id,"category_id":category_id,"bbox":[x,y,w,h],"score":score})
				
				if args.show_result == 'True' and category in less_category:
					cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 1)
					content = "{}:{}".format(category, score)
					img = change_cv2_draw(img, content, (x,y))
					show = True
		if args.show_result == 'True' and show:
			height, width, _ = img.shape
			save_path = os.path.join('vis', filename)
			img = cv2.resize(img, (int(width/2), int(height/2)))
			cv2.imwrite(save_path, img)

		if f_idx % 20 == 0:
			print("{}/{}".format(f_idx, file_count))

	with open('../results.json', 'w') as fp:
		json.dump(results, fp)
		
	print(category_counts)
	print("use time = {}".format(time.time()-start))

if __name__ == '__main__':
	os.system('rm -rf vis/*')
	main()


