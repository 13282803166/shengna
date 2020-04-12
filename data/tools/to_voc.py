import os
import cv2
import xml.etree.ElementTree as ET

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

def create_xml(anno_dir, image_dir, name, bboxes, width, height):
        name_prefix = name.split('.')[0]
        xml_name = name_prefix + ".xml"
        xml_path = os.path.join(anno_dir, xml_name)
        root = ET.Element('annotation')
        xml_jpg_name = ET.SubElement(root, 'filename')
        xml_jpg_name.text = name
        xml_image_path = ET.SubElement(root, 'path')
        xml_image_path.text = os.path.join(image_dir, name_prefix, name)
        xml_size = ET.SubElement(root, 'size')
        xml_width = ET.SubElement(xml_size, 'width')
        xml_width.text = str(width)
        xml_height = ET.SubElement(xml_size, 'height')
        xml_height.text = str(height)
        xml_depth = ET.SubElement(xml_size, 'depth')
        xml_depth.text = '3'
        for bbox in bboxes:
                xml_object = ET.SubElement(root, 'object')
                xml_object_diff = ET.SubElement(xml_object, 'difficult')
                xml_object_diff.text = bbox[-1]
                xml_object_name = ET.SubElement(xml_object, 'name')
                xml_object_name.text = bbox[-2]
                xml_object_bndbox = ET.SubElement(xml_object, 'bndbox')
                xml_bndbox_xmin = ET.SubElement(xml_object_bndbox, 'xmin')
                xml_bndbox_xmin.text = str(bbox[0])
                xml_bndbox_ymin = ET.SubElement(xml_object_bndbox, 'ymin')
                xml_bndbox_ymin.text = str(bbox[1])
                xml_bndbox_xmax = ET.SubElement(xml_object_bndbox, 'xmax')
                xml_bndbox_xmax.text = str(bbox[2])
                xml_bndbox_ymax = ET.SubElement(xml_object_bndbox, 'ymax')
                xml_bndbox_ymax.text = str(bbox[3])

        indent(root)
        tree = ET.ElementTree(root)
        tree.write(xml_path, encoding="utf-8", xml_declaration=True)

def main():
	image_dir = 'JPEGImages'
	anno_dir = 'Annotations'
	if not os.path.exists(anno_dir):
		os.mkdir(anno_dir)

	src_anno_dir = 'box'
	src_anno_files = os.listdir(src_anno_dir)
	for index, src_anno_file in enumerate(src_anno_files):
		image_file = src_anno_file.split('.')[0] + '.jpg'
		image_path = os.path.join(image_dir, image_file)
		im = cv2.imread(image_path)
		if im is None:
			continue

		height, width, _ = im.shape
		path = os.path.join(src_anno_dir, src_anno_file)
		root = ET.parse(path)
		bboxes = []
		objs = root.findall('object')
		for obj in objs:
			cls = obj.find('name').text
			bndbox = obj.find('bndbox')
			xmin = int(float(bndbox.find('xmin').text))
			ymin = int(float(bndbox.find('ymin').text))
			xmax = int(float(bndbox.find('xmax').text))
			ymax = int(float(bndbox.find('ymax').text))
			bboxes.append([xmin, ymin, xmax, ymax, cls, '0'])
		create_xml(anno_dir, image_dir, image_file, bboxes, width, height)
		print(index)

if __name__ == "__main__":
	main()