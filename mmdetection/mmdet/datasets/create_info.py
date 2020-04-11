import os
import cv2
import numpy as np
import numpy.random as random

'''
def get_background_img(image_dir, new_width, new_height):
        images = os.listdir(image_dir)
        index = random.randint(0, len(images))
        image_path = os.path.join(image_dir, images[index])
        im = cv2.imread(image_path)
        height, width, _ = im.shape
        while width != new_width or height != new_height:
                #print("get_background_img failed, need:({},{}), get:({},{})".format(new_width, new_height, width, height))
                index = random.randint(0, len(images))
                image_path = os.path.join(image_dir, images[index])
                im = cv2.imread(image_path)
                height, width, _ = im.shape
                
        kernel_1 = random.randint(500, 900) * 2 + 1
        kernel_2 = random.randint(360, 500) * 2 + 1
        im = cv2.blur(im, (kernel_1, kernel_2))

        return im
'''
def get_background_img(image_dir='../../shengna/train/no_target/image'):
        images = os.listdir(image_dir)
        index = random.randint(0, len(images))
        image_path = os.path.join(image_dir, images[index])
        im = cv2.imread(image_path)
        return im, images[index]

def random_flip(img):
    if random.random() > 0.75:
        return img
    flips = [0, 1, -1]
    index = random.randint(0, len(flips))
    flip_type = flips[index]
    img = cv2.flip(img, flip_type)
    return img

def random_rotate(img):
        if random.random() > 0.5:
                return img

        angle = 90
        (h, w) = img.shape[:2]
        M = cv2.getRotationMatrix2D((w / 2, h / 2), -angle, 1.0)
        newW = int((h * np.abs(M[0, 1])) + (w * np.abs(M[0, 0])))
        newH = int((h * np.abs(M[0, 0])) + (w * np.abs(M[0, 1])))
        M[0, 2] += (newW - w) / 2
        M[1, 2] += (newH - h) / 2
        img = cv2.warpAffine(img, M, (newW, newH))
        img = img[:, 1:, :]
        return img

def random_resize(img):
        scale = random.uniform(0.9, 1.1)
        height, width, _ = img.shape
        height = round(height * scale)
        width = round(width * scale)
        img = cv2.resize(img, (width, height))
        return img

def random_mask(img):
        if random.random() > 0.5:
                return img

        height, width, _ = img.shape
        if height < 8 or width < 8:
                return img

        mask_width = random.randint(width//8, width // 2)
        mask_height = random.randint(height//8, height // 2)
        mask_x = random.randint(0, width-mask_width)
        mask_y = random.randint(0, height-mask_height)
        img[mask_y:mask_y+mask_height, mask_x:mask_x+mask_width, :] = np.random.randint(0, 255, (mask_height, mask_width, 3))
        return img

def random_reset_img(img):
        img = random_resize(img)
        #img = random_rotate(img)
        img = random_flip(img)
        #img = random_mask(img)
        return img

def iou(box1,box2):
        bxmin = max(box1[0],box2[0])
        bymin = max(box1[1],box2[1])
        bxmax = min(box1[2],box2[2])
        bymax = min(box1[3],box2[3])

        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        inter_area = max(bxmax-bxmin, 0) * max(bymax-bymin, 0)
        union_area = box1_area + box2_area - inter_area

        return inter_area / union_area, inter_area

def get_type(img_name):
	if 'ss' in img_name:
		return 'ss'
	if 'gx' in img_name:
		return 'gx'
	if 'flv' in img_name:
		return 'flv'
	raise('error img name', img_name)

# root_dir: object images root dir
def create_new_img(min_obj_count, max_obj_count, new_width, new_height, cls2index, root_dir, background_dir):
        image_dirs = os.listdir(root_dir)
        offset = 10
        objs = []
        img_classes = []
        #new_img = get_background_img(background_dir, new_width, new_height)
        new_img, new_image_name = get_background_img()
        img_type = get_type(new_image_name)
        img_height, img_width, _ = new_img.shape
        obj_count = random.randint(min_obj_count, max_obj_count)
        scale = 1.05
        for i in range(obj_count):
                #class_dir_index = random.randint(0, len(image_dirs))
                #img_cls = image_dirs[class_dir_index]
                #img_dir = os.path.join(root_dir, img_cls)
                img_cls = 'target'
                img_dir = os.path.join(root_dir, img_type)
                single_images = os.listdir(img_dir)
                img_index = random.randint(0, len(single_images))
                img_path = os.path.join(img_dir, single_images[img_index])
                obj = cv2.imread(img_path)
                obj_height, obj_width, _ = obj.shape
                #if ((scale * obj_height) >= img_height) or ((scale * obj_width) >= img_width):
                #        print("obj class = {}, path = {}, shape = {}".format(img_cls, img_path, obj.shape))
                #        continue

                obj = random_reset_img(obj)
                obj_height, obj_width, _ = obj.shape
                if ((scale * obj_height) >= img_height) or ((scale * obj_width) >= img_width):
                        print("obj class = {}, path = {}, shape = {}, base img shape = {}".format(img_cls, img_path, obj.shape, new_img.shape))
                        continue

                objs.append(obj)
                img_classes.append(img_cls)

        for i, obj_i in enumerate(objs):
                for j, obj_j in enumerate(objs):
                        i_height, i_width, _ = obj_i.shape
                        j_height, j_width, _ = obj_j.shape
                        i_area = i_height * i_width
                        j_area = j_height * j_width
                        if i_area > j_area:
                                tmp = objs[i]
                                objs[i] = objs[j]
                                objs[j] = tmp
                                tmp_cls = img_classes[i]
                                img_classes[i] = img_classes[j]
                                img_classes[j] = tmp_cls


        centers = []
        widths = []
        heights = []        
        for obj in objs:
                height, width, depth = obj.shape
                half_h = height // 2
                half_w = width // 2
                iou_threshold = 0.1
                loop_count = 0
                while True:
                        loop_count += 1
                        if loop_count % 5 == 4:
                                if loop_count % 10 == 9:
                                        print("loop count = {}, iou_threshold = {}".format(loop_count, iou_threshold))
                                iou_threshold += 0.05

                        bFlag = True
                        try:
                                center_h = random.randint(half_h+1, img_height - half_h - 1)
                                center_w = random.randint(half_w+1, img_width - half_w -1)
                        except:
                                print("random center failed, half_h = {}, half_w = {}, img_width = {}, img_height = {}".format(half_h, half_w, img_width, img_height))
                                raise('error')

                        box1 = [center_w-half_w, center_h-half_h, center_w+half_w, center_h+half_h]
                        for src_center, src_width, src_height in zip(centers, widths, heights):
                                xmin = src_center[0] - (src_width // 2)
                                ymin = src_center[1] - (src_height // 2)
                                xmax = src_center[0] + (src_width // 2)
                                ymax = src_center[1] + (src_height // 2)
                                box2 = [xmin, ymin, xmax, ymax]
                                if iou(box1, box2)[0] > iou_threshold:
                                        bFlag = False
                                        break

                        if bFlag:
                                break


                center = (center_w, center_h)
                centers.append(center)
                widths.append(width)
                heights.append(height)

        bboxes = []
        labels = []
        for obj, img_cls, center in zip(objs, img_classes, centers):
                height, width, depth = obj.shape
                half_h = height // 2
                half_w = width // 2

                mask = 255 * np.ones(obj.shape, obj.dtype)
                start_w, start_h = center
                try:
                        new_img = cv2.seamlessClone(obj, new_img, mask, center, cv2.NORMAL_CLONE)
                except:
                        print('seamless clone failed')
                        continue
            
                start_x = start_w - half_w + offset
                start_y = start_h - half_h + offset
                end_x = start_w + half_w - offset
                end_y = start_h + half_h - offset
                bboxes.append([start_x, start_y, end_x, end_y])
                label = cls2index[img_cls]
                labels.append(label)

        bboxes = np.array(bboxes, dtype=np.float32)
        labels = np.array(labels)
        return new_img, bboxes, labels
