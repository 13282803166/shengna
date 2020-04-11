import inspect

import mmcv
import numpy as np
from numpy import random
import cv2
import math
from mmdet.core.evaluation.bbox_overlaps import bbox_overlaps
from ..registry import PIPELINES
import xml.etree.ElementTree as ET
import os
import time
from icecream import ic

try:
    from imagecorruptions import corrupt
except ImportError:
    corrupt = None

try:
    import albumentations
    from albumentations import Compose
except ImportError:
    albumentations = None
    Compose = None

def show_img(img, bboxes, filename):
    for bbox in bboxes:
        cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 1)
    cv2.imwrite(filename, img)


@PIPELINES.register_module
class Resize(object):
    """Resize images & bbox & mask.

    This transform resizes the input image to some scale. Bboxes and masks are
    then resized with the same scale factor. If the input dict contains the key
    "scale", then the scale in the input dict is used, otherwise the specified
    scale in the init method is used.

    `img_scale` can either be a tuple (single-scale) or a list of tuple
    (multi-scale). There are 3 multiscale modes:
    - `ratio_range` is not None: randomly sample a ratio from the ratio range
        and multiply it with the image scale.
    - `ratio_range` is None and `multiscale_mode` == "range": randomly sample a
        scale from the a range.
    - `ratio_range` is None and `multiscale_mode` == "value": randomly sample a
        scale from multiple scales.

    Args:
        img_scale (tuple or list[tuple]): Images scales for resizing.
        multiscale_mode (str): Either "range" or "value".
        ratio_range (tuple[float]): (min_ratio, max_ratio)
        keep_ratio (bool): Whether to keep the aspect ratio when resizing the
            image.
    """

    def __init__(self,
                 img_scale=None,
                 multiscale_mode='range',
                 ratio_range=None,
                 keep_ratio=True):
        if img_scale is None:
            self.img_scale = None
        else:
            if isinstance(img_scale, list):
                self.img_scale = img_scale
            else:
                self.img_scale = [img_scale]
            assert mmcv.is_list_of(self.img_scale, tuple)

        if ratio_range is not None:
            # mode 1: given a scale and a range of image ratio
            assert len(self.img_scale) == 1
        else:
            # mode 2: given multiple scales or a range of scales
            assert multiscale_mode in ['value', 'range']

        self.multiscale_mode = multiscale_mode
        self.ratio_range = ratio_range
        self.keep_ratio = keep_ratio

    @staticmethod
    def random_select(img_scales):
        assert mmcv.is_list_of(img_scales, tuple)
        scale_idx = np.random.randint(len(img_scales))
        img_scale = img_scales[scale_idx]
        return img_scale, scale_idx

    @staticmethod
    def random_sample(img_scales):
        assert mmcv.is_list_of(img_scales, tuple) and len(img_scales) == 2
        img_scale_long = [max(s) for s in img_scales]
        img_scale_short = [min(s) for s in img_scales]
        long_edge = np.random.randint(
            min(img_scale_long),
            max(img_scale_long) + 1)
        short_edge = np.random.randint(
            min(img_scale_short),
            max(img_scale_short) + 1)
        img_scale = (long_edge, short_edge)
        return img_scale, None

    @staticmethod
    def random_sample_ratio(img_scale, ratio_range):
        assert isinstance(img_scale, tuple) and len(img_scale) == 2
        min_ratio, max_ratio = ratio_range
        assert min_ratio <= max_ratio
        ratio = np.random.random_sample() * (max_ratio - min_ratio) + min_ratio
        scale = int(img_scale[0] * ratio), int(img_scale[1] * ratio)
        return scale, None

    def _random_scale(self, results):
        if self.ratio_range is not None:
            scale, scale_idx = self.random_sample_ratio(
                self.img_scale[0], self.ratio_range)
        elif len(self.img_scale) == 1:
            scale, scale_idx = self.img_scale[0], 0
        elif self.multiscale_mode == 'range':
            scale, scale_idx = self.random_sample(self.img_scale)
        elif self.multiscale_mode == 'value':
            scale, scale_idx = self.random_select(self.img_scale)
        else:
            raise NotImplementedError

        results['scale'] = scale
        results['scale_idx'] = scale_idx

    def _resize_img(self, results):
        if self.keep_ratio:
            img, scale_factor = mmcv.imrescale(
                results['img'], results['scale'], return_scale=True)
        else:
            img, w_scale, h_scale = mmcv.imresize(
                results['img'], results['scale'], return_scale=True)
            scale_factor = np.array([w_scale, h_scale, w_scale, h_scale],
                                    dtype=np.float32)
        results['img'] = img
        results['img_shape'] = img.shape
        results['pad_shape'] = img.shape  # in case that there is no padding
        results['scale_factor'] = scale_factor
        results['keep_ratio'] = self.keep_ratio

    def _resize_bboxes(self, results):
        img_shape = results['img_shape']
        for key in results.get('bbox_fields', []):
            bboxes = results[key] * results['scale_factor']
            bboxes[:, 0::2] = np.clip(bboxes[:, 0::2], 0, img_shape[1] - 1)
            bboxes[:, 1::2] = np.clip(bboxes[:, 1::2], 0, img_shape[0] - 1)
            results[key] = bboxes

    def _resize_masks(self, results):
        for key in results.get('mask_fields', []):
            if results[key] is None:
                continue
            if self.keep_ratio:
                masks = [
                    mmcv.imrescale(
                        mask, results['scale_factor'], interpolation='nearest')
                    for mask in results[key]
                ]
            else:
                mask_size = (results['img_shape'][1], results['img_shape'][0])
                masks = [
                    mmcv.imresize(mask, mask_size, interpolation='nearest')
                    for mask in results[key]
                ]
            results[key] = np.stack(masks)

    def _resize_seg(self, results):
        for key in results.get('seg_fields', []):
            if self.keep_ratio:
                gt_seg = mmcv.imrescale(
                    results[key], results['scale'], interpolation='nearest')
            else:
                gt_seg = mmcv.imresize(
                    results[key], results['scale'], interpolation='nearest')
            results['gt_semantic_seg'] = gt_seg

    def __call__(self, results):
        if 'scale' not in results:
            self._random_scale(results)

        self._resize_img(results)
        self._resize_bboxes(results)
        self._resize_masks(results)
        self._resize_seg(results)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += ('(img_scale={}, multiscale_mode={}, ratio_range={}, '
                     'keep_ratio={})').format(self.img_scale,
                                              self.multiscale_mode,
                                              self.ratio_range,
                                              self.keep_ratio)
        return repr_str



@PIPELINES.register_module
class Different_Size_Resize(object):
    """Resize images & bbox & mask.
    - `ratio_range` is not None: randomly sample a ratio from the ratio range
        and multiply it with the image scale.
    Args:
        ratio_range (tuple[float]): (min_ratio, max_ratio)
    """

    def __init__(self, ratio_range=None, max_width=1920, max_height=1080):
        self.ratio_range = ratio_range
        self.max_width = max_width
        self.max_height = max_height

    @staticmethod
    def random_sample_ratio(img_scale, ratio_range):
        assert isinstance(img_scale, tuple) and len(img_scale) == 2
        min_ratio, max_ratio = ratio_range
        assert min_ratio <= max_ratio
        ratio = np.random.random_sample() * (max_ratio - min_ratio) + min_ratio
        scale = int(img_scale[0] * ratio), int(img_scale[1] * ratio)
        return scale, None

    def _random_scale(self, results):
        img_shape = results['img_shape']
        if len(self.ratio_range) == 1:
            scale, scale_idx = (img_shape[1]*self.ratio_range[0], img_shape[0]*self.ratio_range[0]), 0
        else:
            scale, scale_idx = self.random_sample_ratio((img_shape[1], img_shape[0]), self.ratio_range)
            ratio = random.uniform(self.ratio_range[0], self.ratio_range[1])
            scale = int(ratio*img_shape[1]), int(ratio*img_shape[0])
            scale_idx = None

        results['scale'] = scale
        results['scale_idx'] = scale_idx

    def _resize_img(self, results):
        img, scale_factor = mmcv.imrescale(results['img'], results['scale'], return_scale=True)
        results['img'] = img
        results['img_shape'] = img.shape
        results['pad_shape'] = img.shape  # in case that there is no padding
        results['scale_factor'] = scale_factor
        results['keep_ratio'] = True

    def _resize_bboxes(self, results):
        img_shape = results['img_shape']
        for key in results.get('bbox_fields', []):
            bboxes = results[key] * results['scale_factor']
            bboxes[:, 0::2] = np.clip(bboxes[:, 0::2], 0, img_shape[1] - 1)
            bboxes[:, 1::2] = np.clip(bboxes[:, 1::2], 0, img_shape[0] - 1)
            results[key] = bboxes

    def __call__(self, results):
        '''
        height, width, _ = results['img_shape']
        if height > self.max_height or width > self.max_width:
            width_scale = width / self.max_width
            height_scale = height / self.max_height
            scale =  height_scale if height_scale > width_scale else width_scale

            results['img'] = cv2.resize(results['img'], (int(width/scale), int(height/scale)))
            results['img_shape'] = results['img'].shape
            for key in results.get('bbox_fields', []):
                results[key] = results[key] / scale
        '''

        #if 'scale' not in results:
        self._random_scale(results)
        #else:
        #    results['scale'] = 1.0
        #    results['scale_idx'] = None

        self._resize_img(results)
        self._resize_bboxes(results)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += ('(ratio_range={})').format(self.ratio_range)
        return repr_str


@PIPELINES.register_module
class RandomFlip(object):
    """Flip the image & bbox & mask.

    If the input dict contains the key "flip", then the flag will be used,
    otherwise it will be randomly decided by a ratio specified in the init
    method.

    Args:
        flip_ratio (float, optional): The flipping probability.
    """

    def __init__(self, flip_ratio=None, direction='horizontal'):
        self.flip_ratio = flip_ratio
        self.direction = direction
        if flip_ratio is not None:
            assert flip_ratio >= 0 and flip_ratio <= 1
        assert direction in ['horizontal', 'vertical']

    def bbox_flip(self, bboxes, img_shape, direction):
        """Flip bboxes horizontally.

        Args:
            bboxes(ndarray): shape (..., 4*k)
            img_shape(tuple): (height, width)
        """
        assert bboxes.shape[-1] % 4 == 0
        flipped = bboxes.copy()
        if direction == 'horizontal':
            w = img_shape[1]
            flipped[..., 0::4] = w - bboxes[..., 2::4] - 1
            flipped[..., 2::4] = w - bboxes[..., 0::4] - 1
        elif direction == 'vertical':
            h = img_shape[0]
            flipped[..., 1::4] = h - bboxes[..., 3::4] - 1
            flipped[..., 3::4] = h - bboxes[..., 1::4] - 1
        else:
            raise ValueError(
                'Invalid flipping direction "{}"'.format(direction))
        return flipped

    def __call__(self, results):
        if 'flip' not in results:
            flip = True if np.random.rand() < self.flip_ratio else False
            results['flip'] = flip
        if 'flip_direction' not in results:
            results['flip_direction'] = self.direction

        if results['flip']:
            # flip image
            results['img'] = mmcv.imflip(results['img'], direction=results['flip_direction']) 

            # flip bboxes
            for key in results.get('bbox_fields', []):
                results[key] = self.bbox_flip(results[key], results['img_shape'], results['flip_direction'])
            # flip masks
            for key in results.get('mask_fields', []):
                results[key] = np.stack([
                    mmcv.imflip(mask, direction=results['flip_direction'])
                    for mask in results[key]
                ])

            # flip segs
            for key in results.get('seg_fields', []):
                results[key] = mmcv.imflip(
                    results[key], direction=results['flip_direction'])

        return results

    def __repr__(self):
        return self.__class__.__name__ + '(flip_ratio={})'.format(
            self.flip_ratio)

@PIPELINES.register_module
class RandomVFlip(object):
    """Flip the image & bbox & mask.
    If the input dict contains the key "vflip", then the flag will be used,
    otherwise it will be randomly decided by a ratio specified in the init
    method.
    Args:
        flip_ratio (float, optional): The flipping probability.
    """

    def __init__(self, flip_ratio=None):
        self.flip_ratio = flip_ratio
        if flip_ratio is not None:
            assert flip_ratio >= 0 and flip_ratio <= 1

    def bbox_flip(self, bboxes, img_shape):
        """Flip bboxes horizontally.
        Args:
            bboxes(ndarray): shape (..., 4*k)
            img_shape(tuple): (height, width)
        """
        assert bboxes.shape[-1] % 4 == 0
        h = img_shape[0]
        flipped = bboxes.copy()
        flipped[..., 1::4] = h - bboxes[..., 3::4] - 1
        flipped[..., 3::4] = h - bboxes[..., 1::4] - 1
        return flipped

    def __call__(self, results):
        if 'vflip' not in results:
            flip = True if np.random.rand() < self.flip_ratio else False
            results['vflip'] = flip
        if results['vflip']:
            # flip image
            results['img'] = mmcv.imflip(results['img'], direction="vertical")
            # flip bboxes
            for key in results.get('bbox_fields', []):
                results[key] = self.bbox_flip(results[key],
                                              results['img_shape'])
        return results

    def __repr__(self):
        return self.__class__.__name__ + '(flip_ratio={})'.format(
            self.flip_ratio)

@PIPELINES.register_module
class GaussianBlur(object):
    def __init__(self, ratio=0.5, kernels=[3, 5, 7, 9, 11]):
        self.ratio = ratio
        self.kernels = kernels

    def __call__(self, results):
        if random.random() > self.ratio:
            return results

        kernel_index = random.randint(0, len(self.kernels))
        kernel_size = self.kernels[kernel_index]
        results['img'] = cv2.GaussianBlur(results['img'], (kernel_size, kernel_size), 0)
        return results

    def __repr__(self):
        return self.__class__.__name__ + '(ratio={})'.format(self.ratio)

@PIPELINES.register_module
class ContrastAndBrightness(object):
    def __init__(self, ratio=0.5, min_alpha=0.8, max_beta=25):
        self.ratio = ratio
        self.min_alpha = min_alpha
        self.max_beta = max_beta

    def __call__(self, results):
        if random.random() > self.ratio:
            return results

        img = results['img']
        blank = np.zeros(img.shape, img.dtype)
        alpha = random.uniform(self.min_alpha, 1.0)
        beta = random.randint(0, self.max_beta)
        results['img'] = cv2.addWeighted(img, alpha, blank, 1-alpha, beta)
        return results

    def __repr__(self):
        return self.__class__.__name__ + '(ratio={})'.format(self.ratio)

@PIPELINES.register_module
class Mixup(object):
    def __init__(self, classes, ratio=0.5, data_dir='../data/'):
        self.ratio = ratio
        self.data_dir = data_dir
        self.cat2label = {cat: i + 1 for i, cat in enumerate(classes)}

    def get_mixup_image(self, src_name, width, height):
        trainset = open(os.path.join(self.data_dir, 'ImageSets/Main/train.txt'))
        images = trainset.read().splitlines()
        while True:
            idx = random.randint(0, len(images))
            mixup_img = images[idx]
            if mixup_img == 'custom':
                continue
            if mixup_img != src_name:
                break
        img_path = os.path.join(self.data_dir, 'JPEGImages/{}.jpg'.format(mixup_img))
        ann_path = os.path.join(self.data_dir, 'Annotations/{}.xml'.format(mixup_img))
        mix_img = cv2.imread(img_path)
        img_height, img_width, _ = mix_img.shape
        if img_width < width:
            mix_img = cv2.copyMakeBorder(mix_img, 0, 0, 0, width-img_width, cv2.BORDER_CONSTANT, value=(0,0,0))
        if img_height < height:
            mix_img = cv2.copyMakeBorder(mix_img, 0, height-img_height, 0, 0, cv2.BORDER_CONSTANT, value=(0,0,0)) 

        root = ET.parse(ann_path)
        objs = root.findall('object')
        bboxes = []
        labels = []
        for obj in objs:
            cls = obj.find('name').text
            label = self.cat2label[cls]
            bndbox = obj.find('bndbox')
            xmin = int(float(bndbox.find('xmin').text))
            xmax = int(float(bndbox.find('xmax').text))
            ymin = int(float(bndbox.find('ymin').text))
            ymax = int(float(bndbox.find('ymax').text))
            bboxes.append([xmin, ymin, xmax, ymax])
            labels.append(label)

        return mix_img, bboxes, labels

    def __call__(self, results):
        if random.random() > self.ratio:
            return results

        img = results['img']
        src_img = results['img_info']['filename'].split('/')[-1].split('.')[0]
        mix_img, bboxes, labels = self.get_mixup_image(src_img, img.shape[1], img.shape[0])

        bboxes = np.array(bboxes).astype('float32')
        labels = np.array(labels)

        results['gt_bboxes'] = np.vstack((results['gt_bboxes'], bboxes))
        results['gt_labels'] = np.hstack((results['gt_labels'], labels))
        img = img*0.5 + mix_img*0.5
        results['img'] = img
        return results

    def __repr__(self):
        return self.__class__.__name__ + '(ratio={})'.format(self.ratio)

@PIPELINES.register_module
class Diff_Mixup(object):
    def __init__(self, classes, ratio=0.5, data_dir='../data/'):
        self.ratio = ratio
        self.data_dir = data_dir
        self.cat2label = {cat: i + 1 for i, cat in enumerate(classes)}

    def diff_get_mixup_image(self, src_name, width, height, src_img):
        trainset = open(os.path.join(self.data_dir, 'ImageSets/Main/train.txt'))
        images = trainset.read().splitlines()
        while True:
            idx = random.randint(0, len(images))
            mixup_img = images[idx]
            if mixup_img != src_name:
                break
        img_path = os.path.join(self.data_dir, 'JPEGImages/{}.jpg'.format(mixup_img))
        ann_path = os.path.join(self.data_dir, 'Annotations/{}.xml'.format(mixup_img))
        mix_img = cv2.imread(img_path)

        root = ET.parse(ann_path)
        objs = root.findall('object')
        bboxes = []
        labels = []
        for obj in objs:
            cls = obj.find('name').text
            label = self.cat2label[cls]
            bndbox = obj.find('bndbox')
            xmin = int(float(bndbox.find('xmin').text))
            xmax = int(float(bndbox.find('xmax').text))
            ymin = int(float(bndbox.find('ymin').text))
            ymax = int(float(bndbox.find('ymax').text))
            bboxes.append([xmin, ymin, xmax, ymax])
            labels.append(label)

        img_height, img_width, _ = mix_img.shape
        if img_width < width:
            mix_img = cv2.copyMakeBorder(mix_img, 0, 0, 0, width-img_width, cv2.BORDER_CONSTANT, value=(0,0,0))
        elif img_width > width:
            src_img = cv2.copyMakeBorder(src_img, 0, 0, 0, img_width-width, cv2.BORDER_CONSTANT, value=(0,0,0))

        if img_height < height:
            mix_img = cv2.copyMakeBorder(mix_img, 0, height-img_height, 0, 0, cv2.BORDER_CONSTANT, value=(0,0,0)) 
        elif img_height > height:
            src_img = cv2.copyMakeBorder(src_img, 0, img_height-height, 0, 0, cv2.BORDER_CONSTANT, value=(0,0,0))

        return mix_img, bboxes, labels, src_img

    def __call__(self, results):
        if random.random() > self.ratio:
            return results

        img = results['img']
        src_img = results['img_info']['filename'].split('/')[-1].split('.')[0]
        mix_img, bboxes, labels, img = self.diff_get_mixup_image(src_img, img.shape[1], img.shape[0], img)

        bboxes = np.array(bboxes).astype('float32')
        labels = np.array(labels)

        results['gt_bboxes'] = np.vstack((results['gt_bboxes'], bboxes))
        results['gt_labels'] = np.hstack((results['gt_labels'], labels))
        img = img*0.5 + mix_img*0.5
        results['img'] = img

        return results

    def __repr__(self):
        return self.__class__.__name__ + '(ratio={})'.format(self.ratio)



@PIPELINES.register_module
class RandomTranslate(object):
    def __init__(self, ratio=0.5):
        self.ratio = ratio

    def random_translate(self, img, bboxes):
        h_img, w_img, _ = img.shape
        min_x = min_y = 9999
        max_x = max_y = 0
        for key in bboxes:
            bbox = bboxes[key]
            if len(bbox) == 0:
                continue
            min_x = min_x if min_x < np.min(bbox[:, 0], axis=0) else np.min(bbox[:, 0], axis=0)
            min_y = min_y if min_y < np.min(bbox[:, 1], axis=0) else np.min(bbox[:, 1], axis=0)            
            max_x = max_x if max_x > np.max(bbox[:, 2], axis=0) else np.max(bbox[:, 2], axis=0)
            max_y = max_y if max_y > np.max(bbox[:, 3], axis=0) else np.max(bbox[:, 3], axis=0)            

        max_l_trans = min_x
        max_u_trans = min_y
        max_r_trans = w_img - max_x
        max_d_trans = h_img - max_y
 
        tx = random.uniform(-(max_l_trans - 1), (max_r_trans - 1))
        ty = random.uniform(-(max_u_trans - 1), (max_d_trans - 1))
 
        M = np.array([[1, 0, tx], [0, 1, ty]])
        img = cv2.warpAffine(img, M, (w_img, h_img))

        for key in bboxes:
            bboxes[key][:, [0, 2]] = bboxes[key][:, [0, 2]] + tx
            bboxes[key][:, [1, 3]] = bboxes[key][:, [1, 3]] + ty

        return img, bboxes

    def __call__(self, results):
        if random.random() > self.ratio:
            return results

        bboxes = {}
        for key in results.get('bbox_fields', []):
            bboxes[key] = results[key]

        results['img'], bboxes = self.random_translate(results['img'], bboxes)
        for key in bboxes:
            results[key] = bboxes[key]

        return results

    def __repr__(self):
        return self.__class__.__name__ + '(ratio={})'.format(self.ratio)



@PIPELINES.register_module
class GridMask(object):
    def __init__(self, ratio=0.25):
        self.ratio = ratio

    def __call__(self, results):
        if random.random() > self.ratio:
            return results

        bboxes = results['gt_bboxes']
        obj_ws = bboxes[:, 2] - bboxes[:, 0]
        obj_hs = bboxes[:, 3] - bboxes[:, 1]
        w_min = int(obj_ws.min())
        h_min = int(obj_hs.min())

        if w_min < 8 or h_min < 8:
            return results

        x = random.randint(w_min // 2, w_min // 1.5)
        y = random.randint(h_min // 2, h_min // 1.5)
        r_x = random.randint(w_min // 1.5, w_min // 1.25)
        r_y = random.randint(h_min // 1.5, h_min // 1.25)
        img = results['img']
        h, w, _ = img.shape
        rows = h // (y + r_y) + 1
        cols = w // (x + r_x) + 1
        for i in range(rows):
            y_start = i * (y + r_y)
            y_end = y_start + y
            for j in range(cols):
                x_start = j * (x + r_x)
                x_end = x_start + x
                if i == rows - 1 and j == cols - 1:
                    h - y_start
                    img[y_start:, x_start:, :] = np.random.randint(0, 255, (h-y_start, w-x_start, 3))
                elif i == rows - 1:
                    img[y_start:, x_start:x_end, :] = np.random.randint(0, 255, (h-y_start, x_end-x_start, 3))
                elif j == cols - 1:
                    img[y_start:y_end, x_start:, :] = np.random.randint(0, 255, (y_end-y_start, w-x_start, 3))
                else:
                    img[y_start:y_end, x_start:x_end, :] = np.random.randint(0, 255, (y_end-y_start, x_end-x_start, 3))

        results['img'] = img
        return results

    def __repr__(self):
        return self.__class__.__name__ + '(ratio={})'.format(self.ratio)

@PIPELINES.register_module
class BBoxJitter(object):
    """
    bbox jitter
    Args:
        min (int, optional): min scale
        max (int, optional): max scale
        ## origin w scale
    """

    def __init__(self, min=0.9, max=1.1):
        self.min_scale = min
        self.max_scale = max
        self.count = 0
        ic("USE BBOX_JITTER")
        ic(min, max)

    def bbox_jitter(self, bboxes, img_shape):
        """Flip bboxes horizontally.
        Args:
            bboxes(ndarray): shape (..., 4*k)
            img_shape(tuple): (height, width)
        """
        assert bboxes.shape[-1] % 4 == 0
        if len(bboxes) == 0:
            return bboxes

        jitter_bboxes = []
        for box in bboxes:
            w = box[2] - box[0]
            h = box[3] - box[1]
            center_x = (box[0] + box[2]) / 2
            center_y = (box[1] + box[3]) / 2
            scale = np.random.uniform(self.min_scale, self.max_scale)
            w = w * scale / 2.
            h = h * scale / 2.
            xmin = center_x - w
            ymin = center_y - h
            xmax = center_x + w
            ymax = center_y + h
            box2 = np.array([xmin, ymin, xmax, ymax], dtype=np.float32)
            jitter_bboxes.append(box2)
        jitter_bboxes = np.array(jitter_bboxes, dtype=np.float32)
        jitter_bboxes[:, 0::2] = np.clip(jitter_bboxes[:, 0::2], 0, img_shape[1] - 1)
        jitter_bboxes[:, 1::2] = np.clip(jitter_bboxes[:, 1::2], 0, img_shape[0] - 1)
        return jitter_bboxes

    def __call__(self, results):
        for key in results.get('bbox_fields', []):
            results[key] = self.bbox_jitter(results[key], results['img_shape'])

        return results

    def __repr__(self):
        return self.__class__.__name__ + '(bbox_jitter={}-{})'.format(
            self.min_scale, self.max_scale)


@PIPELINES.register_module
class AffineTransformation(object):
    def __init__(self, angles=[-3, -2, -1, 1, 2, 3], ratio=0.5):
        self.angles = angles
        self.current_index = 0
        self.ratio=ratio

    def get_angle(self):
        angle = self.angles[self.current_index]
        self.current_index += 1
        if self.current_index >= len(self.angles):
            self.current_index = 0
        return angle

    def rotate_img(self, img, angle, scale=1.0):
        w = img.shape[1]
        h = img.shape[0]
        rangle = np.deg2rad(angle) # angle in radians
        # now calculate new image width and height
        nw = (abs(np.sin(rangle)*h) + abs(np.cos(rangle)*w))*scale
        nh = (abs(np.cos(rangle)*h) + abs(np.sin(rangle)*w))*scale
        # ask OpenCV for the rotation matrix
        rot_mat = cv2.getRotationMatrix2D((nw*0.5, nh*0.5), angle, scale)
        # calculate the move from the old center to the new center combined
        # with the rotation
        rot_move = np.dot(rot_mat, np.array([(nw-w)*0.5, (nh-h)*0.5,0]))
        # the move only affects the translation, so update the translation
        # part of the transform
        rot_mat[0,2] += rot_move[0]
        rot_mat[1,2] += rot_move[1]
        # 仿射变换
        rot_img = cv2.warpAffine(img, rot_mat, (int(math.ceil(nw)), int(math.ceil(nh))), flags=cv2.INTER_LANCZOS4)
        return rot_img, rot_mat

    def crop_img(self, img, src_w, src_h):
        nh, nw, _ = img.shape
        w_start = int((nw - src_w) / 2)
        h_start = int((nh - src_h) / 2)
        w_end = src_w + w_start
        h_end = src_h + h_start
        img = img[h_start:h_end, w_start:w_end, :]
        return img, w_start, h_start

    def rotate_bbox(self, w_start, h_start, rot_mat, bboxes, angle):
        rot_bboxes = list()
        for bbox in bboxes:
            xmin = bbox[0]
            ymin = bbox[1]
            xmax = bbox[2]
            ymax = bbox[3]
            obj_w = xmax - xmin
            obj_h = ymax - ymin

            point1 = np.dot(rot_mat, np.array([(xmin+xmax)/2, ymin, 1]))
            point2 = np.dot(rot_mat, np.array([xmax, (ymin+ymax)/2, 1]))
            point3 = np.dot(rot_mat, np.array([(xmin+xmax)/2, ymax, 1]))
            point4 = np.dot(rot_mat, np.array([xmin, (ymin+ymax)/2, 1]))
		
            concat = np.vstack((point1, point2, point3, point4))
            concat = concat.astype(np.int32)
            rx, ry, rw, rh = cv2.boundingRect(concat)
            rx_min = rx - w_start - abs(angle)
            ry_min = ry - h_start - abs(angle)
            rx_max = rx+rw - w_start + abs(angle)
            ry_max = ry+rh - h_start + abs(angle)

            rot_bboxes.append([rx_min, ry_min, rx_max, ry_max])

        rot_bboxes = np.array(rot_bboxes).astype('float32')
        return rot_bboxes

    def __call__(self, results):
        if random.random() > self.ratio:
            return results

        angle = self.get_angle()
        img = results['img']
        rot_img, rot_mat = self.rotate_img(img, angle)
        new_img, w_start, h_start = self.crop_img(rot_img, img.shape[1], img.shape[0])
        results['img'] = new_img
        for key in results.get('bbox_fields', []):
            results[key] = self.rotate_bbox(w_start, h_start, rot_mat, results[key], angle)

        return results 

    def __repr__(self):
        return self.__class__.__name__ + 'angles = {}'.format(self.angles)

@PIPELINES.register_module
class Pad(object):
    """Pad the image & mask.

    There are two padding modes: (1) pad to a fixed size and (2) pad to the
    minimum size that is divisible by some number.

    Args:
        size (tuple, optional): Fixed padding size.
        size_divisor (int, optional): The divisor of padded size.
        pad_val (float, optional): Padding value, 0 by default.
    """

    def __init__(self, size=None, size_divisor=None, pad_val=0):
        self.size = size
        self.size_divisor = size_divisor
        self.pad_val = pad_val
        # only one of size and size_divisor should be valid
        assert size is not None or size_divisor is not None
        assert size is None or size_divisor is None

    def _pad_img(self, results):
        if self.size is not None:
            padded_img = mmcv.impad(results['img'], self.size, self.pad_val)
        elif self.size_divisor is not None:
            padded_img = mmcv.impad_to_multiple(
                results['img'], self.size_divisor, pad_val=self.pad_val)
        results['img'] = padded_img
        results['pad_shape'] = padded_img.shape
        results['pad_fixed_size'] = self.size
        results['pad_size_divisor'] = self.size_divisor

    def _pad_masks(self, results):
        pad_shape = results['pad_shape'][:2]
        for key in results.get('mask_fields', []):
            padded_masks = [
                mmcv.impad(mask, pad_shape, pad_val=self.pad_val)
                for mask in results[key]
            ]
            if padded_masks:
                results[key] = np.stack(padded_masks, axis=0)
            else:
                results[key] = np.empty((0, ) + pad_shape, dtype=np.uint8)

    def _pad_seg(self, results):
        for key in results.get('seg_fields', []):
            results[key] = mmcv.impad(results[key], results['pad_shape'][:2])

    def __call__(self, results):
        self._pad_img(results)
        self._pad_masks(results)
        self._pad_seg(results)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += '(size={}, size_divisor={}, pad_val={})'.format(
            self.size, self.size_divisor, self.pad_val)
        return repr_str


@PIPELINES.register_module
class Normalize(object):
    """Normalize the image.

    Args:
        mean (sequence): Mean values of 3 channels.
        std (sequence): Std values of 3 channels.
        to_rgb (bool): Whether to convert the image from BGR to RGB,
            default is true.
    """

    def __init__(self, mean, std, to_rgb=True):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.to_rgb = to_rgb

    def __call__(self, results):
        results['img'] = mmcv.imnormalize(results['img'], self.mean, self.std,
                                          self.to_rgb)
        results['img_norm_cfg'] = dict(
            mean=self.mean, std=self.std, to_rgb=self.to_rgb)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += '(mean={}, std={}, to_rgb={})'.format(
            self.mean, self.std, self.to_rgb)
        return repr_str


@PIPELINES.register_module
class RandomCrop(object):
    """Random crop the image & bboxes & masks.

    Args:
        crop_size (tuple): Expected size after cropping, (h, w).
    """

    def __init__(self, crop_size):
        self.crop_size = crop_size

    def __call__(self, results):
        img = results['img']
        margin_h = max(img.shape[0] - self.crop_size[0], 0)
        margin_w = max(img.shape[1] - self.crop_size[1], 0)
        offset_h = np.random.randint(0, margin_h + 1)
        offset_w = np.random.randint(0, margin_w + 1)
        crop_y1, crop_y2 = offset_h, offset_h + self.crop_size[0]
        crop_x1, crop_x2 = offset_w, offset_w + self.crop_size[1]

        # crop the image
        img = img[crop_y1:crop_y2, crop_x1:crop_x2, ...]
        img_shape = img.shape
        results['img'] = img
        results['img_shape'] = img_shape

        # crop bboxes accordingly and clip to the image boundary
        for key in results.get('bbox_fields', []):
            bbox_offset = np.array([offset_w, offset_h, offset_w, offset_h],
                                   dtype=np.float32)
            bboxes = results[key] - bbox_offset
            bboxes[:, 0::2] = np.clip(bboxes[:, 0::2], 0, img_shape[1] - 1)
            bboxes[:, 1::2] = np.clip(bboxes[:, 1::2], 0, img_shape[0] - 1)
            results[key] = bboxes

        # crop semantic seg
        for key in results.get('seg_fields', []):
            results[key] = results[key][crop_y1:crop_y2, crop_x1:crop_x2]

        # filter out the gt bboxes that are completely cropped
        if 'gt_bboxes' in results:
            gt_bboxes = results['gt_bboxes']
            valid_inds = (gt_bboxes[:, 2] > gt_bboxes[:, 0]) & (
                gt_bboxes[:, 3] > gt_bboxes[:, 1])
            # if no gt bbox remains after cropping, just skip this image
            if not np.any(valid_inds):
                return None
            results['gt_bboxes'] = gt_bboxes[valid_inds, :]
            if 'gt_labels' in results:
                results['gt_labels'] = results['gt_labels'][valid_inds]

            # filter and crop the masks
            if 'gt_masks' in results:
                valid_gt_masks = []
                for i in np.where(valid_inds)[0]:
                    gt_mask = results['gt_masks'][i][crop_y1:crop_y2,
                                                     crop_x1:crop_x2]
                    valid_gt_masks.append(gt_mask)
                results['gt_masks'] = np.stack(valid_gt_masks)

        return results

    def __repr__(self):
        return self.__class__.__name__ + '(crop_size={})'.format(
            self.crop_size)


@PIPELINES.register_module
class SegRescale(object):
    """Rescale semantic segmentation maps.

    Args:
        scale_factor (float): The scale factor of the final output.
    """

    def __init__(self, scale_factor=1):
        self.scale_factor = scale_factor

    def __call__(self, results):
        for key in results.get('seg_fields', []):
            if self.scale_factor != 1:
                results[key] = mmcv.imrescale(
                    results[key], self.scale_factor, interpolation='nearest')
        return results

    def __repr__(self):
        return self.__class__.__name__ + '(scale_factor={})'.format(
            self.scale_factor)


@PIPELINES.register_module
class PhotoMetricDistortion(object):
    """Apply photometric distortion to image sequentially, every transformation
    is applied with a probability of 0.5. The position of random contrast is in
    second or second to last.

    1. random brightness
    2. random contrast (mode 0)
    3. convert color from BGR to HSV
    4. random saturation
    5. random hue
    6. convert color from HSV to BGR
    7. random contrast (mode 1)
    8. randomly swap channels

    Args:
        brightness_delta (int): delta of brightness.
        contrast_range (tuple): range of contrast.
        saturation_range (tuple): range of saturation.
        hue_delta (int): delta of hue.
    """

    def __init__(self,
                 brightness_delta=32,
                 contrast_range=(0.5, 1.5),
                 saturation_range=(0.5, 1.5),
                 hue_delta=18):
        self.brightness_delta = brightness_delta
        self.contrast_lower, self.contrast_upper = contrast_range
        self.saturation_lower, self.saturation_upper = saturation_range
        self.hue_delta = hue_delta

    def __call__(self, results):
        img = results['img']
        # random brightness
        if random.randint(2):
            delta = random.uniform(-self.brightness_delta,
                                   self.brightness_delta)
            img += delta

        # mode == 0 --> do random contrast first
        # mode == 1 --> do random contrast last
        mode = random.randint(2)
        if mode == 1:
            if random.randint(2):
                alpha = random.uniform(self.contrast_lower,
                                       self.contrast_upper)
                img *= alpha

        # convert color from BGR to HSV
        img = mmcv.bgr2hsv(img)

        # random saturation
        if random.randint(2):
            img[..., 1] *= random.uniform(self.saturation_lower,
                                          self.saturation_upper)

        # random hue
        if random.randint(2):
            img[..., 0] += random.uniform(-self.hue_delta, self.hue_delta)
            img[..., 0][img[..., 0] > 360] -= 360
            img[..., 0][img[..., 0] < 0] += 360

        # convert color from HSV to BGR
        img = mmcv.hsv2bgr(img)

        # random contrast
        if mode == 0:
            if random.randint(2):
                alpha = random.uniform(self.contrast_lower,
                                       self.contrast_upper)
                img *= alpha

        # randomly swap channels
        if random.randint(2):
            img = img[..., random.permutation(3)]

        results['img'] = img
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += ('(brightness_delta={}, contrast_range={}, '
                     'saturation_range={}, hue_delta={})').format(
                         self.brightness_delta, self.contrast_range,
                         self.saturation_range, self.hue_delta)
        return repr_str


@PIPELINES.register_module
class Expand(object):
    """Random expand the image & bboxes.

    Randomly place the original image on a canvas of 'ratio' x original image
    size filled with mean values. The ratio is in the range of ratio_range.

    Args:
        mean (tuple): mean value of dataset.
        to_rgb (bool): if need to convert the order of mean to align with RGB.
        ratio_range (tuple): range of expand ratio.
        prob (float): probability of applying this transformation
    """

    def __init__(self,
                 mean=(0, 0, 0),
                 to_rgb=True,
                 ratio_range=(1, 4),
                 seg_ignore_label=None,
                 prob=0.5):
        self.to_rgb = to_rgb
        self.ratio_range = ratio_range
        if to_rgb:
            self.mean = mean[::-1]
        else:
            self.mean = mean
        self.min_ratio, self.max_ratio = ratio_range
        self.seg_ignore_label = seg_ignore_label
        self.prob = prob

    def __call__(self, results):
        if random.uniform(0, 1) > self.prob:
            return results

        img, boxes = [results[k] for k in ('img', 'gt_bboxes')]

        h, w, c = img.shape
        ratio = random.uniform(self.min_ratio, self.max_ratio)
        expand_img = np.full((int(h * ratio), int(w * ratio), c),
                             self.mean).astype(img.dtype)
        left = int(random.uniform(0, w * ratio - w))
        top = int(random.uniform(0, h * ratio - h))
        expand_img[top:top + h, left:left + w] = img
        boxes = boxes + np.tile((left, top), 2).astype(boxes.dtype)

        results['img'] = expand_img
        results['gt_bboxes'] = boxes

        if 'gt_masks' in results:
            expand_gt_masks = []
            for mask in results['gt_masks']:
                expand_mask = np.full((int(h * ratio), int(w * ratio)),
                                      0).astype(mask.dtype)
                expand_mask[top:top + h, left:left + w] = mask
                expand_gt_masks.append(expand_mask)
            results['gt_masks'] = np.stack(expand_gt_masks)

        # not tested
        if 'gt_semantic_seg' in results:
            assert self.seg_ignore_label is not None
            gt_seg = results['gt_semantic_seg']
            expand_gt_seg = np.full((int(h * ratio), int(w * ratio)),
                                    self.seg_ignore_label).astype(gt_seg.dtype)
            expand_gt_seg[top:top + h, left:left + w] = gt_seg
            results['gt_semantic_seg'] = expand_gt_seg
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += '(mean={}, to_rgb={}, ratio_range={}, ' \
                    'seg_ignore_label={})'.format(
                        self.mean, self.to_rgb, self.ratio_range,
                        self.seg_ignore_label)
        return repr_str


@PIPELINES.register_module
class MinIoURandomCrop(object):
    """Random crop the image & bboxes, the cropped patches have minimum IoU
    requirement with original image & bboxes, the IoU threshold is randomly
    selected from min_ious.

    Args:
        min_ious (tuple): minimum IoU threshold for all intersections with
        bounding boxes
        min_crop_size (float): minimum crop's size (i.e. h,w := a*h, a*w,
        where a >= min_crop_size).
    """

    def __init__(self, min_ious=(0.1, 0.3, 0.5, 0.7, 0.9), min_crop_size=0.3):
        # 1: return ori img
        self.sample_mode = (1, *min_ious, 0)
        self.min_crop_size = min_crop_size

    def __call__(self, results):
        img, boxes, labels = [
            results[k] for k in ('img', 'gt_bboxes', 'gt_labels')
        ]
        h, w, c = img.shape
        while True:
            mode = random.choice(self.sample_mode)
            if mode == 1:
                return results

            min_iou = mode
            for i in range(50):
                new_w = random.uniform(self.min_crop_size * w, w)
                new_h = random.uniform(self.min_crop_size * h, h)

                # h / w in [0.5, 2]
                if new_h / new_w < 0.5 or new_h / new_w > 2:
                    continue

                left = random.uniform(w - new_w)
                top = random.uniform(h - new_h)

                patch = np.array(
                    (int(left), int(top), int(left + new_w), int(top + new_h)))
                overlaps = bbox_overlaps(
                    patch.reshape(-1, 4), boxes.reshape(-1, 4)).reshape(-1)
                if overlaps.min() < min_iou:
                    continue

                # center of boxes should inside the crop img
                center = (boxes[:, :2] + boxes[:, 2:]) / 2
                mask = ((center[:, 0] > patch[0]) * (center[:, 1] > patch[1]) *
                        (center[:, 0] < patch[2]) * (center[:, 1] < patch[3]))
                if not mask.any():
                    continue
                boxes = boxes[mask]
                labels = labels[mask]

                # adjust boxes
                img = img[patch[1]:patch[3], patch[0]:patch[2]]
                boxes[:, 2:] = boxes[:, 2:].clip(max=patch[2:])
                boxes[:, :2] = boxes[:, :2].clip(min=patch[:2])
                boxes -= np.tile(patch[:2], 2)

                results['img'] = img
                results['gt_bboxes'] = boxes
                results['gt_labels'] = labels

                if 'gt_masks' in results:
                    valid_masks = [
                        results['gt_masks'][i] for i in range(len(mask))
                        if mask[i]
                    ]
                    results['gt_masks'] = np.stack([
                        gt_mask[patch[1]:patch[3], patch[0]:patch[2]]
                        for gt_mask in valid_masks
                    ])

                # not tested
                if 'gt_semantic_seg' in results:
                    results['gt_semantic_seg'] = results['gt_semantic_seg'][
                        patch[1]:patch[3], patch[0]:patch[2]]
                return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += '(min_ious={}, min_crop_size={})'.format(
            self.min_ious, self.min_crop_size)
        return repr_str


@PIPELINES.register_module
class Corrupt(object):

    def __init__(self, ratio=0.8):
        #self.corruptions = ['gaussian_noise', 'shot_noise', 'impulse_noise', 'motion_blur', 'fog', 'brightness',
        #                    'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression', 'gaussian_blur', 'spatter', 'saturate']

        self.corruptions = ['gaussian_noise', 'impulse_noise', 'fog', 'brightness', 
                            'contrast', 'pixelate', 'jpeg_compression', 'gaussian_blur', 'spatter']                        
 

        self.severities = [1, 2]
        self.ratio = ratio

    def __call__(self, results):
        if corrupt is None:
            raise RuntimeError('imagecorruptions is not installed')

        if random.random() > self.ratio:
            return results

        index = random.randint(len(self.corruptions))
        corruption = self.corruptions[index]
        severity = self.severities[random.randint(0,2)]
        results['img'] = corrupt(results['img'].astype(np.uint8), corruption_name=corruption, severity=severity)

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += '(corruption={}, severity={})'.format(
            self.corruptions, self.severitys)
        return repr_str


@PIPELINES.register_module
class Albu(object):

    def __init__(self,
                 transforms,
                 bbox_params=None,
                 keymap=None,
                 update_pad_shape=False,
                 skip_img_without_anno=False):
        """
        Adds custom transformations from Albumentations lib.
        Please, visit `https://albumentations.readthedocs.io`
        to get more information.

        transforms (list): list of albu transformations
        bbox_params (dict): bbox_params for albumentation `Compose`
        keymap (dict): contains {'input key':'albumentation-style key'}
        skip_img_without_anno (bool): whether to skip the image
                                      if no ann left after aug
        """
        if Compose is None:
            raise RuntimeError('albumentations is not installed')

        self.transforms = transforms
        self.filter_lost_elements = False
        self.update_pad_shape = update_pad_shape
        self.skip_img_without_anno = skip_img_without_anno

        # A simple workaround to remove masks without boxes
        if (isinstance(bbox_params, dict) and 'label_fields' in bbox_params
                and 'filter_lost_elements' in bbox_params):
            self.filter_lost_elements = True
            self.origin_label_fields = bbox_params['label_fields']
            bbox_params['label_fields'] = ['idx_mapper']
            del bbox_params['filter_lost_elements']

        self.bbox_params = (
            self.albu_builder(bbox_params) if bbox_params else None)
        self.aug = Compose([self.albu_builder(t) for t in self.transforms],
                           bbox_params=self.bbox_params)

        if not keymap:
            self.keymap_to_albu = {
                'img': 'image',
                'gt_masks': 'masks',
                'gt_bboxes': 'bboxes'
            }
        else:
            self.keymap_to_albu = keymap
        self.keymap_back = {v: k for k, v in self.keymap_to_albu.items()}

    def albu_builder(self, cfg):
        """Import a module from albumentations.
        Inherits some of `build_from_cfg` logic.

        Args:
            cfg (dict): Config dict. It should at least contain the key "type".
        Returns:
            obj: The constructed object.
        """
        assert isinstance(cfg, dict) and "type" in cfg
        args = cfg.copy()

        obj_type = args.pop("type")
        if mmcv.is_str(obj_type):
            if albumentations is None:
                raise RuntimeError('albumentations is not installed')
            obj_cls = getattr(albumentations, obj_type)
        elif inspect.isclass(obj_type):
            obj_cls = obj_type
        else:
            raise TypeError(
                'type must be a str or valid type, but got {}'.format(
                    type(obj_type)))

        if 'transforms' in args:
            args['transforms'] = [
                self.albu_builder(transform)
                for transform in args['transforms']
            ]

        return obj_cls(**args)

    @staticmethod
    def mapper(d, keymap):
        """
        Dictionary mapper.
        Renames keys according to keymap provided.

        Args:
            d (dict): old dict
            keymap (dict): {'old_key':'new_key'}
        Returns:
            dict: new dict.
        """
        updated_dict = {}
        for k, v in zip(d.keys(), d.values()):
            new_k = keymap.get(k, k)
            updated_dict[new_k] = d[k]
        return updated_dict

    def __call__(self, results):
        # dict to albumentations format
        results = self.mapper(results, self.keymap_to_albu)

        if 'bboxes' in results:
            # to list of boxes
            if isinstance(results['bboxes'], np.ndarray):
                results['bboxes'] = [x for x in results['bboxes']]
            # add pseudo-field for filtration
            if self.filter_lost_elements:
                results['idx_mapper'] = np.arange(len(results['bboxes']))

        results = self.aug(**results)

        if 'bboxes' in results:
            if isinstance(results['bboxes'], list):
                results['bboxes'] = np.array(
                    results['bboxes'], dtype=np.float32)
            results['bboxes'] = results['bboxes'].reshape(-1, 4)

            # filter label_fields
            if self.filter_lost_elements:

                results['idx_mapper'] = np.arange(len(results['bboxes']))

                for label in self.origin_label_fields:
                    results[label] = np.array(
                        [results[label][i] for i in results['idx_mapper']])
                if 'masks' in results:
                    results['masks'] = np.array(
                        [results['masks'][i] for i in results['idx_mapper']])

                if (not len(results['idx_mapper'])
                        and self.skip_img_without_anno):
                    return None

        if 'gt_labels' in results:
            if isinstance(results['gt_labels'], list):
                results['gt_labels'] = np.array(results['gt_labels'])
            results['gt_labels'] = results['gt_labels'].astype(np.int64)

        # back to the original format
        results = self.mapper(results, self.keymap_back)

        # update final shape
        if self.update_pad_shape:
            results['pad_shape'] = results['img'].shape

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += '(transformations={})'.format(self.transformations)
        return repr_str
