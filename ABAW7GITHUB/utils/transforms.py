


import cv2
import numpy as np
import torch
import pdb

# Usual dtypes for common modalities
KEYS_TO_DTYPES = {
    'RGB': torch.float,
    'RGB_4': torch.float,
    'RGB_W': torch.float,

    #'FLOWY': torch.float,

}


class Pad(object):
    """Pad image and mask to the desired size.

    Args:
      size (int) : minimum length/width.
      img_val (array) : image padding value.
      msk_val (int) : mask padding value.

    """
    def __init__(self, size, img_val, msk_val):
        assert isinstance(size, int)
        self.size = size
        self.img_val = img_val
        self.msk_val = msk_val

    def __call__(self, sample):
        image = sample['RGB']
        h, w = image.shape[:2]
        h_pad = int(np.clip(((self.size - h) + 1) // 2, 0, 1e6))
        w_pad = int(np.clip(((self.size - w) + 1) // 2, 0, 1e6))
        pad = ((h_pad, h_pad), (w_pad, w_pad))
        #for key in sample['inputs']:
        sample['RGB'] = self.transform_input(sample['RGB'], pad)
        sample['RGB_4'] = self.transform_input(sample['RGB_4'], pad)
        sample['RGB_W'] = self.transform_input(sample['RGB_W'], pad)
        #sample['audio'] = np.pad(sample['audio'], pad, mode='constant', constant_values=self.msk_val)
        return sample 

    def transform_input(self, input, pad):
        input = np.stack([
            np.pad(input[:, :, c], pad, mode='constant',
            constant_values=self.img_val[c]) for c in range(3)
        ], axis=2)
        return input


class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        crop_size (int): Desired output size.

    """
    def __init__(self, crop_size):
        assert isinstance(crop_size, int)
        self.crop_size = crop_size
        if self.crop_size % 2 != 0:
            self.crop_size -= 1

    def __call__(self, sample):
        image = sample['RGB']
        h, w = image.shape[1:3]
        lensb=image.shape[0]
        new_h = min(h, self.crop_size)
        new_w = min(w, self.crop_size)
        top = np.random.randint(0, h - new_h + 1)
        left = np.random.randint(0, w - new_w + 1)
        #for key in sample['inputs']:
        sample['RGB'] = self.transform_input(sample['RGB'], top, new_h, left, new_w,lensb)
        sample['RGB_4'] = self.transform_input(sample['RGB_4'], top, new_h, left, new_w, lensb)
        sample['RGB_W'] = self.transform_input(sample['RGB_W'], top, new_h, left, new_w, lensb)
        #sample['audio'] = sample['audio'][top : top + new_h, left : left + new_w]
        return sample

    def transform_input(self, input, top, new_h, left, new_w,lensb):
        for ij in range(lensb):
            input[ij] = input[ij,top : top + new_h, left : left + new_w]
        return input


class ResizeAndScale(object):

    def __init__(self, side, low_scale, high_scale, shorter=True):
        assert isinstance(side, int)
        assert isinstance(low_scale, float)
        assert isinstance(high_scale, float)
        self.side = side
        self.low_scale = low_scale
        self.high_scale = high_scale
        self.shorter = shorter

    def __call__(self, sample):
        image = sample['RGB']
        scale = np.random.uniform(self.low_scale, self.high_scale)
        if self.shorter:
            min_side = min(image.shape[:2])
            if min_side * scale < self.side:
                scale = (self.side * 1. / min_side)
        else:
            max_side = max(image.shape[:2])
            if max_side * scale > self.side:
                scale = (self.side * 1. / max_side)
        inters = {'RGB': cv2.INTER_CUBIC, 'FLOW': cv2.INTER_NEAREST,'MEL': cv2.INTER_CUBIC, 'DELTA': cv2.INTER_NEAREST}
        #for key in sample['inputs']:
        inter = cv2.INTER_CUBIC
        sample['RGB'] = self.transform_input(sample['RGB'], scale, inter)
        sample['RGB_4'] = self.transform_input(sample['RGB_4'], scale, inter)
        sample['RGB_W'] = self.transform_input(sample['RGB_W'], scale, inter)
        #sample['audio'] = cv2.resize(sample['audio'], None, fx=scale, fy=scale,
        #                            interpolation=cv2.INTER_NEAREST)
        return sample

    def transform_input(self, input, scale, inter):
        input = cv2.resize(input, None, fx=scale, fy=scale, interpolation=inter)
        return input


class CropAlignToMask(object):
    """Crop inputs to the size of the mask."""
    def __call__(self, sample):
        mask_h, mask_w = sample['RGB'].shape[:2]
        #for key in sample['inputs']:
            #print(sample['inputs'])
            #pdb.set_trace()
            #print(key)
        sample['RGB'] = self.transform_input(sample['RGB'], mask_h, mask_w)
        sample['RGB_4'] = self.transform_input(sample['RGB_4'], mask_h, mask_w)
        sample['RGB_W'] = self.transform_input(sample['RGB_W'], mask_h, mask_w)
        return sample

    def transform_input(self, input, mask_h, mask_w):
        input_h, input_w = input.shape[:2]


        if (input_h, input_w) == (mask_h, mask_w):
            return input
        h, w = (input_h - mask_h) // 2, (input_w - mask_w) // 2
        del_h, del_w = (input_h - mask_h) % 2, (input_w - mask_w) % 2
        input = input[h: input_h - h - del_h, w: input_w - w - del_w]


        assert input.shape[:2] == (mask_h, mask_w)
        return input


class ResizeAlignToMask(object):
    """Resize inputs to the size of the mask."""
    def __call__(self, sample):
        mask_h, mask_w = 224,224#sample['mask'].shape[:2]
        assert mask_h == mask_w
        inters = {'RGB': cv2.INTER_CUBIC, 'FLOW': cv2.INTER_NEAREST,'MEL': cv2.INTER_CUBIC, 'DELTA': cv2.INTER_NEAREST}
        #for key in sample['inputs']:
        inter = cv2.INTER_CUBIC
        sample['RGB'] = self.transform_input(sample['RGB'], mask_h, inter)
        sample['RGB_4'] = self.transform_input(sample['RGB_4'], mask_h, inter)
        sample['RGB_W'] = self.transform_input(sample['RGB_W'], mask_h, inter)
        return sample

    def transform_input(self, input, mask_h, inter):
        lensb,input_h, input_w = input.shape[:3]

        scale = mask_h / input_h
        inputsb=np.zeros((lensb, 224, 224, 3))
        for ij in range(lensb):
            inputsb[ij] = cv2.resize(input[ij], dsize =(int(224),int(224)), interpolation=inter)#fx=scale, fy=scale, interpolation=inter)
        input=inputsb
        return input


class ResizeInputs(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, sample):
        #sample['RGB'] = sample['RGB'].numpy()
        if self.size is None:
            return sample
        size = sample['RGB'].shape[1]
        lensb=sample['RGB'].shape[0]
        scale = self.size / size
        # print(sample['rgb'].shape, type(sample['rgb']))
        inters = {'RGB': cv2.INTER_CUBIC, 'FLOW': cv2.INTER_NEAREST,'MEL': cv2.INTER_CUBIC, 'DELTA': cv2.INTER_NEAREST}
        #for key in sample['inputs']:
        inter = cv2.INTER_CUBIC
        sample['RGB'] = self.transform_input(sample['RGB'], scale, inter,lensb)
        sample['RGB_4'] = self.transform_input(sample['RGB_4'], scale, inter, lensb)
        sample['RGB_W'] = self.transform_input(sample['RGB_W'], scale, inter, lensb)
        return sample

    def transform_input(self, input, scale, inter,lensb):
        for ij in range(lensb):
            input[ij] = cv2.resize(input[ij], None, fx=scale, fy=scale, interpolation=inter)
        return input


class ResizeInputsScale(object):
    def __init__(self, scale):
        self.scale = scale

    def __call__(self, sample):
        if self.scale is None:
            return sample
        inters = {'RGB': cv2.INTER_CUBIC, 'FLOW': cv2.INTER_NEAREST,'MEL': cv2.INTER_CUBIC, 'DELTA': cv2.INTER_NEAREST}
        #for key in sample['inputs']:
        inter = cv2.INTER_CUBIC
        sample['RGB'] = self.transform_input(sample['RGB'], self.scale, inter)
        sample['RGB_4'] = self.transform_input(sample['RGB_4'], self.scale, inter)
        sample['RGB_W'] = self.transform_input(sample['RGB_W'], self.scale, inter)
        return sample

    def transform_input(self, input, scale, inter):
        input = cv2.resize(input, None, fx=scale, fy=scale, interpolation=inter)
        return input


class RandomMirror(object):
    """Randomly flip the image and the mask"""
    def __call__(self, sample):
        do_mirror = np.random.randint(2)
        if do_mirror:

            sample['RGB'] = cv2.flip(sample['RGB'], 1)
            sample['RGB_4'] = cv2.flip(sample['RGB_4'], 1)
            sample['RGB_W'] = cv2.flip(sample['RGB_W'], 1)
            #sample['mask'] = cv2.flip(sample['mask'], 1)
        return sample


class Normalise(object):

    def __init__(self, scale, mean, std, depth_scale=1.):
        self.scale = scale
        self.mean = mean
        self.std = std
        self.depth_scale = depth_scale

    def __call__(self, sample):

        sample['RGB'] = (self.scale * sample['RGB'] - self.mean) / self.std
        sample['RGB_4'] = (self.scale * sample['RGB_4'] - self.mean) / self.std
        sample['RGB_W'] = (self.scale * sample['RGB_W'] - self.mean) / self.std

        return sample


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, sample):
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        #for key in sample['inputs']:
        sample['RGB'] = torch.from_numpy(sample['RGB'].transpose((0,3, 1, 2))).to(KEYS_TO_DTYPES['RGB'] )
        sample['RGB_4'] = torch.from_numpy(sample['RGB_4'].transpose((0, 3, 1, 2))).to(KEYS_TO_DTYPES['RGB_4'])
        sample['RGB_W'] = torch.from_numpy(sample['RGB_W'].transpose((0, 3, 1, 2))).to(KEYS_TO_DTYPES['RGB_W'])
        #sample['audio'] = torch.from_numpy(sample['audio']).to(KEYS_TO_DTYPES['audio'])
        return sample


def make_list(x):
    """Returns the given input as a list."""
    if isinstance(x, list):
        return x
    elif isinstance(x, tuple):
        return list(x)
    else:
        return [x]

