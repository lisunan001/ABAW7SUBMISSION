import os
import numpy as np
import cv2
from PIL import Image
from torch.utils.data import Dataset
import torch
import pdb
import csv



class SegDataset(Dataset):

    def __init__(self, dataset, data_file, data_dir,  #input_mask_idxs,
                 transform_trn=None, transform_val=None, stage='train', ignore_label=None):
        self.datalist = []
        self.labellist=[]
        with open(data_file, 'r') as f:
            datalist = csv.reader(f)
            #print(datalist)
            next(datalist)
            #pdb.set_trace()
            isblist=[]
            isblabel=[]
            isb=0
            #pdb.set_trace()
            for row in datalist:
                self.datalist.append(row[0].zfill(5))
                self.labellist.append(int(row[1])-1)

        self.root_dir = data_dir
        self.transform_trn = transform_trn
        self.transform_val = transform_val
        self.step= 15
        self.stage = stage
        #self.input_names = input_names
        self.input_mask_idxs = [0,1]#input_mask_idxs
        self.ignore_label = ignore_label
        #self.read_select_frames()

    def set_stage(self, stage):

        self.stage = stage

    def __len__(self):
        return len(self.datalist)





    def __getitem__(self, idx):
        #idxs = self.input_mask_idxs
        names_RGB = os.path.join(self.root_dir, self.datalist[idx])

        #print(self.datalist[idx])
        sample = {}
        i=0
        sample['RGB'] = np.zeros((self.step, 256, 256, 3))#, dtype=torch.float32)
        sample['RGB_4']= np.zeros((self.step, 256, 256, 3))
        sample['RGB_W']= np.zeros((self.step, 256, 256, 3))
        sample['RGB']=self.read_image(names_RGB,self.step)
        sample['RGB_4'] =sample['RGB']
        sample['RGB_W'] =sample['RGB']

        if self.stage == 'train':
            if self.transform_trn:

                sample = self.transform_trn(sample)
                #pdb.set_trace()
        elif self.stage == 'val':
            if self.transform_val:
                sample = self.transform_val(sample)

        sample['label']=self.labellist[idx]
        sample['names']=self.datalist[idx]
        #del sample['inputs']

        return sample

    @staticmethod
    def read_image_(x, key):
        img = cv2.imread(x)
        if key == 'depth':
            img = cv2.applyColorMap(cv2.convertScaleAbs(255 - img, alpha=1), cv2.COLORMAP_JET)
        return img

    @staticmethod
    def read_image(x,step):

        #img_arr_series = torch.zeros((15, 256, 256, 3), dtype=torch.float32)
        img_arr_series = np.zeros((step, 256, 256, 3))#, dtype=float32)
        image_name_list=os.listdir(x)
        image_name_list.sort()
        for ij in range(step):
            img_arr = np.array(Image.open(os.path.join(x,image_name_list[ij])).resize((256,256)))
            #img_arr_re=img_arr.resize((256,256,3))
            if len(img_arr.shape) == 2:  # grayscale
                img_arr = np.tile(img_arr, [3, 1, 1]).transpose(1, 2, 0)
            #img_arr_series[ij]=torch.from_numpy(img_arr).float()
            img_arr_series[ij,:,:,:]=img_arr
            #import pdb
            #pdb.set_trace()
        return img_arr_series


class SegDataset_va(Dataset):


    def __init__(self, dataset, data_file, data_dir,  # input_mask_idxs,
                 transform_trn=None, transform_val=None, stage='train', ignore_label=None):
        self.datalist = []
        self.labellist_v = []
        self.labellist_a = []
        with open(data_file, 'r') as f:
            datalist = csv.reader(f)
            # print(datalist)
            next(datalist)
            # pdb.set_trace()
            isblist = []
            isblabel = []
            isb = 0
            # pdb.set_trace()
            for row in datalist:
                self.datalist.append(row[0].zfill(5))
                self.labellist_v.append(int(row[1]) )
                self.labellist_a.append(int(row[2]))

        self.root_dir = data_dir
        self.transform_trn = transform_trn
        self.transform_val = transform_val
        self.step = 15
        self.stage = stage
        # self.input_names = input_names
        self.input_mask_idxs = [0, 1]  # input_mask_idxs
        self.ignore_label = ignore_label
        # self.read_select_frames()

    def set_stage(self, stage):
        """Define which set of transformation to use.

        Args:
            stage (str): either 'train' or 'val'

        """
        self.stage = stage

    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, idx):
        # idxs = self.input_mask_idxs
        names_RGB = os.path.join(self.root_dir, self.datalist[idx])

        # print(self.datalist[idx])
        sample = {}
        i = 0
        sample['RGB'] = np.zeros((self.step, 256, 256, 3))  # , dtype=torch.float32)
        sample['RGB_4'] = np.zeros((self.step, 256, 256, 3))
        sample['RGB_W'] = np.zeros((self.step, 256, 256, 3))
        sample['RGB'] = self.read_image(names_RGB, self.step)
        sample['RGB_4'] = sample['RGB']
        sample['RGB_W'] = sample['RGB']

        if self.stage == 'train':
            if self.transform_trn:
                sample = self.transform_trn(sample)
                # pdb.set_trace()
        elif self.stage == 'val':
            if self.transform_val:
                sample = self.transform_val(sample)

        sample['label_v'] = self.labellist_v[idx]
        sample['label_a'] = self.labellist_a[idx]
        sample['names'] = self.datalist[idx]
        # del sample['inputs']

        return sample

    @staticmethod
    def read_image_(x, key):
        img = cv2.imread(x)
        if key == 'depth':
            img = cv2.applyColorMap(cv2.convertScaleAbs(255 - img, alpha=1), cv2.COLORMAP_JET)
        return img

    @staticmethod
    def read_image(x, step):

        # img_arr_series = torch.zeros((15, 256, 256, 3), dtype=torch.float32)
        img_arr_series = np.zeros((step, 256, 256, 3))  # , dtype=float32)
        image_name_list = os.listdir(x)
        image_name_list.sort()
        for ij in range(step):
            img_arr = np.array(Image.open(os.path.join(x, image_name_list[ij])).resize((256, 256)))
            # img_arr_re=img_arr.resize((256,256,3))
            if len(img_arr.shape) == 2:  # grayscale
                img_arr = np.tile(img_arr, [3, 1, 1]).transpose(1, 2, 0)
            # img_arr_series[ij]=torch.from_numpy(img_arr).float()
            img_arr_series[ij, :, :, :] = img_arr
            # import pdb
            # pdb.set_trace()
        return img_arr_series


class SegDataset_TEST(Dataset):


    def __init__(self, dataset, data_file, data_dir,  # input_mask_idxs,
                 transform_trn=None, transform_val=None, stage='train', ignore_label=None):
        self.datalist = []
        self.labellist = []
        with open(data_file, 'r') as f:
            datalist = csv.reader(f)
            # print(datalist)
            next(datalist)
            # pdb.set_trace()
            isblist = []
            isblabel = []
            isb = 0
            # pdb.set_trace()
            for row in datalist:
                self.datalist.append(row[0].zfill(6))
                #self.labellist.append(int(row[1]) - 1)


        # self.datalist = isblist
        # self.labellist = isblabel
        # pdb.set_trace()
        self.root_dir = data_dir
        self.root_dir_4 = '/data/lucheng/lisunan/ABAW7/FACE4ABAWSPLIT4'
        self.root_dir_W = '/data/lucheng/lisunan/ABAW7/FACE4ABAWSPLITwhole'
        self.transform_trn = transform_trn
        self.transform_val = transform_val
        self.step = 15
        self.stage = stage
        # self.input_names = input_names
        self.input_mask_idxs = [0, 1]  # input_mask_idxs
        self.ignore_label = ignore_label
        # self.read_select_frames()

    def set_stage(self, stage):
        """Define which set of transformation to use.

        Args:
            stage (str): either 'train' or 'val'

        """
        self.stage = stage

    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, idx):
        # idxs = self.input_mask_idxs
        names_RGB = os.path.join(self.root_dir, self.datalist[idx])
        names_RGB_4 = os.path.join(self.root_dir_4, self.datalist[idx])
        names_RGB_W = os.path.join(self.root_dir_W, self.datalist[idx])

        # print(self.datalist[idx])
        sample = {}
        i = 0
        sample['RGB'] = np.zeros((self.step, 256, 256, 3))  # , dtype=torch.float32)
        sample['RGB_4'] = np.zeros((self.step, 256, 256, 3))
        sample['RGB_W'] = np.zeros((self.step, 256, 256, 3))
        sample['RGB'] = self.read_image(names_RGB, self.step)
        sample['RGB_4']= self.read_image(names_RGB_4, self.step)
        sample['RGB_W'] = self.read_image(names_RGB_W, self.step)
        # sample['FLOW']=self.read_image(names_RGB,self.step)
        # sample['inputs'] = self.input_names

        # print(names_RGB)
        # pdb.set_trace()
        # sample['mask'] = mask change for emotion
        if self.stage == 'train':
            if self.transform_trn:
                sample = self.transform_trn(sample)
                # pdb.set_trace()
        else:#if self.stage == 'val':
            if self.transform_val:
                sample = self.transform_val(sample)
        # wb2=load_workbook('/mnt2/lisunan/CEN-master/DFEW/Annotation/annotation/annotation.xlsx')
        # ws=wb2.get_sheet_by_name('Sheet1')
        # print(self.datalist[idx][0].split('/')[1].strip('.jpg'))
        # cou=int(self.datalist[idx][1][-9:-4])+1
        # cou=self.datalist[idx][0].split('/')[1].strip('.jpg')
        #sample['label'] = self.labellist[idx]
        sample['names'] = self.datalist[idx]
        # del sample['inputs']

        return sample

    @staticmethod
    def read_image_(x, key):
        img = cv2.imread(x)
        if key == 'depth':
            img = cv2.applyColorMap(cv2.convertScaleAbs(255 - img, alpha=1), cv2.COLORMAP_JET)
        return img

    @staticmethod
    def read_image(x, step):
        """Simple image reader

        Args:
            x (str): path to image.

        Returns image as `np.array`.

        """
        # img_arr_series = torch.zeros((15, 256, 256, 3), dtype=torch.float32)
        img_arr_series = np.zeros((step, 256, 256, 3))  # , dtype=float32)
        image_name_list = os.listdir(x)
        image_name_list.sort()
        for ij in range(step):
            img_arr = np.array(Image.open(os.path.join(x, image_name_list[ij])).resize((256, 256)))
            # img_arr_re=img_arr.resize((256,256,3))
            if len(img_arr.shape) == 2:  # grayscale
                img_arr = np.tile(img_arr, [3, 1, 1]).transpose(1, 2, 0)
            # img_arr_series[ij]=torch.from_numpy(img_arr).float()
            img_arr_series[ij, :, :, :] = img_arr
            # import pdb
            # pdb.set_trace()
        return img_arr_series