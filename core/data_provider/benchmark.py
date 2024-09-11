__author__ = 'mahfuja'
import numpy as np
import os
import cv2
from PIL import Image
import logging
import random
from typing import Iterable, List
from dataclasses import dataclass
import xarray as xr

logger = logging.getLogger(__name__)

class InputHandle:
    def __init__(self, datas, indices, input_param):
        self.name = input_param['name']
        self.input_data_type = input_param.get('input_data_type', 'float32')
        self.minibatch_size = input_param['minibatch_size']
        self.image_height = input_param['image_height']
        self.image_width = input_param['image_width']
        self.datas = datas
        self.indices = indices
        self.current_position = 0
        self.current_batch_indices = []
        self.current_input_length = input_param['seq_length']

    def total(self):
        return len(self.indices)

    def begin(self, do_shuffle=True):
        logger.info("Initialization for read data ")
        if do_shuffle:
            random.shuffle(self.indices)
        self.current_position = 0
        self.current_batch_indices = self.indices[self.current_position:self.current_position + self.minibatch_size]

    def next(self):
        self.current_position += self.minibatch_size
        if self.no_batch_left():
            return None
        self.current_batch_indices = self.indices[self.current_position:self.current_position + self.minibatch_size]

    def no_batch_left(self):
        if self.current_position + self.minibatch_size >= self.total():
            return True
        else:
            return False

    def get_batch(self):
        if self.no_batch_left():
            logger.error(
                "There is no batch left in " + self.name + ". Consider to user iterators.begin() to rescan from the beginning of the iterators")
            return None
        # Create batch of N videos of length L, i.e. shape = (N, L, w, h, c)
        # where w x h is the resolution and c the number of color channels
        input_batch = np.zeros(
            (self.minibatch_size, self.current_input_length, self.image_height, self.image_width, 1)).astype(self.input_data_type)
        for i in range(self.minibatch_size):
            begin = self.current_batch_indices[i]
            end = begin + self.current_input_length
            #data[:,:,:,0] = self.datas[begin:end, :, :] #(cv2.resize(data_slice, (self.image_height, self.image_width))/255.0).astype(np.float32)
            input_batch[i, :self.current_input_length,:, :, 0] = self.datas[begin:end, :, :]
        input_batch = input_batch.astype(self.input_data_type)
        return input_batch

    def print_stat(self):
        logger.info("Iterator Name: " + self.name)
        logger.info("    current_position: " + str(self.current_position))
        logger.info("    Minibatch Size: " + str(self.minibatch_size))
        logger.info("    total Size: " + str(self.total()))
        logger.info("    current_input_length: " + str(self.current_input_length))
        logger.info("    Input Data Type: " + str(self.input_data_type))


class DataProcess:
    def __init__(self, input_param):
        self.paths = input_param['paths']  # path to parent folder containing category dirs
        self.drop_category = ["RA_RAVKDPComp", "RZKDPComp"] #keep only RA_RAHKDPComp
        self.image_height = input_param['image_height']
        self.image_width = input_param['image_width']
        
        # Hard coded training and test persons (prevent same person occurring in train - test set)
        self.train_date = ['20180101', '20180102', '20180103', '20180104', '20180105', '20180106', '20180107', '20180108'] #
        self.test_date = ['20180109' , '20180110'] #

        self.input_param = input_param
        self.seq_len = input_param['seq_length']

    def load_data(self, path, mode='train'):
        '''
        frame -- action -- person_seq(a dir)
        :param paths: action_path list
        :return:
        '''
        path_ = path[0]
        if mode == 'train':
            mode_dates = self.train_date
        elif mode == 'test':
            mode_dates = self.test_date
        else:
            raise Exception("Unexpected mode: " + mode)
        print('Loading data from ' + str(path_))

        file_list=[]
        for date_ in mode_dates:
            tmp_path_ = os.path.join(path_, date_+".nc")
            if os.path.exists(tmp_path_):
                file_list.append(tmp_path_)
        mode_data = xr.open_mfdataset(file_list, combine="nested", concat_dim="time", drop_variables=self.drop_category)
        mode_data = mode_data.RA_RAHKDPComp
        size_ = len(mode_data)
        num_splits = int(size_/288)
        indices = []
        for i in range(num_splits):
            d_start = int(round(i * size_/(num_splits*1.0),0))
            d_end = int(round(i * size_/(num_splits*1.0) + size_/(num_splits*1.0)-1, 0))
            while (d_end-d_start) >= (self.seq_len -1):
                indices.append(d_start)
                d_start += self.seq_len
        
        print("there are " + str(mode_data.shape[0]) + " pictures")
        print("there are " + str(len(indices)) + " sequences")
        return mode_data, indices

    def get_train_input_handle(self):
        train_data, train_indices = self.load_data(self.paths, mode='train')
        return InputHandle(train_data, train_indices, self.input_param)

    def get_test_input_handle(self):
        test_data, test_indices = self.load_data(self.paths, mode='test')
        return InputHandle(test_data, test_indices, self.input_param)

