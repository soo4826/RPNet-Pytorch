# Python script for parse image file into rtmaps .rec file 
# Author: Jinsu ha , soo4826@gmail.com
# Date: 2022.5.30

import rtmaps.core as rt
import rtmaps.types
from rtmaps.base_component import BaseComponent

import numpy as np
import cv2
import os.path


class rtmaps_python(BaseComponent):
    def __init__(self):
        BaseComponent.__init__(self)

    def Dynamic(self):
    
        self.add_output("out", rtmaps.types.IPL_IMAGE)
        self.add_property("dataset_path", "data\Woodscape\\", rtmaps.types.PATH, rtmaps.types.MUST_EXIST)
        # C:\Users\Alien01\Documents\GitHub\RPNet-Pytorch\data\Woodscape\
        self.add_property("image_split", "FV.txt")

    def Birth(self):
        # Read dataset_path
        self.dataset_path = os.path.dirname(self.properties['dataset_path'].data)
        if not os.path.isdir(self.dataset_path):
            rt.report_error(f'Dataset_path "{self.dataset_path}" Not exist')

        ## Load image list
        self.image_split = self.properties['image_split'].data
        img_seq_path = os.path.join(self.dataset_path, self.image_split)

        if not os.path.isfile(img_seq_path):
            rt.report_error(f'Image split file "{img_seq_path}" Not exist')
        
        # Read image list from file
        with open(img_seq_path, "r") as f:
            self.img_path_list = f.read().splitlines()
        self.img_path_list.sort()
        self.num_img = len(self.img_path_list)
        self.curr_img_idx = 0

    def Death(self):
        print("Shuttung down img2rec script")

    def Core(self):

        ## Get each frame of image and make output
        img_path = os.path.join(self.dataset_path, "rgb_images", self.img_path_list[self.curr_img_idx])
        img = cv2.imread(img_path)
        
        # Read all image from image list
        if self.curr_img_idx+1 == self.num_img:
            self.curr_img_idx = 0
        else:    
            self.curr_img_idx += 1

        # Write output.
        out = rtmaps.types.Ioelt()
        out.data = rtmaps.types.IplImage()
        out.data.image_data = img
        out.data.color_model = "COLR"
        out.data.channel_seq = "BGR"
        self.outputs["out"].write(out)