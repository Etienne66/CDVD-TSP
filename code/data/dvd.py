import os
from data import videodata
from pathlib import Path


class DVD(videodata.VIDEODATA):
    def __init__(self, args, name='DVD', train=True):
        super(DVD, self).__init__(args, name=name, train=train)

    def _set_filesystem(self, dir_data, name='DVD'):
        print("Loading {} => {} DataSet".format("train" if self.train else "test", name))
        self.apath = dir_data
        self.dir_gt = self.apath / 'gt'
        self.dir_input = self.apath / 'blur'
        print("DataSet gt path:", self.dir_gt)
        print("DataSet blur path:", self.dir_input)
