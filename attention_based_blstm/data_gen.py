# -*- coding: utf-8 -*-
"""
Created on Mon May 28 15:51:05 2018

@author: Administrator
"""


class Corpus(object):
    def __init__(self, in_file):
        self.in_file=in_file
        self.__iter__()
        
    def __iter__(self):
        for i, line in enumerate(open(self.in_file, encoding='utf8')):
            items=line.split(',')
            text=items[1].strip()
            yield list(text), items[0].strip().split(' ')