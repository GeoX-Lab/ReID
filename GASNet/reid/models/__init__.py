from __future__ import absolute_import

from .rga_model import *
from .rga_model import ResNet50_RGA_Model

__factory = {
	'resnet50_rga': resnet50_rga,
}


def names():
	return sorted(__factory.keys())

def create(name, *args, **kwargs):
	# print("args:",args)
	# print("**kwargs:",kwargs)
	if name not in __factory:
		raise KeyError("Unknown model:", name)
	return __factory[name](*args, **kwargs) # *args: 创建一个元组 **kwargs:创建一个字典
	# return ResNet50_RGA_Model(*args, **kwargs) # *args: 创建一个元组 **kwargs:创建一个字典
