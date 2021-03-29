#!/usr/bin/env python
# -*- coding: utf-8 -*-
########################################################################
# 
# Copyright (c) 2020 Baidu.com, Inc. All Rights Reserved
# 
########################################################################
 
"""
File: test_cos.py
Author: DST(DST@baidu.com)
Date: 2020/12/30 16:48:15
"""
import tensorflow as tf
import numpy as np

a = np.ones((2,3))*0.5
print(a)
b = np.ones((2,3))*0.8

def cos(query_vecs, doc_vecs):
	ret = np.sum(query_vecs*doc_vecs, axis=1)/(np.linalg.norm(query_vecs, axis=1) * np.linalg.norm(doc_vecs, axis=1))
	return ret

print(cos(a,b))

