#!/usr/bin/env python
# -*- coding: utf-8 -*-
########################################################################
# 
# Copyright (c) 2021 Baidu.com, Inc. All Rights Reserved
# 
########################################################################
 
"""
File: test_broadcast.py
Author: DST(DST@baidu.com)
Date: 2021/01/13 10:13:12
"""

import tensorflow as tf
import numpy as np

a = tf.ones((2,1))*2
b = tf.ones((1,2))
c = a-b
with tf.Session() as sess:
    out = sess.run(c)
    print(out)
    print(out.shape)

