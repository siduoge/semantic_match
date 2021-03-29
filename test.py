#!/usr/bin/env python
# -*- coding: utf-8 -*-
########################################################################
# 
# Copyright (c) 2020 Baidu.com, Inc. All Rights Reserved
# 
########################################################################
 
"""
File: test.py
Author: DST(DST@baidu.com)
Date: 2020/12/23 22:02:20
"""
import tensorflow as tf
import numpy as np
import os
from model.cnn_dssm import cnn_dssm
from model.lstm_dssm import lstm_dssm
from model.esim import esim
from model.match_pyramid import match_pyramid
from model.ABCNN import ABCNN
from util.config import *

batch_size = FLAGS.batch_size
max_len = FLAGS.max_len
word_vocab = FLAGS.word_vocab_size
doc_size = FLAGS.num_negs + 1
input_query = np.random.randint(0, word_vocab, (batch_size, max_len))
input_doc = np.random.randint(0, word_vocab, (batch_size, FLAGS.num_negs+1, max_len))
input_single_doc = np.random.randint(0, word_vocab, (batch_size, max_len))
input_query_len = np.random.randint(5, FLAGS.max_len, (batch_size))
input_doc_len = np.random.randint(5, FLAGS.max_len, (batch_size, FLAGS.num_negs+1))
input_single_doc_len = np.random.randint(5, FLAGS.max_len, (batch_size))

label = np.ones([batch_size])

os.environ["CUDA_VISIBLE_DEVICES"] = '{}'.format(FLAGS.gpu_id)
with tf.Graph().as_default():
    sess = tf.Session()
    with sess.as_default():
        model = ABCNN(FLAGS)
        sess.run(tf.global_variables_initializer())
        feed_dict = {
                model.input_query: input_query,
                model.input_doc: input_single_doc,
                model.label: label,
                model.query_seq_len: input_query_len,
                model.doc_seq_len: input_single_doc_len,
                model.keep_prob: 0.5,
        }
        loss, acc = sess.run([model.loss, model.accuracy], feed_dict)
        print(loss)
        print(acc)
        #print(intent_prob)
