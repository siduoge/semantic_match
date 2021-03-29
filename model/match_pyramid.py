#!/usr/bin/env python
# -*- coding: utf-8 -*-
########################################################################
# 
# Copyright (c) 2021 Baidu.com, Inc. All Rights Reserved
# 
########################################################################
 
"""
File: esim.py
Author: DST(DST@baidu.com)
Date: 2021/01/01 11:50:13
"""
import tensorflow as tf
from model.model_base import *
from util.config import *
from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)

class match_pyramid(object):
    def __init__(self, FLAGS):
        self.config = FLAGS
        self.filters = eval(self.config.match_filters)
        self.pools = eval(self.config.match_pools)
        self.out_channels = eval(self.config.match_channels)
        self.add_placeholder()
        self.input2embed()
        self.model()

    def add_placeholder(self):
        self.input_query = tf.placeholder(tf.int32, [None, self.config.max_len], name='query_input')
        self.input_doc = tf.placeholder(tf.int32, [None, self.config.max_len], name='doc_input')
        self.query_seq_len = tf.placeholder(tf.int32, [None], name="query_seq_len")
        self.doc_seq_len = tf.placeholder(tf.int32, [None], name="doc_seq_len")
        self.keep_prob = tf.placeholder(tf.float32, name="keep_prob")
        self.label = tf.placeholder(tf.int32, [None], name="label")

    def input2embed(self):
        self.query_embedding = tf.get_variable(name='word_embedding',
                                                   shape=[self.config.word_vocab_size, self.config.word_embedding_dim],
                                                   dtype=tf.float32,
                                                   initializer=tf.random_uniform_initializer(maxval=-1., minval=1., seed=0))
        self.query_embed = tf.nn.embedding_lookup(self.query_embedding, self.input_query)
        self.doc_embed = tf.nn.embedding_lookup(self.query_embedding, self.input_doc)

    def match_pyramid_model(self, scope):
		# input_encode
        # local inference model, soft align attention seq_mask
        atten_matrix = tf.matmul(self.query_embed, tf.transpose(self.doc_embed, [0, 2, 1])) #[b, seq_len, seq_len]
        mask_matrix = matrix_mask(self.query_seq_len, self.doc_seq_len, self.config.max_len)
        #mask_matrix = tf.Print(mask_matrix, ["mask_matrix", mask_matrix[0]], summarize=1000)
        # query_len and doc_len is samw size, so no need to dynamic pool
        true_atten_matrix = atten_matrix * mask_matrix
        ret_output = cnn_layer_mask(true_atten_matrix, tf.expand_dims(mask_matrix,axis=-1), self.filters, self.pools, self.out_channels, self.config.random_base, self.config.l2_reg, active_func=tf.nn.relu, scope_name="cnn")
        dense_prob = tf.layers.dense(ret_output, self.config.cnn_hidden_dim, activation=tf.nn.relu)
        self.prob = tf.layers.dense(dense_prob, self.config.num_class, activation=tf.nn.relu)
        #self.prob = tf.Print(self.prob, ["self.prob", self.prob], summarize=10000)

    def loss(self):
        with tf.name_scope("loss"):
            label = tf.one_hot(self.label, 2)
            self.origin_loss = tf.nn.softmax_cross_entropy_with_logits(labels=label, logits=self.prob)
            self.loss = tf.reduce_mean(self.origin_loss)
        with tf.name_scope("accuarcy"):
            doc_predict = tf.cast(tf.argmax(self.prob, axis=-1, name="predict"), dtype=tf.int32)
            correct_predictions = tf.equal(doc_predict, self.label, name='correct_predictions')
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
        
    def model(self):
        scope = "match_pyramid"
        self.match_pyramid_model(scope)
        self.loss()	
