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

class lstm_dssm(object):
    def __init__(self, FLAGS):
        self.config = FLAGS
        self.add_placeholder()
        self.input2embed()
        self.model()

    def add_placeholder(self):
        self.input_query = tf.placeholder(tf.int32, [None, self.config.max_len], name='query_input')
        self.input_doc = tf.placeholder(tf.int32, [None, self.config.num_negs+1, self.config.max_len], name='doc_input')
        self.query_seq_len = tf.placeholder(tf.int32, [None], name="query_seq_len")
        self.doc_seq_len = tf.placeholder(tf.int32, [None, self.config.num_negs+1], name="doc_seq_len")
        self.keep_prob = tf.placeholder(tf.float32, name="keep_prob")
        self.label = tf.placeholder(tf.int32, [None], name="label")

    def input2embed(self):
        self.query_embedding = tf.get_variable(name='word_embedding',
                                                   shape=[self.config.word_vocab_size, self.config.word_embedding_dim],
                                                   dtype=tf.float32,
                                                   initializer=tf.random_uniform_initializer(maxval=-1., minval=1., seed=0))
        self.query_embed = tf.nn.embedding_lookup(self.query_embedding, self.input_query)
        self.doc_embed = tf.nn.embedding_lookup(self.query_embedding, self.input_doc)

    def lstm_dssm_model(self, scope):
		# input_encode
        cell_fw = tf.contrib.rnn.LSTMCell(self.config.bilstm_hidden_dim)
        cell_bw = tf.contrib.rnn.LSTMCell(self.config.bilstm_hidden_dim)
        cell_fw = tf.contrib.rnn.DropoutWrapper(cell_fw, input_keep_prob=self.keep_prob,
                                             output_keep_prob=self.keep_prob)
        cell_bw = tf.contrib.rnn.DropoutWrapper(cell_bw, input_keep_prob=self.keep_prob,
                                             output_keep_prob=self.keep_prob)
        self.query_vec = bilstm_encode_final(self.query_embed, self.query_seq_len, cell_fw, cell_bw, "forward")
        doc_embed = tf.reshape(self.doc_embed, [-1, self.config.max_len, self.config.word_embedding_dim])
        doc_seq_len = tf.reshape(self.doc_seq_len, [-1])
        doc_encode_reshape = bilstm_encode_final(doc_embed, doc_seq_len, cell_fw, cell_bw, "forward")
        self.doc_vec = tf.reshape(doc_encode_reshape, [-1, self.config.num_negs+1, 4*self.config.bilstm_hidden_dim])
        
        query_norm = tf.tile(tf.expand_dims(tf.sqrt(tf.reduce_sum(tf.square(self.query_vec), axis=1)), axis=1), [1, self.config.num_negs+1])
        doc_norm = tf.sqrt(tf.reduce_sum(tf.square(self.doc_vec), axis=2))
        norm = query_norm * doc_norm
        #norm = tf.Print(norm, ["norm:", norm], summarize=10000)
        cos_query_vec = tf.expand_dims(self.query_vec, -1)
        cos_unormal = tf.reshape(tf.einsum('abc, acd->abd', self.doc_vec, cos_query_vec), [-1, self.config.num_negs+1])
        #print(cos_unormal)
        self.cosine_dis = tf.truediv(cos_unormal, norm)*20
        #self.cosine_dis = tf.Print(self.cosine_dis, ["cos_dis:", self.cosine_dis[0]], summarize=10000)
        self.prob = tf.nn.softmax(self.cosine_dis)
        #self.prob = tf.Print(self.prob, ["cos_dis:", self.prob], summarize=10000)
 	    	
    def loss(self):
        with tf.name_scope("loss"):
            true_label, false_label = tf.split(self.prob, [1, self.config.num_negs], 1)
            self.loss = -tf.reduce_mean(tf.log(true_label), name="loss")

        with tf.name_scope("accuarcy"):
            doc_predict = tf.cast(tf.argmax(self.prob, axis=-1, name="predict"), dtype=tf.int32)
            correct_predictions = tf.equal(doc_predict, self.label, name='correct_predictions')
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
        
    def model(self):
        scope = "lstm_dssm"
        self.lstm_dssm_model(scope)
        self.loss()	
