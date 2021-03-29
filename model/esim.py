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

class esim(object):
    def __init__(self, FLAGS):
        self.config = FLAGS
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

    def esim_model(self, scope):
		# input_encode
        cell_fw = tf.contrib.rnn.LSTMCell(self.config.bilstm_hidden_dim)
        cell_bw = tf.contrib.rnn.LSTMCell(self.config.bilstm_hidden_dim)
        cell_fw = tf.contrib.rnn.DropoutWrapper(cell_fw, input_keep_prob=self.keep_prob,
                                             output_keep_prob=self.keep_prob)
        cell_bw = tf.contrib.rnn.DropoutWrapper(cell_bw, input_keep_prob=self.keep_prob,
                                             output_keep_prob=self.keep_prob)
        query_encode = bilstm_encode(self.query_embed, self.query_seq_len, cell_fw, cell_bw, "forward")
        doc_encode = bilstm_encode(self.doc_embed, self.doc_seq_len, cell_fw, cell_bw, "forward")
 		
        # local inference model, soft align attention seq_mask
        atten_matrix = tf.matmul(query_encode, tf.transpose(doc_encode, [0, 2, 1])) #[b, seq_len, seq_len]
        query_sequence_mask = tf.sequence_mask(self.query_seq_len, self.config.max_len, name="query_mask")
        doc_sequence_mask = tf.sequence_mask(self.doc_seq_len, self.config.max_len, name="doc_mask")
        tile_doc_mask =  tf.cast(tf.tile(tf.expand_dims(doc_sequence_mask, axis=1), [1, self.config.max_len, 1]) , dtype=tf.float32)#[b, seq_len ,seq_len]
        tile_query_mask = tf.cast(tf.tile(tf.expand_dims(doc_sequence_mask, axis=2), [1, 1, self.config.max_len]),  dtype=tf.float32) #[b, seq_len ,seq_len]
        mask_matrix = tile_doc_mask * tile_query_mask
        #mask_matrix = tf.Print(mask_matrix, ["mask_matrix", mask_matrix[0]], summarize=1000)
        pad_one = tf.ones([tf.shape(self.query_seq_len)[0], self.config.max_len, self.config.max_len])
        mask_pad = (pad_one - mask_matrix)*0.000001 + mask_matrix
        atten_matrix_out = mask_pad * atten_matrix
        query_doc_atten = tf.nn.softmax(atten_matrix_out, axis=2) #[b,s,s]
        doc_query_atten = tf.nn.softmax(atten_matrix_out, axis=1) #[b,s,s]
        query_doc_encode = tf.einsum('abc,acd->abd', query_doc_atten, doc_encode) #[b,s, h]
        doc_query_encode = tf.einsum('abc,acd->abd', tf.transpose(doc_query_atten, [0,2,1]), query_encode) #[b,s,h]
		
        query_embed_out = tf.concat([query_encode, query_doc_encode, query_encode-query_doc_encode, query_encode*query_doc_encode], axis=-1)
        doc_embed_out = tf.concat([doc_encode, doc_query_encode, doc_encode-doc_query_encode, doc_encode*doc_query_encode], axis=-1)
		
		# inference composition
        cell_fw_back = tf.contrib.rnn.LSTMCell(self.config.bilstm_hidden_dim)
        cell_bw_back = tf.contrib.rnn.LSTMCell(self.config.bilstm_hidden_dim)
        cell_fw_back = tf.contrib.rnn.DropoutWrapper(cell_fw_back, input_keep_prob=self.keep_prob,
                                             output_keep_prob=self.keep_prob)
        cell_bw_back = tf.contrib.rnn.DropoutWrapper(cell_bw_back, input_keep_prob=self.keep_prob,
                                             output_keep_prob=self.keep_prob)
        query_final_out = bilstm_encode(query_doc_encode, self.query_seq_len, cell_fw_back, cell_bw_back, "backward")
        doc_final_out = bilstm_encode(doc_query_encode, self.doc_seq_len, cell_fw_back, cell_bw_back, "backward")
        #query_outembd = tf.reshape(tf.reduce_max(query_final_out, axis=1), [-1, self.config.bilstm_hidden_dim*2]) #
        query_outembd = max_and_avg(query_final_out)
        doc_outembed = tf.reshape(tf.reduce_max(doc_final_out, axis=1), [-1, self.config.bilstm_hidden_dim*2]) #
        doc_outembed = max_and_avg(doc_final_out)
        ret_output = tf.concat([query_outembd, doc_outembed], axis=-1)
        ret_output = tf.reshape(ret_output, [-1, self.config.bilstm_hidden_dim*8])
        print(ret_output)
        self.prob = tf.layers.dense(ret_output, self.config.num_class, activation=tf.nn.tanh)
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
        scope = "esim"
        self.esim_model(scope)
        self.loss()	
