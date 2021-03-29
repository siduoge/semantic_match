#!/usr/bin/env python
# -*- coding: utf-8 -*-
########################################################################
# 

# 
########################################################################
 
"""
File: cnn_dssm.py
Author: DST(DST@baidu.com)
Date: 2020/12/22 17:20:12
"""
import tensorflow as tf
from model.model_base import *
from util.config import *
from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)

class cnn_dssm(object):
    def __init__(self, FLAGS):
        self.config = FLAGS
        self.config.filters = eval(self.config.filters)
        self.add_placeholder()
        self.input2embed()
        self.model()
    
    def add_placeholder(self):
        self.input_query = tf.placeholder(tf.int32, [None, self.config.max_len], name='query_input')
        self.input_doc = tf.placeholder(tf.int32, [None, self.config.num_negs+1, self.config.max_len], name='doc_input')
        self.single_input_doc = tf.placeholder(tf.int32, [None, self.config.max_len], name='single_doc_input')
        self.keep_prob = tf.placeholder(tf.float32, name="keep_prob")
        self.label = tf.placeholder(tf.int32, [None], name="label")
   
    def input2embed(self):
       self.query_embedding = tf.get_variable(name='word_embedding',
                                                   shape=[self.config.word_vocab_size, self.config.word_embedding_dim],
                                                   dtype=tf.float32,
                                                   initializer=tf.random_uniform_initializer(maxval=-1., minval=1., seed=0))
       self.query_embed = tf.nn.embedding_lookup(self.query_embedding, self.input_query)
       #self.query_embed = tf.Print(self.query_embed, ["self.query_embed:", self.query_embed], summarize=10000)
       self.doc_embed = tf.nn.embedding_lookup(self.query_embedding, self.input_doc)
       self.single_doc_embed = tf.nn.embedding_lookup(self.query_embedding, self.single_input_doc)
       #self.doc_embed = tf.Print(self.doc_embed, ["self.doc_embed:", self.doc_embed], summarize=10000)

    def cnn_dssm_model(self, scope):
       #self.query_embed = tf.Print(self.query_embed, ["query_emebd", self.query_embed[0,0]], summarize=10000)
       #self.single_doc_embed = tf.Print(self.single_doc_embed, ["self.single_doc_embed", self.single_doc_embed[0,0]], summarize=10000)

       self.query_vec = cnn_layer(self.query_embed, self.config.filters, self.config.out_channles, self.config.cnn_hidden_dim, self.config.random_base, self.config.l2_reg, scope_name="cnn")
       self.single_doc_vec = cnn_layer(self.single_doc_embed, self.config.filters, self.config.out_channles, self.config.cnn_hidden_dim, self.config.random_base, self.config.l2_reg, scope_name="cnn")
       #self.query_vec = tf.Print(self.query_vec, ["query_vec", self.query_vec[0,0]], summarize=10000)
       #self.single_doc_vec = tf.Print(self.single_doc_vec, ["self.single_doc_vec", self.single_doc_vec[0,0]], summarize=10000) 
       doc_reshape = tf.reshape(self.doc_embed, [-1, self.config.max_len, self.config.word_embedding_dim])
       doc_vec = cnn_layer(doc_reshape, self.config.filters, self.config.out_channles, self.config.cnn_hidden_dim, self.config.random_base, self.config.l2_reg, scope_name="cnn")
       self.doc_vec = tf.reshape(doc_vec, [-1, self.config.num_negs+1, self.config.cnn_hidden_dim])
       #self.query_vec = tf.Print(self.query_vec, ["self.query_vec:", self.query_vec], summarize=10000)
       #self.doc_vec = tf.Print(self.doc_vec, ["self.doc_vec:", self.doc_vec], summarize=10000)
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
       #self.prob = tf.Print(self.prob, ["self.prob:", self.prob], summarize=10000)
    def loss(self):
       with tf.name_scope("loss"):
           # cat no use softmax_loss [-1, 1]
           #log loss
           true_label, false_label = tf.split(self.prob, [1, self.config.num_negs], 1)
           loss = tf.concat([tf.log(true_label), tf.log(1-false_label)], axis=1)
           self.origin_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.label, logits=self.cosine_dis*100)
           reg_losses = tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
           self.loss = -tf.reduce_mean(tf.log(true_label), name="loss")-tf.reduce_mean(tf.log(1-false_label))+reg_losses
       with tf.name_scope("accuarcy"):
           doc_predict = tf.cast(tf.argmax(self.cosine_dis, axis=-1, name="predict"), dtype=tf.int32)
           correct_predictions = tf.equal(doc_predict, self.label, name='correct_predictions')
           self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")

    def model(self):
       scope = "cnn_dssm"
       self.cnn_dssm_model(scope)
       self.loss()
       
       
 
        

