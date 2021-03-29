#!/usr/bin/env python
# -*- coding: utf-8 -*-
 
"""
File: train.py
Author: DST(DST@baidu.com)
Date: 2020/12/23 21:30:38
"""
import tensorflow as tf
import numpy as np
import os
import time
from model.cnn_dssm import cnn_dssm
from model.esim import esim
from model.match_pyramid import match_pyramid
from model.ABCNN import ABCNN
from util.data_util import *
from util.config import *
import datetime
import sys

def train_step(train_querys_batch, train_docs_batch, train_query_len, train_doc_len, train_labels_batch, sess, train_op, model):
    #print(train_querys_batch)
    #print(train_docs_batch)
    #print(train_query_len)
    #print(train_doc_len)
    #print(train_labels_batch)
    
    # data use is wrong
    fedd_dict = {
            model.input_query : train_querys_batch,
            model.input_doc : train_docs_batch,
            model.query_seq_len: train_query_len,
            model.doc_seq_len: train_doc_len,
            model.label : train_labels_batch,
            model.keep_prob : FLAGS.keep_prob}
    #print(fedd_dict)
    _, step, loss, acc, _1  = sess.run([train_op, global_step, model.loss, model.accuracy, model.prob], fedd_dict)
    time_str = datetime.datetime.now().isoformat()
    print("{}: step {}, loss: {:g}, acc: {:g},".format(time_str, step, loss, acc))

def dev_step(test_querys_batch, test_docs_batch, test_query_len, test_doc_len, test_labels_batch, sess,  model):
    #print(test_querys_batch)
    #print(test_docs_batch)
    batches = batch_iter(list(zip(test_querys_batch, test_docs_batch, test_query_len, test_doc_len, test_labels_batch)), 1000, 1, False)
    all_acc = []
    for batch in batches:
       querys_batch, docs_batch, querys_len, docs_len, labels_batch = zip(*batch)
       #print(train_labels_batch)
       feed_dict = {
            model.input_query : querys_batch,
            model.input_doc : docs_batch,
            model.query_seq_len: querys_len,
            model.doc_seq_len: docs_len,
            model.label : labels_batch, 
            model.keep_prob : 1.0}
       #print(feed_dict)
       step, acc = sess.run([global_step, model.accuracy], feed_dict)
       #print(acc)
       
       all_acc.append(1000*acc)
    #print(all_acc)
    avg_acc = sum(all_acc)/(len(all_acc)*1000)
    time_str = datetime.datetime.now().isoformat() 
    print("{}: step {},  acc: {:g}".format(time_str, step, avg_acc))


papers, paper_len, paper_ids = load_all_data(FLAGS.paper_path, FLAGS.max_len)
relation = load_relation(FLAGS.train_data)
querys_dev, docs_dev, labels_dev, querys_len_dev, docs_len_dev = load_data_dssm(papers, paper_len, FLAGS.test_data)
#load_data_dssm_train(papers, relation, paper_ids, FLAGS.train_data, FLAGS.num_negs)
os.environ["CUDA_VISIBLE_DEVICES"] = '{}'.format(FLAGS.gpu_id)
with tf.Graph().as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        if FLAGS.model == "cnn_dssm":
            model = cnn_dssm(FLAGS)
        elif FLAGS.model == "esim":
            model = esim(FLAGS)
        elif FLAGS.model == "match_pyramid":
            model = match_pyramid(FLAGS)
        elif FLAGS.model == "ABCNN":
            model = ABCNN(FLAGS)
        elif FLAGS.model == "bilstm_slot_gated":
            print("1efrvervgerv")
            model = bilstm_slot_gated(FLAGS)
        saver = tf.train.Saver(tf.global_variables())
        global_step = tf.Variable(0, name="global_step", trainable=False)
        trainable_variables = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(model.loss, trainable_variables), 10)
        optimizer = tf.train.AdamOptimizer(FLAGS.lr)
        train_op = optimizer.apply_gradients(zip(grads, trainable_variables),global_step=global_step)
        timestamp = str(int(time.time()))
        out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", FLAGS.out_dir))
        print("Writing to {}\n".format(out_dir))
        checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)
        sess.run(tf.global_variables_initializer())
        train_querys, train_docs, train_labels, train_querys_len, train_docs_len = load_data_dssm(papers, paper_len, FLAGS.train_data)
        #print(train_data)
        #sys.exit()
        batches = batch_iter(list(zip(train_querys, train_docs, train_labels, train_querys_len, train_docs_len)), FLAGS.batch_size, FLAGS.num_epochs, True)
        iter_n = 0

        for batch in batches:
            iter_n += 1
            train_querys_batch, train_docs_batch, train_labels_batch, train_querys_len, train_docs_len = zip(*batch)
            #print(data_batch)
            train_step(train_querys_batch, train_docs_batch, train_querys_len, train_docs_len, train_labels_batch, sess, train_op, model)
            current_step = tf.train.global_step(sess, global_step)
            if current_step % FLAGS.evaluate_every == 0:
                print("\nEvaluation {}:".format(iter_n))
                dev_step(querys_dev, docs_dev, querys_len_dev, docs_len_dev, labels_dev, sess, model)
                path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                print("Saved model checkpoint to {}\n".format(path))
