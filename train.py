#! /usr/bin/env python

import tensorflow as tf
import numpy as np
import re
import os
import time
import datetime
import gc
from input_helpers import InputHelper
from siamese_network import SiameseLSTM
from tensorflow.contrib import learn
import gzip
from random import random
import argparse

# python train.py -d dataset

parser = argparse.ArgumentParser()
parser.add_argument("train", help="train dataset") #yahoo or trec
parser.add_argument("dev", help="dev dataset")
#parser.add_argument("model", help="model path")
parser.add_argument("-seq_l","--seq_len", help="sequence length", default=30) #trec30 wiki20
parser.add_argument("-word_l","--word_len", help="word length", default=10)
parser.add_argument("-m","--mode", help="mode")
parser.add_argument("-muti","--muti_mode", help="muti mode") #QA, QC, QAQC
parser.add_argument("-att","--attention_mode", help="attention mode", default=None) #none, static, dynamic
parser.add_argument("-sl","--share_label_mode", type=int ,help="share label mode", default=0) #1, 0
parser.add_argument("-mlp","--mlp_mode", help="mlp mode", default=None)
parser.add_argument("-emb_rec","--emb_recorder", type=int, help="record emb or not", default=0)
args = parser.parse_args()
train = args.train
dev = args.dev

#model = args.model+'/'+args.muti_mode+'/'
max_seq_len = int(args.seq_len)
print(max_seq_len)
max_word_len = int (args.word_len)
mode = args.mode
muti_mode = args.muti_mode
att_mode = args.attention_mode
sl_mode = args.share_label_mode
mlp_mode = args.mlp_mode
emb_rec = args.emb_recorder

# Parameters
# ==================================================
tf.flags.DEFINE_integer("embedding_dim", 300, "Dimensionality of character embedding (default: 300)") #trec300
tf.flags.DEFINE_integer("label_emb_dim", 100, "Dimensionality of label embedding (default: 300)")
tf.flags.DEFINE_float("dropout_keep_prob", 1, "Dropout keep probability (default: 1.0)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularizaion lambda (default: 0.0)")
tf.flags.DEFINE_string("training_files", "./"+train+"QA/"+train+"QA-train.txt", "training file (default: None)")
tf.flags.DEFINE_string("dev_files", "./"+dev+"QA/"+dev+"QA-test.txt", "dev file (default: None)")
tf.flags.DEFINE_string("label_file", "./"+train+"QA/QC_label.txt", "QC label file (default: None)")
tf.flags.DEFINE_string("model", "./model/"+str(muti_mode)+"/", "model file (default: None)")
tf.flags.DEFINE_string("embedding_file", "./embed/glove.6B.300d.txt", "embedding file (default: None)")
tf.flags.DEFINE_integer("hidden_units", 200, "Number of hidden units in softmax regression layer (default:50)")


# QC Parameters
alphabet="abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}"
conv_layers = [
    [256, 7, 3],
    [256, 7, 3],
    [256, 3, None],
    [256, 3, None],
    [256, 3, None],
    [256, 3, 3]
]
fully_layers = [100] 
tf.flags.DEFINE_integer("filter_size", 3, "Size of filter in QC(default: 3)")
tf.flags.DEFINE_integer("num_filters", 20, "Number of the filter in QC(default: 20)")
tf.flags.DEFINE_integer("word_emb", 250, "Dimensionality of word embedding in QC(default: 20)")
tf.flags.DEFINE_integer("char_emb", 30, "Dimensionality of char embedding in QC(default: 30)") #trec150
tf.flags.DEFINE_integer("pos_emb", 30, "Dimensionality of position embedding in QC(default: 6)")#trec30
tf.flags.DEFINE_boolean("emb_concat", True, "Use the concated embedding or not (default: True)")
tf.flags.DEFINE_integer("d_a_size", 400, "Dimensionality of weight1 in self attention (default: 350)") #trec200
tf.flags.DEFINE_integer("r_size", 50, "Dimensionality of weight2 in self attention (default: 30)") #trec50
# Training parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 20, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 50, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 50, "Save model after this many steps (default: 100)")
# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")

if FLAGS.training_files==None:
    print("Input Files List is empty. use --training_files argument.")
    exit()

inpH = InputHelper()
input_set, input_set_dev, vocab_processor, sum_no_of_batches, qc_label_list = inpH.getDataSets(FLAGS.training_files, FLAGS.dev_files, max_seq_len, max_word_len, FLAGS.batch_size, alphabet, FLAGS.label_file)
embedding_matrix = inpH.getEmbeddings(FLAGS.embedding_file,FLAGS.embedding_dim)
FLAGS.embedding_dim = 350
print(emb_rec)

# Training
# ==================================================
print("starting graph def")
with tf.Graph().as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    print("started session")
    with sess.as_default():
        siameseModel = SiameseLSTM(
            max_seq_len=max_seq_len,
            vocab_size=len(vocab_processor.vocabulary_),
            embedding_size=FLAGS.embedding_dim,
            label_emb_size=FLAGS.label_emb_dim,
            hidden_units=FLAGS.hidden_units,
            l2_reg_lambda=FLAGS.l2_reg_lambda,
            batch_size=FLAGS.batch_size,
            embedding_matrix=embedding_matrix,
            mode=mode,
            conv_layers=conv_layers,
            fully_layers=fully_layers,
            alphabet_size=len(alphabet),
            no_of_classes=len(qc_label_list),
            th=1e-6,
            filter_size=FLAGS.filter_size,
            num_filters=FLAGS.num_filters,
            max_word_len=max_word_len,
            word_emb_size=FLAGS.word_emb,
            char_emb_size=FLAGS.char_emb,
            pos_emb_size=FLAGS.pos_emb,
            emb_concat=FLAGS.emb_concat,
            d_a_size=FLAGS.d_a_size,
            r_size=FLAGS.r_size,
            muti_mode=muti_mode,
            att_mode=att_mode,
            sl_mode=sl_mode,
            mlp_mode=mlp_mode)

        # Define Training procedure
        global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(5e-4)
        print("initialized siameseModel object")

    grads_and_vars=optimizer.compute_gradients(siameseModel.loss)
    tr_op_set = optimizer.apply_gradients(grads_and_vars, global_step=global_step)
    print("defined training_ops")
    # Keep track of gradient values and sparsity (optional)
    grad_summaries = []
    for g, v in grads_and_vars:
        if g is not None:
            grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
            sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
            grad_summaries.append(grad_hist_summary)
            grad_summaries.append(sparsity_summary)
    grad_summaries_merged = tf.summary.merge(grad_summaries)
    print("defined gradient summaries")
    # Output directory for models and summaries
    out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", FLAGS.model))
    print("Writing to {}\n".format(out_dir))

    # Summaries for loss and accuracy
    loss_summary = tf.summary.scalar("loss", siameseModel.loss)
    acc_summary = tf.summary.scalar("accuracy", siameseModel.accuracy)

    # Train Summaries
    train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
    train_summary_dir = os.path.join(out_dir, "summaries", "train")
    train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

    # Dev summaries
    dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
    dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
    dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

    # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
    checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
    checkpoint_prefix = os.path.join(checkpoint_dir, "model"+str(muti_mode))
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=100)

    # Write vocabulary
    vocab_processor.save(os.path.join(checkpoint_dir, "vocab"))

    # Initialize all variables
    sess.run(tf.global_variables_initializer())

    print("init all variables")
    graph_def = tf.get_default_graph().as_graph_def()
    graphpb_txt = str(graph_def)
    with open(os.path.join(checkpoint_dir, "graphpb.txt"), 'w') as f:
        f.write(graphpb_txt)


    def train_step(que_batch, ans_batch, que_char_batch, ans_char_batch, y1_batch, y2_batch, add_fea_batch):
        """
        A single training step
        """
        feed_dict = {
            siameseModel.que: que_batch,
            siameseModel.ans: ans_batch,
            siameseModel.que_char: que_char_batch,
            siameseModel.ans_char: ans_char_batch,
            siameseModel.input_y1: y1_batch,
            siameseModel.input_y2: y2_batch,
            siameseModel.add_fea: add_fea_batch,
            siameseModel.dropout_keep_prob: FLAGS.dropout_keep_prob,
            siameseModel.qc_dropout_keep_prob: 1.0
        }
        _, step, summaries, loss, accuracy = sess.run([tr_op_set, global_step, train_summary_op, siameseModel.loss, siameseModel.accuracy], feed_dict)
        time_str = datetime.datetime.now().isoformat()
        if step % 10 == 0:
            print("TRAIN {}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
        train_summary_writer.add_summary(summaries, step)

    def dev_step(que_batch, ans_batch, que_char_batch, ans_char_batch, y1_batch, y2_batch, add_fea_batch, muti_mode):
        """
        A single training step
        """
        feed_dict = {
            siameseModel.que: que_batch,
            siameseModel.ans: ans_batch,
            siameseModel.que_char: que_char_batch,
            siameseModel.ans_char: ans_char_batch,
            siameseModel.input_y1: y1_batch,
            siameseModel.input_y2: y2_batch,
            siameseModel.add_fea: add_fea_batch,
            siameseModel.dropout_keep_prob: FLAGS.dropout_keep_prob,
            siameseModel.qc_dropout_keep_prob: 1.0
        }
        accuracy, qa_soft_prob, que_acc = None, None, 0
        if muti_mode == 'QA':
            step, summaries, loss, accuracy, qa_soft_prob = sess.run([global_step, dev_summary_op, siameseModel.loss, siameseModel.accuracy, siameseModel.qa_soft_prob],  feed_dict)
        elif muti_mode == 'QC':
            step, summaries, loss, accuracy = sess.run([global_step, dev_summary_op, siameseModel.loss, siameseModel.accuracy],  feed_dict)
        elif muti_mode == 'QAQC':
            if emb_rec == 1:
                step, summaries, loss, accuracy, qa_soft_prob, q_att, a_att, qa_pre = sess.run([global_step, dev_summary_op, siameseModel.loss, siameseModel.accuracy, siameseModel.qa_soft_prob, siameseModel.q_att, siameseModel.a_att, siameseModel.qa_predictions],  feed_dict)
            else:
                step, summaries, loss, accuracy, qa_soft_prob = sess.run([global_step, dev_summary_op, siameseModel.loss, siameseModel.accuracy, siameseModel.qa_soft_prob],  feed_dict)
        time_str = datetime.datetime.now().isoformat()
        print("DEV {}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
        dev_summary_writer.add_summary(summaries, step)
        if emb_rec == 1:
            return accuracy, qa_soft_prob, que_acc, q_att, a_att, qa_pre
        else:
            return accuracy, qa_soft_prob, que_acc

    # Generate batches
    batches=inpH.batch_iter(list(zip(input_set[0], input_set[1], input_set[2], input_set[3], input_set[4], input_set[5], input_set[6])), FLAGS.batch_size, FLAGS.num_epochs)

    max_acc = 0.0
    max_que_acc = 0.0 
    max_map = 0.0
    max_mrr = 0.0
    max_p_1 = 0.0
    print('Num of Batches:'+str(sum_no_of_batches))
    for nn in range(sum_no_of_batches*FLAGS.num_epochs):
        batch = batches.__next__()
        if len(batch)<1:
            continue
        que_batch, ans_batch, que_char_batch, ans_char_batch, y1_batch, y2_batch, add_fea_batch = zip(*batch)
        if len(y1_batch)<1:
            continue
        train_step(que_batch, ans_batch, que_char_batch, ans_char_batch, y1_batch, y2_batch, add_fea_batch)
        current_step = tf.train.global_step(sess, global_step)

        if current_step % FLAGS.evaluate_every == 0:
            sum_acc=0.0
            sum_que_acc = 0.0
            num_of_dev_batches = 0.0
            all_predictions = []
            all_pre_y = []
            all_y = []
            all_q = []
            all_a = []
            q_att_qa, a_att_qa, q_att_qc, a_att_qc = None, None, None, None
            q_att_all, a_att_all = [], []
            print("\nEvaluation:")
            dev_batches=inpH.batch_iter(list(zip(input_set_dev[0], input_set_dev[1], input_set_dev[2], input_set_dev[3], input_set_dev[4], input_set_dev[5], input_set_dev[6], input_set_dev[7], input_set_dev[8])), FLAGS.batch_size, 1)
            for db in dev_batches:
                if len(db)<1:
                    continue
                que_batch_dev, ans_batch_dev, que_char_batch_dev, ans_char_batch_dev, y1_batch_dev, y2_batch_dev, add_fea_batch_dev, que_text_batch_dev, ans_text_batch_dev= zip(*db)
                if len(y1_batch_dev)<1:
                    continue
                if emb_rec == 1:
                    acc, qa_soft_prob, que_acc, q_att, a_att, qa_pre = dev_step(que_batch_dev, ans_batch_dev, que_char_batch_dev, ans_char_batch_dev, y1_batch_dev, y2_batch_dev, add_fea_batch_dev, muti_mode)
                else:
                    acc, qa_soft_prob, que_acc = dev_step(que_batch_dev, ans_batch_dev, que_char_batch_dev, ans_char_batch_dev, y1_batch_dev, y2_batch_dev, add_fea_batch_dev, muti_mode)
                #print('################################')
                #print(qa_soft_prob)
                #print(y1_batch_dev)
                #print('################################')
                sum_acc += acc
                if muti_mode == 'QAQC':
                    sum_que_acc += que_acc
                if muti_mode == 'QA' or muti_mode == 'QAQC':
                    all_predictions = np.concatenate([all_predictions, [x[1] for x in qa_soft_prob]])
                    all_pre_y.extend(qa_pre)
                    all_y.extend(y1_batch_dev)
                    all_q.extend(que_text_batch_dev)
                    all_a.extend(ans_text_batch_dev)
                    if emb_rec == 1:
                        q_att_all.extend(q_att)
                        a_att_all.extend(a_att)
                num_of_dev_batches += 1
            print("")

            if muti_mode == 'QA' or muti_mode == 'QAQC':
                result = {}
                for i in range(len(all_y)):
                    if all_q[i] not in result:
                        result[all_q[i]] = []
                    result[all_q[i]].append((all_predictions[i], all_y[i]))
                rank_all = 0
                p_1 = 0
                count = 0
                for key in result.keys():
                    answers = sorted(result[key], key=lambda x:x[0], reverse=True)
                    #print(key)
                    #print(answers[:10])
                    rank = 0
                    for i in range(len(answers)):
                        if answers[i][1] == 1:
                            rank = 1.0/(i+1.0)
                            break
                    if rank != 0:
                        rank_all += rank
                        count +=1
                    if answers[0][1] == 1:
                        p_1 += 1
                print('MRR:' + str(rank_all/count))
                if train == 'yahoo':
                    print('P@1:' + str(p_1/count))
                MRR = rank_all/count
                P_1 = p_1/count
                MAP = 0
                count = 0
                for key in result.keys():
                    answers = sorted(result[key], key=lambda x:x[0], reverse=True)
                    rank = 0
                    rank_all = 0
                    for i in range(len(answers)):
                        if answers[i][1] == 1:
                            rank += 1.0
                            rank_all += rank/(i+1.0)
                    if rank != 0:
                        MAP += rank_all/rank
                        count +=1
                print('MAP:' + str(MAP/count))
            if muti_mode == 'QC' or muti_mode == 'QAQC':
                print("Mean Accuracy: {}".format(sum_acc/num_of_dev_batches))
                print("Mean Accuracy of Question: {}".format(sum_que_acc/num_of_dev_batches))
        if current_step % FLAGS.checkpoint_every == 0:
            if muti_mode == 'QA' or muti_mode == 'QAQC':
                if MAP/count >= max_map or MRR >= max_mrr or P_1 >= max_p_1: 
                    max_map = max(MAP/count, max_map)
                    max_mrr = max(MRR, max_mrr)
                    max_p_1 = max(P_1, max_p_1)
                    if sum_que_acc != 0:
                        max_que_acc = max(sum_que_acc/num_of_dev_batches, max_que_acc)
                    with open('./result_'+muti_mode+'.txt', 'a+') as f:
                        f.write('Step:'+str(current_step)+'\n')
                        #f.write('MAP:'+str(max_map)+'\n')
                        #f.write('MRR:'+str(max_mrr)+'\n')
                        f.write('MAP:'+str(MAP/count)+'\n')
                        f.write('MRR:'+str(MRR)+'\n')
                        if train =='yahoo':
                            #f.write('P@1:'+str(max_p_1)+'\n')
                            f.write('P@1:'+str(P_1)+'\n')
                        f.write('Mean Accuracy:'+str(sum_acc/num_of_dev_batches)+'\n')
                        if sum_que_acc != 0:
                            f.write('Mean Accuracy of Question:'+str(max_que_acc)+'\n')
                        f.write('\n')
                    saver.save(sess, checkpoint_prefix, global_step=current_step)
                    tf.train.write_graph(sess.graph.as_graph_def(), checkpoint_prefix, "graph"+str(nn)+".pb", as_text=False)
                    print("Saved model {} checkpoint to {}\n".format(nn, checkpoint_prefix))

                    if emb_rec == 1:
                        print("Saved emb...")
                        with open('att.txt', 'w') as f:
                            for q,a,q_att,a_att,y1,y2 in zip(all_q, all_a, q_att_all, a_att_all, all_pre_y, all_y):
                                f.write(str(q)+'\t'+str(a)+'\t'+' '.join(map(str,q_att))+'\t'+' '.join(map(str,a_att))+'\t'+str(y1)+'\t'+str(y2)+'\n')

            elif muti_mode == 'QC':
                if sum_acc/num_of_dev_batches >= max_que_acc:
                    with open('./result_'+muti_mode+'.txt', 'a+') as f:
                        f.write('Step:'+str(current_step)+'    ')
                        f.write('Mean Accuracy of Question:'+str(sum_acc/num_of_dev_batches)+'\n')
                        f.write('\n')
                    max_que_acc = sum_acc/num_of_dev_batches
                    saver.save(sess, checkpoint_prefix, global_step=current_step)
                    tf.train.write_graph(sess.graph.as_graph_def(), checkpoint_prefix, "graph"+str(nn)+".pb", as_text=False)
                    print("Saved model {} checkpoint to {}\n".format(nn, checkpoint_prefix))

