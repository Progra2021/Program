import tensorflow as tf
import numpy as np
from math import sqrt

class SiameseLSTM(object):
    """
    A LSTM based deep Siamese network for text similarity.
    Uses an character embedding layer, followed by a biLSTM and Energy Loss layer.
    """

    def BiRNN(self, x, dropout, scope, embedding_size, sequence_length, hidden_units):
        n_input=embedding_size
        n_steps=sequence_length
        n_hidden=hidden_units
        n_layers=1
        # Prepare data shape to match `bidirectional_rnn` function requirements
        # Current data input shape: (batch_size, n_steps, n_input) (?, seq_len, embedding_size)
        # Required shape: 'n_steps' tensors list of shape (batch_size, n_input)
        # Permuting batch_size and n_steps
        x = tf.transpose(x, [1, 0, 2])
        # Reshape to (n_steps*batch_size, n_input)
        x = tf.reshape(x, [-1, n_input])
        #print(x)
        # Split to get a list of 'n_steps' tensors of shape (batch_size, n_input)
        x = tf.split(x, n_steps, 0)
        #print(x)
        # Define lstm cells with tensorflow
        # Forward direction cell
        with tf.name_scope("fw"+scope),tf.variable_scope("fw"+scope):
            stacked_rnn_fw = []
            for _ in range(n_layers):
                fw_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
                lstm_fw_cell = tf.contrib.rnn.DropoutWrapper(fw_cell,output_keep_prob=dropout)
                stacked_rnn_fw.append(lstm_fw_cell)
            lstm_fw_cell_m = tf.nn.rnn_cell.MultiRNNCell(cells=stacked_rnn_fw, state_is_tuple=True)

        with tf.name_scope("bw"+scope),tf.variable_scope("bw"+scope):
            stacked_rnn_bw = []
            for _ in range(n_layers):
                bw_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
                lstm_bw_cell = tf.contrib.rnn.DropoutWrapper(bw_cell,output_keep_prob=dropout)
                stacked_rnn_bw.append(lstm_bw_cell)
            lstm_bw_cell_m = tf.nn.rnn_cell.MultiRNNCell(cells=stacked_rnn_bw, state_is_tuple=True)

        # Get lstm cell output
        with tf.name_scope("bw"+scope),tf.variable_scope("bw"+scope):
            outputs, fw_state, bw_state = tf.nn.static_bidirectional_rnn(lstm_fw_cell_m, lstm_bw_cell_m, x, dtype=tf.float32)
        #outputs, shape = [max_seq_len=30][?, h*2=400]
        #fw_state,h, shape=[?, 200]
        #bw_state,h, shape=[?, 200]
        
        # output transformation to the original tensor type
        state = tf.concat([fw_state[0][1], bw_state[0][1]], 1)

        #print(state)
        outputs = tf.stack(outputs)
        outputs = tf.transpose(outputs, [1, 0, 2])
        return outputs, state

    def overlap(self, embed1, embed2):
        overlap1 = tf.matmul(embed1,tf.transpose(embed2,[0,2,1]))
        overlap1 = tf.expand_dims(tf.reduce_max(overlap1,axis=2),-1)

        overlap2 = tf.matmul(embed2,tf.transpose(embed1,[0,2,1]))
        overlap2 = tf.expand_dims(tf.reduce_max(overlap2,axis=2),-1)
        embed1 = tf.concat([embed1,overlap1],2)
        embed2 = tf.concat([embed2,overlap2],2)
        return embed1,embed2

    # return 1 output of lstm cells after pooling, lstm_out(batch, step, rnn_size * 2)
    def max_pooling(self, lstm_out):
        height, width = int(lstm_out.get_shape()[1]), int(lstm_out.get_shape()[2])       # (step, length of input for one step)
        # do max-pooling to change the (sequence_lenght) tensor to 1-lenght tensor
        lstm_out = tf.expand_dims(lstm_out, -1)
        output = tf.nn.max_pool(
            lstm_out,
            ksize=[1, height, 1, 1],
            strides=[1, 1, 1, 1],
            padding='VALID')
        output = tf.reshape(output, [-1, width])
        return output

    def avg_pooling(self, lstm_out):
        height, width = int(lstm_out.get_shape()[1]), int(lstm_out.get_shape()[2])       # (step, length of input for one step)      
        # do avg-pooling to change the (sequence_lenght) tensor to 1-lenght tensor
        lstm_out = tf.expand_dims(lstm_out, -1)
        output = tf.nn.avg_pool(
            lstm_out,
            ksize=[1, height, 1, 1],
            strides=[1, 1, 1, 1],
            padding='VALID')       
        output = tf.reshape(output, [-1, width])       
        return output


    def attentive_pooling(self, h1, h2, U, att_mode='static'):
        dim = int(h1.get_shape()[2])
        if att_mode == 'static':
            transform_left = tf.einsum('ijk,kl->ijl',h1, U)
        elif att_mode == 'dynamic':
            transform_left = tf.matmul(h1, U)
        att_mat= tf.matmul(transform_left, tf.transpose(h2,[0,2,1]))
        #print(att_mat)
        #row_max = tf.reduce_max(att_mat, axis=1)
        row_max = tf.expand_dims(tf.nn.softmax(tf.reduce_max(att_mat, axis=1)),-1, name='answer_attention')
        column_max = tf.expand_dims(tf.nn.softmax(tf.reduce_max(att_mat, axis=2)),-1, name='question_attention')
        out2 = tf.reshape(tf.matmul(tf.transpose(h2,[0,2,1]),row_max),[-1,dim])
        out1 = tf.reshape(tf.matmul(tf.transpose(h1,[0,2,1]),column_max),[-1,dim])
        row_max = tf.reshape(row_max, [-1, row_max.get_shape()[1]])
        column_max = tf.reshape(column_max, [-1, column_max.get_shape()[1]])
        return out1, out2, column_max, row_max

########################################################################
    def scaled_dot_product_attention(Q, K, V, dropout_rate=0.0):
        scaler = tf.rsqrt(tf.to_float(tf_utils.get_shape(Q)[2])) # depth of the query
        logits = tf.matmul(Q, K, transpose_b=True) * scaler
        weights = tf.nn.softmax(logits)
        weights = tf.nn.dropout(weights, 1.0 - dropout_rate)
        return tf.matmul(weights, V)
########################################################################
    def self_attentive(self, x, x_reshape, W_s1, W_s2, r_size, max_seq_len):
        #print('begin')
        #print(x) #(?, 30, 400)
        #print(x_reshape) #(?, 400)
        x_s1 = tf.nn.tanh(tf.matmul(x_reshape, W_s1))
        #print(x_s1) #(?, 400)
        x_s2 = tf.matmul(x_s1, W_s2)
        #print(x_s2) #(?, 50)
        x_s2_reshape = tf.transpose(tf.reshape(x_s2, [-1, max_seq_len, r_size]), [0, 2, 1])
        #print(x_s2_reshape) # (?, 50, 30)
        A = tf.nn.softmax(x_s2_reshape, name="qc_attention") #[?, 50, 30]
        out = tf.matmul(A, x)
        #print(out) #(?,50, 400)
        #print(A) #(?,50,30)
        #print(x) #(?,30,400)

        return A, out




    def EmbeddingLayer(self, input_word, input_char, vocab_size, alphabet_size, no_of_classes, filter_size, num_filters, max_seq_len, max_word_len, word_emb_size, char_emb_size, pos_emb_size, emb_concat, embedding_matrix):

        def Position_Embedding(inputs, position_size):
            batch_size,seq_len = tf.shape(inputs)[0],tf.shape(inputs)[1]
            position_j = 1. / tf.pow(10000., \
                                     2 * tf.range(position_size / 2, dtype=tf.float32 \
                                    ) / position_size)
            position_j = tf.expand_dims(position_j, 0)
            position_i = tf.range(tf.cast(seq_len, tf.float32), dtype=tf.float32)
            position_i = tf.expand_dims(position_i, 1)
            position_ij = tf.matmul(position_i, position_j)
            position_ij = tf.concat([tf.cos(position_ij), tf.sin(position_ij)], 1)
            position_embedding = tf.expand_dims(position_ij, 0) \
                               + tf.zeros((batch_size, seq_len, position_size))
            return position_embedding

        with tf.variable_scope("word_embedding", reuse=tf.AUTO_REUSE):
            word_emb_W = tf.get_variable("word_emb_W", [vocab_size, word_emb_size], dtype='float32')
            self.word_emb_W = word_emb_W
            word_embedding = tf.nn.embedding_lookup(word_emb_W, input_word)

        with tf.variable_scope("position_embedding", reuse=tf.AUTO_REUSE):
            pos_embedding = Position_Embedding(input_word, pos_emb_size)

        with tf.variable_scope("char_embedding", reuse=tf.AUTO_REUSE):
            char_emb_W = tf.get_variable("char_emb_W", [alphabet_size+1, char_emb_size], dtype='float32')
            self.char_emb_W = char_emb_W
            char_embedding = tf.nn.embedding_lookup(char_emb_W, input_char)

        #Add CNN get filters and combine with word
        with tf.variable_scope("char_conv_maxPool", reuse=tf.AUTO_REUSE):
            filter_shape = [filter_size, char_emb_size, num_filters]
            W_conv = tf.get_variable("W_conv", filter_shape, dtype='float32')
            b_conv = tf.get_variable("b_conv", [num_filters], dtype='float32')
            conv = tf.nn.conv1d(char_embedding,
                        W_conv,
                        stride=1,
                        padding="SAME",
                        name="conv")
            h_expand = tf.expand_dims(conv, -1)
            pooled = tf.nn.max_pool(
                        h_expand,
                        ksize=[1, max_seq_len * max_word_len,1, 1],
                        strides=[1, max_word_len, 1, 1],
                        padding='SAME',
                        name="pooled")
            char_pool_flat = tf.reshape(pooled, [-1, max_seq_len, num_filters], name="char_pool_flat")
            concat_emb = tf.concat([word_embedding, char_pool_flat, pos_embedding], axis=2)  
        x = concat_emb

        return x


    def ShareLabelLayer(self, out_put, label_emb_W):
        #with tf.variable_scope("label_embedding", reuse=tf.AUTO_REUSE):
            #label_emb_W = tf.get_variable("label_emb_W", [no_of_classes, label_emb_size], dtype='float32')
        out_put = tf.expand_dims(out_put, 1)
        comb_repr = tf.multiply(out_put, self.label_emb_W)
        scores = tf.reduce_sum(comb_repr, 2)
        return scores

        
    def LossLayer(self, muti_mode, sl_mode, att_mode, no_of_classes):
        with tf.name_scope("loss"):
            if muti_mode == 'QA':
                qa_losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.qa_prob, labels=tf.one_hot(self.input_y1,2))
                #self.l2_loss = tf.contrib.layers.apply_regularization(regularizer=tf.contrib.layers.l2_regularizer(0.0001),weights_list=tf.trainable_variables())
                self.loss = tf.reduce_mean(qa_losses)
            elif muti_mode == 'QC':
                que_losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.que_prob, labels=tf.one_hot(self.input_y2,no_of_classes))
                self.loss = tf.reduce_mean(que_losses)
            elif muti_mode == 'QAQC':
                qa_losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.qa_prob, labels=tf.one_hot(self.input_y1,2))
                #que_losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.que_prob, labels=tf.one_hot(self.input_y2,no_of_classes)) 
                self.qa_loss = tf.reduce_mean(qa_losses)
                #self.que_loss = tf.reduce_mean(que_losses)
                
                #self.tln_loss = tf.losses.mean_squared_error(self.x_l_emb, tf.matmul(tf.one_hot(self.input_y2,no_of_classes), self.label_emb_W))
                if att_mode == 'static':
                    self.loss_P = tf.reduce_mean(self.P * 1.0)
                    self.loss = self.qa_loss + self.loss_P #+ self.que_loss #+ self.tln_loss
                else:
                    self.loss = self.qa_loss #+ self.que_loss + self.tln_loss
    def AccuracyLayer(self, muti_mode):
        with tf.name_scope("accuracy"):
            if muti_mode == 'QA':
                qa_correct_predictions = tf.equal(self.qa_predictions, self.input_y1)
                self.accuracy = tf.reduce_mean(tf.cast(qa_correct_predictions, "float"), name="qa_accuracy")
            elif muti_mode == 'QC':
                que_correct_predictions = tf.equal(self.que_predictions, self.input_y2)
                self.accuracy = tf.reduce_mean(tf.cast(que_correct_predictions, "float"), name="que_accuracy")
            elif muti_mode == 'QAQC':
                qa_correct_predictions = tf.equal(self.qa_predictions, self.input_y1)
                self.accuracy = tf.reduce_mean(tf.cast(qa_correct_predictions, "float"), name="qa_accuracy")
                #que_correct_predictions = tf.equal(self.que_predictions, self.input_y2)
                #self.que_accuracy = tf.reduce_mean(tf.cast(que_correct_predictions, "float"), name="que_accuracy")

    def qa_private(
            self,
            max_seq_len,
            vocab_size,
            embedding_size,
            hidden_units,
            r_size,
            mode,
            muti_mode,
            att_mode,
            sl_mode,
            mlp_mode):

        # Create a convolution + maxpool layer for each filter size
        # que_emb=[?, 30, 300]
        # embedded_chars1=[?, 30, 301]
        # h1, h2 [batch_size, word_len, emb_size]=[?, 30, 400]
        # out1, out2 [batch_size, 400]
        '''
        if muti_mode == 'QA':
            with tf.name_scope("embedding"):
                self.embedded_chars1 = self.que_emb
                self.embedded_chars2 = self.ans_emb
                self.embedded_chars1,self.embedded_chars2 = self.overlap(self.embedded_chars1,self.embedded_chars2)
            with tf.name_scope("RNN"):
                self.h1=self.BiRNN(self.embedded_chars1, self.dropout_keep_prob,
                        "side1", embedding_size+1, max_seq_len, hidden_units)
                self.h2=self.BiRNN(self.embedded_chars2, self.dropout_keep_prob,
                        "side2", embedding_size+1, max_seq_len, hidden_units)
        elif muti_mode == 'QAQC':
            self.h1 = self.que_fea  #[?, 30, 400]
            self.h2 = self.ans_fea
            print(self.que_fea)
        '''
        if att_mode == 'static' or att_mode == None:
            with tf.name_scope("attentive_pooling"):
                U1 = tf.get_variable(
                        "U1",
                        shape=[2*hidden_units, 2*hidden_units],
                        initializer=tf.contrib.layers.xavier_initializer())
                self.out1, self.out2, self.q_att_qa, self.a_att_qa= self.attentive_pooling(self.h1,self.h2,U1)
                print('hhhhhhhhhhhhhhhhhh')
                print(self.out1)
                print(self.out2)
                print(self.q_att_qa)
        elif att_mode == 'dynamic':
            with tf.name_scope("attentive_pooling"):
                U1 = tf.get_variable(
                          "U1",
                          shape=[2*hidden_units, 2*hidden_units],
                          initializer=tf.contrib.layers.xavier_initializer())
                ##############################################
                U_qc = 0.5
                ##############################################
                #qc_info = tf.add(tf.expand_dims(U_qc * self.avg_pooling(self.qc_info), 1), U1)
                qc_info = tf.add(tf.expand_dims(U_qc * self.qc_info, 1), U1)
                #U2 = tf.einsum('kj,ijk->ikk',U_qc, self.qc_info)
                # self.qc_info, (?, 30, 400)
                self.out1, self.out2, self.q_att_qa, self.a_att_qa= self.attentive_pooling(self.h1, self.h2, qc_info, att_mode)

                #print(self.q_att_qa) q_att_qa(?,100,1)
            #self.out1 = tf.concat([self.out1, self.que_prob], 1)
            #self.out2 = tf.concat([self.out2, self.que_prob], 1)
        #elif att_mode =='None':
        #   self.out1 = self.max_pooling(self.h1)
        #    self.out2 = self.max_pooling(self.h2)

        with tf.name_scope("similarity"):
            if sl_mode == 1:
                self.out1 = tf.concat([self.out1, self.o1], 1)
                self.out2 = tf.concat([self.out2, self.o2], 1)

            sim_size = int(self.out1.get_shape()[1])
            W = tf.get_variable(
                "W",
                shape=[sim_size, sim_size],
                initializer=tf.contrib.layers.xavier_initializer())
            self.transform_left = tf.matmul(self.out1, W)
            self.sims = tf.reduce_sum(tf.multiply(self.transform_left, self.out2), 1, keep_dims=True)

        if mode == 'raw':
            if sl_mode == 1:
                #self.new_input = tf.concat([self.out1, self.o1, self.sims, self.out2, self.o2], 1, name='new_input')
                self.new_input = tf.concat([self.out1, self.sims, self.out2], 1, name='new_input')
            else:
                self.new_input = tf.concat([self.out1, self.sims, self.out2], 1, name='new_input')
        else:
            if sl_mode == 1:
                #self.new_input = tf.concat([self.out1, self.o1, self.sims, self.out2, self.o2, self.add_fea], 1, name='new_input')
                self.new_input = tf.concat([self.out1, self.sims, self.out2, self.add_fea], 1, name='new_input')
            else:
                self.new_input = tf.concat([self.out1, self.sims, self.out2, self.add_fea], 1, name='new_input')
        #self.new_input = tf.reshape(emb_cnt, [-1, emb_cnt.get_shape()[1]*emb_cnt.get_shape()[2]], name="new_input")        
        num_feature = int(self.new_input.get_shape()[1])
        hidden_size = 200
        with tf.name_scope("hidden"):
            W = tf.get_variable(
                "W_hidden",
                shape=[num_feature, hidden_size],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[hidden_size]), name="b")
            self.hidden_output = tf.nn.relu(tf.nn.xw_plus_b(tf.cast(self.new_input, tf.float32), W, b, name="hidden_output"))

        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.hidden_output, self.dropout_keep_prob, name="hidden_output_drop")

        with tf.name_scope("output"):
            W = tf.get_variable(
                "W_output",
                shape=[hidden_size, 2],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[2]), name="b")
            self.qa_prob = tf.nn.xw_plus_b(self.h_drop, W, b)
            self.qa_soft_prob = tf.nn.softmax(self.qa_prob, name='distance')
            self.qa_predictions = tf.argmax(self.qa_soft_prob, 1, name="predictions")

    def qc_private(
            self,
            conv_layers,
            fully_layers,
            no_of_classes,
            th,
            embedding_size,
            label_emb_size,
            max_seq_len,
            hidden_units,
            d_a_size,
            r_size,
            att_mode,
            sl_mode,
            mlp_mode):

        #with tf.name_scope("RNN"):
            #self.que_fea = self.BiRNN(self.que_emb, self.qc_dropout_keep_prob, "q_LSTM", embedding_size, max_seq_len, hidden_units)
            #self.ans_fea = self.BiRNN(self.ans_emb, self.qc_dropout_keep_prob, "a_LSTM", embedding_size, max_seq_len, hidden_units)
        x = self.state1 #(?, 2*hidden=400)
        x2 = self.state2

        if att_mode == 'static' or att_mode == 'dynamic':
            with tf.name_scope("self_attentive"):
                W_s1 = tf.get_variable(
                       "W_s1",
                       shape=[2*hidden_units, d_a_size],
                       initializer=tf.contrib.layers.xavier_initializer())
                W_s2 = tf.get_variable(
                       "W_s2",
                       shape=[d_a_size, r_size],
                       initializer=tf.contrib.layers.xavier_initializer())
                x_reshape = tf.reshape(self.h1, [-1, 2*hidden_units])
                x2_reshape = tf.reshape(self.h2, [-1, 2*hidden_units])
                q_att_qa, q_att_out = self.self_attentive(self.h1, x_reshape, W_s1, W_s2, r_size, max_seq_len) 
                a_att_qa, a_att_out = self.self_attentive(self.h2, x2_reshape, W_s1, W_s2, r_size, max_seq_len)
                x = tf.reshape(q_att_out, shape=[-1, 2*hidden_units*r_size])
                x2 = tf.reshape(a_att_out, shape=[-1, 2*hidden_units*r_size])

                self.q_att = tf.reduce_mean(q_att_qa, 1)
                self.a_att = tf.reduce_mean(a_att_qa, 1)
            with tf.name_scope("penalization"):
                AA_T = tf.matmul(q_att_qa, tf.transpose(q_att_qa, [0, 2, 1]))
                I = tf.reshape(tf.tile(tf.eye(r_size), [tf.shape(q_att_qa)[0], 1]), [-1, r_size, r_size])
                self.P = tf.square(tf.norm(AA_T - I, axis=[-2, -1], ord="fro"))


            fc_size=256
            with tf.name_scope("fully-connected"):
                q_att_flat = tf.reshape(x, shape=[-1, 2*hidden_units*r_size])
                W_fc = tf.get_variable("W_fc", shape=[2*hidden_units*r_size, fc_size], initializer=tf.contrib.layers.xavier_initializer())
                b_fc = tf.Variable(tf.constant(0.1, shape=[fc_size]), name="b_fc")
                x = tf.nn.relu(tf.nn.xw_plus_b(q_att_flat, W_fc, b_fc), name="fc_x1")
                x2 = tf.nn.relu(tf.nn.xw_plus_b(q_att_flat, W_fc, b_fc), name="fc_x2")
            '''
            with tf.name_scope("output"):
                W_output = tf.get_variable("W_output", shape=[fc_size, no_of_classes], initializer=initializer)
                b_output = tf.Variable(tf.constant(0.1, shape=[no_of_classes]), name="b_output")
                prob = tf.nn.xw_plus_b(self.fc, W_output, b_output, name="logits")
                predictions = tf.argmax(prob, 1, name="predictions")
             '''
        qc_info = x #+x2?

        vec_dim = x.get_shape()[1].value
        weights = [vec_dim] + [label_emb_size]
        #print(weights)
        for i, fl in enumerate(fully_layers):
            #var_id += 1
            with tf.name_scope("LinearLayer" ):
                stdv = 1/sqrt(weights[i])
                W = tf.Variable(tf.random_uniform([weights[i], fl], minval=-stdv, maxval=stdv), dtype='float32', name='W')
                b = tf.Variable(tf.random_uniform(shape=[fl], minval=-stdv, maxval=stdv), dtype='float32', name = 'b')
                x = tf.tanh(tf.nn.xw_plus_b(x, W, b))  #[?, 100]
                x2 = tf.tanh(tf.nn.xw_plus_b(x2, W, b))
            '''
            with tf.name_scope("DropoutLayer"):
                x = tf.nn.dropout(x, self.qc_dropout_keep_prob) 
                x2 = tf.nn.dropout(x2, self.qc_dropout_keep_prob)
            '''
        if sl_mode == 1:
            with tf.name_scope("LabelEmbeddingLayer"):
                self.label_emb_W = tf.get_variable("label_emb_W", [no_of_classes, label_emb_size], dtype='float32')               
                scores = self.ShareLabelLayer(x, self.label_emb_W)
                print(scores)
                prob = tf.nn.softmax(scores)
                predictions = tf.argmax(prob, 1)
                #x_l_emb = tf.matmul(tf.one_hot(self.input_y2,no_of_classes), label_emb_W)
                self.x_l_emb = tf.matmul(prob,self.label_emb_W)

                x2_scores = self.ShareLabelLayer(x2, self.label_emb_W)
                x2_prob = tf.nn.softmax(x2_scores)
                x2_l_emb = tf.matmul(x2_prob, self.label_emb_W)
            '''
            with tf.name_scope("MLPLayer"):
                mlp_h = 16
                W1 = tf.Variable(tf.random_uniform([label_emb_size, mlp_h], minval=-stdv, maxval=stdv), dtype='float32', name='W1')
                W2 = tf.Variable(tf.random_uniform([mlp_h, no_of_classes], minval=-stdv, maxval=stdv), dtype='float32', name='W2')
                b1 = tf.Variable(tf.random_uniform(shape=[mlp_h], minval=-stdv, maxval=stdv), name = 'b1')
                b2 = tf.Variable(tf.random_uniform(shape=[no_of_classes], minval=-stdv, maxval=stdv), name = 'b2')
                o1 = tf.einsum('ij,jk->ik', prob, self.label_emb_W)
                o1 = tf.nn.xw_plus_b(o1, W1, b1, name="que_out1")
                self.o1 = tf.nn.xw_plus_b(o1, W2, b2, name="que_out2")
                self.o1_prob = tf.nn.softmax(self.o1)
                o2 = tf.einsum('ij,jk->ik', x2_prob, self.label_emb_W)
                o2 = tf.nn.xw_plus_b(o2, W1, b1, name="ans_out1")
                self.o2 = tf.nn.xw_plus_b(o2, W2, b2, name="ans_out2")
                self.o2_prob = tf.nn.softmax(self.o2)
             
            self.o1 = 
            self.o2=self.o2
            '''
        '''
        #elif sl_mode == 'QC':
        with tf.name_scope("OutputLayer"):
            stdv = 1/sqrt(weights[-1])
            W = tf.Variable(tf.random_uniform([weights[-1], no_of_classes], minval=-stdv, maxval=stdv), dtype='float32', name='W')
            b = tf.Variable(tf.random_uniform(shape=[no_of_classes], minval=-stdv, maxval=stdv), name = 'b')
            prob = tf.nn.xw_plus_b(x, W, b, name="que_scores") #[?, 6]
            predictions = tf.argmax(prob, 1) #[?]
        '''

        self.o1=scores
        self.o2=x2_scores
        return scores, prob, predictions, qc_info


    def __init__(
        self,
        max_seq_len,
        vocab_size,
        embedding_size,
        label_emb_size, #defaul==0
        hidden_units,
        l2_reg_lambda,
        batch_size,
        embedding_matrix,
        mode,
        conv_layers,
        fully_layers,
        alphabet_size,
        no_of_classes,
        th,
        filter_size,
        num_filters,
        max_word_len,
        word_emb_size,
        char_emb_size,
        pos_emb_size,
        emb_concat,
        d_a_size,
        r_size,
        muti_mode,
        att_mode,
        sl_mode,
        mlp_mode):       

        #self.que_accuracy = None
        #self.qa_soft_prob = None

        with tf.name_scope("Input-Layer"):
            self.que = tf.placeholder(tf.int32, [None, max_seq_len], name="que")
            self.ans = tf.placeholder(tf.int32, [None, max_seq_len], name="ans")
            self.que_char = tf.placeholder(tf.int32, [None, max_seq_len, max_word_len], name="que_char")
            self.ans_char = tf.placeholder(tf.int32, [None, max_seq_len, max_word_len], name="ans_char")
            self.que_char_flat = tf.reshape(self.que_char, [-1, max_seq_len*max_word_len], name="que_char_flat")
            self.ans_char_flat = tf.reshape(self.ans_char, [-1, max_seq_len*max_word_len], name="ans_char_flat")
            self.input_y1 = tf.placeholder(tf.int64, [None], name="input_y1")
            self.input_y2 = tf.placeholder(tf.int64, [None], name="input_y2")
            self.add_fea = tf.placeholder(tf.float32, [None, 4], name="add_fea")
            self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
            self.qc_dropout_keep_prob = tf.placeholder(tf.float32, name="qc_dropout_keep_prob")

        # que_emb=[?, 30, 300]
        # embedded_chars1=[?, 30, 301]
        # h1, h2 [batch_size, word_len, emb_size]=[?, 30, 400]
        '''
        # Share embedding layer:
        with tf.name_scope("embedding"):
            self.que_emb = self.EmbeddingLayer(self.que, self.que_char_flat, vocab_size, alphabet_size, no_of_classes, filter_size, num_filters, max_seq_len, max_word_len, word_emb_size, char_emb_size, pos_emb_size, emb_concat, embedding_matrix)
            self.ans_emb = self.EmbeddingLayer(self.ans, self.ans_char_flat, vocab_size, alphabet_size, no_of_classes, filter_size, num_filters, max_seq_len, max_word_len, word_emb_size, char_emb_size, pos_emb_size, emb_concat, embedding_matrix)
            self.embedded_chars1,self.embedded_chars2 = self.overlap(self.que_emb,self.ans_emb)
        print(self.que_emb)
        '''

        with tf.name_scope("word_embedding"):
            if embedding_matrix.all() != None:
                self.word_W = tf.Variable(embedding_matrix, trainable=True, name="emb", dtype=tf.float32)
            else:
                self.word_W = tf.get_variable("emb", [vocab_size, embedding_size])

            embedded_chars1 = tf.nn.embedding_lookup(self.word_W, self.que)
            embedded_chars2 = tf.nn.embedding_lookup(self.word_W, self.ans)
            #self.embedded_chars1,self.embedded_chars2 = self.overlap(self.embedded_chars1,self.embedded_chars2)

        with tf.name_scope("position_embedding"):
            b_s = tf.shape(self.que)[0]
            position_j = 1. / tf.pow(10000., \
                                     2 * tf.range(pos_emb_size / 2, dtype=tf.float32 \
                                    ) / pos_emb_size)
            position_j = tf.expand_dims(position_j, 0)
            position_i = tf.range(tf.cast(max_seq_len, tf.float32), dtype=tf.float32)
            position_i = tf.expand_dims(position_i, 1)
            position_ij = tf.matmul(position_i, position_j)
            position_ij = tf.concat([tf.cos(position_ij), tf.sin(position_ij)], 1)
            pos_embedding = tf.expand_dims(position_ij, 0) \
                               + tf.zeros((b_s, max_seq_len, pos_emb_size))

        with tf.name_scope("char_embedding"):
            char_emb_W = tf.get_variable("char_emb_W", [alphabet_size+1, char_emb_size], dtype='float32')
            self.char_emb_W = char_emb_W
            char_embedding1 = tf.nn.embedding_lookup(self.char_emb_W, self.que_char_flat)
            char_embedding2 = tf.nn.embedding_lookup(self.char_emb_W, self.ans_char_flat)

        #Add CNN get filters and combine with word
        with tf.name_scope("char_conv_maxPool"):
            filter_shape = [filter_size, char_emb_size, num_filters]
            W_conv = tf.get_variable("W_conv", filter_shape, dtype='float32')
            b_conv = tf.get_variable("b_conv", [num_filters], dtype='float32')
            conv = tf.nn.conv1d(char_embedding1,
                        W_conv,
                        stride=1,
                        padding="SAME",
                        name="conv1")
            h_expand = tf.expand_dims(conv, -1)
            pooled = tf.nn.max_pool(
                        h_expand,
                        ksize=[1, max_seq_len * max_word_len,1, 1],
                        strides=[1, max_word_len, 1, 1],
                        padding='SAME',
                        name="pooled1")
            char_pool_flat = tf.reshape(pooled, [-1, max_seq_len, num_filters], name="char_pool_flat1")
            concat_emb = tf.concat([embedded_chars1, char_pool_flat, pos_embedding], axis=2)

            conv2 = tf.nn.conv1d(char_embedding2,
                        W_conv,
                        stride=1,
                        padding="SAME",
                        name="conv2")
            h_expand2 = tf.expand_dims(conv2, -1)
            pooled2 = tf.nn.max_pool(
                        h_expand2,
                        ksize=[1, max_seq_len * max_word_len,1, 1],
                        strides=[1, max_word_len, 1, 1],
                        padding='SAME',
                        name="pooled2")
            char_pool_flat2 = tf.reshape(pooled2, [-1, max_seq_len, num_filters], name="char_pool_flat2")
            concat_emb2 = tf.concat([embedded_chars2, char_pool_flat2, pos_embedding], axis=2)
 
        print(concat_emb)

        self.que_emb = concat_emb
        self.ans_emb = concat_emb2
        self.embedded_chars1,self.embedded_chars2 = self.overlap(self.que_emb,self.ans_emb)


        '''
        with tf.name_scope("embedding"):
            if embedding_matrix.all() != None:
                self.W = tf.Variable(embedding_matrix, trainable=False, name="emb", dtype=tf.float32)
            else:
                self.W = tf.get_variable("emb", [vocab_size, embedding_size])

            self.embedded_chars1 = tf.nn.embedding_lookup(self.W, self.que)
            self.embedded_chars2 = tf.nn.embedding_lookup(self.W, self.ans)
            self.embedded_chars1,self.embedded_chars2 = self.overlap(self.embedded_chars1,self.embedded_chars2)
        '''

        with tf.name_scope("RNN"):
            self.h1, self.state1=self.BiRNN(self.embedded_chars1, self.dropout_keep_prob,
                    "side1", embedding_size+1, max_seq_len, hidden_units)
            self.h2, self.state2=self.BiRNN(self.embedded_chars2, self.dropout_keep_prob,
                    "side2", embedding_size+1, max_seq_len, hidden_units)
        print(self.h1)
        print(self.state1)


        # Muti-mode
        # que_fea, ans_fea [batch_size, emb_size]=[?, 1024]
        if muti_mode == 'QA':
            self.qa_private(max_seq_len, vocab_size, embedding_size, hidden_units, r_size, mode, muti_mode, att_mode, sl_mode, mlp_mode)
        elif muti_mode == 'QC':
            self.que_prob, self.que_predictions, _= self.qc_private(conv_layers, fully_layers, no_of_classes, th, embedding_size, label_emb_size, max_seq_len, hidden_units, d_a_size, r_size, att_mode, sl_mode, mlp_mode)
        elif muti_mode == 'QAQC':
            if sl_mode == 1:
                self.scores, self.que_prob, self.que_predictions, self.qc_info = self.qc_private(conv_layers, fully_layers, no_of_classes, th, embedding_size, label_emb_size, max_seq_len, hidden_units, d_a_size, r_size, att_mode, sl_mode, mlp_mode)
            self.qa_private(max_seq_len, vocab_size, embedding_size, hidden_units, r_size, mode, muti_mode, att_mode, sl_mode, mlp_mode)

        # Calculate mean loss
        self.LossLayer(muti_mode, sl_mode, att_mode, no_of_classes)
        
        # Accuracy
        self.AccuracyLayer(muti_mode)
