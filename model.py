import tensorflow as tf
from tensorflow.contrib import rnn

class RCNN(object):
    def __init__(self, args):
        self.lstm_layers=args.lstm_layers
        self.lstm_units=args.lstm_units
        self.lstm_act=args.lstm_act
        self.conv_act=args.conv_act
        self.fc_act=args.fc_act
        self.keep_prob=args.keep_prob
        self.batch_size=args.batch_size
        self.max_len=args.max_len
        self.top_k=args.top_k
        self.conv_hori_w=args.conv_hori_w
        self.conv_hori_deep=args.conv_hori_deep
        self.conv_vert_k=args.conv_vert_k
        self.conv_vert_deep=args.conv_vert_deep
        self.choose_T=args.choose_T
        self.n_items=args.n_items
        self.is_training=args.is_training
        self.l2_reg=args.l2_reg
        self.learning_rate=args.learning_rate
        self.decay_rate=args.decay_rate
        self.max_to_keep=args.max_to_keep

        self.lstm_act=self.init_act(self.lstm_act)
        self.conv_act = self.init_act(self.conv_act)
        self.fc_act=self.init_act(self.fc_act)

        self.build_model()
        self.metric_at_top_k()
        self.saver = tf.train.Saver(max_to_keep=self.max_to_keep)

    def init_act(self,act_type):
        if act_type=='relu':
            return self.relu
        elif act_type=='tanh':
            return self.tanh
        elif act_type=='sigmoid':
            return self.sigmoid
    def tanh(self, X):
        return tf.nn.tanh(X)
    def relu(self, X):
        return tf.nn.relu(X)
    def sigmoid(self, X):
        return tf.nn.sigmoid(X)
    def softmax(self, X):
        return tf.nn.softmax(X)

    def LSTM_layer(self,input_tensor,sequence_length):
        '''
        :param input_tensor: [batch_size,max_len]
        :return:[batch_size,max_len,lstm_units]
        '''
        embedding = tf.get_variable('embedding', [self.n_items, self.lstm_units])
        with tf.variable_scope('LSTM_layer', reuse=tf.AUTO_REUSE):
            cells = []
            for _ in range(self.lstm_layers):
                cell = rnn.BasicLSTMCell(self.lstm_units, activation=self.lstm_act)
                if self.is_training and self.keep_prob < 1.0:
                    cell = rnn.DropoutWrapper(cell, output_keep_prob=self.keep_prob)
                cells.append(cell)
            cell = rnn.MultiRNNCell(cells)
            zero_state = cell.zero_state(self.batch_size, dtype=tf.float32)
            inputs = tf.nn.embedding_lookup(embedding,input_tensor)
            outputs, state = tf.nn.dynamic_rnn(cell, inputs, sequence_length=sequence_length, initial_state=zero_state,time_major=False)
            self.final_state = state
        return outputs

    def conv_layer(self,layer_name,input_tensor,filter_shape,bias_shape,strides,padding):
        with tf.variable_scope(layer_name, reuse=tf.AUTO_REUSE):
            conv_weights = tf.get_variable(layer_name+"weights",filter_shape,initializer=tf.truncated_normal_initializer())
            conv_biases = tf.get_variable(layer_name+"biases",bias_shape, initializer=tf.constant_initializer())
            conv_out = tf.nn.conv2d(input_tensor, conv_weights, strides=strides, padding=padding)
            conv_out = self.conv_act(tf.nn.bias_add(conv_out, conv_biases))
        return conv_out

    def fc_layer(self,layer_name,input_tensor,W_shape,B_shape):
        with tf.variable_scope(layer_name, reuse=tf.AUTO_REUSE):
            fc_weights = tf.get_variable(layer_name+"weights",W_shape,initializer=tf.truncated_normal_initializer())
            if self.l2_reg > 0:
                tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(self.l2_reg)(fc_weights))
            fc_biases = tf.get_variable(layer_name+"biases",B_shape, initializer=tf.constant_initializer())
            logits = self.fc_act(tf.matmul(input_tensor, fc_weights) + fc_biases)
            if self.is_training:
                logits = tf.nn.dropout(logits, self.keep_prob)
        return logits

    def build_model(self):
        self.X=tf.placeholder(tf.int32, [self.batch_size, self.max_len], name='input')
        self.Y=tf.placeholder(tf.int32, [self.batch_size, self.n_items], name='output_onehot_label')
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        sequence_length=tf.count_nonzero(self.X, 1)
        lstm_out=self.LSTM_layer(self.X,sequence_length)

        #lstm_out=tf.reshape(lstm_out,[self.batch_size,self.max_len*self.lstm_units])
        conv_hori_input = lstm_out[:,self.choose_T- 1:self.choose_T, :]#[batch_size,1,lstm_units]
        conv_hori_input = tf.reshape(conv_hori_input, [-1, 1,self.lstm_units, 1])  # [batch_size,1,lstm_units,1]
        choose_T_value=tf.reshape(conv_hori_input,[-1,self.lstm_units])
        conv_hori_out=self.conv_layer('conv_hori_',conv_hori_input,[1,self.conv_hori_w,1,self.conv_hori_deep],
                                      [self.conv_hori_deep],[1,1,1,1],'VALID')
        conv_hori_out = tf.reduce_sum(conv_hori_out, 2)
        conv_hori_out = tf.reshape(conv_hori_out, [-1, self.conv_hori_deep])  # [batch_size,CONV_H_DEEP]
        conv_vert_input = lstm_out[:, self.choose_T- self.conv_vert_k:self.choose_T, :]  # [batch_size,conv_vert_k,lstm_units]
        conv_vert_input = tf.reshape(conv_vert_input,[-1, self.conv_vert_k,self.lstm_units,1])# [batch_size,CONV_V_SIZE,nums,1]
        conv_vert_out=self.conv_layer('conv_vert',conv_vert_input,[self.conv_vert_k, 1, 1, self.conv_vert_deep],
                                      [self.conv_vert_deep],[1,1,1,1],'VALID')# [batch_size,1,lstm_units,conv_vert_deep]
        conv_vert_out = tf.reshape(conv_vert_out, [-1, self.lstm_units])#[batch_size,lstm_units]
        conv_vert_out = tf.multiply(conv_vert_out, choose_T_value)#对应位置值相乘,#[batch_size,lstm_units]
        fc_input = tf.concat([conv_hori_out, choose_T_value, conv_vert_out], 1)
        fc_input_size = self.conv_hori_deep + self.lstm_units * 2
        #logits=self.fc_layer('fc1_',lstm_out,W_shape=[self.max_len*self.lstm_units,self.n_items],B_shape=[self.n_items])
        logits = self.fc_layer('fc1_', fc_input, W_shape=[fc_input_size, self.n_items],B_shape=[self.n_items])

        self.prediction=logits
        loss = tf.nn.softmax_cross_entropy_with_logits(labels=self.Y, logits=logits)
        self.sum_loss=tf.reduce_mean(loss)+ tf.add_n(tf.get_collection('losses'))
        self.lr = tf.train.exponential_decay(self.learning_rate,self.global_step,self.n_items / self.batch_size,self.decay_rate)
        self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.sum_loss, global_step=self.global_step)

    def metric_at_top_k(self):
        prediction = self.prediction
        prediction_scaled = tf.nn.softmax(prediction)
        top_k_value, top_k_index = tf.nn.top_k(prediction_scaled, k=self.top_k)
        self.top_k_index = tf.reshape(top_k_index, [-1, self.top_k])
        self.y_labels = tf.argmax(self.Y, axis=1)

