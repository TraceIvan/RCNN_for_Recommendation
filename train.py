import numpy as np
import tensorflow as tf
import argparse
import random
import os
from utils import load_train,load_valid,cal_eval
from model import RCNN


def parseArgs():
    parser = argparse.ArgumentParser(description='RCNN args')
    parser.add_argument('--lstm_layers', default=1, type=int)
    parser.add_argument('--lstm_units', default=128, type=int)
    parser.add_argument('--lstm_act', default='tanh', type=str)
    parser.add_argument('--conv_act', default='relu', type=str)
    parser.add_argument('--fc_act', default='relu', type=str)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--max_len', default=16, type=int)
    parser.add_argument('--top_k', default=20, type=int)
    parser.add_argument('--conv_hori_w', default=32, type=int)
    parser.add_argument('--conv_hori_deep', default=10, type=int)
    parser.add_argument('--conv_vert_k', default=4, type=int)
    parser.add_argument('--conv_vert_deep', default=1, type=int)
    parser.add_argument('--choose_T', default=16, type=int)#must <=max_len
    parser.add_argument('--n_items', default=2000, type=int)
    parser.add_argument('--is_training', default=True, type=bool)
    parser.add_argument('--l2_reg', default=0.0001, type=float)
    parser.add_argument('--learning_rate', default=0.001, type=float)
    parser.add_argument('--decay_rate', default=0.99, type=float)
    parser.add_argument('--keep_prob', default=0.8, type=float)
    parser.add_argument('--max_to_keep', default=20, type=int)
    parser.add_argument('--is_store', default=False, type=bool)
    parser.add_argument('--checkpoint_dir', default='save/', type=str)
    args = parser.parse_args()
    return args

def eval_validation(model,sess,batch_size):
    valid_batches = len(valid_x) // batch_size
    valid_loss=0.0
    HR, NDCG, MRR = 0.0, 0.0, 0.0
    for i in range(valid_batches):
        x = valid_x[i * batch_size: (i + 1) * batch_size]
        y = valid_y[i * batch_size: (i + 1) * batch_size]
        fetches = [model.sum_loss,model.top_k_index,model.y_labels]
        feed_dict = {model.X: x, model.Y: y}
        loss,top_k_index,labels= sess.run(fetches, feed_dict)
        valid_loss+=loss
        hr,ndcg,mrr=cal_eval(top_k_index,labels)
        HR += hr
        NDCG += ndcg
        MRR += mrr
    return valid_loss/valid_batches,HR/valid_batches,NDCG/valid_batches,MRR/valid_batches
def train_RCNN(args):
    train_x, train_y, n_items = load_train(args.max_len)
    args.n_items = n_items
    data = list(zip(train_x, train_y))
    random.shuffle(data)
    train_x, train_y = zip(*data)
    num_batches = len(train_x) // args.batch_size
    global valid_x
    global valid_y
    valid_x, valid_y, _ = load_valid(args.max_len)


    print('#Items: {}'.format(n_items))
    print('#Training Nums: {}'.format(len(train_x)))

    gpu_config = tf.ConfigProto()
    gpu_config.gpu_options.allow_growth = True
    with tf.Session(config=gpu_config) as sess:
        model = RCNN(args)
        if args.is_store:
            saver = tf.train.Saver(tf.global_variables())
            ckpt = tf.train.get_checkpoint_state(args.checkpoint_dir)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                print('Restore model from {} successfully!'.format(ckpt.model_checkpoint_path))
            else:
                print('Restore model from {} failed!'.format(args.checkpoint_dir))
                return
        else:
            sess.run(tf.global_variables_initializer())
        best_epoch = -1
        best_step = -1
        best_loss = np.inf
        best_HR = np.inf

        max_stay,stay_cnt=20,0
        losses=0.0
        for epoch in range(args.epochs):
            for i in range(num_batches):
                x=train_x[i*args.batch_size: (i+1)*args.batch_size]
                y=train_y[i*args.batch_size: (i+1)*args.batch_size]
                fetches=[model.sum_loss, model.global_step, model.lr, model.train_op]
                feed_dict={model.X: x, model.Y: y}
                loss, step, lr, _ = sess.run(fetches, feed_dict)
                losses+=loss
                if step%50==0:
                    print('Epoch-{}\tstep-{}\tlr:{:.6f}\tloss: {:.6f}'.format(epoch+1, step,lr, losses/50))
                    losses=0.0
                if step%1000==0:
                    valid_loss,HR, NDCG, MRR=eval_validation(model,sess,args.batch_size)
                    print('step-{}\teval_validation\tloss:{:6f}\tHR@{}:{:.6f}\tNDCG@{}:{:.6f}\tMRR@{}:{:.6f}'
                          .format(step,valid_loss,args.top_k,HR,args.top_k,NDCG,args.top_k,MRR))
                    if HR>best_HR or (valid_loss<best_loss and HR>0.0):
                        best_HR=HR
                        best_loss=valid_loss
                        best_epoch=epoch+1
                        best_step=step
                        stay_cnt=0
                        ckpt_path =args.checkpoint_dir+'model.ckpt'
                        model.saver.save(sess, ckpt_path, global_step=step)
                        print("model saved to {}".format(ckpt_path))
                    else:
                        stay_cnt+=1
                        if stay_cnt>=max_stay:
                            break
            if stay_cnt >= max_stay:
                break
        print("best model at:epoch-{}\tstep-{}\tloss:{:.6f}\tHR@{}:{:.6f}".format(best_epoch,best_step,best_loss,args.top_k,best_HR))

if __name__=='__main__':
    args=parseArgs()
    train_RCNN(args)






