import tensorflow as tf
from utils import load_test,cal_eval
from model import RCNN
from train import parseArgs

def test(args):
    test_x, test_y, n_items = load_test(args.max_len)
    args.n_items = n_items
    test_batches = len(test_x) // args.batch_size
    HR, NDCG, MRR = 0.0, 0.0, 0.0
    test_loss=0.0

    model = RCNN(args)
    gpu_config = tf.ConfigProto()
    gpu_config.gpu_options.allow_growth = True
    with tf.Session(config=gpu_config) as sess:
        saver = tf.train.Saver(tf.global_variables())
        ckpt = tf.train.get_checkpoint_state(args.checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            print('Restore model from {} successfully!'.format(args.checkpoint_dir))
        else:
            print('Restore model from {} failed!'.format(args.checkpoint_dir))
            return
        for i in range(test_batches):
            x = test_x[i * args.batch_size: (i + 1) * args.batch_size]
            y = test_y[i * args.batch_size: (i + 1) * args.batch_size]
            fetches = [model.sum_loss, model.top_k_index, model.y_labels]
            feed_dict = {model.X: x, model.Y: y}
            loss, top_k_index, labels = sess.run(fetches, feed_dict)
            test_loss += loss
            hr, ndcg, mrr = cal_eval(top_k_index, labels)
            HR += hr
            NDCG += ndcg
            MRR += mrr
    print('loss:{:6f}\tHR@{}:{:.6f}\tNDCG@{}:{:.6f}\tMRR@{}:{:.6f}'.format(test_loss,args.top_k,HR,args.top_k,NDCG,args.top_k,MRR))

if __name__=='__main__':
    args=parseArgs()
    args.is_store=True
    args.is_training=False
    test(args)