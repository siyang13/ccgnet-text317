import sys
import os
import multiprocessing  # 多进程支持

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import tensorflow as tf
from ccgnet import experiment as exp
from ccgnet.finetune import *
from ccgnet import layers
from ccgnet.layers import *
import numpy as np
import time
import random
from sklearn.metrics import balanced_accuracy_score
from ccgnet.Dataset import Dataset, DataLoader
from Featurize.Coformer import Coformer
from Featurize.Cocrystal import Cocrystal
from hyperopt import hp, fmin, tpe, Trials


def verify_dir_exists(dirname):
    """确保目录存在，不存在则创建"""
    dir_path = os.path.dirname(dirname)
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)


def make_dataset(fp_size, radii, abs_path):
    data1 = Dataset(os.path.join(abs_path, 'CC_Table/CC_Table.tab'),
                    mol_blocks_dir=os.path.join(abs_path, 'Mol_Blocks.dir'))
    data1.make_embedding_dataset(fp_type='ecfp', nBits=fp_size, radii=radii,
                                 processes=4, make_dataframe=True)
    return data1


def build_model(
        layer_1_size,
        layer_2_size,
        layer_3_size,
        act_func,
        dropout,
        merge,
        forward_layer_1_size,
        forward_layer_2_size,
        forward_layer_3_size,
        forward_act_func,
        forward_dropout
):

    class DNN_5(object):
        def build_model(self, inputs, is_training, global_step=None):
            fps = inputs[0]
            labels = inputs[1]
            tags = inputs[2]

            fps = tf.reshape(fps, [-1, int(fps.get_shape()[-1].value / 2)])
            with tf.compat.v1.variable_scope('FC_1') as scope:
                fps = tf.compat.v1.layers.dense(fps, layer_1_size)
                fps = tf.compat.v1.layers.batch_normalization(fps, training=is_training)
                fps = act_func(fps)
                fps = tf.compat.v1.layers.dropout(fps, dropout, training=is_training)
            if layer_2_size != None:
                with tf.compat.v1.variable_scope('FC_2') as scope:
                    fps = tf.compat.v1.layers.dense(fps, layer_2_size)
                    fps = tf.compat.v1.layers.batch_normalization(fps, training=is_training)
                    fps = act_func(fps)
                    fps = tf.compat.v1.layers.dropout(fps, dropout, training=is_training)
            if layer_3_size != None:
                with tf.compat.v1.variable_scope('FC_3') as scope:
                    fps = tf.compat.v1.layers.dense(fps, layer_3_size)
                    fps = tf.compat.v1.layers.batch_normalization(fps, training=is_training)
                    fps = act_func(fps)
                    fps = tf.compat.v1.layers.dropout(fps, dropout, training=is_training)

            if merge == 'add':
                with tf.compat.v1.variable_scope('merge_add') as scope:
                    fp_size = fps.get_shape()[-1].value
                    fps = tf.reshape(fps, [-1, 2, fp_size])
                    fps = tf.reduce_sum(fps, axis=1)
            elif merge == 'concat':
                with tf.compat.v1.variable_scope('merge_concat') as scope:
                    fp_size = fps.get_shape()[-1].value
                    fps = tf.reshape(fps, [-1, fp_size * 2])

            with tf.compat.v1.variable_scope('Forward_FC_1') as scope:
                fps = tf.compat.v1.layers.dense(fps, forward_layer_1_size)
                fps = tf.compat.v1.layers.batch_normalization(fps, training=is_training)
                fps = forward_act_func(fps)
                fps = tf.compat.v1.layers.dropout(fps, dropout, training=is_training)
            if forward_layer_2_size != None:
                with tf.compat.v1.variable_scope('Forward_FC_2') as scope:
                    fps = tf.compat.v1.layers.dense(fps, forward_layer_2_size)
                    fps = tf.compat.v1.layers.batch_normalization(fps, training=is_training)
                    fps = forward_act_func(fps)
                    fps = tf.compat.v1.layers.dropout(fps, dropout, training=is_training)
            if forward_layer_3_size != None:
                with tf.compat.v1.variable_scope('Forward_FC_3') as scope:
                    fps = tf.compat.v1.layers.dense(fps, forward_layer_3_size)
                    fps = tf.compat.v1.layers.batch_normalization(fps, training=is_training)
                    fps = forward_act_func(fps)
                    fps = tf.compat.v1.layers.dropout(fps, dropout, training=is_training)

            fps = layers.make_fc_layer(fps, 2, is_training=is_training, with_bn=False, act_func=None)
            return fps, labels

    return DNN_5()


def black_box_function(args_dict, train_samples, valid_samples, data_abs_path, root_abs_path):
    tf.compat.v1.reset_default_graph()

    # 解析超参数
    fp_size = args_dict['fp_size']
    radii = args_dict['fp_radii']
    batch_size = args_dict['batch_size']
    layer_1_size = args_dict['layer_1_size']
    layer_2_size = args_dict['layer_2_size']
    layer_3_size = args_dict['layer_3_size']
    act_fun = args_dict['act_fun']
    dropout = args_dict['dropout']
    merge = args_dict['merge']
    forward_layer_1_size = args_dict['forward_layer_1_size']
    forward_layer_2_size = args_dict['forward_layer_2_size']
    forward_layer_3_size = args_dict['forward_layer_3_size']
    forward_act_fun = args_dict['forward_act_fun']
    forward_dropout = args_dict['forward_dropout']

    # 加载数据集
    data = make_dataset(fp_size, radii, data_abs_path)
    train_data, valid_data = data.split(train_samples=train_samples, valid_samples=valid_samples, with_fps=True)

    snapshot_path = os.path.join(root_abs_path, 'bayes_snapshot/')
    model_name = 'BayesOpt-FP/'
    verify_dir_exists(os.path.join(snapshot_path, model_name))

    # 生成保存目录
    model_full_path = os.path.join(snapshot_path, model_name)
    if not os.path.exists(model_full_path):
        os.makedirs(model_full_path)
    if os.listdir(model_full_path) == []:
        dataset_name = 'Step_0/'
    else:
        l_ = [int(i.split('_')[1]) for i in os.listdir(model_full_path) if 'Step_' in i]
        dataset_name = 'Step_{}/'.format(max(l_) + 1)

    # 训练模型
    tf.compat.v1.reset_default_graph()
    model = build_model(layer_1_size, layer_2_size, layer_3_size, act_fun, dropout, merge, forward_layer_1_size,
                        forward_layer_2_size, forward_layer_3_size, forward_act_fun, forward_dropout)
    exp_model = exp.Model(model, train_data, valid_data, with_test=False, snapshot_path=snapshot_path,
                          use_subgraph=False, use_desc=False, build_fc=True, model_name=model_name,
                          dataset_name=dataset_name + '/time_0')
    history = exp_model.fit(num_epoch=100, save_info=True, save_att=False, silence=0,
                            train_batch_size=batch_size, max_to_keep=1, metric='loss')

    # 计算最小验证损失
    loss = min(history['valid_cross_entropy'])
    tf.compat.v1.reset_default_graph()
    print('\nLoss: {}'.format(loss))
    print('当前超参数：', args_dict)
    return loss


if __name__ == '__main__':
    multiprocessing.freeze_support()
    tf.compat.v1.disable_eager_execution()

    # 定义超参数搜索空间
    args_dict = {
        'fp_size': hp.choice('fp_size', [128, 256, 512, 1024, 2048, 4096]),
        'fp_radii': hp.choice('fp_radii', (1, 2, 3)),
        'batch_size': hp.choice('batch_size', (64, 128, 256)),
        'layer_1_size': hp.choice('layer_1_size', (128, 256, 512, 1024, 2048)),
        'layer_2_size': hp.choice('layer_2_size', (128, 256, 512, 1024, 2048, None)),
        'layer_3_size': hp.choice('layer_3_size', (128, 256, 512, 1024, 2048, None)),
        'act_fun': hp.choice('act_fun', (tf.nn.relu, tf.nn.elu, tf.nn.tanh)),
        'dropout': hp.uniform('dropout', 0.0, 0.75),
        'merge': hp.choice('merge', ('add', 'concat')),
        'forward_layer_1_size': hp.choice('forward_layer_1_size', (128, 256, 512, 1024, 2048)),
        'forward_layer_2_size': hp.choice('forward_layer_2_size', (128, 256, 512, 1024, 2048, None)),
        'forward_layer_3_size': hp.choice('forward_layer_3_size', (128, 256, 512, 1024, 2048, None)),
        'forward_act_fun': hp.choice('forward_act_fun', (tf.nn.relu, tf.nn.elu, tf.nn.tanh)),
        'forward_dropout': hp.uniform('forward_dropout', 0.0, 0.75),
    }

    root_abs_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))  # 项目根目录
    data_abs_path = os.path.join(root_abs_path, "data")  # data子目录

    # 检查Fold_10.dir文件是否存在
    fold_10_path = os.path.join(data_abs_path, 'Fold_10.dir')
    if not os.path.exists(fold_10_path):
        raise FileNotFoundError(f"未找到Fold_10.dir文件，请确认路径：{fold_10_path}")
    fold_10 = eval(open(fold_10_path).read())

    print("预加载数据集，过滤无效分子样本...")
    # 预加载数据集，解析指纹以获取有效样本列表
    temp_data = Dataset(os.path.join(data_abs_path, 'CC_Table/CC_Table.tab'),
                        mol_blocks_dir=os.path.join(data_abs_path, 'Mol_Blocks.dir'))
    temp_data.make_embedding_dataset(fp_type='ecfp', nBits=1024, radii=2,
                                     processes=4, make_dataframe=True)
    valid_sample_names = list(temp_data.dataframe.keys())

    # 过滤fold-0的样本，仅保留有效样本
    fold_samples = fold_10['fold-0']['train'] + fold_10['fold-0']['valid']
    Samples = [name for name in fold_samples if name in valid_sample_names]
    print(f"过滤前样本数：{len(fold_samples)}，过滤后有效样本数：{len(Samples)}")

    # 划分训练/验证集
    random.shuffle(Samples)
    num_sample = len(Samples)
    train_num = int(0.9 * num_sample)
    train_samples = Samples[:train_num]
    valid_samples = Samples[train_num:]

    print("开始贝叶斯优化搜索超参数...")
    trials = Trials()
    best = fmin(
        fn=lambda args: black_box_function(args, train_samples, valid_samples, data_abs_path, root_abs_path),
        space=args_dict,
        algo=tpe.suggest,
        max_evals=100,
        trials=trials,
        trials_save_file='trials_save_file-FP')

    # 输出最优超参数
    print('\n========== 最优超参数 ==========')
    print(best)
    # 保存最优超参数
    np.save(os.path.join(root_abs_path, 'bayes_snapshot/BayesOpt-FP/best_params.npy'), best)
    print('最优超参数已保存至：', os.path.join(root_abs_path, 'bayes_snapshot/BayesOpt-FP/best_params.npy'))