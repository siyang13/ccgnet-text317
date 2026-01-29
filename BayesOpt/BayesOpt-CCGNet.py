import sys
import os
import multiprocessing
import warnings

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore')

import tensorflow as tf
from rdkit import Chem
from rdkit.Chem import rdmolops

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
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


def verify_dir_exists(dirname):
    """检查并创建目录"""
    if not os.path.isdir(dirname):
        os.makedirs(dirname)


def is_valid_mol_block(mol_block):
    from rdkit import Chem

    if not mol_block or mol_block.strip() == "":
        return False

    try:
        mol = Chem.MolFromMolBlock(mol_block, sanitize=True)
        return mol is not None

    except Exception as e:
        print(f"分子校验失败: {str(e)}")
        return False


def get_sample_mol_blocks(sample_tag, table, mol_blocks):
    """
    从table中获取样本对应的分子块
    :param sample_tag: 样本标签（tag）
    :param table: Dataset的table属性
    :param mol_blocks: Dataset的mol_blocks属性
    :return: (block1, block2) 分子块元组
    """
    for items in table:
        if items[-1] == sample_tag:  # table中最后一列是tag
            block1 = mol_blocks.get(items[0], "")
            block2 = mol_blocks.get(items[1], "")
            return block1, block2
    return "", ""


def make_dataset(data_abs_path):
    """加载数据集并过滤无效分子"""
    # 1. 初始化Dataset
    data1 = Dataset(os.path.join(data_abs_path, 'CC_Table/CC_Table.tab'),
                    mol_blocks_dir=os.path.join(data_abs_path, 'Mol_Blocks.dir'))

    valid_table = []
    exception_samples = []
    for items in data1.table:
        sample_tag = items[-1]
        block1 = data1.mol_blocks.get(items[0], "")
        block2 = data1.mol_blocks.get(items[1], "")
        if is_valid_mol_block(block1) and is_valid_mol_block(block2):
            valid_table.append(items)
        else:
            exception_samples.append(sample_tag)
            print(f"提前过滤无效样本: {sample_tag}")
    data1.table = valid_table
    print(f"提前过滤完成：保留 {len(valid_table)} / {len(data1.table) + len(exception_samples)} 个样本")
    # ==========================================

    # 2. 先创建graph dataset和dataframe（
    data1.make_graph_dataset(Desc=1, A_type='OnlyCovalentBond', hbond=0, pipi_stack=0, contact=0, make_dataframe=True)

    # 3. 过滤无效分子
    valid_samples = {}
    for sample_name in data1.dataframe.keys():
        try:
            block1, block2 = get_sample_mol_blocks(sample_name, data1.table, data1.mol_blocks)
            if is_valid_mol_block(block1) and is_valid_mol_block(block2):
                valid_samples[sample_name] = data1.dataframe[sample_name]
            else:
                print(f"无效分子结构，跳过样本: {sample_name}")
        except Exception as e:
            print(f"加载样本失败 {sample_name}: {str(e)}，跳过")

    data1.dataframe = valid_samples

    print(f"数据集过滤完成：保留 {len(valid_samples)} 个有效样本")

    return data1


def build_model(
        blcok_1_size,
        blcok_2_size,
        blcok_3_size,
        blcok_4_size,
        blcok_5_size,
        mp_act_func,
        n_head,
        pred_layer_1_size,
        pred_layer_2_size,
        pred_layer_3_size,
        pred_act_func,
        pred_dropout_rate
):
    class Model(object):
        def build_model(self, inputs, is_training, global_step=None):
            V = inputs[0]
            A = inputs[1]
            labels = inputs[2]
            mask = inputs[3]
            graph_size = inputs[4]
            tags = inputs[5]
            global_state = inputs[6]
            subgraph_size = inputs[7]

            V, global_state = CCGBlock(V, A, global_state, subgraph_size, no_filters=blcok_1_size, act_func=mp_act_func,
                                       mask=mask, num_updates=global_step, is_training=is_training)
            if blcok_2_size is not None:
                V, global_state = CCGBlock(V, A, global_state, subgraph_size, no_filters=blcok_2_size,
                                           act_func=mp_act_func, mask=mask, num_updates=global_step,
                                           is_training=is_training)
            if blcok_3_size is not None:
                V, global_state = CCGBlock(V, A, global_state, subgraph_size, no_filters=blcok_3_size,
                                           act_func=mp_act_func, mask=mask, num_updates=global_step,
                                           is_training=is_training)
            if blcok_4_size is not None:
                V, global_state = CCGBlock(V, A, global_state, subgraph_size, no_filters=blcok_4_size,
                                           act_func=mp_act_func, mask=mask, num_updates=global_step,
                                           is_training=is_training)
            if blcok_5_size is not None:
                V, global_state = CCGBlock(V, A, global_state, subgraph_size, no_filters=blcok_5_size,
                                           act_func=mp_act_func, mask=mask, num_updates=global_step,
                                           is_training=is_training)
            # Readout
            V = ReadoutFunction(V, global_state, graph_size, num_head=n_head, is_training=is_training)
            # Prediction
            with tf.compat.v1.variable_scope('Predictive_FC_1') as scope:
                V = layers.make_embedding_layer(V, pred_layer_1_size)
                V = layers.make_bn(V, is_training, mask=None, num_updates=global_step)
                V = pred_act_func(V)
                V = tf.compat.v1.layers.dropout(V, pred_dropout_rate, training=is_training)
            if pred_layer_2_size is not None:
                with tf.compat.v1.variable_scope('Predictive_FC_2') as scope:
                    V = layers.make_embedding_layer(V, pred_layer_2_size)
                    V = layers.make_bn(V, is_training, mask=None, num_updates=global_step)
                    V = pred_act_func(V)
                    V = tf.compat.v1.layers.dropout(V, pred_dropout_rate, training=is_training)
            if pred_layer_3_size is not None:
                with tf.compat.v1.variable_scope('Predictive_FC_3') as scope:
                    V = layers.make_embedding_layer(V, pred_layer_3_size)
                    V = layers.make_bn(V, is_training, mask=None, num_updates=global_step)
                    V = pred_act_func(V)
                    V = tf.compat.v1.layers.dropout(V, pred_dropout_rate, training=is_training)
            # Output
            out = layers.make_embedding_layer(V, 2, name='final')
            return out, labels

    return Model()


def black_box_function(args_dict, root_abs_path, train_data, valid_data):
    """贝叶斯优化目标函数"""
    print('\n' + str(args_dict))
    tf.compat.v1.reset_default_graph()

    batch_size = args_dict['batch_size']
    blcok_1_size = args_dict['blcok_1_size']
    blcok_2_size = args_dict['blcok_2_size']
    blcok_3_size = args_dict['blcok_3_size']
    blcok_4_size = args_dict['blcok_4_size']
    blcok_5_size = args_dict['blcok_5_size']
    mp_act_func = args_dict['mp_act_func']
    n_head = args_dict['n_head']
    pred_layer_1_size = args_dict['pred_layer_1_size']
    pred_layer_2_size = args_dict['pred_layer_2_size']
    pred_layer_3_size = args_dict['pred_layer_3_size']
    pred_act_func = args_dict['pred_act_func']
    pred_dropout_rate = args_dict['pred_dropout_rate']

    # 构建保存路径
    snapshot_path = os.path.join(root_abs_path, 'bayes_snapshot/')
    model_name = 'BayesOpt-CCGNet/'
    verify_dir_exists(os.path.join(snapshot_path, model_name))

    if os.listdir(os.path.join(snapshot_path, model_name)) == []:
        dataset_name = 'Step_0/'
    else:
        l_ = [int(i.split('_')[1]) for i in os.listdir(os.path.join(snapshot_path, model_name)) if 'Step_' in i]
        dataset_name = 'Step_{}/'.format(max(l_) + 1)

    # 训练模型
    model = build_model(
        blcok_1_size,
        blcok_2_size,
        blcok_3_size,
        blcok_4_size,
        blcok_5_size,
        mp_act_func,
        n_head,
        pred_layer_1_size,
        pred_layer_2_size,
        pred_layer_3_size,
        pred_act_func,
        pred_dropout_rate
    )

    model = exp.Model(model, train_data, valid_data, with_test=False, snapshot_path=snapshot_path, use_subgraph=True,
                      model_name=model_name, dataset_name=dataset_name + '/time_0')
    history = model.fit(num_epoch=100, save_info=True, save_att=False, silence=True,  # silence=True减少日志输出
                        train_batch_size=batch_size, max_to_keep=1, metric='loss')
    loss = min(history['valid_cross_entropy'])
    tf.compat.v1.reset_default_graph()
    print('\nLoss: {}'.format(loss))
    return loss


if __name__ == '__main__':
    multiprocessing.freeze_support()
    tf.compat.v1.disable_eager_execution()
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)  # 屏蔽TF日志

    # 基础路径配置
    root_abs_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    data_abs_path = os.path.join(root_abs_path, "data")

    # 检查Fold_10.dir文件
    fold_10_path = os.path.join(data_abs_path, 'Fold_10.dir')
    if not os.path.exists(fold_10_path):
        raise FileNotFoundError(f"未找到Fold_10.dir文件，请确认路径：{fold_10_path}")
    fold_10 = eval(open(fold_10_path, 'r', encoding='utf-8').read())

    # 预加载并过滤数据集
    print("预加载数据集，过滤无效分子样本...")
    temp_data = make_dataset(data_abs_path)
    valid_sample_names = list(temp_data.dataframe.keys())

    # 过滤fold-0样本
    fold_samples = fold_10['fold-0']['train'] + fold_10['fold-0']['valid']
    Samples = [name for name in fold_samples if name in valid_sample_names]
    print(f"过滤前样本数：{len(fold_samples)}，过滤后有效样本数：{len(Samples)}")

    # 数据划分
    random.shuffle(Samples)
    num_sample = len(Samples)
    train_num = int(0.9 * num_sample)
    train_samples = Samples[:train_num]
    valid_samples = Samples[train_num:]

    data = temp_data
    train_data, valid_data = data.split(train_samples=train_samples, valid_samples=valid_samples)

    # 导入hyperopt
    from hyperopt import fmin, tpe, Trials, hp
    import hyperopt.pyll.stochastic

    # 定义超参数搜索空间
    args_dict = {
        'batch_size': hp.choice('batch_size', (128,)),
        'blcok_1_size': hp.choice('blcok_1_size', (16, 32, 64, 128, 256)),
        'blcok_2_size': hp.choice('blcok_2_size', (16, 32, 64, 128, 256, None)),
        'blcok_3_size': hp.choice('blcok_3_size', (16, 32, 64, 128, 256, None)),
        'blcok_4_size': hp.choice('blcok_4_size', (16, 32, 64, 128, 256, None)),
        'blcok_5_size': hp.choice('blcok_5_size', (16, 32, 64, 128, 256, None)),
        'mp_act_func': hp.choice('mp_act_func', (tf.nn.relu,)),
        'n_head': hp.choice('n_head', (1, 2, 3, 4, 5, 6, 7, 8)),  # 减少搜索空间加速测试
        'pred_layer_1_size': hp.choice('pred_layer_1_size', (64, 128, 256)),
        'pred_layer_2_size': hp.choice('pred_layer_2_size', (64, 128, 256, None)),
        'pred_layer_3_size': hp.choice('pred_layer_3_size', (64, 128, 256, None)),
        'pred_act_func': hp.choice('pred_act_func', (tf.nn.relu,)),
        'pred_dropout_rate': hp.uniform('pred_dropout_rate', 0.0, 0.5)  # 缩小dropout范围
    }

    trials = Trials()
    best = fmin(
        fn=lambda args: black_box_function(args, root_abs_path, train_data, valid_data),
        space=args_dict,
        algo=tpe.suggest,
        max_evals=100,
        trials=trials,
        trials_save_file='trials_save_file-ccgnet'
    )

    # 保存最优参数
    print('\nbest parameters:')
    print(best)
    best_params_path = os.path.join(root_abs_path, 'bayes_snapshot/BayesOpt-CCGNet/best_params.npy')
    # 确保保存目录存在
    verify_dir_exists(os.path.dirname(best_params_path))
    np.save(best_params_path, best)
    print(f'最优超参数已保存至：{best_params_path}')