import sys
import os
import multiprocessing
import warnings
import random

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore')

multiprocessing.freeze_support()

import tensorflow as tf

tf.compat.v1.disable_eager_execution()
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

import numpy as np
import time
from sklearn.metrics import balanced_accuracy_score

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from ccgnet import experiment as exp
from ccgnet import layers
from ccgnet.Dataset import Dataset, DataLoader


def verify_dir_exists(dirname):
    if not os.path.isdir(dirname):
        os.makedirs(dirname, exist_ok=True)
        print(f"创建目录：{dirname}")
    else:
        print(f"目录已存在：{dirname}")


def build_model(
        graphcnn_layer_1_size,
        graphcnn_layer_2_size,
        graphcnn_layer_3_size,
        graphcnn_act_fun,
        graph_pool_1_size,
        graph_pool_2_size,
        graph_pool_3_size,
        graph_pool_act_fun,
        dense_layer_1_size,
        dense_layer_2_size,
        dense_layer_3_size,
        dense_act_func,
        dense_dropout
):
    mask_judge = (graph_pool_1_size, graph_pool_2_size, graph_pool_3_size)
    print(f"Pool层配置：{mask_judge}")

    class Model(object):
        def build_model(self, inputs, is_training, global_step):
            V = inputs[0]
            A = inputs[1]
            labels = inputs[2]
            mask = inputs[3]
            graph_size = inputs[4]
            tags = inputs[5]

            # Graph-CNN stage
            V = layers.make_graphcnn_layer(V, A, graphcnn_layer_1_size)
            V = layers.make_bn(V, is_training, mask=mask, num_updates=global_step)
            V = graphcnn_act_fun(V)

            if graph_pool_1_size is not None:
                V_pool, A = layers.make_graph_embed_pooling(V, A, mask=mask, no_vertices=graph_pool_1_size)
                V = layers.make_bn(V_pool, is_training, mask=None, num_updates=global_step)
                V = graph_pool_act_fun(V)

            if graphcnn_layer_2_size is not None:
                m = None if mask_judge[0] is not None else mask
                V = layers.make_graphcnn_layer(V, A, graphcnn_layer_2_size)
                V = layers.make_bn(V, is_training, mask=m, num_updates=global_step)
                V = graphcnn_act_fun(V)

            if graph_pool_2_size is not None:
                m = None if mask_judge[0] is not None else mask
                V_pool, A = layers.make_graph_embed_pooling(V, A, mask=m, no_vertices=graph_pool_2_size)
                V = layers.make_bn(V_pool, is_training, mask=None, num_updates=global_step)
                V = graph_pool_act_fun(V)

            if graphcnn_layer_3_size is not None:
                m = None if (mask_judge[1] is not None or mask_judge[0] is not None) else mask
                V = layers.make_graphcnn_layer(V, A, graphcnn_layer_3_size)
                V = layers.make_bn(V, is_training, mask=m, num_updates=global_step)
                V = graphcnn_act_fun(V)

            # 最终Pool层
            m = None if (mask_judge[1] is not None or mask_judge[0] is not None) else mask
            V_pool, A = layers.make_graph_embed_pooling(V, A, mask=m, no_vertices=graph_pool_3_size)
            V = layers.make_bn(V_pool, is_training, mask=None, num_updates=global_step)
            V = graph_pool_act_fun(V)

            # Predictive Stage
            no_input_features = int(np.prod(V.get_shape()[1:]))
            V = tf.reshape(V, [-1, no_input_features])

            # 全连接层1
            V = layers.make_embedding_layer(V, dense_layer_1_size, name='FC-1')
            V = layers.make_bn(V, is_training, mask=None, num_updates=global_step)
            V = dense_act_func(V)
            V = tf.compat.v1.layers.dropout(V, dense_dropout, training=is_training)

            # 全连接层2（可选）
            if dense_layer_2_size is not None:
                V = layers.make_embedding_layer(V, dense_layer_2_size, name='FC-2')
                V = layers.make_bn(V, is_training, mask=None, num_updates=global_step)
                V = dense_act_func(V)
                V = tf.compat.v1.layers.dropout(V, dense_dropout, training=is_training)

            # 全连接层3（可选）
            if dense_layer_3_size is not None:
                V = layers.make_embedding_layer(V, dense_layer_3_size, name='FC-3')
                V = layers.make_bn(V, is_training, mask=None, num_updates=global_step)
                V = dense_act_func(V)
                V = tf.compat.v1.layers.dropout(V, dense_dropout, training=is_training)

            # 输出层（二分类）
            out = layers.make_embedding_layer(V, 2, name='final')
            return out, labels

    return Model()


# ===================== 贝叶斯优化目标函数 =====================
def black_box_function(args_dict):
    tf.compat.v1.reset_default_graph()

    # 解包超参数
    batch_size = args_dict['batch_size']
    graphcnn_layer_1_size = args_dict['graphcnn_layer_1_size']
    graphcnn_layer_2_size = args_dict['graphcnn_layer_2_size']
    graphcnn_layer_3_size = args_dict['graphcnn_layer_3_size']
    graphcnn_act_fun = args_dict['graphcnn_act_fun']
    graph_pool_1_size = args_dict['graph_pool_1_size']
    graph_pool_2_size = args_dict['graph_pool_2_size']
    graph_pool_3_size = args_dict['graph_pool_3_size']
    graph_pool_act_fun = args_dict['graph_pool_act_fun']
    dense_layer_1_size = args_dict['dense_layer_1_size']
    dense_layer_2_size = args_dict['dense_layer_2_size']
    dense_layer_3_size = args_dict['dense_layer_3_size']
    dense_act_func = args_dict['dense_act_func']
    dense_dropout = args_dict['dense_dropout']

    # 构建保存路径（修复路径拼接）
    snapshot_path = os.path.join(abs_path, 'bayes_snapshot/')
    model_name = 'BayesOpt-GraphCNN/'
    model_dir = os.path.join(snapshot_path, model_name)
    verify_dir_exists(model_dir)

    # 生成步骤名称
    if os.listdir(model_dir) == []:
        dataset_name = 'Step_0/'
    else:
        step_list = [int(i.split('_')[1]) for i in os.listdir(model_dir) if 'Step_' in i]
        dataset_name = f'Step_{max(step_list) + 1}/'

    # 构建模型
    model = build_model(
        graphcnn_layer_1_size, graphcnn_layer_2_size, graphcnn_layer_3_size,
        graphcnn_act_fun, graph_pool_1_size, graph_pool_2_size, graph_pool_3_size,
        graph_pool_act_fun, dense_layer_1_size, dense_layer_2_size, dense_layer_3_size,
        dense_act_func, dense_dropout
    )

    # 初始化实验模型
    exp_model = exp.Model(
        model, train_data, valid_data, with_test=False,
        snapshot_path=snapshot_path, use_subgraph=False, use_desc=False, build_fc=False,
        model_name=model_name, dataset_name=os.path.join(dataset_name, 'time_0')
    )

    # 训练模型
    history = exp_model.fit(
        num_epoch=100, save_info=True, save_att=False, silence=False,
        train_batch_size=batch_size, max_to_keep=1, metric='loss'
    )

    # 获取最小验证损失
    loss = min(history['valid_cross_entropy'])
    tf.compat.v1.reset_default_graph()

    # 打印结果
    print(f'\n当前超参数损失：{loss:.4f}')
    print(f'当前超参数：{args_dict}')
    return loss


def make_dataset():
    """加载数据集，添加文件存在性检查"""
    # 构建数据集路径
    table_path = os.path.join(abs_path, 'data', 'CC_Table', 'CC_Table.tab')
    mol_blocks_path = os.path.join(abs_path, 'data','Mol_Blocks.dir')

    # 检查文件是否存在
    for file_path, file_name in [(table_path, 'CC_Table.tab'), (mol_blocks_path, 'Mol_Blocks.dir')]:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"找不到{file_name}文件，请确认路径：{file_path}")

    # 加载数据集
    print(f"\n开始加载数据集：")
    print(f"- 表格路径：{table_path}")
    print(f"- 分子块路径：{mol_blocks_path}")
    data1 = Dataset(table_path, mol_blocks_dir=mol_blocks_path)

    # 生成图数据集
    data1.make_graph_dataset(
        Desc=0, A_type='OnlyCovalentBond', hbond=0, pipi_stack=0, contact=0,
        make_dataframe=True
    )
    return data1


if __name__ == '__main__':
    # 1. 基础路径配置
    abs_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    print(f"项目根路径：{abs_path}")

    # 2. 检查并加载Fold_10.dir
    fold_10_path = os.path.join(abs_path, 'data', 'Fold_10.dir')
    if not os.path.exists(fold_10_path):
        raise FileNotFoundError(
            f"找不到Fold_10.dir文件！\n"
            f"当前查找路径：{fold_10_path}\n"
            f"请确认文件位置是否为：D:\\课业\\毕业设计\\ccgnet-main_shiyan\\ccgnet-main\\data\\Fold_10.dir"
        )

    # 加载Fold_10.dir
    fold_10 = eval(open(fold_10_path, 'r', encoding='utf-8').read())
    print(f"成功加载Fold_10.dir，共{len(fold_10)}个fold")

    # 3. 加载数据集
    data = make_dataset()

    # 4. 数据划分
    Samples = fold_10['fold-0']['train'] + fold_10['fold-0']['valid']
    print(f"Fold-0总样本数：{len(Samples)}")

    # 随机打乱（固定种子确保可复现）
    random.seed(42)
    random.shuffle(Samples)
    num_sample = len(Samples)
    train_num = int(0.9 * num_sample)
    train_samples = Samples[:train_num]
    valid_samples = Samples[train_num:]

    print(f"训练集样本数：{len(train_samples)}")
    print(f"验证集样本数：{len(valid_samples)}")

    # 划分训练/验证数据
    train_data, valid_data = data.split(
        train_samples=train_samples,
        valid_samples=valid_samples
    )

    # 5. 贝叶斯优化配置
    from hyperopt import fmin, tpe, Trials, hp

    # 定义超参数搜索空间
    args_dict = {
        'batch_size': hp.choice('batch_size', (128,)),
        'graphcnn_layer_1_size': hp.choice('graphcnn_layer_1_size', (16, 32, 64, 128, 256)),
        'graphcnn_layer_2_size': hp.choice('graphcnn_layer_2_size', (16, 32, 64, 128, 256, None)),
        'graphcnn_layer_3_size': hp.choice('graphcnn_layer_3_size', (16, 32, 64, 128, 256, None)),
        'graphcnn_act_fun': hp.choice('graphcnn_act_fun', (tf.nn.relu,)),
        'graph_pool_1_size': hp.choice('graph_pool_1_size', (8, 16, 32, None)),
        'graph_pool_2_size': hp.choice('graph_pool_2_size', (8, 16, 32, None)),
        'graph_pool_3_size': hp.choice('graph_pool_3_size', (8, 16, 32)),
        'graph_pool_act_fun': hp.choice('graph_pool_act_fun', (tf.nn.relu,)),
        'dense_layer_1_size': hp.choice('dense_layer_1_size', (64, 128, 256, 512)),
        'dense_layer_2_size': hp.choice('dense_layer_2_size', (64, 128, 256, 512, None)),
        'dense_layer_3_size': hp.choice('dense_layer_3_size', (64, 128, 256, 512, None)),
        'dense_act_func': hp.choice('dense_act_func', (tf.nn.relu,)),
        'dense_dropout': hp.uniform('dense_dropout', 0.0, 0.75)
    }

    # 运行贝叶斯优化
    trials = Trials()
    print("\n开始贝叶斯优化（max_evals=100）...")
    best = fmin(
        fn=black_box_function,
        space=args_dict,
        algo=tpe.suggest,
        max_evals=100,
        trials=trials,
        trials_save_file='trials_save_file-graphcnn'
    )

    # 打印最优参数
    print('\n最优超参数：')
    print(best)

    # 保存最优参数（可选）
    best_params_path = os.path.join(abs_path, 'bayes_snapshot', 'BayesOpt-GraphCNN', 'best_params.npy')
    verify_dir_exists(os.path.dirname(best_params_path))
    np.save(best_params_path, best)
    print(f"最优参数已保存至：{best_params_path}")