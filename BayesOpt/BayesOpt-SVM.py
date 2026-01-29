import sys
import os
# 新增：导入multiprocessing，添加freeze_support（Windows下推荐）
import multiprocessing

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, accuracy_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from ccgnet.Dataset import Dataset, DataLoader
import numpy as np
from hyperopt import fmin, tpe, Trials, hp, STATUS_OK
import hyperopt.pyll.stochastic
import random


def make_dataset():
    data1 = Dataset(abs_path + '/data/CC_Table/CC_Table.tab', mol_blocks_dir=abs_path + '/data/Mol_Blocks.dir')
    data1.make_graph_dataset(Desc=1, A_type='OnlyCovalentBond', hbond=0, pipi_stack=0, contact=0, make_dataframe=True)
    return data1


def black_box_function(args_dict):
    clf = SVC(**args_dict, probability=0)
    clf.fit(x_train, y_train)
    valid_pred_labels = clf.predict(x_valid)
    valid_acc = accuracy_score(y_valid, valid_pred_labels)
    print((str(valid_acc) + ': {}').format(args_dict))
    return {'loss': 1 - valid_acc, 'status': STATUS_OK}


if __name__ == '__main__':
    multiprocessing.freeze_support()

    # 定义超参数空间
    space4svc = {
        'C': hp.uniform('C', 0, 20),
        'kernel': hp.choice('kernel', ['linear', 'sigmoid', 'poly', 'rbf']),
        'gamma': hp.choice('gamma', ['scale', 'auto', hp.uniform('gamma-1', 0, 1)]),
        'degree': hp.choice('degree', range(1, 21)),
        'coef0': hp.uniform('coef0', 0, 10)
    }

    abs_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    fold_10 = eval(open(abs_path + '/data/Fold_10.dir').read())

    Samples = fold_10['fold-0']['train'] + fold_10['fold-0']['valid']
    random.seed(10)
    random.shuffle(Samples)
    num_sample = len(Samples)
    train_num = int(0.9 * num_sample)
    train_samples = Samples[:train_num]
    valid_samples = Samples[train_num:]

    # 生成数据集
    data = make_dataset()

    # 拆分训练/验证数据
    train_data, valid_data = data.split(train_samples=train_samples, valid_samples=valid_samples)
    x_train, y_train = train_data[-2], train_data[2]
    x_valid, y_valid = valid_data[-2], valid_data[2]

    # 执行贝叶斯优化
    trials = Trials()
    best = fmin(black_box_function, space4svc, algo=tpe.suggest, max_evals=100, trials=trials)
    print("最优超参数索引：", best)


    kernel_list = ['linear', 'sigmoid', 'poly', 'rbf']
    gamma_list = ['scale', 'auto', None]
    best_params = {
        'C': best['C'],
        'kernel': kernel_list[best['kernel']],
        'gamma': gamma_list[best['gamma']] if best['gamma'] < 2 else trials.best_trial['misc']['vals']['gamma-1'][0],
        'degree': best['degree'] + 1,
        'coef0': best['coef0']
    }
    print("最优超参数实际值：", best_params)