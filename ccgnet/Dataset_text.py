"""
支持文本特征的Dataset类
"""

import os
import sys
import numpy as np
import pandas as pd

try:
    from .Dataset import Dataset
except ImportError:
    import sys

    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from ccgnet.Dataset import Dataset


class DatasetText(Dataset):
    """支持文本特征的Dataset类"""

    def __init__(self, table_dir, mol_blocks_dir='./data/Mol_Blocks.dir',
                 text_feature_dir=None):
        """
        Args:
            table_dir: 原CC_Table路径
            mol_blocks_dir: 分子块文件路径
            text_feature_dir: 文本特征目录（包含text_features.npy和feature_mapping.csv）
        """
        super().__init__(table_dir, mol_blocks_dir)

        self.text_feature_dir = text_feature_dir
        self.text_features = None
        self.pair_to_text_idx = {}
        self.text_feature_map = self.pair_to_text_idx  # 添加别名以保持兼容

        if text_feature_dir:
            self._load_text_features()

    def _load_text_features(self):
        """加载文本特征"""
        try:
            # 加载特征矩阵
            text_feat_path = os.path.join(self.text_feature_dir, 'text_features.npy')
            mapping_path = os.path.join(self.text_feature_dir, 'feature_mapping.csv')

            self.text_features = np.load(text_feat_path).astype(np.float32)

            # 加载映射文件
            mapping_df = pd.read_csv(mapping_path)
            for _, row in mapping_df.iterrows():
                pair_id = str(row['pair_id']).strip()
                idx = row['feature_index']
                self.pair_to_text_idx[pair_id] = idx
                self.text_feature_map[pair_id] = idx  # 同时设置别名

            print(f"[文本特征] 加载成功: {self.text_features.shape}")
            print(f"[文本特征] 映射数量: {len(self.pair_to_text_idx)}")

        except Exception as e:
            print(f"[警告] 加载文本特征失败: {e}")
            self.text_features = None

    def _task(self, items):
        """
        重写_task方法，添加文本特征
        """
        # 调用父类的_task方法
        array = super()._task(items)

        if array is None:
            return None

        tag = items[3]  # pair_id

        # 添加文本特征
        if self.text_features is not None:
            pair_id_str = str(tag).strip()
            if pair_id_str in self.pair_to_text_idx:
                idx = self.pair_to_text_idx[pair_id_str]
                array['text_features'] = self.text_features[idx]
            else:
                array['text_features'] = np.zeros(self.text_features.shape[1], dtype=np.float32)

        return array

    def _graph_func(self, samples, dataframe):
        """
        重写_graph_func，添加文本特征
        """
        V, A, labels, tags, desc, graph_size, masks, subgraph_size = [], [], [], [], [], [], [], []
        text_features = []  # 新增：文本特征列表

        for i in samples:
            V.append(dataframe[i]['V'])
            A.append(dataframe[i]['A'])
            labels.append(int(dataframe[i]['label']))
            tags.append(dataframe[i]['tag'])
            graph_size.append(dataframe[i]['graph_size'])
            masks.append(dataframe[i]['mask'])
            subgraph_size.append(dataframe[i]['subgraph_size'])

            if self.Desc:
                desc.append(dataframe[i]['global_state'])

            # 添加文本特征
            if 'text_features' in dataframe[i]:
                text_features.append(dataframe[i]['text_features'])
            else:
                if hasattr(self, 'text_features') and self.text_features is not None:
                    text_features.append(np.zeros(self.text_features.shape[1], dtype=np.float32))
                else:
                    text_features.append(np.zeros(768, dtype=np.float32))  # 默认768维

        if self.Desc:
            data = [V, A, labels, masks, graph_size, tags, desc, subgraph_size, text_features]
            return [np.array(i) for i in data]
        else:
            data = [V, A, labels, masks, graph_size, tags, text_features]
            return [np.array(i) for i in data]

    def _embedding_func(self, samples, dataframe):
        embedding, labels, tags, text_features = [], [], [], []
        for i in samples:
            embedding.append(dataframe[i]['fingerprints'])
            tags.append(dataframe[i]['tag'])
            labels.append(int(dataframe[i]['label']))

            # 获取文本特征
            if hasattr(self, 'text_features') and self.text_features is not None:
                pair_id_str = str(i).strip()
                if pair_id_str in self.pair_to_text_idx:
                    idx = self.pair_to_text_idx[pair_id_str]
                    text_features.append(self.text_features[idx])
                else:
                    text_features.append(np.zeros(self.text_features.shape[1], dtype=np.float32))
            else:
                text_features.append(np.zeros(768, dtype=np.float32))

        return np.array(embedding, dtype=np.float32), np.array(labels), np.array(tags), np.array(text_features,
                                                                                                 dtype=np.float32)

    def split(self, train_samples=None, valid_samples=None, with_test=False,
              test_samples=None, with_fps=False):
        self.train_samples = train_samples
        self.valid_samples = valid_samples
        self.with_test = with_test
        if self.with_test:
            self.test_samples = test_samples

        if with_fps:
            train_data = self._embedding_func(self.train_samples, self.dataframe)
            valid_data = self._embedding_func(self.valid_samples, self.dataframe)
            if self.with_test:
                test_data = self._embedding_func(self.test_samples, self.dataframe)
                return train_data, valid_data, test_data
            else:
                return train_data, valid_data
        else:
            train_data = self._graph_func(self.train_samples, self.dataframe)
            valid_data = self._graph_func(self.valid_samples, self.dataframe)
            if self.with_test:
                test_data = self._graph_func(self.test_samples, self.dataframe)
                return train_data, valid_data, test_data
            else:
                return train_data, valid_data