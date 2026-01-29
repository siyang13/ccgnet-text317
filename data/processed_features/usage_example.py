
# 示例1：修改Dataset类以加载文本特征

class EnhancedDataset:
    def __init__(self, table_dir, text_feature_dir=None, **kwargs):
        # 原有初始化代码...
        
        # 加载文本特征（如果提供了路径）
        if text_feature_dir:
            self.text_features = np.load(os.path.join(text_feature_dir, 'text_features.npy'))
            mapping_df = pd.read_csv(os.path.join(text_feature_dir, 'feature_mapping.csv'))
            
            # 创建pair_id到特征索引的映射
            self.pair_to_feature_idx = {}
            for _, row in mapping_df.iterrows():
                self.pair_to_feature_idx[str(row['pair_id']).strip()] = row['feature_index']
        else:
            self.text_features = None
    
    def _task(self, items):
        # 原有代码...
        
        # 添加文本特征
        pair_id = items[3]  # 假设pair_id在第4个位置
        
        if hasattr(self, 'text_features') and self.text_features is not None:
            pair_id_str = str(pair_id).strip()
            if pair_id_str in self.pair_to_feature_idx:
                idx = self.pair_to_feature_idx[pair_id_str]
                array['text_features'] = self.text_features[idx]
            else:
                # 使用零向量
                array['text_features'] = np.zeros(768)
        else:
            array['text_features'] = np.zeros(768)
        
        return array
