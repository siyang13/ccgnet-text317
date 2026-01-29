import numpy as np
feature_path = r"D:/课业/毕业设计/ccgnet-main_shiyan/ccgnet-main/data/processed_features/text_features.npy"
features = np.load(feature_path)
print(f"特征矩阵形状: {features.shape}")
print(f"第一个样本的特征 (前10维): {features[0][:10]}")
print(f"所有特征的均值 (应远大于0): {np.mean(features)}")
print(f"全零样本数量: {np.all(features == 0, axis=1).sum()} / {len(features)}")