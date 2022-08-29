'''
Author: DongSunPlus dong.sun@plus.ai
Date: 2022-08-26 11:14:29
LastEditTime: 2022-08-26 15:53:39
LastEditors: DongSunPlus dong.sun@plus.ai
FilePath: /ConvNeXt/show_umap_feature.py
Description: 
'''

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import umap
import umap.plot
import random
import pandas as pd
# import plotly.express as px

class_index_map = {
    0 : "blur",
    1 : "covered",
    2 : "dark",
    3 : "exposure",
    4 : "mud",
    5 : "norm",
    6 : "rainy"
}
marker_list = ['.', ',', 'o', 'v', '^', '<', '>', '1', '2', '3', '4', '8', 's', 'p', 'P', '*', 'h', 'H', '+', 'x', 'X', 'D', 'd', '|', '_', 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

def visualization2d(feature_path, class_array_path):
    encoding_array = np.load(feature_path, allow_pickle=True)
    class_array = np.load(class_array_path, allow_pickle=True)
    
    class_list = [i for i in class_index_map.values()]
    num_class = len(class_list)
    palette = sns.hls_palette(num_class)
    random.seed(1234)
    random.shuffle(marker_list)
    random.shuffle(palette)
    
    mapper = umap.UMAP(n_neighbors=10, n_components=2, random_state=12).fit(encoding_array)
    X_umap_2d = mapper.embedding_
    
    plt.figure(figsize=(14, 14))
    for idx, fruit in enumerate(class_list): # 遍历每个类别
        # 获取颜色和点型
        color = palette[idx]
        marker = marker_list[idx%len(marker_list)]

        # 找到所有标注类别为当前类别的图像索引号
        indices = np.where(class_array==fruit)
        plt.scatter(X_umap_2d[indices, 0], X_umap_2d[indices, 1], color=color, marker=marker, label=fruit, s=150)

    plt.legend(fontsize=16, markerscale=1, bbox_to_anchor=(1, 1))
    plt.xticks([])
    plt.yticks([])
    plt.savefig('result/show_semantic_feature.png', dpi=300) # 保存图像
    plt.show()

def visualization3d(feature_path, class_array_path):
    encoding_array = np.load(feature_path, allow_pickle=True)
    class_array = np.load(class_array_path, allow_pickle=True)
    
    class_list = [i for i in class_index_map.values()]
    num_class = len(class_list)
    palette = sns.hls_palette(num_class)
    random.seed(1234)
    random.shuffle(marker_list)
    random.shuffle(palette)
    
    mapper = umap.UMAP(n_neighbors=10, n_components=3, random_state=12).fit(encoding_array)
    X_umap_3d = mapper.embedding_
    df_3d = pd.DataFrame()
    df_3d['X'] = list(X_umap_3d[:, 0].squeeze())
    df_3d['Y'] = list(X_umap_3d[:, 1].squeeze())
    df_3d['Z'] = list(X_umap_3d[:, 2].squeeze())
    df_3d['label class'] = class_array
    fig = px.scatter_3d(df_3d, 
                    x='X', 
                    y='Y', 
                    z='Z',
                    color="label class", 
                    labels="label class",
                    symbol="label class", 
                    opacity=0.6,
                    width=1000, 
                    height=800)
    # 设置排版
    fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
    fig.show() 
    fig.write_html('result/semantic_feature3d.html')

visualization2d("result/semantic_feature.npy", "result/calss_array.npy")