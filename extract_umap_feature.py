'''
Author: DongSunPlus dong.sun@plus.ai
Date: 2022-08-25 15:21:10
LastEditTime: 2022-08-26 11:24:01
LastEditors: DongSunPlus dong.sun@plus.ai
FilePath: /ConvNeXt/extract_umap_feature.py
Description: 

'''
    
    
import os, shutil
from numpy import average

import torch
from torchvision.models.feature_extraction import create_feature_extractor

import argparse
from PIL import Image
from tqdm import tqdm
import numpy as np
from torchvision import transforms
from models.convnext import convnext_tiny as create_model1
import seaborn as sns
import umap
import umap.plot
import random
import matplotlib
import matplotlib.pyplot as plt

os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3,4,5,6,7,9"

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

def visualization(feature_path, class_array_path):
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
    plt.savefig('show_semantic_feature.png', dpi=300) # 保存图像
    plt.show()
    

def main(args):
    num_classes = len(class_index_map)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"using {device} device.")
    data_transform = transforms.Compose(
        [transforms.Resize(int(args.input_size * 1.14)),
         transforms.CenterCrop(args.input_size),
         transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    # load image
    class_folder_list = os.listdir(args.data_path)
    images_list = []
    for class_folder in class_folder_list:
        class_path = os.path.join(args.data_path, class_folder)
        img_list = os.listdir(class_path)
        img_list = [os.path.join(class_path, img) for img in img_list]
        images_list.extend(img_list)
    # create model
    # model = create_model(
    #     args.model, 
    #     pretrained=False, 
    #     num_classes=num_classes, 
    #     drop_path_rate=args.drop_path,
    #     layer_scale_init_value=args.layer_scale_init_value,
    #     head_init_scale=args.head_init_scale,
    # )
    model = create_model1(num_classes=num_classes)
    model.to(device)
    checkpoint = torch.load(args.load_model, map_location='cpu')
    # load model weights
    model.load_state_dict(checkpoint['model'])
    
    model_trunc = create_feature_extractor(model, return_nodes={'norm': 'semantic_feature'})
    
    encoding_array = []
    class_array = []
    for image_path in tqdm(images_list):
        img = Image.open(image_path)
        img_input = data_transform(img)
        class_name = image_path.split('/')[-2]
        class_array.append(class_name)
        # expand batch dimension
        img_input = torch.unsqueeze(img_input, dim=0).to(device)
        # print('---------------\n',img)
        image_name  = image_path.split('/')[-1]
        
        feature = model_trunc(img_input)['semantic_feature'].squeeze().detach().cpu().numpy()
        encoding_array.append(feature)
    np.save(os.path.join(args.save_path, "semantic_feature.npy"), encoding_array)
    np.save(os.path.join(args.save_path,"calss_array.npy"), class_array)
    # visualization("semantic_feature.npy", class_array)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_size', type=int, default=224)
    parser.add_argument('--model', type=str, default='convnext_tiny')
    parser.add_argument('--drop_path', type=float, default=0, metavar='PCT',
                    help='Drop path rate (default: 0.0)')
    parser.add_argument('--layer_scale_init_value', default=1e-6, type=float,
                        help="Layer scale initial values")
    parser.add_argument('--head_init_scale', default=1.0, type=float,
                        help='classifier head initial scale, typically adjusted in fine-tuning')
    parser.add_argument('--load_model', type=str, default='/home/dong.sun/ConvNeXt/exp/plusai_20220825-162615/model/checkpoint-best.pth')
    parser.add_argument('--data_path', type=str, default='/home/dong.sun/data/iqa_datav2/val/')
    parser.add_argument('--save_path', type=str, default='/home/dong.sun/ConvNeXt/result')
    parser.add_argument('--crop_pct', type=float, default=None)
    parser.add_argument('--save_umap_feature', type=bool, default=True)
    parser.add_argument("--save_umap_feature_path", type=str, default="")
    args = parser.parse_args()
    main(args)
    # visualization("result/semantic_feature.npy", "result/calss_array.npy")
    
    
    