import os

import torch
import argparse
from PIL import Image
from tqdm import tqdm
from timm.models import create_model
from datasets import build_transform
from main import str2bool
import time
import rich
from typing import Dict, List

class_index_map = {
    0 : "blur",
    1 : "covered",
    2 : "dark",
    3 : "exposure",
    4 : "mud",
    5 : "norm",
    6 : "rainy"
}
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3,4,5,6,7,9"

"""
get class index from txt file
"""

def folder_create(path):
    if not os.path.exists(path):
        os.makedirs(path)
    else:
        # remove all files in folder
        for file in os.listdir(path):
            os.remove(os.path.join(path, file))
        

def eval_result(eval_dict):
    TP_all = 0
    total_all = 0
    for class_name, info in eval_dict.items():
        TP = info['TP'] * 1.0
        total = info['total'] * 1.0
        TP_all += TP
        total_all += total
        acc_class = TP / total * 100.0
        rich.print(f"{class_name} acc: {acc_class:.3}%")
    acc1 = TP_all / total_all * 100.0
    rich.print("evaluation finished. acc: {:.3}%".format(acc1))
    return info

def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"using {device} device.")
    for class_name in class_index_map.values():
        folder_create(os.path.join(args.save_path, class_name))
    num_classes = len(class_index_map)
    print('num_classes:{}, class_indict:{}'.format(num_classes, class_index_map))

    data_transform = build_transform(is_train=False, args=args)
    # load image
    images_list = []
    class_folder_list = os.listdir(args.data_path)
    for class_folder in class_folder_list:
        class_path = os.path.join(args.data_path, class_folder)
        img_list = os.listdir(class_path)
        img_list = [os.path.join(class_path, img) for img in img_list]
        images_list.extend(img_list)
    # create model
    
    model = create_model(
        args.model, 
        pretrained=False, 
        num_classes=num_classes, 
        drop_path_rate=args.drop_path,
        layer_scale_init_value=args.layer_scale_init_value,
        head_init_scale=args.head_init_scale,
    )
    model.to(device)
    checkpoint = torch.load(args.load_model, map_location='cpu')
    # load model weights
    model.load_state_dict(checkpoint['model'])
    model.eval()
    
    # feed a blank image to model
    blank_image = torch.randn(1, 3, args.input_size, args.input_size)

    with torch.no_grad():
        output = torch.squeeze(model(blank_image.to(device))).cpu()
        
    sum_time = 0
    eval_dict : Dict[str, Dict[str, int]] = {}
    for image_path in tqdm(images_list):
        img = Image.open(image_path)
        img0 = img.copy()
        # [N, C, H, W]
        t0 = time.time()
        img = data_transform(img)
        
        # expand batch dimension
        img = torch.unsqueeze(img, dim=0)
        # print('---------------\n',img)
        image_name  = image_path.split('/')[-1]
        class_name = image_path.split('/')[-2]
        if eval_dict.get(class_name) is None:
            eval_dict.update({class_name: {'total': 1, 'TP': 0}})
        else:
            eval_dict[class_name]['total'] += 1
        with torch.no_grad():
            # predict class
            output = torch.squeeze(model(img.to(device))).cpu()
            predict = torch.softmax(output, dim=0)
            # predict_sort = torch.sort(predict, descending=True)
            predict_cla = torch.argmax(predict).numpy()
        t1 = time.time()
        predict_class_name = class_index_map[int(predict_cla)]
        if predict_class_name == class_name:
            eval_dict[class_name]['TP'] += 1
        sum_time = sum_time + (t1 - t0)
        # rich.print(print_res)
        image_name  = image_path.split('/')[-1]
        img0.save(os.path.join(args.save_path, predict_class_name, image_name))
    
    eval_result(eval_dict)
    average_time = sum_time/len(images_list) *1000
    rich.print("average time: {:.3}ms".format(average_time))

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
    parser.add_argument('--data_path', type=str, default='/home/dong.sun/data/iqa_datav2/val')
    parser.add_argument('--save_path', type=str, default='/home/dong.sun/ConvNeXt/result')
    parser.add_argument('--crop_pct', type=float, default=None)
    parser.add_argument('--imagenet_default_mean_and_std', type=str2bool, default=True)
    args = parser.parse_args()
    main(args)