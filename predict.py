import os, shutil
from numpy import average

import torch
import argparse
from PIL import Image
from tqdm import tqdm
from timm.models import create_model
from datasets import build_transform
from main import str2bool
import time
import rich


os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3,4,5,6,7,9"

"""
get class index from txt file
"""
def get_class_by_txt(file_path):
    class_indict = []
    with open(file_path, "r") as f:
        for line in f:
            class_indict.append(line.strip())
    
    return class_indict

def folder_create(path):
    if not os.path.exists(path):
        os.makedirs(path)
    else:
        # remove all files in folder
        for file in os.listdir(path):
            os.remove(os.path.join(path, file))
        

def eval_result(result_path):
    class_name_list = os.listdir(result_path)
    #sort class name list
    class_name_list = sorted(class_name_list, key=str.lower)
    info  = ''
    for class_name in class_name_list:
        images_num = len(os.listdir(os.path.join(result_path, class_name)))
        info += f'{class_name} : {images_num}\n'
    rich.print(info)
    return info

def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"using {device} device.")
    class_indict = get_class_by_txt(args.class_indict_path)
    for class_name in class_indict:
        folder_create(os.path.join(args.save_path, class_name))
    num_classes = len(class_indict)
    print('num_classes:{}, class_indict:{}'.format(num_classes, class_indict))

    data_transform = build_transform(is_train=False, args=args)
    # load image
    image_list = os.listdir(args.data_path)
    image_list = sorted(image_list, key=str.lower)
    image_list = [os.path.join(args.data_path, x) for x in image_list]
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
        
    # cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    sum_time = 0
    
    for image_path in tqdm(image_list):
        img = Image.open(image_path)
        img0 = img.copy()
        # [N, C, H, W]
        t0 = time.time()
        img = data_transform(img)
        
        # expand batch dimension
        img = torch.unsqueeze(img, dim=0)
        # print('---------------\n',img)
        image_name  = image_path.split('/')[-1]
        
        with torch.no_grad():
            # predict class
            output = torch.squeeze(model(img.to(device))).cpu()
            predict = torch.softmax(output, dim=0)
            # predict_sort = torch.sort(predict, descending=True)
            predict_cla = torch.argmax(predict).numpy()
        t1 = time.time()
        
        sum_time = sum_time + (t1 - t0)
        # rich.print(print_res)
        image_name  = image_path.split('/')[-1]
        img0.save(os.path.join(args.save_path, class_indict[predict_cla], image_name))
    
    eval_result(args.save_path)
    average_time = sum_time/len(image_list) *1000
    rich.print("average time: {:.3}ms".format(average_time))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_size', type=int, default=448)
    parser.add_argument('--model', type=str, default='convnext_tiny')
    parser.add_argument('--drop_path', type=float, default=0, metavar='PCT',
                    help='Drop path rate (default: 0.0)')
    parser.add_argument('--layer_scale_init_value', default=1e-6, type=float,
                        help="Layer scale initial values")
    parser.add_argument('--head_init_scale', default=1.0, type=float,
                        help='classifier head initial scale, typically adjusted in fine-tuning')
    parser.add_argument('--load_model', type=str, default='/home/dong.sun/git/ConvNeXt/exp/plusai_20220809-154016/model/checkpoint-best.pth')
    parser.add_argument('--data_path', type=str, default='/home/dong.sun/git/ConvNeXt/data/datav2/val/exposure')
    parser.add_argument('--class_indict_path', type=str, default='/home/dong.sun/git/ConvNeXt/data/class.txt')
    parser.add_argument('--save_path', type=str, default='/home/dong.sun/git/ConvNeXt/result')
    parser.add_argument('--crop_pct', type=float, default=None)
    parser.add_argument('--imagenet_default_mean_and_std', type=str2bool, default=True)
    args = parser.parse_args()
    main(args)