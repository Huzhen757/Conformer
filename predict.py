import argparse
import time
import torch
import json
import os
from PIL import Image

from timm.data import transforms
from timm.models import create_model
from datasets import build_transform
import models

    
def main(args):
    print(args)
    device = torch.device(args.device)
    print(f"Creating model: {args.model}")
    model = create_model(
        args.model,
        pretrained=False,
        num_classes=args.nb_classes,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        drop_block_rate=args.drop_block,
    )
    model.to(device)
    json_path = './class_indices.json'
    assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)
    json_file = open(json_path, "r")
    class_indict = json.load(json_file)
    
    val_transform = build_transform(is_train=False, args=args)
    # load model weights 载入训练好的权重
    if args.weights:
        model.load_state_dict(torch.load(args.weights, map_location='cpu'))

    # load image 载入测试图片的根目录
    img_root_path = args.image_path
    AD_number = 0
    MCI_number = 0
    NC_number = 0
    test_result = []
    test_result.append(['image_id', 'category_id'])
    for img_file in os.listdir(img_root_path):
        img_path = os.path.join(img_root_path, img_file)
        assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
        img = Image.open(img_path)
        if img.mode != "RGB":
            img = img.convert('RGB')
        # image preprocess
        img = val_transform(img)
        # expand batch dimension
        img = torch.unsqueeze(img, dim=0)

        start_time = time.time()
        model.eval()
        with torch.no_grad():
        # predict class
            outputs = model(img.to(device))
            # conv分支的预测值
            conv_predict = outputs[0]
            prob, predict_conv_cla = conv_predict.topk(1, 1, True, True)
            predict_conv_cla =predict_conv_cla.reshape(-1).item()
            prob = prob.reshape(-1).item()
            # transformer分支的预测值
            # trans_predict = outputs[1]
            # predict_trans_cla = trans_predict.topk(1, 1, True, True)
            # conv + transformer 联合预测值

        print_conv_res = "predict class: {}  predict prob: {:.3}".format(class_indict[str(predict_conv_cla)],
                                                                         prob)
        img_id = os.path.splitext(img_file)[0]
        test_result.append([img_id, class_indict[str(predict_conv_cla)]])
        end_time = time.time()    
        print("inference one image tims: {}".format(end_time - start_time))
        print(print_conv_res)
    
    f = open('./MergePET-384.txt', mode='w', encoding='utf-8')
    for val in test_result:
        img_id = val[0]
        class_id = val[1]
        f.write(str(img_id))
        f.write(str(class_id) + '\n')
        if class_id == 'MCI':
            MCI_number += 1
        elif class_id == 'AD':
            AD_number += 1
        else:
            NC_number += 1
    print("the predict MCI class number is:{}".format(MCI_number))
    print("the predict AD class number is:{}".format(AD_number))
    print("the predict NC class number is:{}".format(NC_number))


if __name__ == '__main__':
    def get_args_parser():
        parser = argparse.ArgumentParser('Conformer test', add_help=False)
        parser.add_argument('--image_path', default='/media/data2/huzhen/讯飞_PETMRI复赛/test',type=str)
        parser.add_argument('--device', default='cuda:2')
        parser.add_argument('--input_size', type=int, default=448)
        parser.add_argument('--model', default='Conformer_base_patch16', type=str, metavar='MODEL',
                        help='Name of model to train')
        parser.add_argument('--nb_classes', type=int, default=3)
        parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                        help='Dropout rate (default: 0.)')
        parser.add_argument('--drop-path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')
        parser.add_argument('--drop-block', type=float, default=None, metavar='PCT',
                        help='Drop block rate (default: None)')
        parser.add_argument('--weights', type=str, default='./MergePET/model-192-84.44444444444444.pth')
        return parser

    parser = argparse.ArgumentParser('Conformer test', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
