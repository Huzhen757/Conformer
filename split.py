import random
import os
import json
import shutil

def main(root: str, val_rate: float=0.2):
    os.environ['CUDA_VISIBLE_DIVICES'] = '2'
    random.seed(42)
    assert os.path.exists(root), "dataset root: {} does not exist.".format(root)

    # 遍历文件夹，一个文件夹对应一个类别
    river_class = [cla for cla in os.listdir(root) if os.path.isdir(os.path.join(root, cla))]
    # 排序，保证顺序一致
    river_class.sort()
    # 生成类别名称以及对应的数字索引
    class_indices = dict((k, v) for v, k in enumerate(river_class))
    json_str = json.dumps(dict((val, key) for key, val in class_indices.items()), indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    train_images_path = []  # 存储训练集的所有图片路径
    train_images_label = []  # 存储训练集图片对应索引信息
    val_images_path = []  # 存储验证集的所有图片路径
    val_images_label = []  # 存储验证集图片对应索引信息
    every_class_num = []  # 存储每个类别的样本总数
    supported = [".jpg", ".JPG", ".png", ".PNG"]  # 支持的文件后缀类型
    # 遍历每个文件夹下的文件
    for cla in river_class:
        cla_path = os.path.join(root, cla)
        # 遍历获取supported支持的所有文件路径
        images = [os.path.join(root, cla, i) for i in os.listdir(cla_path)
                  if os.path.splitext(i)[-1] in supported]
        # 获取该类别对应的索引
        image_class = class_indices[cla]
        # 记录该类别的样本数量
        every_class_num.append(len(images))
        # 按比例随机采样验证样本
        val_path = random.sample(images, k=int(len(images) * val_rate))

        for img_path in images:
            if img_path in val_path:  
                val_images_path.append(img_path)
                val_images_label.append(image_class)
            else:  
                train_images_path.append(img_path)
                train_images_label.append(image_class)

    print("{} images were found in the dataset.".format(sum(every_class_num)))
    print("{} images for training.".format(len(train_images_path)))
    print("{} images for validation.".format(len(val_images_path)))

    train_root = os.path.join(root, 'train')
    val_root = os.path.join(root, 'val')
    for img_file in train_images_path:
        img_class = os.path.split(img_file)[0]
        img_id = os.path.split(img_class)[1]
        if img_id == 'daisy':
            shutil.copy(img_file, os.path.join(train_root, 'daisy'))
        elif img_id == 'roses':
            shutil.copy(img_file, os.path.join(train_root, 'roses'))
        elif img_id == 'tulips':
            shutil.copy(img_file, os.path.join(train_root, 'tulips'))
        elif img_id == 'sunflowers':
            shutil.copy(img_file, os.path.join(train_root, 'sunflowers'))
        else:
            shutil.copy(img_file, os.path.join(train_root, 'dandelion'))

    for img_file in val_images_path:
        img_class = os.path.split(img_file)[0]
        img_id = os.path.split(img_class)[1]
        if img_id == 'daisy':
            shutil.copy(img_file, os.path.join(val_root, 'daisy'))
        elif img_id == 'roses':
            shutil.copy(img_file, os.path.join(val_root, 'roses'))
        elif img_id == 'tulips':
            shutil.copy(img_file, os.path.join(val_root, 'tulips'))
        elif img_id == 'sunflowers':
            shutil.copy(img_file, os.path.join(val_root, 'sunflowers'))
        else:
            shutil.copy(img_file, os.path.join(val_root, 'dandelion'))
    
    return train_images_path, train_images_label, val_images_path, val_images_label
        

if __name__ == "__main__":
    root = '/media/data2/huzhen/flower_data'
    split_data_rate = 0.2
    main(root, split_data_rate)

