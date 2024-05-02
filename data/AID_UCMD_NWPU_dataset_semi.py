import os
import torch
import random
import numpy as np
from PIL import Image
from torchvision import transforms
from torchvision.transforms import autoaugment
from torch.utils.data import Dataset


class Train_Dataset(Dataset):
    def __init__(self, source_lb_list, source_ulb_list, target_list, transforms):
        self.source_lb_list = source_lb_list
        self.source_ulb_list = source_ulb_list
        self.target_list = target_list
        self.transforms = transforms

        self.source_lb_size = len(self.source_lb_list)
        self.source_ulb_size = len(self.source_ulb_list)
        self.target_size = len(self.target_list)

    def __len__(self):
        return max(self.source_lb_size, self.source_ulb_size, self.target_size)
    
    def __getitem__(self, index):
        if index >= self.source_lb_size:
            index_A = random.randint(0, self.source_lb_size - 1)
        else:
            index_A = index

        if index >= self.source_ulb_size:
            index_B = random.randint(0, self.source_ulb_size - 1)
        else:
            index_B = index
        
        if index >= self.target_size:
            index_C = random.randint(0, self.target_size - 1)
        else:
            index_C = index

        source_lb_path = self.source_lb_list[index_A][0]
        source_lb_image = Image.open(source_lb_path).convert("RGB")
        source_lb_image = self.transforms(source_lb_image)
        source_lb_label = self.source_lb_list[index_A][1]
        source_lb_label = int(source_lb_label)
        
        source_ulb_path = self.source_ulb_list[index_B][0]
        source_ulb_image = Image.open(source_ulb_path).convert("RGB")
        source_ulb_image = self.transforms(source_ulb_image)
        source_ulb_label = self.source_ulb_list[index_B][1]
        source_ulb_label = int(source_ulb_label)

        target_path = self.target_list[index_C][0]
        target_image = Image.open(target_path).convert("RGB")
        target_image = self.transforms(target_image)
        target_label = self.target_list[index_C][1]
        target_label = int(target_label)

        return {
            "source_lb_image": source_lb_image,
            "source_lb_label": source_lb_label,
            "source_lb_index": index_A,
            "source_ulb_image": source_ulb_image,
            "source_ulb_label": source_ulb_label,
            "source_ulb_index": index_B,
            "target_image": target_image,
            "target_label": target_label,
            "target_index": index_C
        }


class Test_Dataset(Dataset):
    def __init__(self, data_list, class_num, transforms):
        self.data_list = data_list
        self.class_num = class_num
        self.transforms = transforms

    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, index):
        data_path = self.data_list[index][0]
        data_label = self.data_list[index][1]
        data_label = int(data_label)

        image = Image.open(data_path).convert("RGB")
        image = self.transforms(image)

        one_hot_label = np.eye(self.class_num, dtype=np.uint8)[data_label]
        
        return image, data_label, one_hot_label, data_path


def read_text(args):
    source_domain, target_domain = args.dataset_mode.split("_")
    
    source_train_txt_path = os.path.join(args.root, source_domain, "train.txt")
    source_database_txt_path = os.path.join(args.root, source_domain, "database.txt")
    
    target_train_txt_path = os.path.join(args.root, target_domain, "train.txt")
    target_val_txt_path = os.path.join(args.root, target_domain, "val.txt")
    target_query_txt_path = os.path.join(args.root, target_domain, "test.txt")
    target_database_txt_path = os.path.join(args.root, target_domain, "database.txt")
    

    source_train_data_list = []
    target_train_data_list = []
    target_val_data_list = []
    target_query_data_list = []
    source_database_data_list = []
    target_database_data_list = []

    with open(source_train_txt_path, "r") as f:
        for line in f:
            temp_path, temp_label = line.strip("\n").split(",")
            source_train_data_list.append([temp_path, int(temp_label)])

    with open(target_train_txt_path, "r") as f:
        for line in f:
            temp_path, temp_label = line.strip("\n").split(",")
            target_train_data_list.append([temp_path, int(temp_label)])
    
    with open(target_val_txt_path, "r") as f:
        for line in f:
            temp_path, temp_label = line.strip("\n").split(",")
            target_val_data_list.append([temp_path, int(temp_label)])

    with open(target_query_txt_path, "r") as f:
        for line in f:
            temp_path, temp_label = line.strip("\n").split(",")
            target_query_data_list.append([temp_path, int(temp_label)])
    
    with open(source_database_txt_path, "r") as f:
        for line in f:
            temp_path, temp_label = line.strip("\n").split(",")
            source_database_data_list.append([temp_path, int(temp_label)])
    
    with open(target_database_txt_path, "r") as f:
        for line in f:
            temp_path, temp_label = line.strip("\n").split(",")
            target_database_data_list.append([temp_path, int(temp_label)])
    

    return source_train_data_list, target_train_data_list, target_val_data_list, target_query_data_list, source_database_data_list, target_database_data_list


def split_data(args, data_list):
    num = len(data_list) / args.class_num
    label_num = int(num * args.ratio)
    print("split_per_num =", label_num)

    lb_list = [temp_num[1] for temp_num in data_list]
    lb_list = np.array(lb_list)
    
    label_data = []
    unlabel_data = []

    for lb in range(args.class_num):
        idx = np.where(lb_list == lb)[0]

        label_idx = idx[:label_num]
        unlabel_idx = idx[label_num:]

        for i in label_idx:
            label_data.append(data_list[i])
        
        for i in unlabel_idx:
            unlabel_data.append(data_list[i])

    return label_data, unlabel_data


def get_data(args):
    source_train_list, target_train_list, target_val_list, target_query_list, source_database_list, target_database_list = read_text(args)
    print("before split:", len(source_train_list), len(target_train_list), len(target_val_list), len(target_query_list), len(source_database_list), len(target_database_list))
    
    mu = [0.485, 0.456, 0.406]
    sigma = [0.229, 0.224, 0.225]

    # mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]

    train_transform = transforms.Compose([
        transforms.Resize((args.size, args.size)),
        transforms.RandomHorizontalFlip(),
        autoaugment.TrivialAugmentWide(),
        transforms.ToTensor(),
        transforms.Normalize(mean=mu, std=sigma),
    ])

    test_transform = transforms.Compose([
        transforms.Resize((args.size, args.size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mu, std=sigma)
    ])

    source_label_list, source_unlabel_list = split_data(args, source_train_list)
    print("after split:", len(source_label_list), len(source_unlabel_list), len(target_train_list), len(target_val_list), len(target_query_list), len(source_database_list), len(target_database_list))

    train_dataset = Train_Dataset(source_label_list, source_unlabel_list, target_train_list, train_transform)
    
    val_dataset = Test_Dataset(target_val_list, args.class_num, test_transform)
    query_dataset = Test_Dataset(target_query_list, args.class_num, test_transform)
    source_db_dataset = Test_Dataset(source_database_list, args.class_num, test_transform)
    target_db_dataset = Test_Dataset(target_database_list, args.class_num, test_transform)

    memory_source_label_dataset = Test_Dataset(source_label_list, args.class_num, test_transform)
    memory_source_unlabel_dataset = Test_Dataset(source_unlabel_list, args.class_num, test_transform)
    memory_target_dataset = Test_Dataset(target_train_list, args.class_num, test_transform)

    return train_dataset, val_dataset, query_dataset, source_db_dataset, target_db_dataset, memory_source_label_dataset, memory_source_unlabel_dataset, memory_target_dataset