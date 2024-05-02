import os
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from torchvision.transforms import autoaugment
from torch.utils.data import Dataset

class Train_Dataset(Dataset):
    def __init__(self, source_list, target_list, transforms):
        self.source_list = source_list
        self.target_list = target_list
        self.transforms = transforms


    def __len__(self):
        return len(self.source_list)
    
    def __getitem__(self, index):
        source_path = self.source_list[index][0]
        source_label = self.source_list[index][1]
        source_label = int(source_label)

        target_path = self.target_list[index][0]
        target_label = self.target_list[index][1]
        target_label = int(target_label)

        source_image = Image.open(source_path).convert("RGB")
        source_image = self.transforms(source_image)

        target_image = Image.open(target_path).convert("RGB")
        target_image = self.transforms(target_image)

        return {
            "source_image": source_image,
            "source_label": source_label,
            "target_image": target_image,
            "target_label": target_label,
            "target_index": index,

        }
        # return source_image, source_label, target_image, target_label, index  # target label for evaluate NCC accuracy


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
    source_databse_txt_path = os.path.join(args.root, source_domain, "database.txt")
    
    target_train_txt_path = os.path.join(args.root, target_domain, "train.txt")
    target_val_txt_path = os.path.join(args.root, target_domain, "val.txt")
    target_query_txt_path = os.path.join(args.root, target_domain, "test.txt")
    target_databse_txt_path = os.path.join(args.root, target_domain, "database.txt")
    

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
    
    with open(source_databse_txt_path, "r") as f:
        for line in f:
            temp_path, temp_label = line.strip("\n").split(",")
            source_database_data_list.append([temp_path, int(temp_label)])
    
    with open(target_databse_txt_path, "r") as f:
        for line in f:
            temp_path, temp_label = line.strip("\n").split(",")
            target_database_data_list.append([temp_path, int(temp_label)])
    

    return source_train_data_list, target_train_data_list, target_val_data_list, target_query_data_list, source_database_data_list, target_database_data_list


def get_data(args):
    source_train_list, target_train_list, target_val_list, target_query_list, source_database_list, target_database_list = read_text(args)
    print(len(source_train_list), len(target_train_list), len(target_val_list), len(target_query_list), len(source_database_list), len(target_database_list))
    
    mu = [0.485, 0.456, 0.406]
    sigma = [0.229, 0.224, 0.225]

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


    train_dataset = Train_Dataset(source_train_list, target_train_list, train_transform)
    
    val_dataset = Test_Dataset(target_val_list, args.class_num, test_transform)
    query_dataset = Test_Dataset(target_query_list, args.class_num, test_transform)
    source_db_dataset = Test_Dataset(source_database_list, args.class_num, test_transform)
    target_db_dataset = Test_Dataset(target_database_list, args.class_num, test_transform)

    memory_source_dataset = Test_Dataset(source_train_list, args.class_num, test_transform)
    memory_target_dataset = Test_Dataset(target_train_list, args.class_num, test_transform)

    return train_dataset, val_dataset, query_dataset, source_db_dataset, target_db_dataset, memory_source_dataset, memory_target_dataset