import torch
import argparse
import os

def get_argument():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset_mode", type=str, default="AID_UCMD")
    parser.add_argument("--root", type=str, default="data")

    # training setting
    parser.add_argument("--ratio", type=float, default=0.1)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--max_epoch", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--size", type=int, default=224)


    # model setting
    parser.add_argument('--CLIP_Arch', type=str, default="ViT-B/32")
    parser.add_argument("--lora_r", type=int, default=8, help="LoRA rank value, default 0 means do not use LoRA")
    parser.add_argument("--lora_all", default=False, action="store_true",
                        help="Should we use LoRA for all layers? only valid when lora_r!=0")
    parser.add_argument("--lora_lock_text", default=False, action="store_true",
                        help="Should LoRA only train the image model? only valid when lora_r!=0")

    # loss setting
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--temperature', type=float, default=0.1)

    parser.add_argument('--lambda_1', type=float, default=1.0)  # s2proto
    parser.add_argument('--lambda_2', type=float, default=1.0)  # t2proto
    parser.add_argument('--lambda_3', type=float, default=1.0)  # s2text
    parser.add_argument('--lambda_4', type=float, default=1.0)  # t2text
    parser.add_argument('--lambda_5', type=float, default=1.0)  # disc
    parser.add_argument('--lambda_6', type=float, default=1.0)  # ent
    
    parser.add_argument("--phase", type=str, default="no", choices=["no", "train_uda", "train_semi", "retrieval", "analyse"])
    parser.add_argument("--test_freq", type=int, default=5)
    parser.add_argument("--topk", type=int, default=-1)
    parser.add_argument("--log_dir", type=str, default="logs")

    args = parser.parse_args()

    # DB1(1)
    if args.dataset_mode == "AID_UCMD":
        args.class_num = 10
        args.class_list = ['farmland', 'baseball filed', 'beach', 'dense residential', 'forest', 
                           'medium residential', 'parking lot', 'river', 'sparse residential', 'storage tanks']

    # DB1(2)
    elif args.dataset_mode == "UCMD_AID":
        args.class_num = 10
        args.class_list = ['farmland', 'baseball filed', 'beach', 'dense residential', 'forest', 
                           'medium residential', 'parking lot', 'river', 'sparse residential', 'storage tanks']

    # DB1(3)
    elif args.dataset_mode == "AID_NWPU":
        args.class_num = 10
        args.class_list = ['farmland', 'baseball filed', 'beach', 'dense residential', 'forest', 
                           'medium residential', 'parking lot', 'river', 'sparse residential', 'storage tanks']

    # DB1(4)
    elif args.dataset_mode == "NWPU_AID":
        args.class_num = 10
        args.class_list = ['farmland', 'baseball filed', 'beach', 'dense residential', 'forest', 
                           'medium residential', 'parking lot', 'river', 'sparse residential', 'storage tanks']

    
    # DB1(5)
    elif args.dataset_mode == "UCMD_NWPU":
        args.class_num = 10
        args.class_list = ['farmland', 'baseball filed', 'beach', 'dense residential', 'forest', 
                           'medium residential', 'parking lot', 'river', 'sparse residential', 'storage tanks']
 
    # DB1(6)
    elif args.dataset_mode == "NWPU_UCMD":
        args.class_num = 10
        args.class_list = ['farmland', 'baseball filed', 'beach', 'dense residential', 'forest', 
                           'medium residential', 'parking lot', 'river', 'sparse residential', 'storage tanks']
    
    
    ########################################################################
    
    # DB2(1)
    elif args.dataset_mode == "PatternNet_VBing":
        args.class_num = 38
        args.class_list = ['airplane', 'baseball field', 'basketball court', 'beach', 'bridge', 
                           'cemetery', 'chaparral','christmas tree farm', 'closed road', 'coastal mansion', 
                           'crosswalk', 'dense residential', 'ferry terminal', 'football field', 'forest', 
                           'freeway', 'golf course', 'harbor', 'intersection', 'mobile home park', 'nursing home', 
                           'oil gas field', 'oil well', 'overpass', 'parking lot', 'parking space', 'railway', 
                           'river', 'runway', 'runway marking', 'shipping yard', 'solar panel', 'sparse residential', 
                           'storage tank', 'swimming pool', 'tennis court', 'transformer station', 'wastewater treatment plant']

    
    # DB2(2)
    elif args.dataset_mode == "VBing_PatternNet":
        args.class_num = 38
        args.class_list = ['airplane', 'baseball field', 'basketball court', 'beach', 'bridge', 
                           'cemetery', 'chaparral','christmas tree farm', 'closed road', 'coastal mansion', 
                           'crosswalk', 'dense residential', 'ferry terminal', 'football field', 'forest', 
                           'freeway', 'golf course', 'harbor', 'intersection', 'mobile home park', 'nursing home', 
                           'oil gas field', 'oil well', 'overpass', 'parking lot', 'parking space', 'railway', 
                           'river', 'runway', 'runway marking', 'shipping yard', 'solar panel', 'sparse residential', 
                           'storage tank', 'swimming pool', 'tennis court', 'transformer station', 'wastewater treatment plant']

    
    # DB2(3)
    elif args.dataset_mode == "PatternNet_VArcGIS":
        args.class_num = 38
        args.class_list = ['airplane', 'baseball field', 'basketball court', 'beach', 'bridge', 
                           'cemetery', 'chaparral','christmas tree farm', 'closed road', 'coastal mansion', 
                           'crosswalk', 'dense residential', 'ferry terminal', 'football field', 'forest', 
                           'freeway', 'golf course', 'harbor', 'intersection', 'mobile home park', 'nursing home', 
                           'oil gas field', 'oil well', 'overpass', 'parking lot', 'parking space', 'railway', 
                           'river', 'runway', 'runway marking', 'shipping yard', 'solar panel', 'sparse residential', 
                           'storage tank', 'swimming pool', 'tennis court', 'transformer station', 'wastewater treatment plant']


    # DB2(4)
    elif args.dataset_mode == "VArcGIS_PatternNet":
        args.class_num = 38
        args.class_list = ['airplane', 'baseball field', 'basketball court', 'beach', 'bridge', 
                           'cemetery', 'chaparral','christmas tree farm', 'closed road', 'coastal mansion', 
                           'crosswalk', 'dense residential', 'ferry terminal', 'football field', 'forest', 
                           'freeway', 'golf course', 'harbor', 'intersection', 'mobile home park', 'nursing home', 
                           'oil gas field', 'oil well', 'overpass', 'parking lot', 'parking space', 'railway', 
                           'river', 'runway', 'runway marking', 'shipping yard', 'solar panel', 'sparse residential', 
                           'storage tank', 'swimming pool', 'tennis court', 'transformer station', 'wastewater treatment plant']
    
    
    # DB2(5)
    elif args.dataset_mode == "VBing_VArcGIS":
        args.class_num = 38
        args.class_list = ['airplane', 'baseball field', 'basketball court', 'beach', 'bridge', 
                           'cemetery', 'chaparral','christmas tree farm', 'closed road', 'coastal mansion', 
                           'crosswalk', 'dense residential', 'ferry terminal', 'football field', 'forest', 
                           'freeway', 'golf course', 'harbor', 'intersection', 'mobile home park', 'nursing home', 
                           'oil gas field', 'oil well', 'overpass', 'parking lot', 'parking space', 'railway', 
                           'river', 'runway', 'runway marking', 'shipping yard', 'solar panel', 'sparse residential', 
                           'storage tank', 'swimming pool', 'tennis court', 'transformer station', 'wastewater treatment plant']

    
    # DB2(6)
    elif args.dataset_mode == "VArcGIS_VBing":
        args.class_num = 38
        args.class_list = ['airplane', 'baseball field', 'basketball court', 'beach', 'bridge', 
                           'cemetery', 'chaparral','christmas tree farm', 'closed road', 'coastal mansion', 
                           'crosswalk', 'dense residential', 'ferry terminal', 'football field', 'forest', 
                           'freeway', 'golf course', 'harbor', 'intersection', 'mobile home park', 'nursing home', 
                           'oil gas field', 'oil well', 'overpass', 'parking lot', 'parking space', 'railway', 
                           'river', 'runway', 'runway marking', 'shipping yard', 'solar panel', 'sparse residential', 
                           'storage tank', 'swimming pool', 'tennis court', 'transformer station', 'wastewater treatment plant']


    else:
        ValueError("dataset mode name error!")
        exit(0)


    # check
    if args.phase == "train_semi":
        print("semi-supervised setting")
        assert args.ratio > 0.0 and args.ratio < 1.0, "train_semi ratio must in (0.0, 1.0)"
    
    if args.phase == "train_uda":
        print("uda setting")
        args.ratio = 1.0
    
    assert args.lora_r >= 0, "lora_r must more than 0"

    return args
