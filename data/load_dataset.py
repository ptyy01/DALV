from data import AID_UCMD_NWPU_dataset as DB1
from data import PatternNet_VBing_VArcGIS_dataset as DB2

def load_data(args):
    train_dataset, val_dataset, query_dataset, s_db_dataset, t_db_dataset, memory_source_dataset, memory_target_dataset = [], [], [], [], [], [], []
    if args.dataset_mode == "AID_UCMD" or args.dataset_mode == "UCMD_AID" \
    or args.dataset_mode == "AID_NWPU" or args.dataset_mode == "NWPU_AID" \
    or args.dataset_mode == "UCMD_NWPU" or args.dataset_mode == "NWPU_UCMD":
        print("DB1 ===>", args.dataset_mode)
        train_dataset, val_dataset, query_dataset, s_db_dataset, t_db_dataset, memory_source_dataset, memory_target_dataset = DB1.get_data(args)

    if args.dataset_mode == "PatternNet_VBing" or args.dataset_mode == "VBing_PatternNet" \
    or args.dataset_mode == "PatternNet_VArcGIS" or args.dataset_mode == "VArcGIS_PatternNet" \
    or args.dataset_mode == "VBing_VArcGIS" or args.dataset_mode == "VArcGIS_VBing":
        print("DB2 ===>", args.dataset_mode)
        train_dataset, val_dataset, query_dataset, s_db_dataset, t_db_dataset, memory_source_dataset, memory_target_dataset = DB2.get_data(args)

    return train_dataset, val_dataset, query_dataset, s_db_dataset, t_db_dataset, memory_source_dataset, memory_target_dataset