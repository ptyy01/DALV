import os
import time
import torch
import shutil
import numpy as np
from loguru import logger
import torch.optim as optim
from torch.utils.data import DataLoader
from scipy.io import savemat

import loss
import m_models
import utils as ut
import metricsutils as metricuts
import data.load_dataset_semi as mdsets_semi
import data.load_dataset as mdsets
from m_argument import get_argument

torch.set_printoptions(threshold=np.inf)


def training_stage_semi(train_loader, model, pred_source_labels, pred_target_labels, source_centroids, source_weights, target_weights, ccl_criterion, wccl_criterion, disc_criterion, ent_criterion, optimizer, args, epoch):
    model.train()

    total_loss = 0.0
    total_s2centroid_loss = 0.0
    total_t2centroid_loss = 0.0
    total_s2text_loss = 0.0
    total_t2text_loss = 0.0
    total_disc_loss = 0.0
    total_ent_loss = 0.0
    
    for idx, loader in enumerate(train_loader):
        # print("idx =", idx)
        source_images = loader["source_lb_image"]
        source_labels = loader["source_lb_label"]

        un_source_images = loader["source_ulb_image"]
        un_source_idx = loader["source_ulb_index"]

        target_images = loader["target_image"]
        target_idx = loader["target_index"]

        images = torch.cat([source_images, un_source_images, target_images]).to(args.device)
        
        source_labels = source_labels.to(args.device)
        un_source_pseudo_labels = pred_source_labels[un_source_idx].to(args.device)
        un_target_pseudo_labels = pred_target_labels[target_idx].to(args.device)

        un_source_w = source_weights[un_source_idx].to(args.device)
        un_target_w = target_weights[target_idx].to(args.device)
        
        source_centroids = source_centroids.to(args.device)

        bcz = source_images.shape[0]
        
        img_features, txt_features = model(images)

        s2centroid_loss = 0.5 * (ccl_criterion(img_features[:bcz], source_labels, source_centroids) \
                    + wccl_criterion(img_features[bcz:2*bcz], un_source_pseudo_labels, source_centroids, un_source_w))

        t2centroid_loss = wccl_criterion(img_features[2*bcz:], un_target_pseudo_labels, source_centroids, un_target_w)

        s2text_loss = 0.5 * (ccl_criterion(img_features[:bcz], source_labels, txt_features) \
                    + wccl_criterion(img_features[bcz:2*bcz], un_source_pseudo_labels, txt_features, un_source_w))

        t2text_loss = wccl_criterion(img_features[2*bcz:], un_target_pseudo_labels, txt_features, un_target_w)
        
        disc_loss = disc_criterion(source_centroids, txt_features)

        ent_loss = ent_criterion(img_features, source_centroids, txt_features)

        loss = args.lambda_1 * s2centroid_loss + args.lambda_2 * t2centroid_loss + args.lambda_3 * s2text_loss + args.lambda_4 * t2text_loss + args.lambda_5 * disc_loss + args.lambda_6 * ent_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_s2centroid_loss += s2centroid_loss.item()
        total_t2centroid_loss += t2centroid_loss.item()
        total_s2text_loss += s2text_loss.item()
        total_t2text_loss += t2text_loss.item()
        total_disc_loss += disc_loss.item()
        total_ent_loss += ent_loss.item()

    total_loss /= len(train_loader)
    total_s2centroid_loss /= len(train_loader)
    total_t2centroid_loss /= len(train_loader)
    total_s2text_loss /= len(train_loader)
    total_t2text_loss /= len(train_loader)
    total_disc_loss /= len(train_loader)
    total_ent_loss /= len(train_loader)

    return total_loss, total_s2centroid_loss, total_t2centroid_loss, total_s2text_loss, total_t2text_loss, total_disc_loss, total_ent_loss


def training_stage_uda(train_loader, model, pred_target_labels, source_centroids, target_weights, ccl_criterion, wccl_criterion, disc_criterion, ent_criterion, optimizer, args, epoch):
    model.train()

    total_loss = 0.0
    total_s2centroid_loss = 0.0
    total_t2centroid_loss = 0.0
    total_s2text_loss = 0.0
    total_t2text_loss = 0.0
    total_disc_loss = 0.0
    total_ent_loss = 0.0
    
    for idx, loader in enumerate(train_loader):
        source_images = loader["source_image"]
        source_labels = loader["source_label"]

        target_images = loader["target_image"]
        target_idx = loader["target_index"]
        target_w = target_weights[target_idx]

        images = torch.cat([source_images, target_images]).to(args.device)
        
        source_labels = source_labels.to(args.device)
        un_target_pseudo_labels = pred_target_labels[target_idx].to(args.device)
        target_w = target_w.to(args.device)
        
        source_centroids = source_centroids.to(args.device)

        bcz = source_images.shape[0]
        
        img_features, txt_features = model(images)
        
        s2centroid_loss = ccl_criterion(img_features[:bcz], source_labels, source_centroids)        
        t2centroid_loss = wccl_criterion(img_features[bcz:], un_target_pseudo_labels, source_centroids, target_w)

        s2text_loss = ccl_criterion(img_features[:bcz], source_labels, txt_features)
        
        t2text_loss = wccl_criterion(img_features[bcz:], un_target_pseudo_labels, txt_features, target_w)
                
        disc_loss = disc_criterion(source_centroids, txt_features)

        ent_loss = ent_criterion(img_features, source_centroids, txt_features)

        loss = args.lambda_1 * s2centroid_loss + args.lambda_2 * t2centroid_loss + args.lambda_3 * s2text_loss + args.lambda_4 * t2text_loss + args.lambda_5 * disc_loss + args.lambda_6 * ent_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_s2centroid_loss += s2centroid_loss.item()
        total_t2centroid_loss += t2centroid_loss.item()
        total_s2text_loss += s2text_loss.item()
        total_t2text_loss += t2text_loss.item()
        total_disc_loss += disc_loss.item()
        total_ent_loss += ent_loss.item()

    total_loss /= len(train_loader)
    total_s2centroid_loss /= len(train_loader)
    total_t2centroid_loss /= len(train_loader)
    total_s2text_loss /= len(train_loader)
    total_t2text_loss /= len(train_loader)
    total_disc_loss /= len(train_loader)
    total_ent_loss /= len(train_loader)

    return total_loss, total_s2centroid_loss, total_t2centroid_loss, total_s2text_loss, total_t2text_loss, total_disc_loss, total_ent_loss


if __name__ == '__main__':
    args = get_argument()
    print(args)

    if args.phase == 'no':
        print("setting the training phase")
        exit(0)

    log_dir = os.path.join(args.log_dir, "ratio="+str(args.ratio))
    ckpt_dir = os.path.join(log_dir, args.dataset_mode, "checkpoits")
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    now_times = time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime(time.time()))
    log_path = os.path.join(log_dir, args.dataset_mode, args.phase + "-" + now_times + ".log")
    logger.add(log_path, rotation='500MB', level="INFO")
    logger.info(args)

    # load datasets
    if args.phase == "train_semi":
        train_dataset, val_dataset, query_dataset, source_db_dataset, _, source_label_only_dataset, source_unlabel_only_dataset, target_only_dataset = mdsets_semi.load_data(args)
        print("dataset =", len(train_dataset), len(val_dataset), len(query_dataset), len(source_db_dataset), len(source_label_only_dataset), len(source_unlabel_only_dataset), len(target_only_dataset))

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=4)
        query_loader = DataLoader(query_dataset, batch_size=args.batch_size, num_workers=4)
        db_loader = DataLoader(source_db_dataset, batch_size=args.batch_size, num_workers=4)
        source_label_only_loader = DataLoader(source_label_only_dataset, batch_size=args.batch_size, num_workers=4)
        source_unlabel_only_loader = DataLoader(source_unlabel_only_dataset, batch_size=args.batch_size, num_workers=4)
        target_only_loader = DataLoader(target_only_dataset, batch_size=args.batch_size, num_workers=4)
        print("dataloader =", len(train_loader), len(val_loader), len(query_loader), len(db_loader), len(source_label_only_loader), len(source_unlabel_only_loader), len(target_only_loader))

        clip_model = m_models.MYCLIP(args)
        clip_model = clip_model.to(args.device)
    
        ccl_criterion = loss.CentroidsConLoss(args.temperature)
        wccl_criterion = loss.WCentroidsConLoss(args.temperature)
        disc_criterion = loss.DiscriminationLoss(args.temperature)
        ent_criterion = loss.EntropyMinLoss(args.temperature)

        optimizer = optim.Adam(clip_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        mAP_list = []
        mAP = ut.validate_mAP(args, val_loader, db_loader, clip_model)
        mAP_list.append(mAP)
        message = "begin training mAP = {:.4f} \n".format(mAP)
        logger.info(message)

        for epoch in range(args.max_epoch): 
            preds_source_labels, source_centroids, source_weights, s_pred_acc = ut.pseudo_labeling(args, clip_model, source_label_only_loader, source_unlabel_only_loader)
            preds_target_labels, _, target_weights, t_pred_acc = ut.pseudo_labeling(args, clip_model, source_label_only_loader, target_only_loader)
            
            message = "s_acc = {:.4f}, t_acc = {:.4f}".format(s_pred_acc, t_pred_acc)
            logger.info(message)

            train_loss, s2centroid_loss, t2centroid_loss, s2text_loss, t2text_loss, disc_loss, ent_loss = training_stage_semi(train_loader, clip_model, preds_source_labels, preds_target_labels, source_centroids, source_weights, target_weights, ccl_criterion, wccl_criterion, disc_criterion, ent_criterion, optimizer, args, epoch)
            
            mAP = ut.validate_mAP(args, val_loader, db_loader, clip_model)

            message = "epoch = {} | train loss = {:.4f}, s2centroid_loss = {:.4f}, t2centroid_loss = {:.4f}, s2text_loss = {:.4f}, t2text_loss = {:.4f}, disc_loss = {:.4f}, ent_loss = {:.4f} | mAP = {:.4f}\n".format(epoch, train_loss, s2centroid_loss, t2centroid_loss, s2text_loss, t2text_loss, disc_loss, ent_loss, mAP)
            logger.info(message)

            torch.save(clip_model.state_dict(), os.path.join(ckpt_dir, "last.pth"))
            if mAP > max(mAP_list):
                shutil.copy(os.path.join(ckpt_dir, "last.pth"), os.path.join(ckpt_dir, "best.pth"))
                
            mAP_list.append(mAP)

        message = "{} best_mAP = {:.4f} \n".format(args.dataset_mode, max(mAP_list))
        logger.info(message)    
    
    if args.phase == "train_uda":
        train_dataset, val_dataset, query_dataset, source_db_dataset, _, source_only_dataset, target_only_dataset = mdsets.load_data(args)
        print("dataset =", len(train_dataset), len(val_dataset), len(query_dataset), len(source_db_dataset), len(source_only_dataset), len(target_only_dataset))


        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=4)
        query_loader = DataLoader(query_dataset, batch_size=args.batch_size, num_workers=4)
        db_loader = DataLoader(source_db_dataset, batch_size=args.batch_size, num_workers=4)
        source_only_loader = DataLoader(source_only_dataset, batch_size=args.batch_size, num_workers=4)
        target_only_loader = DataLoader(target_only_dataset, batch_size=args.batch_size, num_workers=4)
        print("dataloader =", len(train_loader), len(val_loader), len(query_loader), len(db_loader), len(source_only_loader), len(target_only_loader))

        clip_model = m_models.MYCLIP(args)
        clip_model = clip_model.to(args.device)
    
        ccl_criterion = loss.CentroidsConLoss(args.temperature)
        wccl_criterion = loss.WCentroidsConLoss(args.temperature)
        disc_criterion = loss.DiscriminationLoss(args.temperature)
        ent_criterion = loss.EntropyMinLoss(args.temperature)
        optimizer = optim.Adam(clip_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        mAP_list = []
        mAP = ut.validate_mAP(args, val_loader, db_loader, clip_model)
        # mAP = ut.validate_mAP(args, query_loader, db_loader, clip_model)
        mAP_list.append(mAP)
        message = "begin training mAP = {:.4f} \n".format(mAP)
        logger.info(message)

        for epoch in range(args.max_epoch):
            preds_target_labels, source_centroids, target_weights, t_pred_acc = ut.pseudo_labeling(args, clip_model, source_only_loader, target_only_loader)

            message = "t_acc = {:.4f}".format(t_pred_acc)
            logger.info(message)

            train_loss, s2centroid_loss, t2centroid_loss, s2text_loss, t2text_loss, disc_loss, ent_loss = training_stage_uda(train_loader, clip_model, preds_target_labels, source_centroids, target_weights, ccl_criterion, wccl_criterion, disc_criterion, ent_criterion, optimizer, args, epoch)
            mAP = ut.validate_mAP(args, val_loader, db_loader, clip_model)

            message = "epoch = {} | train loss = {:.4f}, s2centroid_loss = {:.4f}, t2centroid_loss = {:.4f}, s2text_loss = {:.4f}, t2text_loss = {:.4f}, disc_loss = {:.4f}, ent_loss = {:.4f} | mAP = {:.4f}\n".format(epoch, train_loss, s2centroid_loss, t2centroid_loss, s2text_loss, t2text_loss, disc_loss, ent_loss, mAP)
            logger.info(message)

            torch.save(clip_model.state_dict(), os.path.join(ckpt_dir, "last.pth"))
            if mAP > max(mAP_list):
                shutil.copy(os.path.join(ckpt_dir, "last.pth"), os.path.join(ckpt_dir, "best.pth"))
                
            mAP_list.append(mAP)

        message = "{} best_mAP = {:.4f} \n".format(args.dataset_mode, max(mAP_list))
        logger.info(message)    
    
    if args.phase == "retrieval":
        train_dataset, val_dataset, query_dataset, source_db_dataset, _, source_only_dataset, target_only_dataset = mdsets.load_data(args)
        print("dataset =", len(train_dataset), len(val_dataset), len(query_dataset), len(source_db_dataset), len(source_only_dataset), len(target_only_dataset))

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=4)
        query_loader = DataLoader(query_dataset, batch_size=args.batch_size, num_workers=4)
        db_loader = DataLoader(source_db_dataset, batch_size=args.batch_size, num_workers=4)
        source_only_loader = DataLoader(source_only_dataset, batch_size=args.batch_size, num_workers=4)
        target_only_loader = DataLoader(target_only_dataset, batch_size=args.batch_size, num_workers=4)
        print("dataloader =", len(train_loader), len(val_loader), len(query_loader), len(db_loader), len(source_only_loader), len(target_only_loader))
        
        clip_model = m_models.MYCLIP(args)
        ckpt_path = os.path.join(ckpt_dir, "best.pth")
        clip_model.load_state_dict(torch.load(ckpt_path))
        clip_model = clip_model.to(args.device)
        save_dir = os.path.join(log_dir, args.dataset_mode, "retrieval")
        os.makedirs(save_dir, exist_ok=True)

        query_index = os.path.join(save_dir, 'query_index.h5')
        database_index = os.path.join(save_dir, 'db_index.h5')

        print("valid map, anmrr, p@k...")

        metricuts.get_rs_feature(query_loader, clip_model, args.device, query_index)
        metricuts.get_rs_feature(db_loader, clip_model, args.device, database_index)

        ANMRR, mAP, Pk = metricuts.execute_retrieval(save_dir, pools=10, classes=args.class_num)
        
        message = "{} ANMRR = {:.4f}, mAP = {:.4f}\n".format(args.dataset_mode, ANMRR, mAP)
        logger.info(message)


    if args.phase == "analyse":
        train_dataset, val_dataset, query_dataset, source_db_dataset, _, source_only_dataset, target_only_dataset = mdsets.load_data(args)
        print("dataset =", len(train_dataset), len(val_dataset), len(query_dataset), len(source_db_dataset), len(source_only_dataset), len(target_only_dataset))

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=4)
        query_loader = DataLoader(query_dataset, batch_size=args.batch_size, num_workers=4)
        db_loader = DataLoader(source_db_dataset, batch_size=args.batch_size, num_workers=4)
        source_only_loader = DataLoader(source_only_dataset, batch_size=args.batch_size, num_workers=4, shuffle=False)
        target_only_loader = DataLoader(target_only_dataset, batch_size=args.batch_size, num_workers=4, shuffle=False)
        print("dataloader =", len(train_loader), len(val_loader), len(query_loader), len(db_loader), len(source_only_loader), len(target_only_loader))
        
        clip_model = m_models.MYCLIP(args)
        ckpt_path = os.path.join(ckpt_dir, "best.pth")
        clip_model.load_state_dict(torch.load(ckpt_path))
        clip_model = clip_model.to(args.device)
        analyse_dir = os.path.join(log_dir, args.dataset_mode, "retrieval")
        os.makedirs(analyse_dir, exist_ok=True)

        source_feature, source_label, _ = ut.extract_features(args, clip_model, source_only_loader)
        target_feature, target_label, _ = ut.extract_features(args, clip_model, target_only_loader)

        print(source_feature.shape, source_label.shape)
        print(target_feature.shape, target_label.shape)

        dict = {"source_feature": source_feature.numpy(), 
                "source_label": source_label.numpy(), 
                "target_feature": target_feature.numpy(), 
                "target_label": target_label.numpy()}

        save_file = os.path.join(analyse_dir, "features.mat")
        savemat(save_file, dict)
    
    
