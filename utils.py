import torch
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
import scipy.spatial.distance as ssd
from torch.utils.data import DataLoader


import numpy as np


def get_subset_idx(dataset, args, m_num):
    per_num = int(len(dataset) / args.class_num)
    print("per num =", per_num)

    idx_list = []

    for lb in range(args.class_num):
        for i in range(m_num):
            temp_idx = lb * per_num + i
            idx_list.append(temp_idx)

    idx_list = torch.tensor(idx_list)
    
    
    # # lb_list = [temp_dsets[1] for temp_dsets in dataset]
    # # lb_list = np.array(lb_list)
    # # lb_list = torch.from_numpy(lb_list)

    # idx_list = []

    # for lb in range(args.class_num):
    #     idx = torch.where(lb_list == lb)[0]
    #     idx_list.append(idx[:m_num])
    
    return idx_list    


def compute_result(dataloader, model, args):
    bs, clses = [], []

    model.eval()
    with torch.no_grad():  
        for data in tqdm(dataloader):
            image = data[0]
            one_hot_labels = data[2]

            image = image.to(args.device)
            image_feature, _ = model(image)
            clses.append(one_hot_labels)
            bs.append(image_feature.data.cpu())             
    
    return torch.cat(bs), torch.cat(clses)


def CalcDist(B1, B2):
    q = B2.shape[1]
    distH = 0.5 * (q - np.dot(B1, B2.transpose()))
    return distH


def Euclidean_Distances(f1, f2):
    f1 = torch.from_numpy(f1).float()
    f2 = torch.from_numpy(f2).float()
    
    dist = torch.cdist(f1.unsqueeze(0), f2.unsqueeze(0))
    dist = dist.squeeze(0).squeeze(0)

    dist = dist.numpy()
    
    return dist


def CalcTopMap(db_feature, query_feature, db_one_hot_label, query_one_hot_label, topk):
    print("sshape =", db_feature.shape, query_feature.shape, db_one_hot_label.shape, query_one_hot_label.shape)
    num_query = query_one_hot_label.shape[0]
    topkmap = 0

    i = 0
    for iter in tqdm(range(num_query)):
        gnd = (np.dot(query_one_hot_label[iter, :], db_one_hot_label.transpose()) > 0).astype(np.float32)
        # m_dist = CalcDist(query_feature[iter, :], db_feature)
        # i += 1
        # print("i =", i, "mdist.shape =", m_dist.shape)
        # print(m_dist)
        
        m_dist = Euclidean_Distances(query_feature[iter, :], db_feature)

        ind = np.argsort(m_dist)
        gnd = gnd[ind]
        tgnd = gnd[0:topk]
        tsum = np.sum(tgnd).astype(int)
        if tsum == 0:
            continue
        count = np.linspace(1, tsum, tsum)

        tindex = np.asarray(np.where(tgnd == 1)) + 1.0
        topkmap_ = np.mean(count / (tindex))
        topkmap = topkmap + topkmap_
    
    topkmap = topkmap / num_query
    return topkmap


def validate_mAP(args, query_loader, db_loader, model):
    print("calculating query_loader feature......")
    query_feature, query_one_hot_label = compute_result(query_loader, model, args)

    print("calculating db_loader feature......")
    db_feature, db_one_hot_label = compute_result(db_loader, model, args)

    print("calculating mAP......")
    mAP = CalcTopMap(db_feature.numpy(), query_feature.numpy(), db_one_hot_label.numpy(), query_one_hot_label.numpy(), args.topk)

    return mAP


def validate_acc(args, db_loader, feature_extractor, classifier):
    feature_extractor.eval()
    classifier.eval()
    total_num = 0.0
    for loader in db_loader:
        images = loader[0].to(args.device)
        labels = loader[1].to(args.device)

        logits = classifier(feature_extractor(images))
        _, pred = torch.max(logits, 1)
        total_num += pred.eq(labels).float().sum().item()
    
    acc = total_num / len(db_loader.dataset)
    return acc


def extract_features(args, model, loader):
    # print("extract features...")

    img_features_list = []
    labels_list = []
    
    model.eval()
    with torch.no_grad():
        for idx, data in enumerate(loader):
            images = data[0].to(args.device)
            labels = data[1]

            img_features, txt_features = model(images)

            img_features_list.append(img_features.data.cpu())
            labels_list.append(labels)
    
    return torch.cat(img_features_list), torch.cat(labels_list), txt_features.data.cpu()


def select_samples_source(args, features, txt_features, labels):
    features = F.normalize(features, dim=-1)
    txt_features = F.normalize(txt_features, dim=-1)

    select_idx = []
    for lb in range(args.class_num):
        idx = torch.where(labels==lb)[0]
        temp_features = features[idx]
        logits = (100 * txt_features[lb] @ temp_features.T).softmax(dim=-1)

        if logits.shape[0] < args.select_num:
            _, temp_select_idx = torch.topk(logits, k=logits.shape[0], dim=-1)
        else:
            _, temp_select_idx = torch.topk(logits, k=args.select_num, dim=-1)
        select_idx.append(idx[temp_select_idx])
    
    select_idx = torch.cat(select_idx)
    print("select source:", select_idx.shape)
    return select_idx


def select_samples_target(args, features, txt_features):
    features = F.normalize(features, dim=-1)
    txt_features = F.normalize(txt_features, dim=-1)
    
    logits = (100 * txt_features @ features.T).softmax(dim=-1)
    print(logits.shape)

    _, select_idx = torch.topk(logits, k=args.select_num, dim=-1)
    print("select target:", select_idx.shape)
    return select_idx


def update_centroids(args, centroids, features, labels):
    batch_centroids = []
    for lb in range(args.class_num):
        idx = torch.where(labels==lb)[0]
        if len(idx) == 0:
            temp_centroids = centroids[lb]
            # temp_centroids = torch.zeros_like(features[0])
            batch_centroids.append(temp_centroids.unsqueeze(0))
        else:
            temp_features = features[idx]
            temp_centroids = temp_features.mean(0)
            batch_centroids.append(temp_centroids.unsqueeze(0))

    batch_centroids = torch.cat(batch_centroids)
    centroids = args.m * centroids + (1 - args.m) * batch_centroids

    return centroids


def update_centroids2(args, centroids, select_datasets, model):
    select_num = int(len(select_datasets) / args.class_num)
    select_loader = DataLoader(select_datasets, batch_size=args.batch_size)
    labels = torch.arange(0, args.class_num)
    labels = labels.repeat(select_num)
    labels, _ = torch.sort(labels)

    features, _, _ = extract_features(args, model, select_loader)

    batch_centroids = []
    for lb in range(args.class_num):
        idx = torch.where(labels==lb)[0]
        temp_features = features[idx]
        temp_centroids = temp_features.mean(0)
        batch_centroids.append(temp_centroids.unsqueeze(0))

    batch_centroids = torch.cat(batch_centroids)
    centroids = args.m * centroids + (1 - args.m) * batch_centroids

    return centroids


def get_pseudo_labels(features, centroids, labels, class_num):
    features = F.normalize(features, dim=-1)
    centroids = F.normalize(centroids, dim=-1)
    
    logits = (100 * features @ centroids.T).softmax(dim=-1)
    # logits = (100 * features @ text_centroids.T).softmax(dim=-1)
    _, pred_target_label = torch.max(logits, dim=-1)
    acc = torch.sum(pred_target_label == labels) / len(pred_target_label)
    print("zsc acc = {:.4f}".format(acc))

    features = features.float().detach().cpu()
    # features = (features.t() / torch.norm(features, p=2, dim=1)).t()
    
    features = features.numpy()
    aff = logits.float().detach().cpu().numpy()
    initc = aff.transpose().dot(features)
    initc = initc / (1e-8 + aff.sum(axis=0)[:, None])
    dd = ssd.cdist(features, initc, 'cosine')
    pred_target_label = dd.argmin(axis=1)

    pred_target_label = torch.from_numpy(pred_target_label)

    acc = torch.sum(pred_target_label == labels) / len(pred_target_label)
    print("1 st acc = {:.4f}".format(acc))

    initc = torch.from_numpy(initc)
    # for round in range(1):
    #     aff = np.eye(class_num)[pred_target_label]
    #     initc = aff.transpose().dot(features)
    #     initc = initc / (1e-8 + aff.sum(axis=0)[:, None])
        
    #     dd = ssd.cdist(features, initc, 'cosine')
    #     pred_target_label = dd.argmin(axis=1)
    
    # pred_target_label = torch.from_numpy(pred_target_label)

    return pred_target_label, initc, acc


def semantic_guided_pseudo_labeling(args, model, unlabel_loader):
    all_features, old_labels, text_features = extract_features(args, model, unlabel_loader)
    print(all_features.shape, old_labels.shape, text_features.shape)
    
    select_idx = select_samples_target(args, all_features, text_features)
    pred_labels, centroids = get_pseudo_labels(all_features, text_features, old_labels, args.class_num)
    acc = torch.sum(pred_labels == old_labels) / len(pred_labels)

    return pred_labels, centroids, acc, select_idx


def source_centroids_pseudo_labeling(args, model, label_loader, unlabel_loader):
    all_label_features, all_label_label, text_features = extract_features(args, model, label_loader)
    all_features, old_labels, _ = extract_features(args, model, unlabel_loader)

    select_idx = select_samples_source(args, all_label_features, text_features, all_label_label)

    centroids = []
    for lb in range(args.class_num):
        idx = torch.where(all_label_label==lb)[0]
        temp_features = all_label_features[idx]

        temp_centroids = temp_features.mean(0)
        centroids.append(temp_centroids.unsqueeze(0))

    centroids = torch.cat(centroids)

    print(all_features.shape, old_labels.shape, centroids.shape)
    pred_labels, _ = get_pseudo_labels(all_features, centroids, old_labels, args.class_num)
    acc = torch.sum(pred_labels == old_labels) / len(pred_labels)
    return pred_labels, centroids, acc, select_idx


def compute_entropy(p, axis=1):
    return -torch.sum(p * torch.log(p+1e-5), dim=axis)


def compute_weight(avg_logits, class_num):
    ent = compute_entropy(avg_logits)
    max_ent = torch.log(torch.tensor(float(class_num)))
    ent = ent / max_ent
    w = torch.exp(-ent)
    return w


def pseudo_labeling(args, model, label_loader, unlabel_loader):
    all_label_features, all_label_label, text_features = extract_features(args, model, label_loader)
    all_unlabel_features, all_old_labels, _ = extract_features(args, model, unlabel_loader)

    source_centroids = []
    for lb in range(args.class_num):
        idx = torch.where(all_label_label==lb)[0]
        temp_features = all_label_features[idx]
        temp_centroids = temp_features.mean(0)
        source_centroids.append(temp_centroids.unsqueeze(0))

    source_centroids = torch.cat(source_centroids)

    pred_labels1, centroids1, acc = get_pseudo_labels(all_unlabel_features, source_centroids, all_old_labels, args.class_num)
    pred_labels2, centroids2, _ = get_pseudo_labels(all_unlabel_features, text_features, all_old_labels, args.class_num)

    logits1 = (all_unlabel_features @ centroids1.T).softmax(dim=-1)
    logits2 = (all_unlabel_features @ centroids2.T).softmax(dim=-1)

    avg_logits = (logits1 + logits2) / 2
    weight = compute_weight(avg_logits, args.class_num)

    return pred_labels1, source_centroids, weight, acc


def nearest_centroids_classification(args, model, label_loader, unlabel_loader):
    all_label_features, all_label_label, _ = extract_features(args, model, label_loader)
    all_unlabel_features, all_unlabel_label, _ = extract_features(args, model, unlabel_loader)

    centroids = []
    for lb in range(args.class_num):
        idx = torch.where(all_label_label==lb)[0]
        temp_features = all_label_features[idx]

        temp_centroids = temp_features.mean(0)
        centroids.append(temp_centroids.unsqueeze(0))

    centroids = torch.cat(centroids)

    features = F.normalize(all_unlabel_features, dim=-1)
    centroids = F.normalize(centroids, dim=-1)
    
    logits = (features @ centroids.T).softmax(dim=-1)
    _, pred_labels = torch.max(logits, dim=-1)
    acc = torch.sum(pred_labels == all_unlabel_label) / len(pred_labels)

    return pred_labels, acc


def knn_classification(args, model, label_loader, unlabel_loader):
    all_label_features, all_label_label, _ = extract_features(args, model, label_loader)
    all_unlabel_features, all_unlabel_label, _ = extract_features(args, model, unlabel_loader)

    label_features = F.normalize(all_label_features, dim=-1)
    unlabel_features = F.normalize(all_unlabel_features, dim=-1)

    sim = unlabel_features @ label_features.T
    _, idx = torch.max(sim, dim=-1)
    pred_labels = all_label_label[idx]

    acc = torch.sum(pred_labels == all_unlabel_label) / len(pred_labels)

    return pred_labels, acc



