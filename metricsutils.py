import csv
import h5py
import torch
import numpy as np
from tqdm import tqdm
from collections import Counter
from multiprocessing import Pool


def get_rs_feature(data_loader, feature_extractor, device, index_file):

    feature_extractor.eval()

    img_paths = []
    preds = []
    labels = []
    with torch.no_grad():
        for i, (img, label, onehot_label, img_path) in enumerate(tqdm(data_loader)):

            img = img.to(device)
            # feature = feature_extractor.get_feature(img).detach().cpu()
            feature, _ = feature_extractor(img)
            feature = feature.detach().cpu()
            
            img_paths.append(img_path)
            labels.append(label)
            preds.append(feature)
            
        print('Writing features information to the file')

        img_paths = np.concatenate(img_paths, axis=0)
        labels = np.concatenate(labels, axis=0)
        preds = torch.cat(preds, dim=0).numpy()

        print(labels.shape, preds.shape)
        img_paths_encode = []
        for word in img_paths:
            img_paths_encode.append(word.encode())

        h5f = h5py.File(index_file, 'w')
        h5f.create_dataset('img_paths_encode', data=img_paths_encode)
        h5f.create_dataset('labels', data=labels)
        h5f.create_dataset('preds', data=preds)
        h5f.close()
        print('done!')


def cal_distance(featureA, featureB):
    '''
    Euclidean distance of two feature
    :param featureA:
    :param featureB:
    :return:Euclidean distance (float)
    '''
    featureA = featureA.flatten()
    featureB = featureB.flatten()

    return np.sqrt(np.sum(np.square(featureA - featureB)))


def retrieval_h5py_thread(query_feature, query_labels,query_img_names,database_feature,database_labels,database_img_names):
    """
    Functions constructed to implement multi-threading: a class of queries a thread
    :param query_feature:
    :param query_labels:
    :param query_img_names:
    :param database_feature:
    :param database_labels:
    :param database_img_names:
    :return:
    """
    print(str(query_labels[0]) + ' category image start retrieval...')
    accus = []
    for index, (query_temp_feature, query_temp_label, query_temp_name) in enumerate(zip(query_feature, query_labels, query_img_names)):

        query_result_dict = {}
        query_result_dict.setdefault(query_temp_name, []).append(-1)
        query_result_dict.setdefault(query_temp_name, []).append(query_temp_label)
        query_result_dict.setdefault(query_temp_name, []).append(1)

        for database_temp_feature, database_temp_label, data_temp_name in zip(database_feature,database_labels,database_img_names):
            relevance_flag = 0

            distance = cal_distance(database_temp_feature, query_temp_feature)

            query_result_dict.setdefault(data_temp_name, []).append(distance)
            query_result_dict.setdefault(data_temp_name, []).append(database_temp_label)
            if query_temp_label == database_temp_label:
                relevance_flag = 1
            query_result_dict.setdefault(data_temp_name, []).append(relevance_flag)

        sort_result = sorted(query_result_dict.items(), key=lambda x: x[1][0])

        accu = []
        for i, val in enumerate(sort_result):
            rele = val[1][2]
            accu.append(rele)

        if index == 0:
            accu_np = np.array(accu)
            accu_np = np.expand_dims(accu_np, axis=0)
            accus = accu_np
        else:
            accu_np = np.array(accu)
            accu_np = np.expand_dims(accu_np, axis=0)
            accus = np.concatenate((accus, accu_np), axis=0)

    accus = np.delete(accus, [0], axis=1)

    print(str(query_labels[0]) + 'category image finish retrieval')

    return accus, str(query_labels[0])



def retrieval_h5py_by_thread(db_index_file, query_index_file, classes, pools=10):
    """
    Calculate the distance of all queries using h5f file retrieval
    :param db_index_file: Index file path of the image database
    :param distance_file: Path to the distance dictionary to be saved, distance dictionary file of the form [(image name, [distance, actual class label, whether relevant])]
    :return: None
    """
    h5f = h5py.File(db_index_file, 'r')
    img_paths_encode = h5f['img_paths_encode'][:]
    database_feature = h5f['preds'][:]
    database_labels = h5f['labels'][:]
    h5f.close()
    database_img_names = np.char.decode(img_paths_encode.astype(np.string_))

    h5f = h5py.File(query_index_file, 'r')
    query_img_names = h5f['img_paths_encode'][:]
    query_feature = h5f['preds'][:]
    query_labels = h5f['labels'][:]
    h5f.close()
    query_img_names = np.char.decode(query_img_names.astype(np.string_))

    query_count_dict = Counter(query_labels)
    print(query_count_dict)
    query_class_count = len(query_count_dict)
    threads = []
    start_query = 0
    end_query = 0

    main_pool = Pool(pools)
    results = []
    print("query_class_count =", query_class_count)
    for i in range(query_class_count):
        if i == 0:
            result = main_pool.apply_async(retrieval_h5py_thread,args=(
                query_feature[:query_count_dict[i]],query_labels[:query_count_dict[i]],
                query_img_names[:query_count_dict[i]],database_feature,database_labels,database_img_names, ))

            results.append(result)
            start_query = start_query + query_count_dict[i]
        else:
            end_query = start_query + query_count_dict[i]

            result = main_pool.apply_async(retrieval_h5py_thread, args=(
                query_feature[start_query:end_query], query_labels[start_query:end_query],
                query_img_names[start_query:end_query], database_feature, database_labels, database_img_names,))

            results.append(result)
            start_query = end_query
    main_pool.close()
    main_pool.join()

    names = locals()
    for i in results:
        rele, ind = i.get()
        names['prec' + str(ind)] = rele

    matrix = []
    for i in range(classes):
        if i == 0:
            temp = names.get('prec' + str(i))
            matrix = temp
        else:
            temp = names.get('prec' + str(i))
            matrix = np.concatenate((matrix, temp), axis=0)

    print('..........Finish Retrieval..........')
    return matrix

def get_ng_k(database_file):
    """
    Calculate the number of ng actually associated with the query qi in the image library
    Calculate k=min(4*ng,2M) m=max{Ng(q1),Ng(q2)，....,Ng(qn)}
    :param database_file: Indexing of the database
    :return: Tags: number of dictionaries and maximum number of similarities corresponding to all queries
    """
    h5f = h5py.File(database_file, 'r')
    database_labels = h5f['labels'][:]
    h5f.close()
    label_count_dict = Counter(database_labels)
    all_query_max_count = max(label_count_dict.values())
    return label_count_dict, all_query_max_count


def get_query_label(query_file):
    """
    Return to the list of tags for the image to be queried
    :param query_file: Index list of images to be queried
    :return: Back to the list of tags
    """
    h5f = h5py.File(query_file, 'r')
    query_labels = h5f['labels'][:]
    h5f.close()
    return query_labels

def cal_precision(all_rel_list, database_index_path, query_index_path, metric_path):
    """
    Calculate various accuracies based on saved distance files, image data feature libraries, and image feature libraries to be queried
    including ANMRR/mAP/P@5/P@10/P@20/P@50/P@100/P@1000
    :param distance_path: Sorted distance file
        :param database_index_path: Image data feature library file path
    :param query_index_path: Feature library file of the image to be queried
    :param metric_path: Path to the csv file for saving accuracy
    :return: ANMRR, mAP, [P@K]
    """
    writer_metric = csv.writer(open(metric_path, 'a', newline='', encoding='utf8'))
    label_count, max_count = get_ng_k(database_index_path)
    query_label_list = get_query_label(query_index_path)

    nmrr_list = []
    writer_metric.writerow(['NMRR', 'AP', 'P@5', 'P@10', 'P@20', 'P@50', 'P@100', 'P@1000'])
    p_5_list = []
    p_10_list = []
    p_20_list = []
    p_50_list = []
    p_100_list = []
    p_1000_list = []
    q_ap_list = []
    for q_rel, q_label in zip(all_rel_list, query_label_list):
        p_5 = precision_at_k(q_rel, 5)
        p_10 =precision_at_k(q_rel, 10)
        p_20 = precision_at_k(q_rel, 20)
        p_50 = precision_at_k(q_rel, 50)
        p_100 = precision_at_k(q_rel, 100)
        p_1000 = precision_at_k(q_rel, 1000)
        q_ap = average_precision(q_rel)
        k_value = min(4 * label_count[q_label], 2 * max_count)
        q_avr, q_mrr, q_nmrr = nmrr(q_rel, label_count[q_label], k_value)
        writer_metric.writerow([q_nmrr, q_ap, p_5, p_10, p_20, p_50, p_100, p_1000])
        nmrr_list.append(q_nmrr)
        p_5_list.append(p_5)
        p_10_list.append(p_10)
        p_20_list.append(p_20)
        p_50_list.append(p_50)
        p_100_list.append(p_100)
        p_1000_list.append(p_1000)
        q_ap_list.append(q_ap)
    a_p_5 = np.mean(p_5_list)
    a_p_10 = np.mean(p_10_list)
    a_p_20 = np.mean(p_20_list)
    a_p_50 = np.mean(p_50_list)
    a_p_100 = np.mean(p_100_list)
    a_p_1000 = np.mean(p_1000_list)
    m_q_ap = np.mean(q_ap_list)
    q_anmrr = anmrr(nmrr_list)
    writer_metric.writerow(['ANMRR', 'mAP', 'a_P@5', 'a_P@10', 'a_P@20', 'a_P@50', 'a_P@100', 'a_P@1000'])
    writer_metric.writerow([q_anmrr, m_q_ap, a_p_5, a_p_10, a_p_20, a_p_50, a_p_100, a_p_1000])

    return q_anmrr, m_q_ap, [a_p_5, a_p_10, a_p_20, a_p_50, a_p_100]

def execute_retrieval(save_path, pools=10, classes=38):

    query_index_file = r'' + save_path + '/query_index.h5'
    database_index_file = r'' + save_path + '/db_index.h5'
    metric_file_path = r'' + save_path + '/metric.csv'

    # 1. Calculate image feature distance and sorting
    matrix = retrieval_h5py_by_thread(database_index_file, query_index_file, classes, pools)
    # 2. Calculate retrieval accuracy
    ANMRR, mAP, Pk = cal_precision(matrix, database_index_file, query_index_file, metric_file_path)

    return ANMRR, mAP, Pk



def precision_at_k(r, k):
    """
    计算查准率P@k
    :param r: 单个查询返回的排序相关列表 [1, 1, 0, 1, 0, 1, 0, 0, 0, 1]
    :param k: 计算前几个查准率
    :return: P@k
    """
    assert k >= 1
    r = np.asarray(r)[:k] != 0
    if r.size != k:
        r = 0.0
        # raise ValueError('Relevance score length < k')
    return np.mean(r)


def average_precision(r):
    """
    计算AP
    :param r: 单个查询返回的排序相关列表 [1, 1, 0, 1, 0, 1, 0, 0, 0, 1]
    :return: AP
    """
    r = np.asarray(r) != 0
    out = [precision_at_k(r, k + 1) for k in range(r.size) if r[k]]
    if not out:
        return 0.
    return np.mean(out)


def mean_average_precision(rs):
    """
    计算mAP
    :param rs: 各个查询排序后的相关列表[[1, 1, 0, 1, 0, 1, 0, 0, 0, 1]]或是[[1, 1, 0, 1, 0, 1, 0, 0, 0, 1], [0]]
    :return:返回mAP
    """
    return np.mean([average_precision(r) for r in rs])


def nmrr(r, ng, k):
    """
    计算归一化修正后的检索秩
    :param r:代表单个查询是否相关的列表[1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0]
    :param ng:在图像库中与查询qi实际相关的个数
    :param k: k=min(4*ng,2M) m=max{Ng(q1),Ng(q2)，....,Ng(qn)}
    :return:归一化修正后的检索秩
    """
    sum = 0
    rank = [index + 1 for index, value in enumerate(r) if value]
    # print(rank)
    for i in range(0,ng):
        if i < len(rank):
            if rank[i] <= k:
                sum = sum +  rank[i]
            else:
                sum = sum + (k + 1)
        else:
            sum = sum + (k + 1)
    avr = sum/ng
    mrr = avr - 0.5 * (1 + ng)
    nmrr = mrr / (k + 0.5 - 0.5 * ng)
    return avr, mrr, nmrr

def anmrr(nmrr):
    """
    计算平均归一化修正后的检索秩
    :param nmrr: 包含nq个查询q的nmrr列表
    :return: anmrr
    """
    anmrr =  np.mean(nmrr)

    return  anmrr
