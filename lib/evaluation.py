from tqdm import tqdm
import torch
from sklearn import metrics
from sklearn.cluster import KMeans
from scipy.spatial.distance import squareform, pdist, cdist
import numpy as np
from typing import List


def get_feature_embeddings(model,loader, device, type = 'src'):
    """
    input:
    output:
    - features
    - labels: 
    """
    torch.cuda.empty_cache()
    with torch.no_grad():
        ### For all test images, extract features
        model.eval()
        target_labels, feature_coll = [],[]
        final_iter = tqdm(loader, desc='Computing Evaluation Metrics...')
        for idx,inp in enumerate(final_iter): # get all 
            input_img,target = inp[0], inp[1] # image, label
            target_labels.extend(target.numpy().tolist())
            f, g = model(input_img.to(device))
            out = g if type == 'src' else f
            feature_coll.extend(out.cpu().detach().numpy().tolist()) # list of feature[num, feat_dim]
        
        target_labels = np.hstack(target_labels).reshape(-1,1)
        feature_coll  = np.vstack(feature_coll).astype('float32')
    return feature_coll, target_labels

def recall_k_eval(model, test_dataloader, k_vals:List[int], device):
    feature_coll, target_labels = get_feature_embeddings(model, test_dataloader, device)
    dist_mat = squareform(pdist(feature_coll)) # num, num (distance matrix)
    k_closest_points  = dist_mat.argsort(1)[:, :int(np.max(k_vals)+1)]
    k_closest_classes = target_labels.reshape(-1)[k_closest_points[:, 1:]]

    ### Compute Recall
    recall_all_k = []
    for k in k_vals:
        recall_at_k = np.sum([1 for target, recalled_predictions in zip(target_labels, k_closest_classes) if target in recalled_predictions[:k]])/len(target_labels)
        recall_all_k.append(recall_at_k)

    return recall_all_k

def NMI_eval(model, test_dataloader, n_classes, device, type):
    feature_coll, target_labels = get_feature_embeddings(model, test_dataloader, device, type)
    kmeans = KMeans(n_clusters=n_classes, random_state=0).fit(feature_coll)
    model_generated_cluster_labels = kmeans.labels_
    computed_centroids = kmeans.cluster_centers_ # used to calculate the f1 score

    ### Compute NMI
    NMI = metrics.cluster.normalized_mutual_info_score(model_generated_cluster_labels.reshape(-1), target_labels.reshape(-1))
    return NMI
    

if __name__ == "__main__":
    # run test
    import sys
    sys.path.append('.')
    import os
    from lib.model import FeatureExtractor
    from lib.mnist_loader import get_val_loader
    device = torch.device('cuda')
    model = FeatureExtractor().to(device)
    
    tar_eval_loader = get_val_loader(
        dataset_root = './dataset/MNIST', 
        train=True,
        label_filter=lambda x : x in [5, 6, 7, 8, 9],
        batch_size=256
    )
    src_eval_loader = get_val_loader(
        dataset_root = './dataset/MNIST-M', 
        train=True, 
        label_filter=lambda x : x in [0, 1, 2, 3, 4],
        batch_size=256
    )

    exp_root = 'exp/mnist'
    model_name = 'minst_backbone_09180906.pth'
    model_path = os.path.join(exp_root, model_name)
    model.load_state_dict(torch.load(model_path))

    # recall k test
    # print("10 class, no training model, val on target domain")
    # print(recall_k_eval(model, tar_eval_loader, [1, 5, 10], device))

    # nmi test
    # print("source only test on target domain")
    # print(NMI_eval(model, tar_eval_loader, 5, device))
    # print("source only test on source domain")
    # print(NMI_eval(model, src_eval_loader, 5, device))
    
    print("dann test on target domain")
    print(NMI_eval(model, tar_eval_loader, 5, device))
    print("dann test on source domain")
    print(NMI_eval(model, src_eval_loader, 5, device))