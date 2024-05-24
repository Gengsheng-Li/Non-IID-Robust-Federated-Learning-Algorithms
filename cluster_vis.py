import argparse
import torch

import matplotlib.pyplot as plt

from ultils import *
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description="Train model in a FL manner.")
    parser.add_argument('--user', type=str, default='shuang')
    
    parser.add_argument('--real_world', type=int, default=1)
    parser.add_argument('--dim_reduction', type=str, default='pca', choices=["pca", "t-sne"])
    parser.add_argument('--divide_method', type=str, default='number', choices=["number", "silhouette", "both"])
    parser.add_argument('--perplexity', type=int, default=1)
    
    parser.add_argument('--num_clients', type=int, default=20)
    parser.add_argument('--num_clusters', type=int, default=5) 
    parser.add_argument('--n_iid', type=int, default=10)
    parser.add_argument('--classes_pc', type=int, default=2)

    parser.add_argument('--plot_silhouette_curve', action="store_true")
    parser.add_argument('--round_for_silhouette', type=int, default=10)
    
    parser.add_argument('--verbose', action="store_true")

    return parser.parse_args()

def plot_silhouette_curve(class_distribution, possible_k_values):
    silhouette_scores = []

    for k in possible_k_values:
        kmeans = KMeans(n_clusters=k, random_state=0).fit(class_distribution)
        labels = kmeans.labels_
        score = silhouette_score(class_distribution, labels)
        silhouette_scores.append(score)
        # print(f"Silhouette Score for k={k}: {score}")

    return silhouette_scores

def divide_via_number(args, client_labels, reduced_data, iid_idx, noniid_idx):
    max_cluster_label = max(range(args.num_clusters), key=lambda x: sum(client_labels == x))
    client_types = ['iid' if label == max_cluster_label else 'non-iid' for label in client_labels]
    
    # 初始化混淆矩阵元素
    TP = sum(1 for i in iid_idx if client_types[i] == 'iid')
    FP = sum(1 for i in noniid_idx if client_types[i] == 'iid')
    TN = sum(1 for i in noniid_idx if client_types[i] == 'non-iid')
    FN = sum(1 for i in iid_idx if client_types[i] == 'non-iid')
    
    # 计算评估指标
    accuracy = (TP + TN) / (TP + FP + TN + FN) if (TP + FP + TN + FN) > 0 else 0
    precision = TP / (TP + FP) if TP + FP > 0 else 0
    recall = TP / (TP + FN) if TP + FN > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0

    
    # 绘制 'iid' 和 'non-iid' 客户端的分布图
    plt.figure(figsize=(10, 6))
    # 首先绘制所有的 IID 点
    for i, client_type in enumerate(client_types):
        if client_type == 'iid':
            plt.scatter(reduced_data[i, 0], reduced_data[i, 1], c='blue', label='IID' if 'IID' not in plt.gca().get_legend_handles_labels()[1] else "")

    # 然后绘制所有的 Non-IID 点
    for i, client_type in enumerate(client_types):
        if client_type == 'non-iid':
            plt.scatter(reduced_data[i, 0], reduced_data[i, 1], c='red', label='Non-IID' if 'Non-IID' not in plt.gca().get_legend_handles_labels()[1] else "")

    plt.title(f'Distribution of IID and Non-IID Clients based on Number ({args.dim_reduction}, Client Num={args.num_clients}, K={args.num_clusters})')
    plt.legend()
    plt.savefig(f'./data_distribution_results/iid_non_iid_distribution_rw{args.real_world}_{args.dim_reduction}_nc{args.num_clients}_c{args.num_clusters}_dmnumber.png')
    plt.close()
    
    return {'TP': TP, 'FP': FP, 'TN': TN, 'FN': FN, 'Accuracy': accuracy, 'Precision': precision, 'Recall': recall, 'F1-Score': f1_score}

def divide_via_silhouette(args, client_labels, reduced_data, cluster_scores):
    # 找出具有最高平均 Silhouette 分数的聚类
    max_silhouette_cluster = max(cluster_scores, key=cluster_scores.get)
    client_types = ['iid' if label == max_silhouette_cluster else 'non-iid' for label in client_labels]
    
    # 绘制 'iid' 和 'non-iid' 客户端的分布图
    plt.figure(figsize=(10, 6))
    # 首先绘制所有的 IID 点
    for i, client_type in enumerate(client_types):
        if client_type == 'iid':
            plt.scatter(reduced_data[i, 0], reduced_data[i, 1], c='blue', label='IID' if 'IID' not in plt.gca().get_legend_handles_labels()[1] else "")

    # 然后绘制所有的 Non-IID 点
    for i, client_type in enumerate(client_types):
        if client_type == 'non-iid':
            plt.scatter(reduced_data[i, 0], reduced_data[i, 1], c='red', label='Non-IID' if 'Non-IID' not in plt.gca().get_legend_handles_labels()[1] else "")

    plt.title(f'Distribution of IID and Non-IID Clients based on Silhouette Scores ({args.dim_reduction}, Client Num={args.num_clients}, K={args.num_clusters})')
    plt.legend()
    plt.savefig(f'./data_distribution_results/iid_non_iid_distribution_rw{args.real_world}_{args.dim_reduction}_nc{args.num_clients}_c{args.num_clusters}_dmsilhouette.png')   
    plt.close()
    
def main(args):
    # 获取数据
    # Get CIFAR-10 dataset
    x_train, y_train, _, _ = get_cifar10()

    if args.real_world == 0:
        split = split_image_data_realwd(x_train, y_train, n_clients=args.num_clients, verbose=args.verbose)
    elif args.real_world == 1:
        split, iid_idx, noniid_idx = split_balanced_non_iid_w_division(x_train, y_train, n_clients=args.num_clients, n_iid=args.n_iid, max_classes_per_client=args.classes_pc)
    else:  
        split = split_image_data(x_train, y_train, n_clients=args.num_clients, classes_per_client=args.classes_pc, shuffle=True, verbose=args.verbose)

    # Store the client category distribution matrix
    class_distribution_matrix = create_class_distribution_matrix(split)
    class_distribution_matrix.to_csv(f'./data_distribution_results/class_distribution_matrix-ba_rw{args.real_world}.csv')
    class_distribution = get_class_distribution(split)
    
    
    # 进行聚类
    # Using kmeans to cluster the clients
    kmeans = KMeans(n_clusters=args.num_clusters, random_state=0).fit(class_distribution)
    client_labels = kmeans.labels_

    for i in range(args.num_clusters):
        print(f"Cluster {i}: {sum(client_labels == i)}")
    for i, label in enumerate(client_labels):
        print(f"Client {i} distributrd to cluster {label}")


    # 计算指标
    cluster_sizes = [sum(client_labels == i) for i in range(args.num_clusters)]
    
    # Calculate individual silhouette scores for each sample
    silhouette_values = silhouette_samples(class_distribution, client_labels)

    # Calculate average silhouette score for each cluster
    cluster_scores = {}
    for i in range(args.num_clusters):
        cluster_indices = (client_labels == i)
        cluster_scores[i] = silhouette_values[cluster_indices].mean()
        print(f"Average Silhouette Score for Cluster {i}: {cluster_scores[i]}")


    # 降维可视化
    if args.dim_reduction == "pca":
        # Apply PCA for dimensionality reduction
        pca = PCA(n_components=2)
        reduced_data = pca.fit_transform(class_distribution)
    elif args.dim_reduction == "t-sne":
        # Apply T-SNE for dimensionality reduction
        tsne = TSNE(n_components=2, random_state=0, perplexity=args.perplexity)
        reduced_data = tsne.fit_transform(class_distribution)
    else:
        raise ValueError("Unsupported dimension reduction algorithm!")

    # After dimensionality reduction, print the 2D coordinates of each point
    for i, coord in enumerate(reduced_data):
        print(f"Client {i} Coordinates: {coord}")

    # 使用内置的色彩映射来生成颜色列表
    cmap = plt.get_cmap('tab10')  # 'tab10' 是一个好的起点，因为它提供了10种明显不同的颜色
    colors = [cmap(i) for i in range(args.num_clusters)]

    # Adjust the size of figures
    plt.figure(figsize=(10, 6))  
    for i in range(args.num_clusters):
        cluster_data = reduced_data[client_labels == i]
        # label = f'Cluster {i} (Clients: {cluster_sizes[i]}, Silhouette: {cluster_scores[i]:.2f})'
        label = f'Cluster {i} (Num. of Clients: {cluster_sizes[i]})'
        plt.scatter(cluster_data[:, 0], cluster_data[:, 1], c=colors[i], label=label)

    plt.title('Client clustering results')
    plt.legend()
    plt.savefig(f'./data_distribution_results/cluster_results_rw{args.real_world}_{args.dim_reduction}_niid{args.n_iid}_ncli{args.num_clients}_nclu{args.num_clusters}.png')
    plt.close()
    
    
    # 划分IID, Non-IID 
    # Assign iid and non-iid labels based on the cluster size
    if args.divide_method == 'number':
        eval_dict = divide_via_number(args, client_labels, reduced_data, iid_idx, noniid_idx)
    elif args.divide_method == 'silhouette':
        divide_via_silhouette(args, client_labels, reduced_data, cluster_scores)
    elif args.divide_method == 'both':
        divide_via_number(args, client_labels, reduced_data)
        divide_via_silhouette(args, client_labels, reduced_data, cluster_scores)
    else:
        raise ValueError("Unsupported dividing algorithm!")
    
    print(eval_dict)
    
    
    # 选择k值
    if args.plot_silhouette_curve:
        print("You are plotting Silhouette curve")
        possible_k_values = range(2, 20)  # Assuming you want to try k from 1 to 20
        silhouette_scores_total = [0] * (20 - 2)
        
        # 随机执行多次取平均
        for i in tqdm(range(args.round_for_silhouette)):
            if args.real_world == 0:
                split = split_image_data_realwd(x_train, y_train, n_clients=args.num_clients, verbose=args.verbose)
            elif args.real_world == 1:
                split = split_balanced_non_iid(x_train, y_train, n_clients=args.num_clients, n_iid=args.n_iid)
            else:  
                split = split_image_data(x_train, y_train, n_clients=args.num_clients, classes_per_client=args.classes_pc, shuffle=True, verbose=args.verbose)
        
            class_distribution = get_class_distribution(split)
            silhouette_scores = plot_silhouette_curve(class_distribution, possible_k_values)
            
            silhouette_scores_total = [total + current for total, current in zip(silhouette_scores_total, silhouette_scores)]
        
        silhouette_scores_mean = [total / args.round_for_silhouette for total in silhouette_scores_total]
        
        # Plot Silhouette scores
        plt.figure()
        plt.plot(possible_k_values, silhouette_scores_mean, 'o-')
        plt.title('Silhouette Score vs. Number of Clusters')
        plt.xlabel('Number of clusters (k)')
        plt.ylabel('Silhouette Score')
        plt.grid(True)
        plt.xticks(possible_k_values)
        plt.savefig(f'./data_distribution_results/silhouette_scores_rw{args.real_world}_r{args.round_for_silhouette}_{args.n_iid}_{args.num_clients}.png')
        plt.close()

if __name__ == "__main__":
    args = parse_args()

    main(args)

    # python .\cluster_vis.py --verbose --dim_reduction='pca' --real_world=0 --plot_silhouette_curve --round_for_silhouette=40
    # python .\cluster_vis.py --verbose --dim_reduction='pca' --real_world=0 --divide_method='number' --num_clients=20 --num_clusters=5
