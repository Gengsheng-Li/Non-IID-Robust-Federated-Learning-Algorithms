import os
import wandb
import argparse
import numpy as np

import torch
import torch.optim as optim

from tqdm import tqdm
from ultils import *
from model import VGG
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pandas as pd

torch.backends.cudnn.benchmark=True


def parse_args():
    parser = argparse.ArgumentParser(description="Train model in a FL manner.")
    parser.add_argument('--user', type=str, default='shuang')
    
    parser.add_argument('--agg_mth', type=str, default="clustering",
                        choices=['FA', 'client-client', 'client-server', 'ligengwk', 'ligengwok', 'kmeans', 'clustering'])
    parser.add_argument('--retrain', action="store_true")
    parser.add_argument('--real_world', type=int, default=2)
    
    parser.add_argument('--num_clients', type=int, default=20)
    parser.add_argument('--num_selected', type=int, default=6)
    parser.add_argument('--num_clusters', type=int, default=5) 
    parser.add_argument('--validation_split', type=float, default=0.2)
    parser.add_argument('--classes_pc', type=int, default=2)
    parser.add_argument('--baseline_num', type=int, default=100)

    parser.add_argument('--num_rounds', type=int, default=150)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--lr', type=float, default=1e-1)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--retrain_epochs', type=int, default=5)
    
    parser.add_argument('--verbose', action="store_true")

    return parser.parse_args()

def train(args):

    # Get CIFAR-10 dataset
    x_train, y_train, x_test, y_test = get_cifar10()


    if args.real_world == 0:
        # 全随机
        split = split_image_data_realwd(x_train, y_train, n_clients=args.num_clients, verbose=True)
    elif args.real_world == 1:
        # 自定义iid数量
        split = split_balanced_non_iid(x_train, y_train, n_clients=args.num_clients, n_iid=2)
    else:  
        # 最极端
        split = split_image_data(x_train, y_train, n_clients=args.num_clients, classes_per_client=args.classes_pc, shuffle=True, verbose=True)


    # Store the client category distribution matrix
    class_distribution_matrix = create_class_distribution_matrix(split)
    class_distribution_matrix.to_csv('./result/class_distribution_matrix-ba.csv')


    # For the algorithm of k-means clustering. Assuming clustering based on the distribution of client data

    # Get the data distribution of each client
    class_distribution = get_class_distribution(split)

    # Using kmeans to cluster the clients
    kmeans = KMeans(n_clusters=args.num_clusters, random_state=0).fit(class_distribution)


    # Obtain cluster labels for each client
    client_labels = kmeans.labels_

    for i in range(args.num_clusters):
        print(f"Cluster {i}: {sum(client_labels == i)}")

    # Print clustering labels for each client
    for i, label in enumerate(client_labels):
        print(f"Client {i} distributrd to cluster {label}")

    # Applying PCA to Reduce Category Distribution to 2D
    pca = PCA(n_components=2)
    reduced_data = pca.fit_transform(class_distribution)

    # After PCA dimensionality reduction, print the coordinates of each point
    for i, coord in enumerate(reduced_data):
        print(f"Client {i} Coordinates: {coord}")


    # Select different colors for each cluster
    colors = ['b', 'g', 'orange', 'purple', 'red']

    # Adjust the size of figures
    plt.figure(figsize=(8, 6))  
    for i in range(args.num_clusters):
        cluster_data = reduced_data[client_labels == i]
        plt.scatter(cluster_data[:, 0], cluster_data[:, 1], c=colors[i], label=f'Cluster {i}')

    plt.title('Client clustering results')
    plt.legend()
    plt.savefig('./result/cluster_results-ba.png')


if __name__ == "__main__":
  args = parse_args()
  
  train(args)
