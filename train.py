import os
import wandb
import argparse
import numpy as np

import torch
import torch.optim as optim

from tqdm import tqdm
from ultils import *
from models.vgg import VGG
from datetime import datetime

torch.backends.cudnn.benchmark=True

def parse_args():
    parser = argparse.ArgumentParser(description="Train model in a FL manner.")
    parser.add_argument('--user', type=str, default='shuang')
    
    parser.add_argument('--agg_mth', type=str, default="FA",
                        choices=['FA', 'client-client', 'client-server', 'ligengwk', 'ligengwok', 'kmeans', 'FedProx'])
    parser.add_argument('--retrain', action="store_true")
    parser.add_argument('--real_world', type=int, default=0)
    
    parser.add_argument('--num_clients', type=int, default=20)
    parser.add_argument('--num_selected', type=int, default=6)
    parser.add_argument('--num_clusters', type=int, default=6) 
    parser.add_argument('--validation_split', type=float, default=0.2)
    parser.add_argument('--classes_pc', type=int, default=2)
    parser.add_argument('--baseline_num', type=int, default=100)

    parser.add_argument('--num_rounds', type=int, default=150)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--lr', type=float, default=1e-1)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--retrain_epochs', type=int, default=5)
    parser.add_argument('--proximal_mu', type=float, default=0.01)
    
    parser.add_argument('--verbose', action="store_true")

    return parser.parse_args()

def local_train_FedProx(client_model, global_model, train_loader, criterion, optimizer, epochs, mu, device):
    """
    Local training function for FedProx
    """
    global_params = global_model.state_dict()
    client_model.train()
    
    for epoch in range(epochs):
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = client_model(data)
            loss = criterion(output, target)
            
            # Add proximal term
            prox_term = 0
            for param_key in client_model.state_dict().keys():
                prox_term += ((client_model.state_dict()[param_key] - global_params[param_key].to(device)) ** 2).sum()
            
            loss += (mu / 2) * prox_term
            loss.backward()
            optimizer.step()
    
    return client_model

def train(args, agg_method, data_loaders, test_loader, original_train_data):
    transforms_train, transforms_eval = get_default_data_transforms(verbose=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    args.agg_mth = agg_method
        
    wandb.init(
        project="federated_learningg",
        name=f"{args.user}_am{args.agg_mth}_r{args.retrain}_rw{args.real_world}_nc{args.num_clients}_ns{args.num_selected}_k{args.num_clusters}_val{args.validation_split}_cpc{args.classes_pc}_nb{args.baseline_num}_rounds{args.num_rounds}_lr{args.lr}_bs{args.batch_size}_t{current_time}",
        config=args
    )

    print("\nStart training")
    print(f"The aggregation algorithm chosen is: {args.agg_mth}\n")
    
    # Initialize models and optimizers 
    global_model = VGG('VGG19').to(device)
    client_models = [VGG('VGG19').to(device) for _ in range(args.num_selected)]
    for model in client_models:
        model.load_state_dict(global_model.state_dict())  # Initial synchronizing with global model 

    opt = [optim.SGD(model.parameters(), lr=args.lr) for model in client_models]

    # Define the criterion (loss function)
    criterion = torch.nn.CrossEntropyLoss()

    # Unpack data loaders
    train_loader, val_loader, _, class_distribution, client_labels, validation_data_sizes = data_loaders

    if args.retrain:
        print("NOTE: Retraining is active!")
        loader_fixed = baseline_data(args.baseline_num)  # Load baseline data

    # Train in a FL manner
    losses_train = []
    losses_retrain = []
    losses_test = []
    accs_test = []

    for r in range(args.num_rounds):
        # Randomly select clients
        client_idx = np.random.permutation(args.num_clients)[:args.num_selected]
        print("Selected clients: ", client_idx)
        client_lens = [len(original_train_data[idx][0]) for idx in client_idx]  # Use original train data length
        print("Each original train_loader's length (num of batch): ", client_lens)

        # Train clients
        loss_train = 0
        for i in tqdm(range(args.num_selected)):
            if args.agg_mth in ['FA', 'FedProx']:
                # Use the original training data for FA and FedProx
                original_train_loader = torch.utils.data.DataLoader(
                    CustomImageDataset(original_train_data[client_idx[i]][0], original_train_data[client_idx[i]][1], transforms_train),
                    batch_size=args.batch_size, shuffle=True
                )
                if args.agg_mth == 'FedProx':
                    client_models[i] = local_train_FedProx(client_models[i], global_model, original_train_loader, 
                                                           criterion, opt[i], args.epochs, args.proximal_mu, device)
                else:
                    loss_train += client_update(client_models[i], opt[i], original_train_loader, args.epochs)
            else:
                loss_train += client_update(client_models[i], opt[i], train_loader[client_idx[i]], args.epochs)
        losses_train.append(loss_train)

        # Retrain clients
        loss_retrain = 0
        if args.retrain:
            for i in tqdm(range(args.num_selected)):
                loss_retrain += client_update(client_models[i], opt[i], loader_fixed, args.retrain_epochs)
        losses_retrain.append(loss_retrain)
        
        # Aggregate clients
        if args.agg_mth == 'FA':
            server_aggregate_FA(global_model, client_models, client_lens)
        elif args.agg_mth == 'client-client':
            server_aggregate_mean(global_model, client_models, client_lens)
        elif args.agg_mth == 'client-server':
            server_aggregate_tocenter(global_model, client_models, client_lens)
        elif args.agg_mth == 'ligengwok':
            server_aggregate_ligeng_wok(global_model, client_models, [val_loader[i] for i in client_idx], 
                                        [validation_data_sizes[i] for i in client_idx], client_labels, client_idx, args.num_clusters)
        elif args.agg_mth == 'ligengwk':
            server_aggregate_ligeng_wk(global_model, client_models, [val_loader[i] for i in client_idx], 
                                       [validation_data_sizes[i] for i in client_idx], client_labels, client_idx, args.num_clusters)
        elif args.agg_mth == 'kmeans':
            server_aggregate_kmeans(global_model, client_models, client_labels, client_idx, args.num_clusters, args.num_clients)
        elif args.agg_mth == 'FedProx':
            server_aggregate_FedProx(global_model, client_models, client_lens)
        else:
            raise ValueError(f"Unsupported aggregation method: {args.agg_mth}")

        # Calculate metrics
        loss_test, acc_test = test(global_model, test_loader)
        losses_test.append(loss_test)
        accs_test.append(acc_test)

        # Log metrics to wandb
        wandb.log({"round": r, "loss_train": loss_train, "loss_retrain": loss_retrain, "loss": loss_test, "acc": acc_test})

        print(f'{r}-th round completed!')
        print(f'loss_train {loss_train:.3g} | loss_test {loss_test:.3g} | acc_test: {acc_test:.3f} \n')
        
    print("\nFinish training\n")
    wandb.finish()  # End the current W&B run

if __name__ == "__main__":
    args = parse_args()

    # Ensure wandb login is called once at the beginning
    wandb.login()

    # Create the data split once and save it to a file
    train_loader, val_loader, test_loader, class_distribution, client_labels, validation_data_sizes, original_train_data = get_data_loaders_val(
        args.num_clients, args.batch_size, args.num_clusters, args.classes_pc, args.real_world, args.validation_split, args.verbose
    )

    data_loaders = (train_loader, val_loader, test_loader, class_distribution, client_labels, validation_data_sizes)

    train(args, 'FA', data_loaders, test_loader, original_train_data)

    # Train with FedProx method
    train(args, 'FedProx', data_loaders, test_loader, original_train_data)
    
    # Train with ligengwok method
    train(args, 'ligengwok', data_loaders, test_loader, original_train_data)

    train(args, 'kmeans', data_loaders, test_loader, original_train_data)

    train(args, 'ligengwk', data_loaders, test_loader, original_train_data)
