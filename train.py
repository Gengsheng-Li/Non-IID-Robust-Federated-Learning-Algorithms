import os
import wandb
import argparse
import numpy as np

import torch
import torch.optim as optim

from tqdm import tqdm
from ultils import *
from model import VGG
from datetime import datetime

torch.backends.cudnn.benchmark=True


def parse_args():
    parser = argparse.ArgumentParser(description="Train model in a FL manner.")
    parser.add_argument('--user', type=str, default='shuang')
    
    parser.add_argument('--agg_mth', type=str, default="ligengwk",
                        choices=['FA', 'client-client', 'client-server', 'ligengwk', 'ligengwok'])
    parser.add_argument('--retrain', action="store_true")
    parser.add_argument('--real_world', type=int, default=2)
    
    parser.add_argument('--num_clients', type=int, default=20)
    parser.add_argument('--num_selected', type=int, default=6)
    parser.add_argument('--num_clusters', type=int, default=8) 
    parser.add_argument('--validation_split', type=float, default=0.1)
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        
    wandb.login()
    wandb.init(
        project="federated_learningg",
        name=f"{args.user}_am{args.agg_mth}_r{args.retrain}_rw{args.real_world}_nc{args.num_clients}_ns{args.num_selected}_cpc{args.classes_pc}_nb{args.baseline_num}_lr{args.lr}_bs{args.batch_size}_t{current_time}",
        config=args
    )

    print("\nStart training")
    print(f"The aggregation algorithm chose is: {args.agg_mth}\n")
    
    # Initialize models and optimizers 
    global_model =  VGG('VGG19').to(device)
    client_models = [ VGG('VGG19').to(device) for _ in range(args.num_selected)]
    for model in client_models:
        model.load_state_dict(global_model.state_dict()) ### initial synchronizing with global modle 

    opt = [optim.SGD(model.parameters(), lr=args.lr) for model in client_models]

    # Load data loaders
    if args.agg_mth=='ligengwk' or 'ligengwok':
        train_loader, val_loader, test_loader, class_distribution, client_labels, validation_data_sizes = get_data_loaders_val(args.num_clients, args.batch_size, 
                                                                                                                                args.num_clusters, args.classes_pc, 
                                                                                                                                args.real_world, args.validation_split,
                                                                                                                                args.verbose)
    else:
        train_loader, test_loader = get_data_loaders(classes_pc=args.classes_pc, nclients= args.num_clients, 
                                                batch_size=args.batch_size, real_wd=args.real_world, verbose=args.verbose)
    
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
        print("Slected clinets: ", client_idx)
        client_lens = [len(train_loader[idx]) for idx in client_idx]
        print("Each train_loader's length (num of batch): ", client_lens)

        # Train clients
        loss_train = 0
        for i in tqdm(range(args.num_selected)):
            client_syn(client_models[i], global_model)
            loss_train += client_update(client_models[i], opt[i], train_loader[client_idx[i]], args.epochs)
        # loss_train = loss_train / args.num_selected
        losses_train.append(loss_train)

        # Retrain clients
        loss_retrain = 0
        if args.retrain:
            for i in tqdm(range(args.num_selected)):
                loss_retrain += client_update(client_models[i], opt[i], loader_fixed, args.retrain_epochs)
        # loss_retrain = loss_retrain / args.num_selected
        losses_retrain.append(loss_retrain)
        
        # Aggregate clients
        if args.agg_mth == 'FA':
            server_aggregate_FA(global_model, client_models, client_lens)
        elif args.agg_mth == 'client-client':
            server_aggregate_mean(global_model, client_models, client_lens)
        elif args.agg_mth == 'client-server':
            server_aggregate_tocenter(global_model, client_models, client_lens)
        elif args.agg_mth == 'ligengwok':
            server_aggregate_ligeng_wok(global_model, client_models, [val_loader[i] for i in client_idx], [validation_data_sizes[i] for i in client_idx], client_labels, client_idx, args.num_clusters)
        elif args.agg_mth == 'ligengwk':
            server_aggregate_ligeng_wk(global_model, client_models, [val_loader[i] for i in client_idx], [validation_data_sizes[i] for i in client_idx], client_labels, client_idx, args.num_clusters)
        else:
            raise ValueError(f"Unsupported aggregation method: {args.agg_mth}")

        # Calculate metrics
        loss_test, acc_test = test(global_model, test_loader)
        losses_test.append(loss_test)
        accs_test.append(acc_test)

        # Log metrics to wandb
        # wandb.log({"loss_train": loss_train, "loss_retrain": loss_retrain, "loss_test": loss_test, "acc_test": acc_test})
        wandb.log({"loss_train": loss_train, "loss_retrain": loss_retrain, "loss": loss_test, "acc": acc_test})


        print('%d-th round completed!' % r)
        print('loss_train %0.3g | loss_test %0.3g | acc_test: %0.3f \n' % (loss_train, loss_test, acc_test))
        
    print("\nFinish training\n")


if __name__ == "__main__":
  args = parse_args()
  
  train(args)
