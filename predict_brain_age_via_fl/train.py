import os
import wandb
import argparse
import numpy as np

import torch
import torch.optim as optim

from tqdm import tqdm
from datetime import datetime

from ultils import *
from predict_brain_age_via_fl.mri_data import generate_training_dataloders, generate_testing_dataloder
from models.resnet import generate_resnet

torch.backends.cudnn.benchmark=True


def parse_args():
    parser = argparse.ArgumentParser(description="Train brain age predictor in a FL manner.")
    # Use or forbid Weights & Biases (recommand to use it)
    parser.add_argument('--forbid_wandb', action='store_true', help='Set this flag to forbid Weights & Biases')

    # Basic setting
    parser.add_argument('--name_experiment', type=str, default='multi_site')
    parser.add_argument('--data_dir', type=str, default='../../../data/OpenBHB')    
    parser.add_argument('--user', type=str, default='gsli')
    parser.add_argument('--agg_mth', type=str, default="FA",
                        choices=['FA', 'client-client', 'client-server', 'ligengwk', 'ligengwok'])

    # MRI dataset setting
    parser.add_argument('--num_clients', type=int, default=58)
    parser.add_argument('--validation_split', type=float, default=0.1)

    # Training setting
    parser.add_argument('--num_selected', type=int, default=10)
    parser.add_argument('--num_rounds', type=int, default=150)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--lr', type=float, default=1e-1)
    parser.add_argument('--batch_size', type=int, default=32)

    parser.add_argument('--retrain', action="store_true")
    parser.add_argument('--baseline_num', type=int, default=100)
    parser.add_argument('--retrain_epochs', type=int, default=5)
   
    parser.add_argument('--verbose', action="store_true")

    return parser.parse_args()

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        
    wandb.login()
    wandb.init(
        project="federated_learningg",
        name=f"{args.user}_am{args.agg_mth}_r{args.retrain}_rw{args.real_world}_nc{args.num_clients}_ns{args.num_selected}_cpc{args.classes_pc}_nb{args.baseline_num}_rounds{args.num_rounds}_lr{args.lr}_bs{args.batch_size}_t{current_time}",
        config=args
    )

    print("\nStart training")
    print(f"The aggregation algorithm chose is: {args.agg_mth}\n")
    
    # Initialize models and optimizers TODO
    global_model =  generate_resnet(18).to(device)
    client_models = [ generate_resnet(18).to(device) for _ in range(args.num_selected)]
    for model in client_models:
        model.load_state_dict(global_model.state_dict()) ### initial synchronizing with global modle 

    opt = [torch.optim.Adam(model.parameters(), lr=args.lr) for model in client_models]

    # Load data loaders TODO
    train_loaders = generate_training_dataloders(data_dir=args.data_dir, num_clients=args.num_clients, batch_size=args.batch_size, name_experiment=args.name_experiment)
    test_loader = generate_testing_dataloder(data_dir=args.data_dir, batch_size=args.batch_size, name_experiment=args.name_experiment)
    
    # if args.retrain:
    #     print("NOTE: Retraining is active!")
    #     loader_fixed = baseline_data(args.baseline_num)  # Load baseline data

    # Train in a FL manner
    losses_train = []
    losses_retrain = []
    losses_test = []
    accs_test = []

    for r in range(args.num_rounds):
        # Randomly select clients 
        client_idx = np.random.permutation(args.num_clients)[:args.num_selected]
        print("Slected clinets: ", client_idx)
        client_lens = [len(train_loaders[idx]) for idx in client_idx]
        print("Each train_loader's length (num of batch): ", client_lens)

        # Train clients
        loss_train = 0
        for i in tqdm(range(args.num_selected)):
            client_syn(client_models[i], global_model)
            loss_train += client_update(client_models[i], opt[i], train_loaders[client_idx[i]], args.epochs)
        # loss_train = loss_train / args.num_selected
        losses_train.append(loss_train)

        # # Retrain clients
        loss_retrain = 0
        # if args.retrain:
        #     for i in tqdm(range(args.num_selected)):
        #         loss_retrain += client_update(client_models[i], opt[i], loader_fixed, args.retrain_epochs)
        # # loss_retrain = loss_retrain / args.num_selected
        # losses_retrain.append(loss_retrain)
        
        # Aggregate clients TODO
        if args.agg_mth == 'FA':
            server_aggregate_FA(global_model, client_models, client_lens)
        elif args.agg_mth == 'client-client':
            server_aggregate_mean(global_model, client_models, client_lens)
        elif args.agg_mth == 'client-server':
            server_aggregate_tocenter(global_model, client_models, client_lens)
        # elif args.agg_mth == 'ligengwok':
        #     server_aggregate_ligeng_wok(global_model, client_models, [val_loader[i] for i in client_idx], [validation_data_sizes[i] for i in client_idx])
        # elif args.agg_mth == 'ligengwk':
        #     server_aggregate_ligeng_wk(global_model, client_models, [val_loader[i] for i in client_idx], [validation_data_sizes[i] for i in client_idx], client_labels, client_idx, args.num_clusters)
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
