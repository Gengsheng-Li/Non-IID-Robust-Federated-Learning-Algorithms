import os
import numpy as np
import torch
import torch.nn as nn
import argparse
import wandb


from torch import nn
from tqdm import tqdm

from predict_brain_age_via_fl.data import generate_training_dataloders, generate_testing_dataloder
from models.resnet import generate_resnet

torch.backends.cudnn.benchmark=True

def parse_args():
    parser = argparse.ArgumentParser(description='Train a classifier on top of CLIP embeddings.')
    # Use or forbid Weights & Biases (recommand to use it)
    parser.add_argument('--forbid_wandb', action='store_true', help='Set this flag to forbid Weights & Biases')

    # Arguments for initializing FL 
    parser.add_argument('--name_experiment', type=str, default='multi_site')
    parser.add_argument('--num_clients', type=int, default=58)
    parser.add_argument('--num_selected', type=int, default=10)
    parser.add_argument('--num_rounds', type=int, default=10)
    
    # Arguments for initializing training in each client in each epoch
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-2)
    
    # Arguments for dataset
    parser.add_argument('--data_dir', type=str, default='../../../../Dataset/OpenBHB')
    
    return parser.parse_args()

def client_update(client_model, optimizer, train_loader, loss_func=nn.L1Loss(reduction='sum'), epoch=5, device='cuda'):
    """
    This function updates/trains client model on client data
    """
    
    client_model.train()
    for _ in range(epoch):
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            data, target = data.unsqueeze(1), target.unsqueeze(1)
            optimizer.zero_grad()
            output = client_model(data)
            # loss = F.nll_loss(output, target)
            loss = loss_func(output, target)
            loss.backward()
            optimizer.step()
            
    return loss.item()

def server_aggregate(global_model, client_models):
    """
    This function has aggregation method 'mean'
    """
    
    ### This will take simple mean of the weights of models ###
    global_dict = global_model.state_dict()
    for k in global_dict.keys():
        global_dict[k] = torch.stack([client_models[i].state_dict()[k].float() for i in range(len(client_models))],
                                     0).mean(0)
    global_model.load_state_dict(global_dict)
    for model in client_models:
        model.load_state_dict(global_model.state_dict())

def test(global_model, test_loader, loss_func=nn.L1Loss(reduction='sum')):
    """This function test the global model on test data and returns test loss and test accuracy """
    
    global_model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.cuda(), target.cuda()
            output = global_model(data)
            # test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            test_loss += loss_func(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    acc = correct / len(test_loader.dataset)

    return test_loss, acc

def main(args):
    wandb.login()
    wandb.init(
        project="federated-brain-age-prediction",
        config = {
            "num_clients": args.num_clients,
            "num_selected": args.num_selected,
            "num_rounds": args.num_rounds,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "name_experiment": args.name_experiment,
            },
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    train_loaders = generate_training_dataloders(data_dir=args.data_dir, num_clients=args.num_clients, batch_size=args.batch_size, name_experiment=args.name_experiment)
    test_loader = generate_testing_dataloder(data_dir=args.data_dir, batch_size=args.batch_size, name_experiment=args.name_experiment)

    global_model = generate_resnet(18).to(device)
    client_models = [ generate_resnet(18).to(device) for _ in range(args.num_selected)]
    for model in client_models:
        model.load_state_dict(global_model.state_dict()) # Synchronize with global model

    optimizers = [torch.optim.Adam(model.parameters(), lr=args.lr) for model in client_models]

    # Start federated training
    for r in tqdm(range(args.num_rounds)):
        # Randomly select clients
        client_idx = np.random.permutation(args.num_clients)[:args.num_selected]
        
        # Update clients
        loss = 0
        for i in range(args.num_selected):
            loss += client_update(client_models[i], optimizers[i], train_loaders[client_idx[i]], epoch=args.epochs, device=device)

        # Aggregate clients
        server_aggregate(global_model, client_models)

        # Calculate metrics
        average_train_loss = loss / args.num_selected
        test_loss, test_acc = test(global_model, test_loader)
        
        # Log to Weights & Biases
        wandb.log({"average_train_loss": average_train_loss, "test_loss": test_loss, "test_acc": test_acc})

        print(f'{r}-th round')
        print(f'average training loss {average_train_loss} | test loss {test_loss} | test acc: {test_acc * 100}')
    

if __name__ == '__main__':
    args = parse_args()
    if args.forbid_wandb:
        os.environ['WANDB_DISABLED'] = 'true'
    
    main(args)