import os
import random
from tqdm import tqdm
import numpy as np
import torch, torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from torchvision.transforms import Compose
torch.backends.cudnn.benchmark=True
from sklearn.cluster import KMeans
import pandas as pd
import pickle


#### get cifar dataset in x and y form

def get_cifar10():
    '''Return CIFAR10 train/test data and labels as numpy arrays'''
    data_train = torchvision.datasets.CIFAR10('./data', train=True, download=True)
    data_test = torchvision.datasets.CIFAR10('./data', train=False, download=True) 
    
    x_train, y_train = data_train.data.transpose((0,3,1,2)), np.array(data_train.targets)
    x_test, y_test = data_test.data.transpose((0,3,1,2)), np.array(data_test.targets)
    
    return x_train, y_train, x_test, y_test

def print_image_data_stats(data_train, labels_train, data_test, labels_test):
    print("\nData: ")
    print(" - Train Set: ({},{}), Range: [{:.3f}, {:.3f}], Labels: {},..,{}".format(
      data_train.shape, labels_train.shape, np.min(data_train), np.max(data_train),
        np.min(labels_train), np.max(labels_train)))
    print(" - Test Set: ({},{}), Range: [{:.3f}, {:.3f}], Labels: {},..,{}".format(
      data_test.shape, labels_test.shape, np.min(data_train), np.max(data_train),
        np.min(labels_test), np.max(labels_test)))
  
  
def clients_rand(train_len, nclients):
    '''
    train_len: size of the train data
    nclients: number of clients
    
    Returns: to_ret
    
    This function creates a random distribution 
    for the clients, i.e. number of images each client 
    possess.
    '''
    client_tmp=[]
    sum_=0
    #### creating random values for each client ####
    for i in range(nclients-1):
        tmp=random.randint(10,100)
        sum_+=tmp
        client_tmp.append(tmp)
  
    client_tmp= np.array(client_tmp)
    #### using those random values as weights ####
    clients_dist= ((client_tmp/sum_)*train_len).astype(int)
    num  = train_len - clients_dist.sum()
    to_ret = list(clients_dist)
    to_ret.append(num)
    return to_ret


def split_image_data_realwd(data, labels, n_clients=100, verbose=True):
    '''
    Splits (data, labels) among 'n_clients s.t. every client can holds any number of classes which is trying to simulate real world dataset
    Input:
      data : [n_data x shape]
      labels : [n_data (x 1)] from 0 to n_labels(10)
      n_clients : number of clients
      verbose : True/False => True for printing some info, False otherwise
    Output:
      clients_split : splitted client data into desired format
    '''
  
    n_labels = np.max(labels) + 1
  
    def break_into(n,m):
        ''' 
        return m random integers with sum equal to n 
        '''
        to_ret = [1 for i in range(m)]
        for i in range(n-m):
            ind = random.randint(0,m-1)
            to_ret[ind] += 1
        return to_ret
  
    #### constants ####
    n_classes = len(set(labels))
    classes = list(range(n_classes))
    np.random.shuffle(classes)
    label_indcs  = [list(np.where(labels==class_)[0]) for class_ in classes]
    
    #### classes for each client ####
    tmp = [np.random.randint(1,10) for i in range(n_clients)]
    total_partition = sum(tmp)
  
    #### create partition among classes to fulfill criteria for clients ####
    class_partition = break_into(total_partition, len(classes))
  
    #### applying greedy approach first come and first serve ####
    class_partition = sorted(class_partition,reverse=True)
    class_partition_split = {}
  
    #### based on class partition, partitioning the label indexes ###
    for ind, class_ in enumerate(classes):
        class_partition_split[class_] = [list(i) for i in np.array_split(label_indcs[ind],class_partition[ind])]
        
  #   print([len(class_partition_split[key]) for key in  class_partition_split.keys()])
  
    clients_split = []
    count = 0
    for i in range(n_clients):
      n = tmp[i]
      j = 0
      indcs = []
  
      while n>0:
          class_ = classes[j]
          if len(class_partition_split[class_])>0:
              indcs.extend(class_partition_split[class_][-1])
              count+=len(class_partition_split[class_][-1])
              class_partition_split[class_].pop()
              n-=1
          j+=1
  
      ##### sorting classes based on the number of examples it has #####
      classes = sorted(classes,key=lambda x:len(class_partition_split[x]),reverse=True)
      if n>0:
          raise ValueError(" Unable to fulfill the criteria ")
      clients_split.append([data[indcs], labels[indcs]])
  #   print(class_partition_split)
  #   print("total example ",count)
  
  
    def print_split(clients_split): 
        print("Data split:")
        for i, client in enumerate(clients_split):
            split = np.sum(client[1].reshape(1,-1)==np.arange(n_labels).reshape(-1,1), axis=1)
            print(" - Client {}: {}".format(i,split))
        print()
        
    if verbose:
        print_split(clients_split)
    
    clients_split = np.array(clients_split, dtype=object)
    
    return clients_split



def split_image_data(data, labels, n_clients, classes_per_client, shuffle, verbose):
  '''
  Splits (data, labels) among 'n_clients s.t. every client can holds 'classes_per_client' number of classes
  Input:
    data : [n_data x shape]
    labels : [n_data (x 1)] from 0 to n_labels
    n_clients : number of clients
    classes_per_client : number of classes per client
    shuffle : True/False => True for shuffling the dataset, False otherwise
    verbose : True/False => True for printing some info, False otherwise
  Output:
    clients_split : client data into desired format
  '''

  n_labels = np.max(labels) + 1

  ### client distribution ####
  data_per_client = clients_rand(len(data), n_clients)
  data_per_client_per_class = [np.maximum(1, nd // classes_per_client) for nd in data_per_client]
  
  # sort for labels
  data_idcs = [[] for i in range(n_labels)]
  for j, label in enumerate(labels):
    data_idcs[label] += [j]
  if shuffle:
    for idcs in data_idcs:
      np.random.shuffle(idcs)
    
  # split data among clients
  clients_split = []
  c = 0
  for i in range(n_clients):
    client_idcs = []
        
    budget = data_per_client[i]
    c = np.random.randint(n_labels)
    while budget > 0:
      take = min(data_per_client_per_class[i], len(data_idcs[c]), budget)
      
      client_idcs += data_idcs[c][:take]
      data_idcs[c] = data_idcs[c][take:]
      
      budget -= take
      c = (c + 1) % n_labels
      
    clients_split += [(data[client_idcs], labels[client_idcs])]

  def print_split(clients_split): 
    print("Data split:")
    for i, client in enumerate(clients_split):
      split = np.sum(client[1].reshape(1,-1)==np.arange(n_labels).reshape(-1,1), axis=1)
      print(" - Client {}: {}".format(i,split))
    print()

    # Assuming clients_split is already populated and is in scope
    for idx, (client_data, client_labels) in enumerate(clients_split):
        print(f"Client {idx} data shape: {client_data.shape}")
        print(f"Client {idx} labels shape: {client_labels.shape}")

      
  if verbose:
    print_split(clients_split)
  
  clients_split = np.array(clients_split, dtype=object)
  
  return clients_split

# def split_balanced_non_iid(data, labels, n_clients, n_iid, skewness=0.5, max_classes_per_client=2, verbose=True):
#     data_per_client = [len(data) // n_clients] * n_clients
#     data_per_client[-1] += len(data) - sum(data_per_client)  # 处理不能整除的情况

#      #### constants #### 
#     n_data = data.shape[0]
#     n_labels = np.max(labels) + 1

#     clients_split = []

#     # 分配IID数据
#     data_indices = np.arange(len(data))
#     np.random.shuffle(data_indices)
#     start = 0
#     for i in range(n_iid):
#         end = start + data_per_client[i]
#         client_indices = data_indices[start:end]
#         clients_split.append((data[client_indices], labels[client_indices]))
#         start = end

#     # 分配non-IID数据
#     num_classes = len(np.unique(labels))
#     for i in range(n_iid, n_clients):
#         # 选择这个客户端将拥有的类别
#         if max_classes_per_client is not None and max_classes_per_client < num_classes:
#             chosen_classes = np.random.choice(num_classes, max_classes_per_client, replace=False)
#         else:
#             chosen_classes = range(num_classes)

#         # 每个类别的数据量按照随机比例分配，但确保每个类别至少有一个样本
#         proportions = np.random.dirichlet(np.repeat(skewness, len(chosen_classes)))
#         proportions = proportions / proportions.sum() * data_per_client[i]
#         proportions = np.ceil(proportions).astype(int)  # 向上取整，确保至少有一个样本

#         # 调整比例以确保总数与data_per_client[i]一致
#         while proportions.sum() > data_per_client[i]:
#             proportions[np.argmax(proportions)] -= 1

#         client_indices = []
#         for j, class_idx in enumerate(chosen_classes):
#             class_indices = np.where(labels == class_idx)[0]
#             class_choices = np.random.choice(class_indices, proportions[j], replace=False)
#             client_indices.extend(class_choices)

#         #clients_split.append((data[client_indices], labels[client_indices]))
#         clients_split += [(data[client_indices], labels[client_indices])]

#     def print_split(clients_split): 
#       print("Data split:")
#       for i, client in enumerate(clients_split):
#         split = np.sum(client[1].reshape(1,-1)==np.arange(n_labels).reshape(-1,1), axis=1)
#         print(" - Client {}: {}".format(i,split))
#       print()
#       # Assuming clients_split is already populated and is in scope
#       for idx, (client_data, client_labels) in enumerate(clients_split):
#           print(f"Client {idx} data shape: {client_data.shape}")
#           print(f"Client {idx} labels shape: {client_labels.shape}")

      
#     if verbose:
#       print_split(clients_split)

#     #clients_split = np.array(clients_split, dtype=object)

#     return clients_split






# def shuffle_list(data):
#   '''
#   This function returns the shuffled data
#   '''
#   for i in range(len(data)):
#     tmp_len= len(data[i][0])
#     index = [i for i in range(tmp_len)]
#     random.shuffle(index)
#     data[i][0],data[i][1] = shuffle_list_data(data[i][0],data[i][1])
#   return data

# def shuffle_list_data(x, y):
#   '''
#   This function is a helper function, shuffles an
#   array while maintaining the mapping between x and y
#   '''
#   inds = list(range(len(x)))
#   random.shuffle(inds)
#   return x[inds],y[inds]


def shuffle_list(clients_data):
    '''
    This function shuffles each client's data and labels.
    '''
    for i in range(len(clients_data)):
        clients_data[i] = shuffle_list_data(clients_data[i][0], clients_data[i][1])
    return clients_data

def shuffle_list_data(x, y):
    '''
    This helper function shuffles an array while maintaining the mapping between x and y.
    '''
    inds = list(range(len(x)))
    random.shuffle(inds)
    x_shuffled = x[inds]
    y_shuffled = y[inds]
    return x_shuffled, y_shuffled


class CustomImageDataset(Dataset):
    '''
    A custom Dataset class for images
    inputs : numpy array [n_data x shape]
    labels : numpy array [n_data (x 1)]
    '''
    def __init__(self, inputs, labels, transforms=None):
        assert inputs.shape[0] == labels.shape[0]
        self.inputs = torch.Tensor(inputs)
        self.labels = torch.Tensor(labels).long()
        self.transforms = transforms 
  
    def __getitem__(self, index):
        img, label = self.inputs[index], self.labels[index]
  
        if self.transforms is not None:
          img = self.transforms(img)
  
        return (img, label)
  
    def __len__(self):
        return self.inputs.shape[0]
          

def get_default_data_transforms(train=True, verbose=True):
    transforms_train = {
    'cifar10' : transforms.Compose([
      transforms.ToPILImage(),
      transforms.RandomCrop(32, padding=4),
      transforms.RandomHorizontalFlip(),
      transforms.ToTensor(),
      transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]),#(0.24703223, 0.24348513, 0.26158784)
    }
    transforms_eval = {    
    'cifar10' : transforms.Compose([
      transforms.ToPILImage(),
      transforms.ToTensor(),
      transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    }
    if verbose:
      print("\nData preprocessing: ")
      for transformation in transforms_train['cifar10'].transforms:
        print(' -', transformation)
      print()
  
    return (transforms_train['cifar10'], transforms_eval['cifar10'])

# def get_data_loaders(nclients,batch_size,classes_pc=10 ,verbose=True ):
  
#   x_train, y_train, x_test, y_test = get_cifar10()

#   if verbose:
#     print_image_data_stats(x_train, y_train, x_test, y_test)

#   transforms_train, transforms_eval = get_default_data_transforms(verbose=True)
  
#   split = split_image_data(x_train, y_train, n_clients=nclients, 
#         classes_per_client=classes_pc, verbose=verbose)
  
#   split_tmp = shuffle_list(split)
  
#   client_loaders = [torch.utils.data.DataLoader(CustomImageDataset(x, y, transforms_train), 
#                                                                 batch_size=batch_size, shuffle=True) for x, y in split_tmp]

#   test_loader  = torch.utils.data.DataLoader(CustomImageDataset(x_test, y_test, transforms_eval), batch_size=100, shuffle=False) 

#   return client_loaders, test_loader



def baseline_data(num):
    '''
    Returns baseline data loader to be used on retraining on global server
    Input:
          num : size of baseline data
    Output:
          loader: baseline data loader
    '''
    xtrain, ytrain, _, _ = get_cifar10()
    x, y = shuffle_list_data(xtrain, ytrain)
  
    x, y = x[:num], y[:num]
    transform, _ = get_default_data_transforms(train=True, verbose=True)
    loader = torch.utils.data.DataLoader(CustomImageDataset(x, y, transform), batch_size=16, shuffle=True)
  
    return loader

def client_update(client_model, optimizer, train_loader, epoch=5):
    """
    This function updates/trains client model on client data
    """
    client_model.train()
    for e in range(epoch):
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.cuda(), target.cuda()
            optimizer.zero_grad()
            output = client_model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            #torch.nn.utils.clip_grad_norm_(client_model.parameters(), max_norm=1)
            optimizer.step()
    return loss.item()

def client_syn(client_model, global_model):
  '''
  This function synchronizes the client model with global model
  '''
  client_model.load_state_dict(global_model.state_dict())



def test(global_model, test_loader):
    """
    This function test the global model on test 
    data and returns test loss and test accuracy 
    """
    global_model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.cuda(), target.cuda()
            output = global_model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    acc = correct / len(test_loader.dataset)

    return test_loss, acc

# A portion of the above code code is inspired from the work of others, 
# the link is: https://towardsdatascience.com/preserving-data-privacy-in-deep-learning-part-1-a04894f78029
#############################################################################################################


def split_balanced_non_iid(data, labels, n_clients, n_iid, skewness=0.5, max_classes_per_client=2):
    data_per_client = [len(data) // n_clients] * n_clients
    data_per_client[-1] += len(data) - sum(data_per_client)  # 处理不能整除的情况

    clients_split = []

    # 分配IID数据
    data_indices = np.arange(len(data))
    np.random.shuffle(data_indices)
    start = 0
    for i in range(n_iid):
        end = start + data_per_client[i]
        client_indices = data_indices[start:end]
        clients_split.append((data[client_indices], labels[client_indices]))
        start = end

    # 分配non-IID数据
    num_classes = len(np.unique(labels))
    for i in range(n_iid, n_clients):
        # 选择这个客户端将拥有的类别数量
        if max_classes_per_client is not None:
            num_classes_for_client = np.random.randint(1, max_classes_per_client + 1)
            chosen_classes = np.random.choice(num_classes, num_classes_for_client, replace=False)
        else:
            chosen_classes = range(num_classes)
        
        # 每个类别的数据量按照随机比例分配，但确保每个类别至少有一个样本
        proportions = np.random.dirichlet(np.repeat(skewness, len(chosen_classes)))
        proportions = proportions / proportions.sum() * data_per_client[i]
        proportions = np.ceil(proportions).astype(int)  # 向上取整，确保至少有一个样本

        # 调整比例以确保总数与data_per_client[i]一致
        while proportions.sum() > data_per_client[i]:
            proportions[np.argmax(proportions)] -= 1
        
        client_indices = []
        for j, class_idx in enumerate(chosen_classes):
            class_indices = np.where(labels == class_idx)[0]
            class_choices = np.random.choice(class_indices, proportions[j], replace=False)
            client_indices.extend(class_choices)
        
        clients_split.append((data[client_indices], labels[client_indices]))

    return clients_split


# def split_balanced_non_iid(data, labels, n_clients, n_iid, skewness=0.5, max_classes_per_client=2):
#     total_data = len(data)
#     data_per_client = np.random.randint(1, total_data, size=n_clients)
#     data_per_client = np.floor(data_per_client / data_per_client.sum() * total_data).astype(int)
#     data_per_client[-1] = total_data - data_per_client[:-1].sum()  # Adjust last client to sum correctly

#     clients_split = []
#     data_indices = np.arange(total_data)
#     np.random.shuffle(data_indices)
#     start = 0

#     for i in range(n_clients):
#         end = start + data_per_client[i]
#         if i < n_iid:
#             client_indices = data_indices[start:end]
#             clients_split.append((data[client_indices], labels[client_indices]))
#         else:
#             num_classes = len(np.unique(labels))
#             if max_classes_per_client is not None:
#                 num_classes_for_client = np.random.randint(1, max_classes_per_client + 1)
#                 chosen_classes = np.random.choice(num_classes, num_classes_for_client, replace=False)
#             else:
#                 chosen_classes = range(num_classes)

#             proportions = np.random.dirichlet(np.repeat(skewness, len(chosen_classes)))
#             proportions = proportions / proportions.sum() * data_per_client[i]
#             proportions = np.ceil(proportions).astype(int)

#             client_indices = []
#             for j, class_idx in enumerate(chosen_classes):
#                 class_indices = np.where(labels == class_idx)[0]
#                 if proportions[j] > len(class_indices):  # Ensure we do not request more samples than available
#                     proportions[j] = len(class_indices)
#                 class_choices = np.random.choice(class_indices, proportions[j], replace=False)
#                 client_indices.extend(class_choices)

#             clients_split.append((data[client_indices], labels[client_indices]))

#         start = end

#     return clients_split



def get_data_loaders_FA(nclients, batch_size, classes_pc, real_wd, verbose=True ):
  
    x_train, y_train, x_test, y_test = get_cifar10()

    if verbose:
        print_image_data_stats(x_train, y_train, x_test, y_test)

    transforms_train, transforms_eval = get_default_data_transforms(verbose=False)
  
    if real_wd == 0:
        # 全随机
        split = split_image_data_realwd(x_train, y_train, n_clients=nclients, verbose=verbose)
    elif real_wd == 1:
        # 自定义iid数量
        split = split_balanced_non_iid(x_train, y_train, n_clients=nclients, n_iid=2)
    else:  
        # 最极端
        split = split_image_data(x_train, y_train, n_clients=nclients, classes_per_client=classes_pc, shuffle=True, verbose=verbose)

    # 统计每个客户端的类别分布
    class_distribution = get_class_distribution(split)

    # Store the client category distribution matrix
    class_distribution_matrix = create_class_distribution_matrix(split)
    class_distribution_matrix.to_csv('class_distribution_matrix-FA.csv')
  
    split_tmp = shuffle_list(split)
  
    client_loaders = [torch.utils.data.DataLoader(CustomImageDataset(x, y, transforms_train), batch_size=batch_size, shuffle=True) for x, y in split_tmp]

    test_loader  = torch.utils.data.DataLoader(CustomImageDataset(x_test, y_test, transforms_eval), batch_size=100, shuffle=False) 

    return client_loaders, test_loader



# def get_data_loaders_val(nclients, batch_size, k, classes_pc=10, real_wd=False, validation_split=0.2, verbose=True):
    
    
#     x_train, y_train, x_test, y_test = get_cifar10()

#     if verbose:
#         print_image_data_stats(x_train, y_train, x_test, y_test)

#     transforms_train, transforms_eval = get_default_data_transforms(verbose=False)

#     if real_wd == 0:
        
#         # 全随机
#         split = split_image_data_realwd(x_train, y_train, n_clients=nclients, verbose=verbose)
#     elif real_wd == 1:
#         # 自定义iid数量
#         split = split_balanced_non_iid(x_train, y_train, n_clients=nclients, n_iid=2)
#     else:  
#         # 最极端
#         split = split_image_data(x_train, y_train, n_clients=nclients, classes_per_client=classes_pc, shuffle=True, verbose=verbose)

#         # 统计每个客户端的类别分布
#     class_distribution = get_class_distribution(split)

#     # Store the client category distribution matrix
#     class_distribution_matrix = create_class_distribution_matrix(split)
#     class_distribution_matrix.to_csv('class_distribution_matrix-acc.csv')
    
#     # 应用K-means聚类
#     kmeans = KMeans(n_clusters=k, random_state=0).fit(class_distribution)
#     client_labels = kmeans.labels_

#     split_tmp = shuffle_list(split)

#     # # Store the client category distribution matrix
#     # class_distribution_matrix1 = create_class_distribution_matrix(split_tmp)
#     # class_distribution_matrix1.to_csv('class_distribution_matrix-ba1.csv')

#         # 在每个客户端上分割出验证集
#     client_loaders = []
#     val_loaders = []
#     validation_data_sizes = []  # List to store the size of each validation dataset
#     for client_data in split_tmp:
#         x_client, y_client = client_data
#         if len(x_client) == 0:
#             continue
#         val_size = int(len(x_client) * validation_split)
#         val_size = max(1, min(val_size, len(x_client) - 1))
#         # 验证集
#         x_val_client = x_client[-val_size:]
#         y_val_client = y_client[-val_size:]
#         # 训练集
#         x_train_client = x_client[:-val_size]
#         y_train_client = y_client[:-val_size]

#         # Store the size of each validation dataset
#         validation_data_sizes.append(len(x_val_client))


#             # 检查分割后的训练集是否为空
#         if len(x_train_client) > 0 and len(x_val_client) > 0:
#             # 创建训练和验证 DataLoader
#             train_loader = torch.utils.data.DataLoader(
#                 CustomImageDataset(x_train_client, y_train_client, transforms_train),
#                 batch_size=batch_size, shuffle=True
#             )
#             val_loader = torch.utils.data.DataLoader(
#                 CustomImageDataset(x_val_client, y_val_client, transforms_eval),
#                 batch_size=batch_size, shuffle=False
#             )

#             client_loaders.append(train_loader)
#             val_loaders.append(val_loader)
#         else:
#             print(f"Skipping client with insufficient data. Training samples: {len(x_train_client)}, Validation samples: {len(x_val_client)}")
#     # client_loaders = [torch.utils.data.DataLoader(CustomImageDataset(x, y, transforms_train),
#     #                                               batch_size=batch_size, shuffle=True) for x, y in split_tmp]

#     # val_loader = torch.utils.data.DataLoader(CustomImageDataset(x_val, y_val, transforms_eval),
#     #                                          batch_size=batch_size, shuffle=False)

#     test_loader = torch.utils.data.DataLoader(CustomImageDataset(x_test, y_test, transforms_eval), batch_size=100,
#                                               shuffle=False)

#    # 返回客户端数据加载器、测试数据加载器、类别分布和客户端标签
#     return client_loaders, val_loaders, test_loader, class_distribution, client_labels, validation_data_sizes

#     #return client_loaders, test_loader



# def get_data_loaders_val(nclients, batch_size, k, classes_pc=10, real_wd=False, validation_split=0.2, verbose=True):
#     x_train, y_train, x_test, y_test = get_cifar10()

#     if verbose:
#         print_image_data_stats(x_train, y_train, x_test, y_test)

#     transforms_train, transforms_eval = get_default_data_transforms(verbose=False)

#     if real_wd == 0:
#         # 全随机
#         split = split_image_data_realwd(x_train, y_train, n_clients=nclients, verbose=verbose)
#     elif real_wd == 1:
#         # 自定义iid数量
#         split = split_balanced_non_iid(x_train, y_train, n_clients=nclients, n_iid=2)
#     else:
#         # 最极端
#         split = split_image_data(x_train, y_train, n_clients=nclients, classes_per_client=classes_pc, shuffle=True, verbose=verbose)

#     # 统计每个客户端的类别分布
#     class_distribution = get_class_distribution(split)

#     # Store the client category distribution matrix
#     class_distribution_matrix = create_class_distribution_matrix(split)
#     class_distribution_matrix.to_csv('class_distribution_matrix.csv')

#     # 应用K-means聚类
#     kmeans = KMeans(n_clusters=k, random_state=0).fit(class_distribution)
#     client_labels = kmeans.labels_

#     split_tmp = shuffle_list(split)

#     client_loaders = []
#     val_loaders = []
#     validation_data_sizes = []
#     for client_data in split_tmp:
#         x_client, y_client = client_data
#         if len(x_client) == 0:
#             continue
#         val_size = int(len(x_client) * validation_split)
#         val_size = max(1, min(val_size, len(x_client) - 1))
#         x_val_client = x_client[-val_size:]
#         y_val_client = y_client[-val_size:]
#         x_train_client = x_client[:-val_size]
#         y_train_client = y_client[:-val_size]

#         validation_data_sizes.append(len(x_val_client))

#         if len(x_train_client) > 0 and len(x_val_client) > 0:
#             train_loader = torch.utils.data.DataLoader(
#                 CustomImageDataset(x_train_client, y_train_client, transforms_train),
#                 batch_size=batch_size, shuffle=True
#             )
#             val_loader = torch.utils.data.DataLoader(
#                 CustomImageDataset(x_val_client, y_val_client, transforms_eval),
#                 batch_size=batch_size, shuffle=False
#             )
#             client_loaders.append(train_loader)
#             val_loaders.append(val_loader)
#         else:
#             print(f"Skipping client with insufficient data. Training samples: {len(x_train_client)}, Validation samples: {len(x_val_client)}")

#     test_loader = torch.utils.data.DataLoader(CustomImageDataset(x_test, y_test, transforms_eval), batch_size=100, shuffle=False)

#     return client_loaders, val_loaders, test_loader, class_distribution, client_labels, validation_data_sizes



def get_data_loaders_val(nclients, batch_size, k, classes_pc=10, real_wd=False, validation_split=0.2, verbose=True, save_path="data_split"):
    save_file = os.path.join(save_path, f'data_split_{nclients}_clients_real_world_k={k}_1.pkl')
    
    # Check if the data has already been split and saved
    if os.path.exists(save_file):
        print(f"Loading previously saved data split from {save_file}")
        client_loaders, val_loaders, test_loader, class_distribution, client_labels, validation_data_sizes, original_train_data = load_data(save_file)
    else:
        x_train, y_train, x_test, y_test = get_cifar10()

        if verbose:
            print_image_data_stats(x_train, y_train, x_test, y_test)

        transforms_train, transforms_eval = get_default_data_transforms(verbose=False)

        if real_wd == 0:
            # 全随机
            split = split_image_data_realwd(x_train, y_train, n_clients=nclients, verbose=verbose)
        elif real_wd == 1:
            # 自定义iid数量
            split = split_balanced_non_iid(x_train, y_train, n_clients=nclients, n_iid=2)
        else:
            # 最极端
            split = split_image_data(x_train, y_train, n_clients=nclients, classes_per_client=classes_pc, shuffle=True, verbose=verbose)

        # 统计每个客户端的类别分布
        class_distribution = get_class_distribution(split)

        # 应用K-means聚类
        kmeans = KMeans(n_clusters=k, random_state=0).fit(class_distribution)
        client_labels = kmeans.labels_

        split_tmp = shuffle_list(split)

        client_loaders = []
        val_loaders = []
        validation_data_sizes = []
        original_train_data = []

        for x_client, y_client in split_tmp:
            if len(x_client) == 0:
                continue
            val_size = int(len(x_client) * validation_split)
            val_size = max(1, min(val_size, len(x_client) - 1))
            x_val_client = x_client[-val_size:]
            y_val_client = y_client[-val_size:]
            x_train_client = x_client[:-val_size]
            y_train_client = y_client[:-val_size]

            validation_data_sizes.append(len(x_val_client))
            original_train_data.append((x_client, y_client))

            if len(x_train_client) > 0 and len(x_val_client) > 0:
                train_loader = torch.utils.data.DataLoader(
                    CustomImageDataset(x_train_client, y_train_client, transforms_train),
                    batch_size=batch_size, shuffle=True
                )
                val_loader = torch.utils.data.DataLoader(
                    CustomImageDataset(x_val_client, y_val_client, transforms_eval),
                    batch_size=batch_size, shuffle=False
                )
                client_loaders.append(train_loader)
                val_loaders.append(val_loader)
            else:
                print(f"Skipping client with insufficient data. Training samples: {len(x_train_client)}, Validation samples: {len(x_val_client)}")

        test_loader = torch.utils.data.DataLoader(CustomImageDataset(x_test, y_test, transforms_eval), batch_size=100, shuffle=False)

        # Save the split data including train/val/test loaders
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        save_data((client_loaders, val_loaders, test_loader, class_distribution, client_labels, validation_data_sizes, original_train_data), save_file)
        print(f"Data split saved to {save_file}")

    return client_loaders, val_loaders, test_loader, class_distribution, client_labels, validation_data_sizes, original_train_data




def server_aggregate_weightedsize(global_model, client_models, client_lens):
    """
    This function has aggregation method 'wmean'
    wmean takes the weighted mean of the weights of models
    """
    total = sum(client_lens)
    n = len(client_models)
    global_dict = global_model.state_dict()
    for k in global_dict.keys():

        global_dict[k] = torch.stack([client_models[i].state_dict()[k].float()*(n*client_lens[i]/total) for i in range(len(client_models))], 0).mean(0)
        

    global_model.load_state_dict(global_dict)
    for model in client_models:
        model.load_state_dict(global_model.state_dict())



def server_aggregate_FA(global_model, client_models, client_lens):
    """
    This function has aggregation method 'mean'
    """
    ### This will take simple mean of the weights of models ###
    global_dict = global_model.state_dict()
    for k in global_dict.keys():
        global_dict[k] = torch.stack([client_models[i].state_dict()[k].float() for i in range(len(client_models))], 0).mean(0)
    global_model.load_state_dict(global_dict)
    
    for model in client_models:
        model.load_state_dict(global_model.state_dict())


def server_aggregate_FedProx(global_model, client_models, client_lens):
    """
    Aggregation method for FedProx
    """
    global_dict = global_model.state_dict()
    total_data_points = sum(client_lens)
    
    for k in global_dict.keys():
        global_dict[k] = sum([client_models[i].state_dict()[k].float() * client_lens[i] for i in range(len(client_models))]) / total_data_points
    
    global_model.load_state_dict(global_dict)
    
    for model in client_models:
        model.load_state_dict(global_model.state_dict())


# def server_aggregate_FedProxCustom(global_model, client_models, client_lens, proximal_mu):
#     """
#     This function implements FedProx without learning rate decay.
#     """
#     global_dict = global_model.state_dict()

#     for k in global_dict.keys():
#         global_dict[k] = torch.stack([client_models[i].state_dict()[k].float() for i in range(len(client_models))], 0).mean(0)
#     global_model.load_state_dict(global_dict)
    
#     for model in client_models:
#         model.load_state_dict(global_model.state_dict())

#     # Apply proximal term if needed
#     for model in client_models:
#         for param_name, param in model.named_parameters():
#             param.data.add_(-proximal_mu, param.data - global_model.state_dict()[param_name].data)



    
def server_aggregate_mean(global_model, client_models, client_lens):
    n = len(client_models)
    global_dict = global_model.state_dict()
    mean_model_dict = {k: torch.stack([client_models[i].state_dict()[k].float() for i in range(n)], 0).mean(0) for k in global_dict.keys()}
    
    # Calculate deviations and determine weights
    deviations = []
    for model in client_models:
        deviation = sum((model.state_dict()[k] - mean_model_dict[k]).norm(2) for k in global_dict.keys())
        deviations.append(deviation)
    weights = [1/d if d != 0 else 0 for d in deviations]  # Prevent division by zero
    weight_sum = sum(weights)
    normalized_weights = [w / weight_sum for w in weights]

    # Weighted aggregation of models
    for k in global_dict.keys():
        weighted_params = torch.stack([client_models[i].state_dict()[k].float() * normalized_weights[i] for i in range(n)], 0).sum(0)
        global_dict[k] = weighted_params

    global_model.load_state_dict(global_dict)

    # Optionally update client models
    for model in client_models:
        model.load_state_dict(global_model.state_dict())



def server_aggregate_tocenter(global_model, client_models, client_lens):
    n = len(client_models)
    global_dict = global_model.state_dict()

    # Calculate deviations from the global model
    deviations = []
    for model in client_models:
        deviation = sum((model.state_dict()[k].float() - global_dict[k].float()).norm(2) for k in global_dict.keys())
        deviations.append(deviation)
    weights = [1/d if d != 0 else 0 for d in deviations]  # Prevent division by zero
    weight_sum = sum(weights)
    normalized_weights = [w / weight_sum for w in weights]

    # Weighted aggregation of models
    for k in global_dict.keys():
        weighted_params = torch.stack([client_models[i].state_dict()[k].float() * normalized_weights[i] for i in range(n)], 0).sum(0)
        global_dict[k] = weighted_params

    global_model.load_state_dict(global_dict)

    # Optionally update client models
    for model in client_models:
        model.load_state_dict(global_model.state_dict())



def evaluate_model_on_multiple_loaders(model, data_loaders, data_sizes):
    model.eval()
    total_correct = 0
    total_samples = 0
    weighted_accuracy = 0

    with torch.no_grad():
        for data_loader, data_size in zip(data_loaders, data_sizes):
            correct = 0
            total = 0
            for batch in data_loader:
                data, labels = batch  # 正确地解包数据和标签
                data, labels = data.cuda(), labels.cuda()
                outputs = model(data)
                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
            # 如果数据集为空，避免除以零
            if total > 0:
                weighted_accuracy += (correct / total) * data_size
            total_samples += data_size

    if total_samples > 0:
        weighted_accuracy /= total_samples
    else:
        weighted_accuracy = 0

    return weighted_accuracy



# Proposed by ligeng, without k-means
def server_aggregate_ligeng_wok(global_model, client_models, client_data_loaders, validation_data_sizes, client_labels, client_indices, k):
    # 确保这里使用的是包含多个 DataLoader 的列表，而不是单个 DataLoader

    client_accuracies = [evaluate_model_on_multiple_loaders(client_models[i], [client_data_loaders[j] for j in range(len(client_models))], [validation_data_sizes[j] for j in range(len(client_models))]) for i in range(len(client_models))]

    # Avoid division by zero
    total_accuracy = sum(client_accuracies)
    if total_accuracy == 0:
        print("Warning: Total accuracy is zero, adjusting weights to avoid division by zero.")
        total_accuracy = 1  # Set to 1 or a small number to avoid division by zero

    # Calculate performance weights based on accuracies
    performance_weights = [accuracy / total_accuracy for accuracy in client_accuracies]

    # Normalize weights
    total_weight = sum(performance_weights)
    normalized_weights = [w / total_weight for w in performance_weights]
    print("Performance Weights", performance_weights)
    print("Normalized Weights:", normalized_weights)  # Print normalized weights for debugging

    # Aggregate the models using normalized performance weights
    global_dict = global_model.state_dict()
    for key in global_dict.keys():
        layer_weights = torch.stack([
            client_models[i].state_dict()[key].float() * normalized_weights[i]
            for i in range(len(client_models))
        ])
        global_dict[key] = torch.sum(layer_weights, dim=0)
    
    global_model.load_state_dict(global_dict)

    # Optionally, update all client models to the new global model
    for model in client_models:
        model.load_state_dict(global_model.state_dict())


# Proposed by ligeng, with k-means
def server_aggregate_ligeng_wk(global_model, client_models, client_data_loaders, validation_data_sizes, client_labels, client_indices, k):
    # 确保这里使用的是包含多个 DataLoader 的列表，而不是单个 DataLoader

    client_accuracies = [evaluate_model_on_multiple_loaders(client_models[i], [client_data_loaders[j] for j in range(len(client_models))], [validation_data_sizes[j] for j in range(len(client_models))]) for i in range(len(client_models))]
    
    # Avoid division by zero
    total_accuracy = sum(client_accuracies)
    if total_accuracy == 0:
        print("Warning: Total accuracy is zero, adjusting weights to avoid division by zero.")
        total_accuracy = 1  # Set to 1 or a small number to avoid division by zero
    
    # Calculate cluster weights based on the number of clients in each cluster
    client_counts = [sum(client_labels == i) for i in range(k)]
    cluster_weights = [count / sum(client_counts) for count in client_counts]

    # Adjust cluster weights by incorporating performance metrics
    performance_weights = [accuracy / total_accuracy for accuracy in client_accuracies]
    #combined_weights = [cluster_weights[client_labels[i]] * performance_weights[i] for i in range(len(client_models))]
    combined_weights = [cluster_weights[client_labels[client_indices[i]]] * performance_weights[i] for i in range(len(client_models))]
    
    # Normalize weights
    total_weight = sum(combined_weights)
    normalized_weights = [w / total_weight for w in combined_weights]
    print("Performance Weights", performance_weights)
    print("Combined Weights:", combined_weights)  # 打印选中的客户端权重
    print("Normalized Weights:", normalized_weights)  # 打印标准化后的权重

    # Aggregate the models
    global_dict = global_model.state_dict()
    for key in global_dict.keys():
        layer_weights = torch.stack([
            client_models[i].state_dict()[key].float() * normalized_weights[i]
            for i in range(len(client_models))
        ])
        global_dict[key] = torch.sum(layer_weights, dim=0)
    
    global_model.load_state_dict(global_dict)

    # Optionally, update all client models to the new global model
    for model in client_models:
        model.load_state_dict(global_model.state_dict())


def server_aggregate_kmeans(global_model, client_models, client_labels, client_indices, k, num_clients):
    client_indices = list(client_indices) if isinstance(client_indices, np.ndarray) else client_indices
    
    
    client_counts = [sum(client_labels == i) for i in range(k)]
    cluster_weights = {i: count / sum(client_counts) for i, count in enumerate(client_counts)}
    
    all_client_weights = [cluster_weights[client_labels[i]] for i in range(num_clients)]
    selected_client_weights = [all_client_weights[i] for i in client_indices]
    
    total_weight = sum(selected_client_weights)
    normalized_client_weights = [w / total_weight for w in selected_client_weights]
    
    print("Selected Client Weights:", selected_client_weights)  # 打印选中的客户端权重
    print("Normalized Weights:", normalized_client_weights)  # 打印标准化后的权重
    
    
    global_dict = global_model.state_dict()
    for key in global_dict.keys():
       # print(f"Key: {key}")  # 打印处理的键
        try:
            layer_weights = torch.stack([
                client_models[i].state_dict()[key].float() * normalized_client_weights[i]
                #for j, i in enumerate(client_indices)
                for i in range(len(client_models))
            ])
        except IndexError as e:
            print(f"Error accessing index: {e}")
            break  # 中断循环以避免进一步的错误
        
        global_dict[key] = torch.sum(layer_weights, dim=0)
    
    global_model.load_state_dict(global_dict)
    
    for model in client_models:
        model.load_state_dict(global_model.state_dict())



def create_class_distribution_matrix(clients_data):
    # 获取客户端数量和类别数量
    num_clients = len(clients_data)
    num_classes = 10  # 对于CIFAR-10数据集
    
    # 初始化一个零矩阵，行数为客户端数，列数为类别数
    class_distribution_matrix = np.zeros((num_clients, num_classes), dtype=int)

    # 填充矩阵
    for i, client in enumerate(clients_data):
        _, labels = client
        class_counts = np.bincount(labels, minlength=num_classes)
        class_distribution_matrix[i, :] = class_counts

    # 转换成DataFrame
    class_distribution_df = pd.DataFrame(class_distribution_matrix,
                                         index=[f'客户端 {i}' for i in range(num_clients)],
                                         columns=[f'类别 {i}' for i in range(num_classes)])
    return class_distribution_df



# def evaluate_model(model, data_loader):
#     model.eval()
#     correct = 0
#     total = 0
#     with torch.no_grad():
#         for data, labels in data_loader:
#             data, labels = data.cuda(), labels.cuda()
#             outputs = model(data)
#             _, predicted = torch.max(outputs.data, 1)
#             total += labels.size(0)
#             correct += (predicted == labels).sum().item()
#     accuracy = correct / total
#     return accuracy

# def server_aggregate(global_model, client_models, client_data_loaders, client_labels, client_indices, k):
#     #     # Evaluate each selected client model
#     # client_accuracies = [evaluate_model(client_models[i], client_data_loaders[i]) for i in range(len(client_models))]
#     # # for client_model in client_models:
#     # #     client_accuracies = evaluate_model(client_model, client_data_loaders)
#     # # Avoid division by zero
#     # total_accuracy = sum(client_accuracies)
#     # if total_accuracy == 0:
#     #     print("Warning: Total accuracy is zero, adjusting weights to avoid division by zero.")
#     #     total_accuracy = 1  # Set to 1 or a small number to avoid division by zero

#     # # Calculate performance weights based on accuracies
#     # performance_weights = [accuracy / total_accuracy for accuracy in client_accuracies]

#     # # Normalize weights
#     # total_weight = sum(performance_weights)
#     # normalized_weights = [w / total_weight for w in performance_weights]
#     # print("Performance Weights", performance_weights)
#     # print("Normalized Weights:", normalized_weights)  # Print normalized weights for debugging

#     # # Aggregate the models using normalized performance weights
#     # global_dict = global_model.state_dict()
#     # for key in global_dict.keys():
#     #     layer_weights = torch.stack([
#     #         client_models[i].state_dict()[key].float() * normalized_weights[i]
#     #         for i in range(len(client_models))
#     #     ])
#     #     global_dict[key] = torch.sum(layer_weights, dim=0)
    
#     # global_model.load_state_dict(global_dict)

#     # # Optionally, update all client models to the new global model
#     # for model in client_models:
#     #     model.load_state_dict(global_model.state_dict())
        
#     # Evaluate each client model
#     client_accuracies = [evaluate_model(client_models[i], client_data_loaders[i]) for i in range(len(client_models))]
    
#     # Avoid division by zero
#     total_accuracy = sum(client_accuracies)
#     if total_accuracy == 0:
#         print("Warning: Total accuracy is zero, adjusting weights to avoid division by zero.")
#         total_accuracy = 1  # Set to 1 or a small number to avoid division by zero
    
#     # Calculate cluster weights based on the number of clients in each cluster
#     client_counts = [sum(client_labels == i) for i in range(k)]
#     cluster_weights = [count / sum(client_counts) for count in client_counts]

#     # Adjust cluster weights by incorporating performance metrics
#     performance_weights = [accuracy / total_accuracy for accuracy in client_accuracies]
#     combined_weights = [cluster_weights[client_labels[i]] * performance_weights[i] for i in range(len(client_models))]
#     combined_weights = [cluster_weights[client_labels[client_indices[i]]] * performance_weights[i] for i in range(len(client_models))]

#     # Normalize weights
#     total_weight = sum(combined_weights)
#     normalized_weights = [w / total_weight for w in combined_weights]
#     print("Performance Weights", performance_weights)
#     print("Combined Weights:", combined_weights)  # 打印选中的客户端权重
#     print("Normalized Weights:", normalized_weights)  # 打印标准化后的权重

#     # Aggregate the models
#     global_dict = global_model.state_dict()
#     for key in global_dict.keys():
#         layer_weights = torch.stack([
#             client_models[i].state_dict()[key].float() * normalized_weights[i]
#             for i in range(len(client_models))
#         ])
#         global_dict[key] = torch.sum(layer_weights, dim=0)
    
#     global_model.load_state_dict(global_dict)

#     # Optionally, update all client models to the new global model
#     for model in client_models:
#         model.load_state_dict(global_model.state_dict())


# 统计每个客户端的类别分布
def get_class_distribution(clients_split):
    class_distribution = []
    for client_data in clients_split:
        _, labels = client_data
        counts = np.bincount(labels, minlength=10)  # 假设有10个类别
        counts = counts / counts.sum()  # 归一化
        class_distribution.append(counts)
    return np.array(class_distribution)


def print_model_parameters(model, layer_name):
    """打印指定层的参数"""
    try:
        param = next(model.state_dict()[layer_name].flatten().numpy())
        print(f"Parameters of {layer_name}: {param}")
    except KeyError:
        print(f"Layer {layer_name} not found in model.")


def save_data(split_data, filepath):
    with open(filepath, 'wb') as f:
        pickle.dump(split_data, f)


def load_data(filepath):
    with open(filepath, 'rb') as f:
        return pickle.load(f)
