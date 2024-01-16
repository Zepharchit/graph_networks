import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from load_graph_dataset import MoleculeGraph
from code_graph_nn import GraphN
import pandas as pd
import numpy as np
from tqdm import tqdm
from torch.optim import SGD

batch_size = 64
epochs = 1
device =  torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_data = MoleculeGraph(root_dir='D:\\Books\\python_practice\\models\\Graphs\\data\\',filename="HIV.csv")
train_loader = DataLoader(train_data,batch_size=64,shuffle=True)
# feature_size = node_feature_size (39)
model = GraphN(feature_size=train_data[1].x.shape[1])
opt = SGD(params=model.parameters(),lr=0.01,weight_decay=1e-7,momentum=0.9)
ls_fn = nn.CrossEntropyLoss()
#print(train_data[1].edge_attrs)
def train():
    running_loss = 0.0
    for _,batch in enumerate(tqdm(train_loader)):
        batch.to(device)
        opt.zero_grad()
        pred = model(batch.x.float(),batch.edge_attr.float(),batch.edge_index,batch.batch)
        loss = ls_fn(torch.squeeze(pred),batch.y.long())
        loss.backward()
        opt.step()
        running_loss += loss.item()
    return running_loss / batch_size

for epoch in range(epochs):
    train_loss = train()
    print(f"epoch number : {epoch}  || training loss : {train_loss}")