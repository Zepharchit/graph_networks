import torch 
import torch.nn as nn
from torch.nn import Linear,LeakyReLU,Dropout
from torch_geometric.nn import TopKPooling,GATConv,global_max_pool as gmp, global_mean_pool as gap
from torch_geometric.nn import GCNConv

class GraphN(nn.Module):
    def __init__(self,feature_size):
        super(GraphN, self).__init__()
        self.embedding_size = 256
        self.linear_1 = Linear(self.embedding_size *2 ,1024)
        self.linear_2 = Linear(1024,2)
        self.inter_linear = Linear(self.embedding_size *3,self.embedding_size)
        self.gcn_1 = GATConv(feature_size,out_channels=self.embedding_size,heads=3,concat=True,dropout=0.2)
        self.gcn_2 = GATConv(self.embedding_size,self.embedding_size,heads=3,concat=True,dropout=0.2)
        self.gcn_3 = GATConv(self.embedding_size,self.embedding_size,heads=3,concat=True,dropout=0.2)
        self.pool_1 = TopKPooling(self.embedding_size*3,ratio=0.8)
        self.pool_2 = TopKPooling(self.embedding_size*3,ratio=0.5)
        self.pool_3 = TopKPooling(self.embedding_size*3,ratio=0.3)
        self.lrelu = nn.LeakyReLU()
        self.dout = nn.Dropout(p=0.3)

    def forward(self,x,edge_attr,edge_index,batch_idx):
        #first block
        x = self.gcn_1(x,edge_index)
        x,edge_index,edge_attr,batch_idx,_,_= self.pool_1(x,edge_index,None,batch_idx)
        x = self.inter_linear(x)
        x1 = torch.cat([gmp(x,batch_idx),gap(x,batch_idx)],dim=1)
        #second block
        x = self.gcn_2(x,edge_index)
        x,edge_index,edge_attr,batch_idx,_,_= self.pool_1(x,edge_index,None,batch_idx)
        x = self.inter_linear(x)
        x2 = torch.cat([gmp(x,batch_idx),gap(x,batch_idx)],dim=1)
        #third block
        x = self.gcn_3(x,edge_index)
        x,edge_index,edge_attr,batch_idx,_,_= self.pool_1(x,edge_index,None,batch_idx)
        x = self.inter_linear(x)
        x3 = torch.cat([gmp(x,batch_idx),gap(x,batch_idx)],dim=1)
        x  =  x1 + x2 + x3
        x = self.lrelu(self.linear_1(x))
        x = self.dout(self.linear_2(x))

        return x

 

#obj = GraphN(2)
#print(obj)
