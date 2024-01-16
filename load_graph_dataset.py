import pandas as pd
import os
import torch
import torch.nn as nn
import rdkit
from rdkit import Chem
from rdkit.Chem import Draw
from torch_geometric.data import Data,Dataset,download_url
from tqdm import tqdm
from rdkit.Chem import rdmolops
import numpy as np

class MoleculeGraph(Dataset):
    def __init__(self,root_dir,filename,transform=None,pre_transform=None):
        self.root_dir = root_dir
        self.filename = filename 
        super(MoleculeGraph, self).__init__(root_dir,transform,pre_transform)

    @property
    def raw_file_names(self):
        return self.filename
    
    @property
    def processed_file_names(self):
        print(self.raw_paths)
        self.data = pd.read_csv(self.raw_paths[0]).reset_index()
        return [f"data_{i}.pt" for i in list(self.data.index)]
    
    def download(self):
        pass
    
    def process(self):
        self.data = pd.read_csv(self.raw_paths[0])
        for idx, mole in tqdm(self.data.iterrows(),total=self.data.shape[0]):
            mol_obj = Chem.MolFromSmiles(mole['smiles'])
            #nodes features
            node_feature = self._get_node_features(mol_obj)
            #edge features
            edge_attr = self._get_edge_features(mol_obj)
            #Adjacency info
            edge_info = self._get_adj_info(mol_obj)
            #Label info
            label = self._get_labels(mole["HIV_active"])
            
            #creating a data object
            data = Data(x = node_feature,edge_attr=edge_attr,edge_index=edge_info,y=label,)
            torch.save(data,os.path.join(self.processed_dir,f"data_{idx}.pt"))

    def _get_node_features(self,mol):
        " Returns a matrix of the shape (nodes,node feature size)"

        all_features = []
        for atom in mol.GetAtoms():
            node_feat = []
            #atomic_numbers
            node_feat.append(atom.GetAtomicNum())
            #Atom degree
            node_feat.append(atom.GetDegree())
            #Formal Charge
            node_feat.append(atom.GetFormalCharge())
            #Hybridization
            node_feat.append(atom.GetHybridization())
            #Aromaticity
            node_feat.append(atom.GetIsAromatic())
            #Radical Electrons
            node_feat.append(atom.GetNumRadicalElectrons())
            #In ring
            node_feat.append(atom.IsInRing())
            #chirality
            node_feat.append(atom.GetChiralTag())
            #total number of H
            node_feat.append(atom.GetTotalNumHs())
            all_features.append(node_feat)
        all_features = np.asarray(all_features)
        all_features = torch.tensor(all_features, dtype=torch.float)
        return all_features
    
    def _get_edge_features(self,mol):
        "Will iterate over bonds"
        "returns a matrix (num_edges,edge_features_size)" 
        all_edge_features = []

        for bond in mol.GetBonds():
            edge = []
            edge.append(bond.GetBondTypeAsDouble())
            edge.append(bond.IsInRing())
            all_edge_features.append(edge)
        
        all_edge_features = np.array(all_edge_features)
        return torch.tensor(all_edge_features,dtype=torch.float)
    
    def _get_adj_info(self,mol):
        adj = rdmolops.GetAdjacencyMatrix(mol)
        r,c = np.where(adj)
        # converting the adjacency matrix to (i,j,val) format[COO]
        ijv =np.array(list(zip(r,c)))
        ijv = np.reshape(ijv,(2,-1))
        return torch.tensor(ijv,dtype=torch.long)
    
    def _get_labels(self,label):
        label = np.array([label])
        return torch.tensor(label,dtype=torch.int64)

    def len(self):
        return self.data.shape[0]

    def get(self,idx):
        data = torch.load(os.path.join(self.processed_dir,f"data_{idx}.pt"))
        return data
    

#dataset = MoleculeGraph(root_dir='D:\\Books\\python_practice\\models\\Graphs\\data\\',filename="HIV.csv")
#print(dataset[0].x)
#print(dataset[0].y)
#print(dataset[0].edge_index.t())
#print(dataset[0].edge_attrs)
