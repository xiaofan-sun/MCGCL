import torch
from torch_geometric.data import InMemoryDataset
from utils.read_graph import *


class MyDataset(InMemoryDataset):
    def __init__(self, root, data_num, transform=None, pre_transform=None):
        self.data_path = root
        super().__init__(root, transform, pre_transform)
        print("self.processed_paths[0]",self.processed_paths[0])
        self.data, self.slices = torch.load(self.processed_paths[0])
        self.data_num = {"s":data_num[0], "e":data_num[1], "k":data_num[2]}
    @property
    def raw_file_names(self):
        return [f'u_e.json'] + ['HpGraph_u_k.json']+[f'Hpgraph_e_k.json'] +[f'Smgraph_e_e.json'] + [f'Smgraph_undirected_k_k.json'] +[f'Smgraph_u_u_scores.json'] + [] +[]+[]
    @property
    def processed_file_names(self):
        return 'data.pt'

    def process(self):
        data_list = []
        HpGraph_e_k, HpGraph_k_e = read_hgraph_e_k(self.data_path,self.data_num)
        HpGraph_s_k, HpGraph_k_s = read_hgraph_s_k(self.data_path,self.data_num)
        SmGraph_s_s = read_sgraph_s_s(self.data_path,self.data_num)
        SmGraph_e_e = read_sgraph_e_e(self.data_path,self.data_num)
        SmGraph_directed_k_k = read_sgraph_directed_k_k(self.data_path,self.data_num)
        SmGraph_undirected_k_k = read_sgraph_undirected_k_k(self.data_path,self.data_num)
        SmGraph_s_e = read_sgraph_s_e(self.data_path,self.data_num)

        data_list.append([HpGraph_e_k,HpGraph_k_e,HpGraph_s_k, HpGraph_k_s,SmGraph_s_s,SmGraph_e_e,SmGraph_directed_k_k,SmGraph_undirected_k_k,SmGraph_s_e])
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None: 
            data_list = [self.pre_transform(data) for data in data_list]
     
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
