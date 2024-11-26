import os
import torch
torch.cuda.current_device()
from torch import nn
import math
from .Attention import wAttention,sAttention
from dataset import MyDataset
from torch_geometric.nn import HypergraphConv,GATv2Conv
import torch.nn.functional as F

class MCGCL(nn.Module):
    def __init__(self,args,data_num):
        super(MCGCL, self).__init__()
        
        self.device = torch.device(('cuda:%d' % (args.gpu)) if torch.cuda.is_available() else 'cpu')
        self.stu_num, self.exer_num, self.knowledge_num = data_num
        self.emb_size = args.emb_size
        self.beta = args.beta
        self.gamma = args.gamma
        self.layers = args.layers
        self.L2 = args.l2
        self.lr = args.lr
        current_path = os.getcwd()
        data_path = current_path.split('code')[0]+'dataset/'
        self.data =  MyDataset(os.path.join(data_path, args.dataset, 'graph'), data_num)
        self.HpGraph_e_k, self.HpGraph_k_e, self.HpGraph_s_k, self.HpGraph_k_s, self.SmGraph_s_s, self.SmGraph_e_e, self.SmGraph_directed_k_k, self.SmGraph_undirected_k_k, self.SmGraph_s_e = self.data[0]
        self.HpGraph_e_k, self.HpGraph_k_e, self.HpGraph_s_k, self.HpGraph_k_s, self.SmGraph_s_s, self.SmGraph_e_e, self.SmGraph_directed_k_k, self.SmGraph_undirected_k_k, self.SmGraph_s_e = self.HpGraph_e_k.to(self.device), self.HpGraph_k_e.to(self.device), self.HpGraph_s_k.to(self.device), self.HpGraph_k_s.to(self.device), self.SmGraph_s_s.to(self.device), self.SmGraph_e_e.to(self.device), self.SmGraph_directed_k_k.to(self.device), self.SmGraph_undirected_k_k.to(self.device), self.SmGraph_s_e.to(self.device)
        self.student_emb = nn.Embedding(self.stu_num, self.emb_size)
        self.exercise_emb = nn.Embedding(self.exer_num, self.emb_size)
        self.knowledge_emb = nn.Embedding(self.knowledge_num, self.emb_size)
        self.k_index = torch.LongTensor(list(range(self.knowledge_num))).to(self.device)
        self.stu_index = torch.LongTensor(list(range(self.stu_num))).to(self.device)
        self.exer_index = torch.LongTensor(list(range(self.exer_num))).to(self.device)
        self.HyperGraph_edge_s_k1 = HypergraphConv(in_channels=self.emb_size, out_channels=self.emb_size,use_attention=True, dropout=0.1)
        self.HyperGraph_edge_s_k2 = HypergraphConv(in_channels=self.emb_size, out_channels=self.emb_size,use_attention=True, dropout=0.1)
        self.HyperGraph_node_s_k1 = HypergraphConv(in_channels=self.emb_size, out_channels=self.emb_size,use_attention=True, dropout=0.1)
        self.HyperGraph_node_s_k2 = HypergraphConv(in_channels=self.emb_size, out_channels=self.emb_size,use_attention=True, dropout=0.1)
        self.HyperGraph_edge_e_k1 = HypergraphConv(in_channels=self.emb_size, out_channels=self.emb_size,use_attention=True, dropout=0.1)
        self.HyperGraph_edge_e_k2 = HypergraphConv(in_channels=self.emb_size, out_channels=self.emb_size,use_attention=True, dropout=0.1)
        self.HyperGraph_node_e_k1 = HypergraphConv(in_channels=self.emb_size, out_channels=self.emb_size,use_attention=True, dropout=0.1)
        self.HyperGraph_node_e_k2 = HypergraphConv(in_channels=self.emb_size, out_channels=self.emb_size,use_attention=True, dropout=0.1)
        self.LineGraph_undirect_k = GATv2Conv(in_channels=self.emb_size,hidden_channels=self.emb_size, num_layers=self.layers, out_channels=self.emb_size, dropout = 0.1)
        self.LineGraph_direct_k = GATv2Conv(in_channels=self.emb_size,hidden_channels=self.emb_size, num_layers=self.layers, out_channels=self.emb_size, dropout = 0.1)
        self.LineGraph_e = GATv2Conv(in_channels=self.emb_size, hidden_channels=self.emb_size, num_layers=self.layers,out_channels=self.emb_size, dropout = 0.1)
        self.LineGraph_s = GATv2Conv(in_channels=self.emb_size,hidden_channels=self.emb_size, num_layers=self.layers, out_channels=self.emb_size, dropout = 0.1)
        self.Attention_e = wAttention()
        self.Attention_s = sAttention()
        self.LN = nn.LayerNorm(self.emb_size).to(self.device)
        self.FC1 = nn.Linear(self.emb_size*2,self.emb_size)
        self.FC2 = nn.Linear(self.emb_size*2,self.emb_size)
        self.FC4 = nn.Linear(self.emb_size*2 ,self.emb_size)
        self.FC5 = nn.Linear(self.emb_size*2 ,self.emb_size)
        self.FC6 = nn.Linear(self.emb_size ,self.emb_size)
        self.FC_LN = nn.Linear(self.emb_size*2 ,self.emb_size)

        self.bn1 = nn.BatchNorm1d(self.emb_size)
        self.bn2 = nn.BatchNorm1d(self.emb_size)
        self.bn4 = nn.BatchNorm1d(self.emb_size)
        self.bn5 = nn.BatchNorm1d(self.emb_size)
        self.bn6 = nn.BatchNorm1d(self.emb_size)
        self.bn_diff = nn.BatchNorm1d(1)
        self.bn_disc = nn.BatchNorm1d(self.emb_size)

        self.loss_function = nn.NLLLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.L2)

        self.diff=nn.Linear(self.emb_size , 1)
        self.disc=nn.Linear(self.emb_size , self.emb_size)

        self.init_parameters()

        self.mlp = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Linear(64, 1), 
            nn.Sigmoid()
        )

    def init_parameters(self):
        stdv = 1.0 / math.sqrt(self.emb_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)
     
    def score(self, x1, x2):
        return torch.matmul(x1, x2)

    def SSL_e(self, e_emb_lgcn, e_emb_hgnn):
        score = self.score(e_emb_lgcn, e_emb_hgnn.transpose(1, 0))
        pos = torch.diag(score)
        neg1 = score - torch.diag_embed(pos)
        pos_loss = F.binary_cross_entropy_with_logits(pos, torch.ones_like(pos))
        neg_loss = F.binary_cross_entropy_with_logits(neg1, torch.zeros_like(neg1))
        con_loss = pos_loss + neg_loss
        return con_loss

    def forward(self, stu_id, exer_id, stage, epoch, batch_cnt, save_folder, method="MCGCL", ifsave=True, user_emb=None, exer_emb=None):
        if stage=="train" or batch_cnt == 1:
            all_stu_emb = self.student_emb(self.stu_index)
            exer_emb = self.exercise_emb(self.exer_index)
            kn_emb = self.knowledge_emb(self.k_index)

            s_embeddings_hg_s_k = self.HyperGraph_edge_s_k1(x=all_stu_emb, hyperedge_index=self.HpGraph_s_k['user','master','knowledge_code'].edge_index, hyperedge_attr=kn_emb)
            s_embeddings_hg_s_k = self.HyperGraph_edge_s_k2(x=s_embeddings_hg_s_k, hyperedge_index=self.HpGraph_s_k['user','master','knowledge_code'].edge_index, hyperedge_attr=kn_emb)
            s_embeddings_hg_s_k = torch.relu(s_embeddings_hg_s_k)
            edge_index = self.HpGraph_s_k['user','master','knowledge_code'].edge_index[[1,0]]
            k_embeddings_hg_s_k = self.HyperGraph_node_s_k1(x=kn_emb,hyperedge_index=edge_index,hyperedge_attr=s_embeddings_hg_s_k)
            k_embeddings_hg_s_k = self.HyperGraph_node_s_k2(x=k_embeddings_hg_s_k,hyperedge_index=edge_index,hyperedge_attr=s_embeddings_hg_s_k)
            k_embeddings_hg_s_k = torch.relu(k_embeddings_hg_s_k)

            e_embeddings_hg_e_k = self.HyperGraph_edge_e_k1(x=exer_emb, hyperedge_index=self.HpGraph_e_k['exer', 'investigate', 'knowledge_code'].edge_index,hyperedge_attr=kn_emb)
            e_embeddings_hg_e_k = self.HyperGraph_edge_e_k2(x=e_embeddings_hg_e_k, hyperedge_index=self.HpGraph_e_k['exer', 'investigate', 'knowledge_code'].edge_index,hyperedge_attr=kn_emb)
            e_embeddings_hg_e_k = torch.relu(e_embeddings_hg_e_k)
            edge_index = self.HpGraph_e_k['exer', 'investigate', 'knowledge_code'].edge_index[[1,0]]
            k_embeddings_hg_e_k = self.HyperGraph_node_e_k1(x=kn_emb,hyperedge_index=edge_index,hyperedge_attr=e_embeddings_hg_e_k)
            k_embeddings_hg_e_k = self.HyperGraph_node_e_k2(x=k_embeddings_hg_e_k,hyperedge_index=edge_index,hyperedge_attr=e_embeddings_hg_e_k)
            k_embeddings_hg_e_k = torch.relu(k_embeddings_hg_e_k)

            k_embeddings_sg_k_k_direction = self.LineGraph_direct_k(kn_emb, self.SmGraph_directed_k_k.edge_index)
            k_embeddings_sg_k_k_undirection = self.LineGraph_undirect_k(kn_emb, self.SmGraph_undirected_k_k.edge_index)
            k_emb1 = self.bn4(self.FC4(torch.cat((k_embeddings_hg_s_k,k_embeddings_hg_e_k),dim=-1)))
            k_emb2 = self.bn5(self.FC5(torch.cat((k_embeddings_sg_k_k_direction,k_embeddings_sg_k_k_undirection),dim=-1)))
            k_embeddings_sg_k_k = self.LN(self.FC_LN(torch.cat((k_emb1, k_emb2), dim=-1)))
            e_embeddings_sg_e_e = self.LineGraph_e(exer_emb, self.SmGraph_e_e.edge_index)
            edge_shape_ek= torch.zeros((self.exer_num,self.knowledge_num)).shape
            value_ek = torch.FloatTensor(torch.ones(self.HpGraph_e_k['exer', 'investigate', 'knowledge_code'].num_edges)).to(self.device)
            adj_e_k = torch.sparse_coo_tensor(self.HpGraph_e_k['exer', 'investigate', 'knowledge_code'].edge_index,value_ek,edge_shape_ek).to_dense()
            exer_embeddings_sg = self.Attention_e(e_embeddings_sg_e_e, k_embeddings_sg_k_k, adj_e_k)
            exer_embeddings_sg = torch.sigmoid(self.bn6(self.FC6(exer_embeddings_sg)))

            s_embeddings_sg_s_s = self.LineGraph_s(all_stu_emb, self.SmGraph_s_s.edge_index)
            edge_shape_sk= torch.zeros((self.stu_num,self.knowledge_num)).shape
            value_sk = torch.FloatTensor(torch.ones(self.HpGraph_s_k['user','master', 'knowledge_code'].num_edges)).to(self.device)
            adj_s_k = torch.sparse_coo_tensor(self.HpGraph_s_k['user','master', 'knowledge_code'].edge_index,value_sk,edge_shape_sk).to_dense()
            stu_embeddings_sg = self.Attention_s(s_embeddings_sg_s_s, k_embeddings_sg_k_k,adj_s_k)
            stu_embeddings_sg = torch.sigmoid(stu_embeddings_sg)
            
            con_loss_e = self.SSL_e(exer_embeddings_sg, e_embeddings_hg_e_k)
            con_loss_s = self.SSL_e(stu_embeddings_sg,s_embeddings_hg_s_k)

            exer_emb = torch.tanh(self.FC1(torch.cat((exer_embeddings_sg,e_embeddings_hg_e_k), dim=-1)))
            user_emb = torch.tanh(self.FC2(torch.cat((stu_embeddings_sg,s_embeddings_hg_s_k), dim=-1)))

        batch_stu_emb = user_emb[stu_id]
        batch_exer_emb = exer_emb[exer_id]
        batch_stu_vector = batch_stu_emb.reshape(batch_stu_emb.shape[0],  batch_stu_emb.shape[1])
        batch_exer_vector = batch_exer_emb.reshape(batch_exer_emb.shape[0],  batch_exer_emb.shape[1])
        if method == "MCGCL":
            output  = self.MCGCL(batch_exer_vector,batch_stu_vector)
        else:
            if method == "IRT":
                output = self.IRT(batch_exer_vector,batch_stu_vector)
            elif method == "MIRT":
                output = self.MIRT(batch_exer_vector,batch_stu_vector,exer_id)
            elif method == "NCD":
                output = self.NCD(batch_exer_vector,batch_stu_vector,exer_id)
            else:
                output = ValueError
        if stage == "train":
            return output, self.beta*con_loss_e, self.gamma*con_loss_s
        else:
            return output, user_emb, exer_emb
        
    def MCGCL(self, batch_exer_vector,batch_stu_vector):
        diff = self.bn_diff(self.diff(batch_exer_vector))
        disc = self.bn_disc(self.disc(batch_exer_vector))
        res_mul = torch.mul(batch_stu_vector, disc)
        e = diff.view(-1) + torch.sum(res_mul, 1)
        res = self.mlp(e.unsqueeze(1))
        return res.view(len(res), -1)
    
    def IRT(self, batch_exer_vector,batch_stu_vector):
        hpro = self.diff(batch_stu_vector)
        hdiff = self.diff(batch_exer_vector)
        e = self.hdisc(hpro-hdiff)
        output = torch.sigmoid(e)
        return output

    def MIRT(self, batch_exer_vector, batch_stu_vector,exer_id):
        stu_emb = torch.sigmoid(batch_stu_vector)
        k_difficulty = torch.sigmoid(batch_exer_vector)
        e_discrimination = torch.sigmoid(self.e_discrimination(exer_id)) * 10
        input_x = e_discrimination * (stu_emb - k_difficulty)
        input_x = self.drop_1(torch.sigmoid(self.prednet_full1(input_x)))
        output = torch.sigmoid(self.prednet_full4(input_x))
        return output
    
    def NCD(self, batch_exer_vector, batch_stu_vector,exer_id):
        stu_emb = torch.sigmoid(batch_stu_vector)
        k_difficulty = torch.sigmoid(batch_exer_vector)
        e_discrimination = torch.sigmoid(self.e_discrimination(exer_id)) * 10
        input_x = e_discrimination * (stu_emb - k_difficulty)
        input_x = self.drop_1(torch.sigmoid(self.prednet_full1(input_x)))
        input_x = self.drop_2(torch.sigmoid(self.prednet_full2(input_x)))
        output = torch.sigmoid(self.prednet_full3(input_x))
        return output