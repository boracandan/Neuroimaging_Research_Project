import torch.nn.functional as F
import torch
import torch.nn as nn
import numpy as np
import MAHGCN

gpu = 1

class fMRINet(nn.Module):
    def __init__(self, ROInum, activation, hidden_dim=1, degree_normalize=False, gcn_mode="gcn", num_heads=1, num_gnn_layers=1, feat_dim=None):
        super(fMRINet, self).__init__()
        self.hidden_dim = hidden_dim
        self.ROInum = ROInum
        # feat_dim defaults to ROInum (FC-profile rows as features). Pass an explicit value for
        # multimodal where node features come from sMRI (in_dim = num sMRI features).
        self.feat_dim = feat_dim if feat_dim is not None else ROInum
        self.GNN = MAHGCN.build_single_res_fmri(gcn_mode, nn.ReLU(), 0.3, self.feat_dim, hidden_dim, num_heads, degree_normalize, num_gnn_layers)
        self.paranum = self.GNN.output_dim

        self.bn1 = torch.nn.BatchNorm1d(self.paranum)
        self.fl1 = nn.Linear(self.paranum, 64)
        self.bn2 = torch.nn.BatchNorm1d(64)

        self.fl2 = nn.Sequential(nn.Linear(64,1), nn.Sigmoid()) if activation == "sigmoid" else nn.Linear(64,1)


    def forward(self, g_matrix, node_features=None):
        batch_size = g_matrix.shape[0]

        fea = node_features if node_features is not None else g_matrix  # default: FC profiles as node features

        out = torch.zeros(batch_size, self.paranum, device='cuda')

        for s in range(batch_size):
            out[s, :] = self.GNN(g_matrix[s, :, :], fea[s, :, :])

        out = self.bn1(out)
        out = F.relu(out)

        out = self.fl1(out)
        out = self.bn2(out)
        out = F.relu(out)

        out = self.fl2(out)

        return out

class MAHGCNNET(nn.Module):
    def __init__(self, layer, activation, hidden_dim=1, degree_normalize=False, gcn_mode="original", num_heads=1, num_gat_layers=1):
        super(MAHGCNNET, self).__init__()
        self.layer = layer
        self.hidden_dim = hidden_dim
        possible_rois = [1000, 400, 300, 200]
        self.roi_start = possible_rois[4 - layer]
        #self.mMAHGCNs = nn.ModuleList()
        #for t in range(tsize):
            #self.mMAHGCNs.append(MAHGCN.MultiresolutionMAHGCN(nn.ReLU(),0.3))
        self.MAHGCN = MAHGCN.build_mahgcn(gcn_mode, nn.ReLU(), 0.3, self.layer, hidden_dim, num_heads, degree_normalize, num_gat_layers)
        self.paranum = self.MAHGCN.output_dim
        #self.gcrn = GLSTM_multi.ConvLSTM(ROInum, 1)

        self.bn1 = torch.nn.BatchNorm1d(self.paranum)
        self.fl1 = nn.Linear(self.paranum, 64)
        self.bn2 = torch.nn.BatchNorm1d(64)

        self.fl2 = nn.Sequential(nn.Linear(64,1), nn.Sigmoid()) if activation == "sigmoid" else nn.Linear(64,1)


    def forward(self, *g_matrices):
        batch_size = g_matrices[0].shape[0]
        ROInum = self.roi_start

        fea = torch.eye(ROInum, device='cuda').unsqueeze(0).expand(batch_size, -1, -1)

        out = torch.zeros(batch_size, self.paranum, device='cuda')

        for s in range(batch_size):
            out[s, :] = torch.squeeze(self.MAHGCN(*[g[s, :, :] if g is not None else None for g in g_matrices], fea[s, :, :]))

        out = self.bn1(out)
        out = F.relu(out)

        out = self.fl1(out)
        out = self.bn2(out)
        out = F.relu(out)

        out = self.fl2(out)

        return out

class GCN_base(nn.Module):
    def __init__(self,ROInum,num_class=2):
        super(GCN_base, self).__init__()

        #self.mMAHGCNs = nn.ModuleList()
        #for t in range(tsize):
            #self.mMAHGCNs.append(MAHGCN.MultiresolutionMAHGCN(nn.ReLU(),0.3))
        self.gcn = MAHGCN.GCN(ROInum, 1, nn.ReLU(),0.3)
        #self.gcrn = GLSTM_multi.ConvLSTM(ROInum, 1)

        self.bn1 = torch.nn.BatchNorm1d(ROInum)
        self.fl1 = nn.Linear(ROInum,64)
        self.bn2 = torch.nn.BatchNorm1d(64)
        self.fl2 = nn.Linear(64,num_class)


        #self.dropout = nn.Dropout(0.6)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, g):
        batch_size = g.shape[0]
        ROInum = g.shape[2]

        fea = torch.zeros(g.size())
        for s in range(g.shape[0]):
            fea[s,:,:] = torch.eye(ROInum)
        fea = fea.cuda()
        g = g.cuda()
        out = torch.zeros(batch_size, ROInum)

        for s in range(batch_size):
            temp = self.gcn(g[s, :, :], fea[s, :, :])
            temp.cuda()
            out[s, :] = torch.squeeze(temp)
        out = out.cuda()

        out = self.bn1.cuda()(out)
        out = F.relu(out)

        out = self.fl1(out)
        out = self.bn2.cuda()(out)
        out = F.relu(out)

        out = self.fl2(out)
        out = self.softmax(out)

        return out

class GCN_gpool(nn.Module):
    def __init__(self,ROInum,num_class=2):
        super(GCN_gpool, self).__init__()

        #self.mMAHGCNs = nn.ModuleList()
        #for t in range(tsize):
            #self.mMAHGCNs.append(MAHGCN.MultiresolutionMAHGCN(nn.ReLU(),0.3))
        self.gcn = MAHGCN.GraphUnet([4/5,3/4,2/3,1/2],ROInum, 1, 1, nn.ReLU(),0.3)
        #self.gcrn = GLSTM_multi.ConvLSTM(ROInum, 1)

        self.bn1 = torch.nn.BatchNorm1d(1500)
        self.fl1 = nn.Linear(1500,64)
        self.bn2 = torch.nn.BatchNorm1d(64)
        self.fl2 = nn.Linear(64,num_class)


        #self.dropout = nn.Dropout(0.6)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, g):
        batch_size = g.shape[0]
        ROInum = g.shape[2]

        fea = torch.zeros(g.size())
        for s in range(g.shape[0]):
            fea[s,:,:] = torch.eye(ROInum)
        fea = fea.cuda()
        g = g.cuda()
        out = torch.zeros(batch_size, 1500)

        for s in range(batch_size):
            temp = self.gcn(g[s, :, :], fea[s, :, :])
            temp.cuda()
            out[s, :] = torch.squeeze(temp)
        out = out.cuda()

        out = self.bn1.cuda()(out)
        out = F.relu(out)

        out = self.fl1(out)
        out = self.bn2.cuda()(out)
        out = F.relu(out)

        out = self.fl2(out)
        out = self.softmax(out)

        return out

class GCN_diffpool(nn.Module):
    def __init__(self,ROInum,num_class=2):
        super(GCN_diffpool, self).__init__()

        #self.mMAHGCNs = nn.ModuleList()
        #for t in range(tsize):
            #self.mMAHGCNs.append(MAHGCN.MultiresolutionMAHGCN(nn.ReLU(),0.3))
        self.gcn = MAHGCN.GraphDiif([4/5,3/4,2/3,1/2],ROInum, 1, 1, nn.ReLU(),0.3)
        #self.gcrn = GLSTM_multi.ConvLSTM(ROInum, 1)

        self.bn1 = torch.nn.BatchNorm1d(1500)
        self.fl1 = nn.Linear(1500,64)
        self.bn2 = torch.nn.BatchNorm1d(64)
        self.fl2 = nn.Linear(64,num_class)


        #self.dropout = nn.Dropout(0.6)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, g):
        batch_size = g.shape[0]
        ROInum = g.shape[2]

        fea = torch.zeros(g.size())
        for s in range(g.shape[0]):
            fea[s,:,:] = torch.eye(ROInum)
        fea = fea.cuda()
        g = g.cuda()
        out = torch.zeros(batch_size, 1500)

        for s in range(batch_size):
            temp = self.gcn(g[s, :, :], fea[s, :, :])
            temp.cuda()
            out[s, :] = torch.squeeze(temp)
        out = out.cuda()

        out = self.bn1.cuda()(out)
        out = F.relu(out)

        out = self.fl1(out)
        out = self.bn2.cuda()(out)
        out = F.relu(out)

        out = self.fl2(out)
        out = self.softmax(out)

        return out
    def Loss(self):
        return self.gcn.Loss()

class GCN_SAG(nn.Module):
    def __init__(self,ROInum,num_class=2):
        super(GCN_SAG, self).__init__()

        #self.mMAHGCNs = nn.ModuleList()
        #for t in range(tsize):
            #self.mMAHGCNs.append(MAHGCN.MultiresolutionMAHGCN(nn.ReLU(),0.3))
        self.gcn = MAHGCN.GraphSAG([4/5,3/4,2/3,1/2],ROInum, 1, 1, nn.ReLU(),0.3)
        #self.gcrn = GLSTM_multi.ConvLSTM(ROInum, 1)

        self.bn1 = torch.nn.BatchNorm1d(1500)
        self.fl1 = nn.Linear(1500,64)
        self.bn2 = torch.nn.BatchNorm1d(64)
        self.fl2 = nn.Linear(64,num_class)


        #self.dropout = nn.Dropout(0.6)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, g):
        batch_size = g.shape[0]
        ROInum = g.shape[2]

        fea = torch.zeros(g.size())
        for s in range(g.shape[0]):
            fea[s,:,:] = torch.eye(ROInum)
        fea = fea.cuda()
        g = g.cuda()
        out = torch.zeros(batch_size, 1500)

        for s in range(batch_size):
            temp = self.gcn(g[s, :, :], fea[s, :, :])
            temp.cuda()
            out[s, :] = torch.squeeze(temp)
        out = out.cuda()

        out = self.bn1.cuda()(out)
        out = F.relu(out)

        out = self.fl1(out)
        out = self.bn2.cuda()(out)
        out = F.relu(out)

        out = self.fl2(out)
        out = self.softmax(out)

        return out
