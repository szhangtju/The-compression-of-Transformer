import torch
import torch.nn as nn
import numpy as np

__author__ = "Yu-Hsiang Huang"

class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, mask=None):

        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature

        if mask is not None:
            attn = attn.masked_fill(mask, -np.inf)

        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = torch.bmm(attn, v)

        return output, attn

__author__ = "Xindian Ma"

# class SingleCoreAttention(nn.Module):
#     ''' Single Core Attention (Single Block Attention)'''
#     def __init__(self, temperature, d_v, n_head, atten_dropout):
#         super().__init__()
#         self.temperature = temperature
#         self.dropout = nn.Dropout(atten_dropout)
#         self.softmax = nn.Softmax(dim=2)
#         core_1 = torch.randn(d_v)
#         core_2 = torch.randn(d_v)
#         self.vectors = nn.Parameter(torch.stack((core_1,core_2),dim=0))

#     def forward(self, q, k, v, mask=None):
#         mb_size, len_q, dimen = q.size()
#         full_matrix_1 = torch.einsum('d, bid, bjd, bkd->bijk', [self.vectors[0], q, k, v]).contiguous().cuda()
#         full_matrix_2 = torch.einsum('d, bid, bjd, bkd->bijk', [self.vectors[1], q, k, v]).contiguous().cuda()

#         attn1 = torch.sum(full_matrix_1,dim=2)/self.temperature
#         attn2 = torch.sum(full_matrix_2,dim=2)/self.temperature
#         attn1 = self.softmax(attn1)
#         output1 = torch.bmm(self.dropout(attn1),v)
#         attn2 = self.softmax(attn2)
#         output2 = torch.bmm(self.dropout(attn2),v)
#         output = torch.cat([output1, output2],1)
#         atten = torch.cat([output1, output2],1)
#         return output, atten
class SingleCoreAttention(nn.Module):
    ''' Single Core Attention (Single Block Attention)'''
    def __init__(self, temperature, d_v, n_head, atten_dropout):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(atten_dropout)
        self.softmax = nn.Softmax(dim=0)
        core_1 = self.softmax(torch.randn(d_v))
        core_2 = self.softmax(torch.randn(d_v))
        self.vectors = torch.stack((core_1,core_2),dim=0)

    def forward(self, q, k, v, mask=None):
        mb_size, len_q, dimen = q.size()

        cores_1 = torch.zeros(dimen,dimen,len_q).cuda()
        cores_2 = torch.zeros(dimen,dimen,len_q).cuda()
        for i in range(int(min(dimen,len_q))):
            cores_1[i][i][i] = self.vectors[0][i].cuda()
            cores_2[i][i][i] = self.vectors[1][i].cuda()
        full_matrix_1 = torch.einsum('pqk, bip,bjq,bkr->bijr', [cores_1, q, k, v]).contiguous().cuda()
        full_matrix_2 = torch.einsum('pqk, bip,bjq,bkr->bijr', [cores_2, q, k, v]).contiguous().cuda()
        average_tensor = (torch.sum(full_matrix_1, dim=2)+torch.sum(full_matrix_2, dim=2)).mul_(0.5)
        average_tensor = average_tensor/self.temperature
        # output = torch.stack(average_tensor).cuda().float()
        output = self.dropout(average_tensor)
        attn = torch.bmm(q, k.transpose(1, 2)).cuda()
        del(cores_1)
        del(cores_2)
        return output, attn

