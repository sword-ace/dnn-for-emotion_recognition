# -*- coding: utf-8 -*-

import os
import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable
from torch_geometric.utils import softmax
from torch_scatter import scatter_add


class MatchingAttention(nn.Module):

    def __init__(self, mem_dim, cand_dim, alpha_dim=None, att_type='general'):
        super(MatchingAttention, self).__init__()
        assert att_type != 'concat' or alpha_dim != None
        assert att_type != 'dot' or mem_dim == cand_dim
        self.mem_dim = mem_dim
        self.cand_dim = cand_dim
        self.att_type = att_type
        if att_type == 'general':
            self.transform = nn.Linear(cand_dim, mem_dim, bias=False)
        if att_type == 'general2':
            self.transform = nn.Linear(cand_dim, mem_dim, bias=True)
            # torch.nn.init.normal_(self.transform.weight,std=0.01)
        elif att_type == 'concat':
            self.transform = nn.Linear(cand_dim + mem_dim, alpha_dim, bias=False)
            self.vector_prod = nn.Linear(alpha_dim, 1, bias=False)

    def forward(self, M, x, mask=None):
        """
        M -> (seq_len, batch, mem_dim)
        x -> (batch, cand_dim)
        mask -> (batch, seq_len)
        """
        if type(mask) == type(None):
            mask = torch.ones(M.size(1), M.size(0)).type(M.type())

        if self.att_type == 'dot':
            # vector = cand_dim = mem_dim
            M_ = M.permute(1, 2, 0)  # batch, vector, seqlen
            x_ = x.unsqueeze(1)  # batch, 1, vector
            alpha = F.softmax(torch.bmm(x_, M_), dim=2)  # batch, 1, seqlen
        elif self.att_type == 'general':
            M_ = M.permute(1, 2, 0)  # batch, mem_dim, seqlen
            x_ = self.transform(x).unsqueeze(1)  # batch, 1, mem_dim
            alpha = F.softmax(torch.bmm(x_, M_), dim=2)  # batch, 1, seqlen
        elif self.att_type == 'general2':
            M_ = M.permute(1, 2, 0)  # batch, mem_dim, seqlen
            x_ = self.transform(x).unsqueeze(1)  # batch, 1, mem_dim
            alpha_ = F.softmax((torch.bmm(x_, M_)) * mask.unsqueeze(1), dim=2)  # batch, 1, seqlen
            alpha_masked = alpha_ * mask.unsqueeze(1)  # batch, 1, seqlen
            alpha_sum = torch.sum(alpha_masked, dim=2, keepdim=True)  # batch, 1, 1
            alpha = alpha_masked / alpha_sum  # batch, 1, 1 ; normalized
            # import ipdb;ipdb.set_trace()
        else:
            M_ = M.transpose(0, 1)  # batch, seqlen, mem_dim
            x_ = x.unsqueeze(1).expand(-1, M.size()[0], -1)  # batch, seqlen, cand_dim
            M_x_ = torch.cat([M_, x_], 2)  # batch, seqlen, mem_dim+cand_dim
            mx_a = F.tanh(self.transform(M_x_))  # batch, seqlen, alpha_dim
            alpha = F.softmax(self.vector_prod(mx_a), 1).transpose(1, 2)  # batch, 1, seqlen

        attn_pool = torch.bmm(alpha, M.transpose(0, 1))[:, 0, :]  # batch, mem_dim

        return attn_pool, alpha


def pad(tensor, length, cuda_flag):
    if isinstance(tensor, Variable):
        var = tensor
        if length > var.size(0):
            if cuda_flag:
                return torch.cat([var, torch.zeros(length - var.size(0), *var.size()[1:]).cuda()])
            else:
                return torch.cat([var, torch.zeros(length - var.size(0), *var.size()[1:])])
        else:
            return var
    else:
        if length > tensor.size(0):
            if cuda_flag:
                return torch.cat([tensor, torch.zeros(length - tensor.size(0), *tensor.size()[1:]).cuda()])
            else:
                return torch.cat([tensor, torch.zeros(length - tensor.size(0), *tensor.size()[1:])])
        else:
            return tensor


def feature_transfer(bank_s_, bank_p_, seq_lengths, cuda_flag=False):
    input_conversation_length = torch.tensor(seq_lengths)
    start_zero = input_conversation_length.data.new(1).zero_()
    if cuda_flag:
        input_conversation_length = input_conversation_length.cuda()
        start_zero = start_zero.cuda()

    max_len = max(seq_lengths)
    start = torch.cumsum(torch.cat((start_zero, input_conversation_length[:-1])), 0)
    # (l,b,h)
    bank_s = torch.stack(
        [pad(bank_s_.narrow(0, s, l), max_len, cuda_flag) for s, l in zip(start.data.tolist(), input_conversation_length.data.tolist())], 0
    ).transpose(0, 1)
    bank_p = torch.stack(
        [pad(bank_p_.narrow(0, s, l), max_len, cuda_flag) for s, l in zip(start.data.tolist(), input_conversation_length.data.tolist())], 0
    ).transpose(0, 1)

    return bank_s, bank_p



class MultiTurnTransformer(nn.Module):
    def __init__(self, input_dim, model_dim):
        super(MultiTurnTransformer, self).__init__()
        self.embedding = nn.Linear(input_dim, model_dim)
        transformer_layer = nn.TransformerEncoderLayer(d_model=model_dim, nhead=4, dropout=0.2)
        self.transformer_encoder = nn.TransformerEncoder(transformer_layer, num_layers=1)


    def positional_encoding(self, seq_len, d_model):
         pe = torch.zeros(seq_len, d_model)
         position = torch.arange(0, seq_len, dtype=torch.float32).unsqueeze(1)
         div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
         pe[:, 0::2] = torch.sin(position * div_term)
         pe[:, 1::2] = torch.cos(position * div_term)
         return pe

    def forward(self, x):
        x = self.embedding(x)  # Embed the input
        x = x + self.positional_encoding(x.size(1), x.size(2)).to(x.device)  # Add positional encoding before transformer
        x = x+ self.transformer_encoder(x)
        return x


class ReasonMod(nn.Module):
    def __init__(self, in_channels):

        super(ReasonMod, self).__init__()

        self.in_channels =  in_channels
       
        self.out_channels =  in_channels

  
        self.transformer = MultiTurnTransformer(self.in_channels, self.out_channels)


    def forward(self,context_buffer):
        # Use enriched_context in attention and reasoning
      
        enriched_context = self.transformer(context_buffer)

        return enriched_context 

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels, self.out_channels)


class DetectiveNetwork(nn.Module):
    def __init__(self, n_features=100, n_classes=7,  cuda_flag=False,  reason_steps=None):
        """
        modeling detection phase
        """
        super(DetectiveNetwork, self).__init__()
        dropout=0.5

        self.cuda_flag = cuda_flag
        self.in_channels = n_features

        self.fc = nn.Linear(n_features*2, n_features * 2)

        self.dropout = nn.Dropout(dropout)

        self.reason = ReasonMod(in_channels=n_features)

        self.smax_fc = nn.Linear(n_features * 2, n_classes)

        self.gelu = nn.GELU()


    def forward(self, U_s, U_p, seq_lengths):
        # (b) <== (l,b,h=200)
        batch_size = U_s.size(1)
        batch_index, context_s_, context_p_ = [], [], []
        # print("u_s size",U_s.shape, U_p.shape ) #size seq_len, batch, dim

        for j in range(batch_size):
            batch_index.extend([j] * seq_lengths[j])
            context_s_.append(U_s[:seq_lengths[j], j, :])
            context_p_.append(U_p[:seq_lengths[j], j, :])

        batch_index = torch.tensor(batch_index)
        bank_s_ = torch.cat(context_s_, dim=0)
        bank_p_ = torch.cat(context_p_, dim=0)
        if self.cuda_flag:
            batch_index = batch_index.cuda()
            bank_s_ = bank_s_.cuda()
            bank_p_ = bank_p_.cuda()

        # (l,b,h) << (l*b,h)

        # print("bs, bp, seq", bank_s_.shape, bank_p_.shape, seq_lengths)
        bank_s, bank_p = feature_transfer(bank_s_, bank_p_, seq_lengths, self.cuda_flag)
        # print(bank_s.shape, bank_p.shape)

        feature_ = []
        # feature_1 = []

        # context_buffer = []
        for t in range(bank_p.size(0)):

            q = self.reason(bank_s)


        ################### this part is for corss -attention ####################
            batch_size = batch_index.max().item() + 1


# ###################################################################################
            q_star = q[t].view(batch_size, self.in_channels)
           
            e = (bank_p_ * q_star[batch_index]).sum(dim=-1, keepdim=True)
            
            a = softmax(e, batch_index, num_nodes=batch_size)
            
            r = scatter_add(a * bank_p_, batch_index, dim=0, dim_size=batch_size)
         
            r = torch.cat([bank_s[t], r], dim=-1)

            r =  F.relu(self.fc(r))

            feature_.append(r.unsqueeze(0))

        hidden = torch.cat(feature_, dim=0)

        log_prob = F.log_softmax(self.smax_fc(hidden), 2)
        log_prob = torch.cat([log_prob[:, j, :][:seq_lengths[j]] for j in range(len(seq_lengths))])
        return log_prob



class DetectiveNN(nn.Module):
    def __init__(self, base_model='LSTM', base_layer=2, input_size=None, hidden_size=None, n_speakers=2,
                 n_classes=7, dropout=0.2, cuda_flag=False, reason_steps=None):
        """
        designing for recall-detect-prediction framework
        """

        super(DetectiveNN, self).__init__()
        self.base_model = base_model
        self.n_speakers = n_speakers
        self.hidden_size = hidden_size
        self.gelu = nn.GELU()

        if self.base_model == 'LSTM':
            self.rnn = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=base_layer, bidirectional=True, dropout=dropout)
            self.rnn_parties = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=base_layer, bidirectional=True, dropout=dropout)
        elif self.base_model == 'GRU':
            self.rnn = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=base_layer, bidirectional=True, dropout=dropout)
            self.rnn_parties = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=base_layer, bidirectional=True, dropout=dropout)

        elif self.base_model == 'conv1d':
            # self.conv1d = MyConv1D(nf=hidden_size * 2, nx=input_size, ...)
            # self.conv1d_parties = MyConv1D(nf=hidden_size * 2, nx=input_size, ...)

            self.rnn = MyConv1D(nf=hidden_size * 2, nx= input_size)
            self.rnn_parties = MyConv1D(nf=hidden_size * 2, nx=input_size)


        elif self.base_model == 'Linear':
            self.base_linear = nn.Linear(input_size, 2 * hidden_size)
        else:
            print('Base model must be one of LSTM/GRU/Linear')
            raise NotImplementedError

        self.detect_net = DetectNetwork(n_features=2 * hidden_size, n_classes=n_classes,  cuda_flag=cuda_flag, reason_steps=reason_steps)
        print(self)

    def forward(self, U, qmask, seq_lengths):
        U_s, U_p = None, None

        if self.base_model == 'LSTM':
            # (b,l,h), (b,l,p)
            # print('U', U.shape, qmask.shape)
            ############################################
            U_, qmask_ = U.transpose(0, 1), qmask #qmask.transpose(0, 1)
            U_p_ = torch.zeros(U_.size()[0], U_.size()[1], self.hidden_size * 2).type(U.type())
            U_parties_ = [torch.zeros_like(U_).type(U_.type()) for _ in range(self.n_speakers)]
            for b in range(U_.size(0)):
                for p in range(len(U_parties_)):
                    index_i = torch.nonzero(qmask_[b][:, p]).squeeze(-1)
                    if index_i.size(0) > 0:
                        U_parties_[p][b][:index_i.size(0)] = U_[b][index_i]
            E_parties_ = [self.rnn_parties(U_parties_[p].transpose(0, 1))[0].transpose(0, 1) for p in range(len(U_parties_))]

            for b in range(U_p_.size(0)):
                for p in range(len(U_parties_)):
                    index_i = torch.nonzero(qmask_[b][:, p]).squeeze(-1)
                    if index_i.size(0) > 0: U_p_[b][index_i] = E_parties_[p][b][:index_i.size(0)]
            U_p = U_p_.transpose(0, 1)

            # (l,b,2*h) [(2*bi,b,h) * 2]
            U_s, hidden = self.rnn(U)

        elif self.base_model == 'GRU':
            U_, qmask_ = U.transpose(0, 1), qmask
            U_p_ = torch.zeros(U_.size()[0], U_.size()[1], self.hidden_size * 2).type(U.type())
            U_parties_ = [torch.zeros_like(U_).type(U_.type()) for _ in range(self.n_speakers)]  # default 2
            for b in range(U_.size(0)):
                for p in range(len(U_parties_)):
                    index_i = torch.nonzero(qmask_[b][:, p]).squeeze(-1)
                    if index_i.size(0) > 0: U_parties_[p][b][:index_i.size(0)] = U_[b][index_i]
            E_parties_ = [self.rnn_parties(U_parties_[p].transpose(0, 1))[0].transpose(0, 1) for p in range(len(U_parties_))]

            for b in range(U_p_.size(0)):
                for p in range(len(U_parties_)):
                    index_i = torch.nonzero(qmask_[b][:, p]).squeeze(-1)
                    if index_i.size(0) > 0: U_p_[b][index_i] = E_parties_[p][b][:index_i.size(0)]
            U_p = U_p_.transpose(0, 1)
            U_s, hidden = self.rnn(U)

        # elif self.base_model == 'conv1d':
        #     U_, qmask_ = U.transpose(0, 1), qmask
        #     # (b,l,h), (b,l,p)
        #     U_s = self.rnn(U) # Process all utterances with MyConv1D

        #     # Process speaker-specific utterances
        #     U_p_ = torch.zeros(U.size(0), U.size(1), self.hidden_size * 2).type(U.type())
        #     U_parties_ = [torch.zeros_like(U_).type(U_.type()) for _ in range(self.n_speakers)]  # default 2

        #     for b in range(U.size(0)):
        #         for p in range(self.n_speakers):
        #             index_i = torch.nonzero(qmask[b][:, p]).squeeze(-1)
        #             if index_i.size(0) > 0:
        #                 U_parties_[p][b][:index_i.size(0)] = U_[b][index_i]
        #                 # U_p_[b][index_i] = U[b][index_i]

        #     E_parties_ = [self.rnn_parties(U_p_[:, p]) for p in range(self.n_speakers)] # Process each speaker's utterance with MyConv1D

        #     for b in range(U_p_.size(0)):
        #         for p in range(self.n_speakers):
        #             index_i = torch.nonzero(qmask[b][:, p]).squeeze(-1)
        #             if index_i.size(0) > 0:
        #                 U_p_[b][index_i] = E_parties_[p][b][:index_i.size(0)]

        #     U_p = U_p_.transpose(0, 1) # Combine speaker-specific representations


        elif self.base_model == 'Linear':
            # TODO
            U = self.base_linear(U)
            # U = self.dropout(F.relu(U))
            ##if apply geru instead of relu
            U = self.dropout(self.gelu(U))
            hidden = self.smax_fc(U)
            log_prob = F.log_softmax(hidden, 2)
            logits = torch.cat([log_prob[:, j, :][:seq_lengths[j]] for j in range(len(seq_lengths))])
            return logits

        
        logits = self.detect_net(U_s, U_p, seq_lengths)
        return logits
