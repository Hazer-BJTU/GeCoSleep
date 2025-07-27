import torch
import torch.nn as nn
from .GCN import GCN
from .base_func import *

#import warnings
#warnings.filterwarnings('ignore')

##############################################################################
################### implemenations of the Graph Builder ######################
##############################################################################

class GraphBuilder(nn.Module):
    def __init__(self, hidden_size, graph_node_dim, num_nodes, dropout):
        super(GraphBuilder, self).__init__()
        self.h_dim = hidden_size
        self.g_dim = graph_node_dim
        self.dropout_rate = dropout
        self.num_nodes = num_nodes
        self.total_edges = int((num_nodes * (num_nodes-1)) / 2)

        # recurrence
        self.gcn1 = GCN(self.g_dim, self.g_dim*4, num_node=num_nodes, input_vector=True)
        self.gcn2 = GCN(self.g_dim*4, self.g_dim*4, num_node=num_nodes, input_vector=False)

        # prior
        self.prior_enc = encode_mean_std(self.g_dim, self.h_dim,
                                         self.dropout_rate)
        self.prior_mij = nn.Linear(self.h_dim, 1)

        # post
        self.post_enc = encode_mean_std(self.g_dim, self.h_dim,
                                        self.dropout_rate)
        self.post_mean_approx_g = nn.Linear(self.h_dim, 1)
        self.post_std_approx_g = nn.Sequential(nn.Linear(self.h_dim, 1), nn.Softplus())

        # graph
        self.node_emb = nn.Sequential(nn.Linear(self.h_dim, self.g_dim), nn.ReLU())
        self.transform = nn.Sequential(nn.Linear(self.g_dim, self.g_dim))
        self.gen_edge_emb = nn.Sequential(
            nn.Linear(self.g_dim * 2, self.g_dim), nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.g_dim, self.g_dim), nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.g_dim, self.g_dim))

    def forward(self, x):
        batch_size = x.size()[0]

        x_node_emb = self.node_emb(x.clone())

        # build node pairs
        node_pairs = torch.zeros(
            batch_size, self.total_edges,
            self.g_dim * 2).cuda()
        adj_node_emb = x_node_emb.clone()
        for i in range(self.num_nodes - 1):
            start = int((self.num_nodes - i - 2) * (self.num_nodes - i - 1) / 2)
            end = int((self.num_nodes - i) * (self.num_nodes - i - 1) / 2)
            one = adj_node_emb[:, self.num_nodes-i-1, :].unsqueeze(1)\
                .repeat(1, self.num_nodes-i-1, 1)
            two = adj_node_emb[:, 0:self.num_nodes - i - 1, :]
            node_pairs[:, start:end, :] = torch.cat([one, two], dim=2)

        # node2edge
        node_pairs = torch.reshape(
            node_pairs,
            (batch_size * self.total_edges, self.g_dim * 2))
        edge_emb_2 = self.gen_edge_emb(node_pairs)


        input4prior = edge_emb_2.clone()
        input4post  = edge_emb_2.clone()

        # prior
        prior_mean_g, prior_std_g, prior_b = self.prior_enc(input4prior)
        prior_mij = self.prior_mij(prior_b)
        prior_mean_g = prior_mean_g.reshape(batch_size, self.total_edges)
        prior_std_g = prior_std_g.reshape(batch_size, self.total_edges)
        prior_mij = prior_mij.reshape(batch_size, self.total_edges)
        prior_mij = 0.4 * sigmoid(prior_mij)

        # post
        post_mean_g, post_std_g, post_b = self.post_enc(input4post)
        post_mean_approx_g = self.post_mean_approx_g(post_b)
        post_std_approx_g = self.post_std_approx_g(post_b)
        post_mean_g = post_mean_g.reshape(batch_size, self.total_edges)
        post_std_g = post_std_g.reshape(batch_size, self.total_edges)
        post_mean_approx_g = post_mean_approx_g.reshape(batch_size, self.total_edges)
        post_std_approx_g = post_std_approx_g.reshape(batch_size, self.total_edges)

        # estimate post mij for Binomial Dis
        eps = 1e-6
        nij = 2.0 * post_mean_approx_g - 1.0
        nij_ = nij*nij + 8.0*post_std_approx_g*post_std_approx_g
        post_mij = 0.25 * (nij + torch.sqrt(nij_)) + eps
        
        # reparameterization: sampling alpha_tilda and alpha_bar
        alpha_bar, alpha_tilde = self.sample_repara(post_mean_g, post_std_g, post_mij)

        # graph embedding
        H_g, A = self.gcn1(x, alpha_bar)
        H_g = self.gcn2(H_g, A)

        # regularization
        kl_g = self.kld_loss_gauss(alpha_tilde * post_mean_g,
                                   torch.sqrt(alpha_tilde) * post_std_g,
                                   alpha_tilde * prior_mean_g,
                                   torch.sqrt(alpha_tilde) * prior_std_g)
        kl_b = self.kld_loss_binomial_upper_bound(post_mij, prior_mij)

        # return dic for next iter and optimization
        outputs = {
            'H_g': H_g,
            'kl_g': kl_g,
            'kl_b': kl_b,
            'summ_graph': alpha_tilde,
            'spec_graph': alpha_bar
        }
        return outputs

    def reset_parameters(self, stdv=1e-1):
        for weight in self.parameters():
            weight.data.normal_(0, stdv)

    def _init_weights(self, stdv):
        pass

    def sample_repara(self, mean, std, mij):
        mean_alpha = mij
        std_alpha = torch.sqrt(mij)
        eps = torch.FloatTensor(std.size()).normal_().cuda()
        alpha_tilde = eps * std_alpha + mean_alpha
        alpha_tilde = softplus(alpha_tilde)

        mean_sij = alpha_tilde * mean
        std_sij = torch.sqrt(alpha_tilde) * std
        eps_2 = torch.FloatTensor(std.size()).normal_().cuda()
        s_ij = eps_2 * std_sij + mean_sij
        alpha_bar = s_ij * alpha_tilde

        return alpha_bar, alpha_tilde

    def kld_loss_gauss(self, mean_post, std_post, mean_prior, std_prior, eps=1e-6):
        kld_element = (2 * torch.log(std_prior + eps) - 2 * torch.log(std_post + eps) +
                       ((std_post).pow(2) + (mean_post - mean_prior).pow(2)) /
                       (std_prior + eps).pow(2) - 1)
        return 0.5 * torch.sum(torch.abs(kld_element))

    def kld_loss_binomial_upper_bound(self, mij_post, mij_prior, eps=1e-6):
        kld_element = mij_prior - mij_post + \
                       mij_post * (torch.log(mij_post+eps) - torch.log(mij_prior+eps))
        return torch.sum(torch.abs(kld_element))


class encode_mean_std(nn.Module):
    def __init__(self, g_dim, h_dim, dropout=0.1):
        super(encode_mean_std, self).__init__()

        self.g_dim = g_dim
        self.h_dim = h_dim
        self.dropout = dropout

        self.enc = nn.Sequential(nn.Linear(g_dim, h_dim),
                                 nn.ReLU(), nn.Dropout(dropout))
        self.enc_g = nn.Sequential(nn.Linear(h_dim, h_dim), nn.ReLU(), nn.Dropout(dropout))
        self.enc_b = nn.Sequential(nn.Linear(h_dim, h_dim), nn.ReLU(), nn.Dropout(dropout))

        self.enc_mean = nn.Linear(h_dim, 1)
        self.enc_std = nn.Sequential(nn.Linear(h_dim, 1), nn.Softplus())

    def forward(self, x):
        enc_ = self.enc(x)
        enc_g = self.enc_g(enc_)
        enc_b = self.enc_b(enc_)
        mean = self.enc_mean(enc_g)
        std = self.enc_std(enc_g)
        return mean, std, enc_b
