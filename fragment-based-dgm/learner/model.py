import numpy as np
import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class Encoder(nn.Module):
    def __init__(self, input_size, embed_size,
                 hidden_size, hidden_layers, latent_size,
                 dropout, use_gpu):
        super().__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.hidden_layers = hidden_layers
        self.latent_size = latent_size
        self.use_gpu = use_gpu

        self.rnn = nn.GRU(
            input_size=self.embed_size,
            hidden_size=self.hidden_size,
            num_layers=self.hidden_layers,
            dropout=dropout,
            batch_first=True)

        self.rnn2mean = nn.Linear(
            in_features=self.embed_size,
            out_features=self.latent_size)
        #nn.Sequential(
        #    nn.Linear(self.embed_size, 64),
        #    nn.ReLU(),
        #    nn.Linear(64, self.latent_size)
        #)
        #    # nn.Dropout(0.2),
        #    nn.ReLU(),
        #    # nn.Dropout(0.2),
        #    nn.Linear(500, self.latent_size)
        #    # nn.Sigmoid()
        #)


        self.rnn2logv =  nn.Linear(
            in_features=self.embed_size,
            out_features=self.latent_size)
        #nn.Sequential(
        #    nn.Linear(self.embed_size, 64),
        #    nn.ReLU(),
        #    nn.Linear(64, self.latent_size)
        #)
        # nn.Linear(
        #    in_features=self.embed_size,
        #    out_features=self.latent_size)


    def forward(self, vec_frag_arr):
        batch_size = vec_frag_arr.size(0)
        #state = self.init_state(dim=batch_size)
        #packed = pack_padded_sequence(embeddings, lengths, batch_first=True, enforce_sorted=False)
        #_, state = self.rnn(packed, state)
        #state = state.view(batch_size, self.hidden_size * self.hidden_layers)
        mean = self.rnn2mean(vec_frag_arr)
        logv = self.rnn2logv(vec_frag_arr)
        std = torch.exp(0.5 * logv)
        z = self.sample_normal(dim=batch_size)
        latent_sample = z * std + mean
        return latent_sample, mean, std

    def sample_normal(self, dim):
        z = torch.randn((self.hidden_layers, dim, self.latent_size))
        return Variable(z).cuda() if self.use_gpu else Variable(z)

    def init_state(self, dim):
        state = torch.zeros((self.hidden_layers, dim, self.hidden_size))
        return Variable(state).cuda() if self.use_gpu else Variable(state)


### MLP predictor class
class MLP(nn.Module):
    def __init__(self, latent_size, use_gpu):
        super(MLP, self).__init__()
        self.latent_size = latent_size
        self.use_gpu = use_gpu
        #self.relu = nn.ReLU()
        #self.softplus = nn.Softplus()
        #self.linear1 = nn.Linear(self.latent_size, 16)
        #self.linear2 = nn.Linear(16, 8)
        #self.linear3 = nn.Linear(8, 1)

        self.layers_qed = nn.Sequential(
            nn.Linear(latent_size, 32),
            nn.ReLU(),
            #nn.Dropout(0.2),
            nn.Linear(32, 1),
            nn.ReLU(),
            #nn.Dropout(0.2),
            nn.Sigmoid()
        )
        self.layers_logp = nn.Sequential(
            nn.Linear(latent_size, 200),
            nn.ReLU(),
            #nn.Dropout(0.2),
            nn.Linear(200, 100),
            nn.ReLU(),
            #nn.Dropout(0.2),
            nn.Linear(100, 1)
            #nn.Sigmoid()
        )
        self.layers_sas = nn.Sequential(
            nn.Linear(latent_size, 200),
            nn.ReLU(),
            #nn.Dropout(0.2),
            nn.Linear(200, 100),
            nn.ReLU(),
            #nn.Dropout(0.2),
            nn.Linear(100, 1)
            #nn.ReLU(),
            #nn.Dropout(0.2),
            #nn.Sigmoid()
        )

    def forward(self, x):
        #x = self.linear1(x)
        #x = self.relu(x)
        #x = self.linear2(x)
        #x = self.relu(x)
        #x = self.linear3(x)
        #y_qed = self.layers_qed(x)
        y_logp = self.layers_logp(x)
        y_sas = self.layers_sas(x)
        #return x.view(-1)
        #return y_logp.view(-1).cuda()
        return y_logp.view(-1).cuda(), y_sas.view(-1).cuda()

class Decoder(nn.Module):
    def __init__(self, embed_size, latent_size, hidden_size,
                 hidden_layers, dropout, output_size):
        super().__init__()
        self.embed_size = embed_size
        self.latent_size = latent_size
        self.hidden_size = hidden_size
        self.hidden_layers = hidden_layers
        self.output_size = output_size
        self.dropout = dropout

        self.rnn = nn.GRU(
            input_size=self.embed_size,
            hidden_size=self.hidden_size,
            num_layers=self.hidden_layers,
            dropout=self.dropout,
            batch_first=True)

        self.rnn2out = nn.Linear(
            in_features=hidden_size,
            out_features=output_size)

    def forward(self, embeddings, state, lengths):
        batch_size = embeddings.size(0)
        packed = pack_padded_sequence(embeddings, lengths, batch_first=True, enforce_sorted=True)
        hidden, state = self.rnn(packed, state)
        state = state.view(self.hidden_layers, batch_size, self.hidden_size)
        hidden, _ = pad_packed_sequence(hidden, batch_first=True)
        output = self.rnn2out(hidden)
        return output, state

class Frag2Mol(nn.Module):
    def __init__(self, config, vocab):
        super().__init__()
        self.config = config
        self.vocab = vocab
        self.input_size = vocab.get_size()
        self.embed_size = config.get('embed_size')
        self.hidden_size = config.get('hidden_size')
        self.hidden_layers = config.get('hidden_layers')
        self.latent_size = config.get('latent_size')
        self.dropout = config.get('dropout')
        self.use_gpu = config.get('use_gpu')

        embeddings = self.load_embeddings()
        self.embedder = nn.Embedding.from_pretrained(embeddings)

        self.latent2rnn = nn.Linear(
            in_features=self.latent_size,
            out_features=self.hidden_size)

        self.encoder = Encoder(
            input_size=self.input_size,
            embed_size=self.embed_size,
            hidden_size=self.hidden_size,
            hidden_layers=self.hidden_layers,
            latent_size=self.latent_size,
            dropout=self.dropout,
            use_gpu=self.use_gpu)

        self.decoder = Decoder(
            embed_size=self.embed_size,
            latent_size=self.latent_size,
            hidden_size=self.hidden_size,
            hidden_layers=self.hidden_layers,
            dropout=self.dropout,
            output_size=self.input_size)
        ### MLP predictor initialise
        self.mlp = MLP(
            latent_size=self.latent_size,
            use_gpu=self.use_gpu
        )

    def forward(self, inputs, lengths):
        batch_size = inputs.size(0)
        #print(inputs)
        vec_frag_arr = torch.zeros(100)
        for idx2, (tgt_i) in enumerate(inputs):
            vec_frag_sum = torch.sum(self.embedder(tgt_i[tgt_i > 2]), 0)
            if idx2 == 0:
                vec_frag_arr = vec_frag_sum
            else:
                vec_frag_arr = torch.vstack((vec_frag_arr, vec_frag_sum))
        embeddings = self.embedder(inputs)
        #print(vec_frag_arr)
        #print(vec_frag_arr.size())
        #embeddings1 = F.dropout(embeddings, p=self.dropout, training=self.training)
        #vec_frag_sum = np.sum(embeddings, 0)
        #print(vec_frag_sum)
        #print(vec_frag_sum.shape())
        z, mu, sigma = self.encoder(vec_frag_arr)
        ### Add Property Predictor
        #mu_norm = F.normalize(mu)
        #pred_1 = self.mlp(Variable(mu_norm[0, :, :]))
        #pred_2 = self.mlp(Variable(mu_norm[1, :, :]))
        #pred = (pred_1 + pred_2)/2
        pred_logp, pred_sas = self.mlp(Variable(mu))
        #pred_logp = self.mlp(Variable(mu))
        ###
        state = self.latent2rnn(z)
        state = state.view(self.hidden_layers, batch_size, self.hidden_size)
        embeddings2 = F.dropout(embeddings, p=self.dropout, training=self.training)
        output, state = self.decoder(embeddings2, state, lengths)
        #return output, mu, sigma
        ### Teddy Code
        #return output, mu, sigma, z, pred_logp
        return output, mu, sigma, z, pred_logp, pred_sas

    def load_embeddings(self):
        filename = f'emb_{self.embed_size}.dat'
        path = self.config.path('config') / filename
        embeddings = np.loadtxt(path, delimiter=",")
        return torch.from_numpy(embeddings).float()

    def calc_mi(self, inputs, lengths):
        """Approximate the mutual information between x and z
        I(x, z) = E_xE_{q(z|x)}log(q(z|x)) - E_xE_{q(z|x)}log(q(z))
        Returns: Float
        """
        # [x_batch, nz]
        batch_size = inputs.size(0)
        embeddings = self.embedder(inputs)
        #print(embeddings)
        embeddings1 = F.dropout(embeddings, p=self.dropout, training=self.training)
        z_samples, mu, sigma = self.encoder(inputs, embeddings1, lengths)
        #_, mu, logvar, z_samples = self.encoder_sample(x)
        #sigma = torch.exp(0.5 * logv)
        logvar = 2 * torch.log(sigma)
        x_batch, nz = mu.size()
        # E_{q(z|x)}log(q(z|x)) = -0.5*nz*log(2*\pi) - 0.5*(1+logvar).sum(-1)
        neg_entropy = (-0.5 * nz * math.log(2 * math.pi)- 0.5 * (1 + logvar).sum(-1)).mean()
        var = logvar.exp()
        # (z_batch, 1, nz)
        z_samples = z_samples.unsqueeze(1)
        # (1, x_batch, nz)
        mu = mu.unsqueeze(0)
        logvar = logvar.unsqueeze(0)
        # (z_batch, x_batch, nz)
        dev = z_samples - mu
        # (z_batch, x_batch)
        log_density = -0.5 * ((dev ** 2) / var).sum(dim=-1) - \
            0.5 * (nz * math.log(2 * math.pi) + logvar.sum(-1))
        # log q(z): aggregate posterior
        log_qz = self.log_sum_exp(log_density, dim=1) - math.log(x_batch)
        #print(log_qz)
        #print(log_qz.size())
        return (neg_entropy - torch.flatten(log_qz).mean(-1)).item()

    def log_sum_exp(self, value, dim=None, keepdim=False):
        """Numerically stable implementation of the operation
        value.exp().sum(dim, keepdim).log()
        """
        if dim is not None:
            m, _ = torch.max(value, dim=dim, keepdim=True)
            value0 = value - m
            if keepdim is False:
                m = m.squeeze(dim)
            return m + torch.log(torch.sum(torch.exp(value0), dim=dim, keepdim=keepdim))
        else:
            m = torch.max(value)
            sum_exp = torch.sum(torch.exp(value - m))
            return m + torch.log(sum_exp)


class Loss(nn.Module):
    def __init__(self, config, vocab, pad):
        super().__init__()
        self.config = config
        self.pad = pad
        ## Insert loss function
        self.loss_fn = nn.MSELoss()
        self.vocab = vocab

    def forward(self, output, target, mu, sigma, pred_logp, labels_logp, pred_sas, labels_sas, epoch, tgt_str_lst,penalty_weights, beta):
        #def forward(self, output, target, mu, sigma, pred_logp, labels_logp, labels_sas, epoch, tgt_str_lst,penalty_weights, beta):
        output = F.log_softmax(output, dim=1)
        #output_mse = F.softmax(output, dim=1)
        #print("molecules logP", labels)
        #print("Original Output Size:", output.size())
        #print("Original Output Sample:", output)
        # flatten all predictions and targets
        #print("Original translated Target Size:", target.size())
        #print("Original translated Target Sample:", target)
        #print("Original Target Sample:", tgt_str_lst)
        target_str_lst = [self.vocab.translate(target_i) for target_i in target.cpu().detach().numpy()]
        #print("target: ", target_str_lst)
        #print([[penalty_weights[tgt_str_lst_i].values] for tgt_str_lst_i in tgt_str_lst])
        target_pen_weight_lst = []
        for target_i in target.cpu().detach().numpy():
            target_pen_weight_i = penalty_weights[self.vocab.translate(target_i)].values
            if len(target_pen_weight_i) < target.size(1):
                pad_len = target.size(1) - len(target_pen_weight_i)
                target_pen_weight_pad_i = np.pad(target_pen_weight_i, (0, pad_len), 'constant')
                target_pen_weight_pad_i[len(target_pen_weight_i)] = penalty_weights.values[-1]
                target_pen_weight_lst.append(target_pen_weight_pad_i)
            else:
                target_pen_weight_lst.append(target_pen_weight_i)
        #target_pen_weight_lst = [penalty_weights[self.vocab.translate(target_i)].values for target_i in target.cpu().detach().numpy()]
        #print("penalty: ", torch.Tensor(target_pen_weight_lst).view(-1))
        target_pen_weight = torch.Tensor(target_pen_weight_lst).view(-1)
        #print("target 2: ", [tgt_str_lst[i] for i in range(len(tgt_str_lst))])
        target = target.view(-1)
        #target_str_lst = [self.vocab.translate(target.cpu().detach().numpy())]
        #print("target: ", target_str_lst)
        #print("Flattened translated Target Size:", target.size())
        #print("Flattened translated Target sample:", target)
        output = output.view(-1, output.size(2))
        #output_mse = output_mse.view(-1, output_mse.size(2))
        #print("Flattened Output Size:", output.size())
        #print("Flattened Output sample:", output)

        # create a mask filtering out all tokens that ARE NOT the padding token
        mask = (target > self.pad).float()
        #print("Padding Mask:", mask)

        # count how many tokens we have
        nb_tokens = int(torch.sum(mask).item())

        # pick the values for the label and zero out the rest with the mask
        #output = output[range(output.size(0)), target] * target_pen_weight.cuda() * mask
        output = output[range(output.size(0)), target] * mask

        #output_mse = output_mse[range(output_mse.size(0)), target] * mask
        #print("output -log:", output)
        #print("output probaility:", output_mse)
        #print("target",target)

        # compute cross entropy loss which ignores all <PAD> tokens
        CE_loss = -torch.sum(output) / nb_tokens
        #CE_loss = -torch.sum(output)

        #try mse ***Teddy***
        #CE_loss = F.mse_loss(output, torch.zeros(len(output)).cuda())

        # compute KL Divergence
        KL_loss = -0.5 * torch.sum(1 + sigma - mu.pow(2) - sigma.exp())
        # alpha = (epoch + 1)/(self.config.get('num_epochs') + 1)
        # return alpha * CE_loss + (1-alpha) * KL_loss

        ### Compute prediction loss
        #pred_qed_loss = F.binary_cross_entropy(pred_qed.type(torch.float64), labels_qed.cuda())
        pred_logp_loss = F.mse_loss(pred_logp.type(torch.float64), labels_logp.cuda())
        pred_sas_loss = F.mse_loss(pred_sas.type(torch.float64), labels_sas.cuda())
        if KL_loss > 10000000:
            total_loss = CE_loss + pred_logp_loss + pred_sas_loss
            #total_loss = CE_loss + pred_logp_loss
            #total_loss = CE_loss
        else:
            total_loss = CE_loss + beta[epoch]*KL_loss + pred_logp_loss + pred_sas_loss
            #total_loss = CE_loss + beta[epoch]*KL_loss + pred_logp_loss
            #total_loss = CE_loss + pred_logp_loss + pred_sas_loss
            #total_loss = CE_loss + pred_logp_loss
            #total_loss = CE_loss
        #return total_loss, CE_loss, KL_loss, pred_logp_loss
        return total_loss, CE_loss, KL_loss, pred_sas_loss, pred_logp_loss