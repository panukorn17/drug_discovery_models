import numpy as np

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
            in_features=self.hidden_size * self.hidden_layers,
            out_features=self.latent_size)

        self.rnn2logv = nn.Linear(
            in_features=self.hidden_size * self.hidden_layers,
            out_features=self.latent_size)

    def forward(self, inputs, embeddings, lengths):
        batch_size = inputs.size(0)
        state = self.init_state(dim=batch_size)
        packed = pack_padded_sequence(embeddings, lengths, batch_first=True)
        _, state = self.rnn(packed, state)
        state = state.view(batch_size, self.hidden_size * self.hidden_layers)
        mean = self.rnn2mean(state)
        logv = self.rnn2logv(state)
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

        self.layers = nn.Sequential(
            nn.Linear(latent_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        #x = self.linear1(x)
        #x = self.relu(x)
        #x = self.linear2(x)
        #x = self.relu(x)
        #x = self.linear3(x)
        x = self.layers(x)
        #return x.view(-1)
        return x.view(-1).cuda() if self.use_gpu else x.view(-1)

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
        packed = pack_padded_sequence(embeddings, lengths, batch_first=True)
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
        embeddings = self.embedder(inputs)
        embeddings1 = F.dropout(embeddings, p=self.dropout, training=self.training)
        z, mu, sigma = self.encoder(inputs, embeddings1, lengths)
        ### Add Property Predictor
        z_norm = F.normalize(z)
        pred_1 = self.mlp(Variable(z_norm[0, :, :]))
        pred_2 = self.mlp(Variable(z_norm[1, :, :]))
        pred = (pred_1 + pred_2)/2
        ###
        state = self.latent2rnn(z)
        state = state.view(self.hidden_layers, batch_size, self.hidden_size)
        embeddings2 = F.dropout(embeddings, p=self.dropout, training=self.training)
        output, state = self.decoder(embeddings2, state, lengths)
        #return output, mu, sigma
        ### Teddy Code
        return output, mu, sigma, z, pred

    def load_embeddings(self):
        filename = f'emb_{self.embed_size}.dat'
        path = self.config.path('config') / filename
        embeddings = np.loadtxt(path, delimiter=",")
        return torch.from_numpy(embeddings).float()


class Loss(nn.Module):
    def __init__(self, config, pad):
        super().__init__()
        self.config = config
        self.pad = pad
        ## Insert loss function
        self.loss_fn = nn.MSELoss()

    def forward(self, output, target, mu, sigma, pred, labels, epoch, frament_count_df):
        output = F.log_softmax(output, dim=1)
        print(output)
        # flatten all predictions and targets
        target = target.view(-1)
        output = output.view(-1, output.size(2))
        print(output)

        # create a mask filtering out all tokens that ARE NOT the padding token
        mask = (target > self.pad).float()

        # count how many tokens we have
        nb_tokens = int(torch.sum(mask).item())

        # pick the values for the label and zero out the rest with the mask
        output = output[range(output.size(0)), target] * mask
        print(output)

        # compute cross entropy loss which ignores all <PAD> tokens
        CE_loss = -torch.sum(output) / nb_tokens

        # compute KL Divergence
        KL_loss = -0.5 * torch.sum(1 + sigma - mu.pow(2) - sigma.exp())
        # alpha = (epoch + 1)/(self.config.get('num_epochs') + 1)
        # return alpha * CE_loss + (1-alpha) * KL_loss

        ### Compute prediction loss
        pred_loss = F.mse_loss(pred.type(torch.float64), labels.cuda())
        return CE_loss + KL_loss + pred_loss, CE_loss, KL_loss, pred_loss
