import time
import numpy as np
import pandas as pd

import torch
from torch.autograd import Variable
from torch.nn.utils import clip_grad_norm_
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.nn import functional as F
from tqdm import tqdm
from tensorboardX import SummaryWriter


from .model import Loss, Frag2Mol
from .sampler import Sampler
from utils.filesystem import load_dataset
from utils.postprocess import score_samples
from molecules.fragmentation import reconstruct

### Import dataset
from learner.dataset import FragmentDataset

SCORES = ["validity", "novelty", "uniqueness"]


def save_ckpt(trainer, epoch, filename):
    path = trainer.config.path('ckpt') / filename
    torch.save({
        'epoch': epoch,
        'best_loss': trainer.best_loss,
        'losses': trainer.losses,
        'best_score': trainer.best_score,
        'scores': trainer.scores,
        'model': trainer.model.state_dict(),
        'optimizer': trainer.optimizer.state_dict(),
        'scheduler': trainer.scheduler.state_dict(),
        'criterion': trainer.criterion.state_dict()
    }, path)


def load_ckpt(trainer, last=True):
    filename = 'last.pt' if last is True else 'best_loss.pt'
    path = trainer.config.path('ckpt') / filename

    if trainer.config.get('use_gpu') is False:
        checkpoint = torch.load(
            path, map_location=lambda storage, loc: storage)
    else:
        checkpoint = torch.load(path)

    print(f"loading {filename} at epoch {checkpoint['epoch']+1}...")

    trainer.model.load_state_dict(checkpoint['model'])
    trainer.optimizer.load_state_dict(checkpoint['optimizer'])
    trainer.scheduler.load_state_dict(checkpoint['scheduler'])
    trainer.criterion.load_state_dict(checkpoint['criterion'])
    trainer.best_loss = checkpoint['best_loss']
    trainer.losses = checkpoint['losses']
    trainer.best_score = checkpoint['best_score']
    trainer.scores = checkpoint['scores']
    return checkpoint['epoch']


def get_optimizer(config, model):
    return Adam(model.parameters(), lr=config.get('optim_lr'))


def get_scheduler(config, optimizer):
    return StepLR(optimizer,
                  step_size=config.get('sched_step_size'),
                  gamma=config.get('sched_gamma'))


def dump(config, losses, CE_loss, KL_loss, pred_sas_loss, pred_logP_loss, beta_list, scores):
    #def dump(config, losses, CE_loss, KL_loss, pred_logP_loss, beta_list, scores):
    df = pd.DataFrame(list(zip(losses, CE_loss, KL_loss, pred_sas_loss, pred_logP_loss, beta_list)),
                      columns=["Total loss", "CE loss", "KL loss", "pred sas loss", "pred logP loss", "beta"])
    #df = pd.DataFrame(list(zip(losses, CE_loss, KL_loss, pred_logP_loss, beta_list)),
    #                  columns=["Total loss", "CE loss", "KL loss", "pred logP loss", "beta"])
    filename = config.path('performance') / "loss.csv"
    df.to_csv(filename)

    if scores != []:
        df = pd.DataFrame(scores, columns=SCORES)
        filename = config.path('performance') / "scores.csv"
        df.to_csv(filename)


class TBLogger:
    def __init__(self, config):
        self.config = config
        self.writer = SummaryWriter(config.path('tb').as_posix())
        config.write_summary(self.writer)

    def log(self, name, value, epoch):
        self.writer.add_scalar(name, value, epoch)


class Trainer:
    @classmethod
    def load(cls, config, vocab, last):
        trainer = Trainer(config, vocab)
        epoch = load_ckpt(trainer, last=last)
        return trainer, epoch

    def __init__(self, config, vocab):
        self.config = config
        self.vocab = vocab

        self.model = Frag2Mol(config, vocab)
        self.optimizer = get_optimizer(config, self.model)
        #torch.optim.Adam(self.model.parameters(), lr=0.0001)
        self.scheduler = get_scheduler(config, self.optimizer)
        self.criterion = Loss(config, vocab, pad=vocab.PAD)

        if self.config.get('use_gpu'):
            self.model = self.model.cuda()

        self.losses = []
        self.CE_loss = []
        self.KL_loss = []
        self.pred_logP_loss = []
        self.pred_sas_loss = []
        self.beta_list = []
        self.mutual_information = []
        self.best_loss = np.float('inf')
        self.scores = []
        self.best_score = - np.float('inf')

    def _train_epoch(self, epoch, loader, penalty_weights, beta):
        ###Teddy Code
        dataset = FragmentDataset(self.config)
        ###
        self.model.train()
        epoch_loss = 0
        epoch_CE_loss = 0
        epoch_KL_loss = 0
        epoch_pred_sas_loss = 0
        epoch_pred_logP_loss = 0

        if epoch > 0 and self.config.get('use_scheduler'):
            self.scheduler.step()
        #for idx, (src, tgt, lengths) in enumerate(loader):
        ### Teddy Code
        total_mutual_info = 0
        for idx, (src, tgt, lengths, data_index, tgt_str) in enumerate(loader):
            ###
            self.optimizer.zero_grad()
            #tgt_str_lst = list(tgt_str)
            #print(tgt_str_lst)
            #print([[penalty_weights[tgt_str_lst_i].values] for tgt_str_lst_i in tgt_str_lst])
            #print(penalty_weights[tgt_str])
            tgt_str_lst = [self.vocab.translate(target_i) for target_i in tgt.cpu().detach().numpy()]
            target_str_ls_2 = [" ".join(self.vocab.translate(target_i)) for target_i in tgt.cpu().detach().numpy()]
            src_str_ls_2 = [self.vocab.translate(target_i) for target_i in src.cpu().detach().numpy()]
            #print("target string list src", tgt_str_lst)
            #print("target string list tgt", target_str_ls_2)
            #print("lengths:", lengths)
            src, tgt = Variable(src), Variable(tgt)
            if self.config.get('use_gpu'):
                src = src.cuda()
                tgt = tgt.cuda()

            output, mu, sigma, z, pred_logp, pred_sas = self.model(src, lengths)
            #output, mu, sigma, z, pred_logp = self.model(src, lengths)
            #mutual_info = self.model.calc_mi(src, lengths)
            #total_mutual_info += mutual_info * src.size(0)
            #print("mu:", mu)
            #print("mu size:", mu.size())
            #print(output.size())
            ### Insert Label
            #print(data_index)
            molecules = dataset.data.iloc[list(data_index)]
            data_index_correct = [molecules[molecules['fragments'] == target_str_ls_2_i].index.values[0] for target_str_ls_2_i in target_str_ls_2]
            molecules_correct = dataset.data.iloc[data_index_correct]
            #print("molecules: ", molecules_correct)
            #rint("target string list", tgt_str_lst)
            #labels_qed = torch.tensor(molecules_correct.qed.values)
            labels_logp = torch.tensor(molecules_correct.logP.values)
            labels_sas = torch.tensor(molecules_correct.SAS.values)
            #print("labels: ", labels)
            loss, CE_loss, KL_loss, pred_sas_loss, pred_logp_loss = self.criterion(output, tgt, mu, sigma, pred_logp, labels_logp, pred_sas, labels_sas, epoch, tgt_str_lst, penalty_weights, beta)
            #loss, CE_loss, KL_loss, pred_logp_loss = self.criterion(output, tgt, mu, sigma, pred_logp, labels_logp, labels_sas, epoch, tgt_str_lst, penalty_weights, beta)
            #pred_loss.backward()
            loss.backward()
            clip_grad_norm_(self.model.parameters(),
                            self.config.get('clip_norm'))

            epoch_loss += loss.item()
            epoch_CE_loss += CE_loss.item()
            epoch_KL_loss += KL_loss.item()
            epoch_pred_sas_loss += pred_sas_loss.item()
            epoch_pred_logP_loss += pred_logp_loss.item()
            #epoch_loss += pred_loss.item()

            self.optimizer.step()
            ### Teddy Code
            if idx == 0 or idx % 200 == 0:
                print("Epoch: ", epoch, "beta: ", beta[epoch])
                print("index:", data_index)
                print("index correct: ", data_index_correct)
                print("batch ", idx, " loss: ", epoch_loss/(idx+1))
                #print("pred qed", pred_qed, " labels qed: ", labels_qed, "loss qed:", F.binary_cross_entropy(pred_qed.type(torch.float64), labels_qed.cuda()))
                print("pred logp", pred_logp, " labels logp: ", labels_logp, "loss logp:", F.mse_loss(pred_logp.type(torch.float64), labels_logp.cuda()))
                print("pred sas", pred_sas, " labels sas: ", labels_sas, "loss sas:", F.mse_loss(pred_sas.type(torch.float64), labels_sas.cuda()))
                #print("CE Loss ", CE_loss, " KL Loss: ", KL_loss, "Prediction Loss:", pred_logp_loss)
                print("CE Loss ", CE_loss, " KL Loss: ", KL_loss, "Prediction Loss:", pred_logp_loss + pred_sas_loss)
            ###
        #return epoch_loss / len(loader), epoch_CE_loss / len(loader), epoch_KL_loss / len(loader), epoch_pred_logP_loss / len(loader)
        return epoch_loss / len(loader), epoch_CE_loss / len(loader), epoch_KL_loss / len(loader), epoch_pred_sas_loss / len(loader), epoch_pred_logP_loss / len(loader)

    def _valid_epoch(self, epoch, loader):
        use_gpu = self.config.get('use_gpu')
        self.config.set('use_gpu', False)

        num_samples = self.config.get('validation_samples')
        trainer, _ = Trainer.load(self.config, self.vocab, last=True)
        sampler = Sampler(self.config, self.vocab, trainer.model)
        samples = sampler.sample(num_samples, save_results=False)
        dataset = load_dataset(self.config, kind="test")
        _, scores = score_samples(samples, dataset)

        self.config.set('use_gpu', use_gpu)
        return scores

    def log_epoch(self, start_time, epoch, epoch_loss, epoch_scores):
        end = time.time() - start_time
        elapsed = time.strftime("%H:%M:%S", time.gmtime(end))

        print(f'epoch {epoch:06d} - '
              f'loss {epoch_loss:6.4f} - ',
              end=' ')

        if epoch_scores is not None:
            for (name, score) in zip(SCORES, epoch_scores):
                print(f'{name} {score:6.4f} - ', end='')

        print(f'elapsed {elapsed}')

    def train(self, loader, start_epoch):
        num_epochs = self.config.get('num_epochs')

        logger = TBLogger(self.config)

        ### Get counts of each fragments
        dataset = FragmentDataset(self.config)
        fragment_list = []
        for frag in tqdm(dataset.data.fragments):
            fragment_list.extend(frag.split())
        fragment_counts = pd.Series(fragment_list).value_counts()
        fragment_counts = fragment_counts.append(pd.Series(len(dataset.data)))
        penalty = np.sum(np.log(fragment_counts + 1)) / np.log(fragment_counts + 1)
        penalty_weights = penalty / np.linalg.norm(penalty)
        ###
        total_mutual_info_list = []
        #KL weights anneal
        #beta = []
        #beta.extend(list(np.zeros(10)))
        #while len(beta) < num_epochs:
        #    #beta.extend(list((np.arange(11)) / 10))
        #    beta.extend(list(np.ones(10)*0.01))
        ##beta[-10:] = list(np.zeros(10))
        #beta = beta[0:num_epochs]
        beta = [0, 0, 0, 0, 0, 0.002, 0.004, 0.006, 0.008, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01]
        # to train the benchmark model uncomment below
        #beta = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        print('beta:', beta)
        self.beta_list = beta

        for epoch in range(start_epoch, start_epoch + num_epochs):
            start = time.time()

            ### Teddy code
            #mu_stack = self._train_epoch(epoch, loader)
            #return mu_stack
            ###
            epoch_loss, CE_epoch_loss, KL_epoch_loss, sas_epoch_loss, logP_epoch_loss = self._train_epoch(epoch, loader, penalty_weights, beta)
            #epoch_loss, CE_epoch_loss, KL_epoch_loss, logP_epoch_loss = self._train_epoch(epoch, loader, penalty_weights, beta)
            #self.mutual_information.append(total_mutual_info)
            self.losses.append(epoch_loss)
            self.CE_loss.append(CE_epoch_loss)
            self.KL_loss.append(KL_epoch_loss)
            self.pred_sas_loss.append(sas_epoch_loss)
            self.pred_logP_loss.append(logP_epoch_loss)
            #print("epoch: "+str(epoch)+", mutual_information: "+str(total_mutual_info))
            logger.log('loss', epoch_loss, epoch)
            save_ckpt(self, epoch, filename="last.pt")
            
            if epoch_loss < self.best_loss:
                self.best_loss = epoch_loss
                save_ckpt(self, epoch, filename=f'best_loss.pt')

            epoch_scores = None

            if epoch_loss < self.config.get('validate_after'):
                epoch_scores = self._valid_epoch(epoch, loader)
                self.scores.append(epoch_scores)

                if epoch_scores[2] >= self.best_score:
                    self.best_score = epoch_scores[2]
                    save_ckpt(self, epoch, filename=f'best_valid.pt')

                logger.log('validity', epoch_scores[0], epoch)
                logger.log('novelty', epoch_scores[1], epoch)
                logger.log('uniqueness', epoch_scores[2], epoch)

            self.log_epoch(start, epoch, epoch_loss, epoch_scores)

        dump(self.config, self.losses, self.CE_loss, self.KL_loss, self.pred_sas_loss, self.pred_logP_loss, self.beta_list, self.scores)
        #dump(self.config, self.losses, self.CE_loss, self.KL_loss, self.pred_logP_loss, self.beta_list, self.scores)
