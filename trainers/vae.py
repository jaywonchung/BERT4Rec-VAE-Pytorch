from .base import AbstractTrainer
from .utils import recalls_and_ndcgs_for_ks
from loggers import MetricGraphPrinter

import torch
import torch.nn as nn
import torch.nn.functional as F


class VAETrainer(AbstractTrainer):
    def __init__(self, args, model, train_loader, val_loader, test_loader, export_root):
        super().__init__(args, model, train_loader, val_loader, test_loader, export_root)

        # Finding or using given optimal beta
        self.__beta = 0.0
        self.finding_best_beta = args.find_best_beta
        self.anneal_amount = 1.0 / args.total_anneal_steps
        if self.finding_best_beta:
            self.current_best_metric = 0.0
            self.anneal_cap = 1.0
        else:
            self.anneal_cap = args.anneal_cap

    @classmethod
    def code(cls):
        return 'vae'

    def add_extra_loggers(self):
        cur_beta_logger = MetricGraphPrinter(self.writer, key='cur_beta', graph_name='Beta', group_name='Train')
        self.train_loggers.append(cur_beta_logger)

        if self.args.find_best_beta:
            best_beta_logger = MetricGraphPrinter(self.writer, key='best_beta', graph_name='Best_beta', group_name='Validation')
            self.val_loggers.append(best_beta_logger)

    def log_extra_train_info(self, log_data):
        log_data.update({'cur_beta': self.__beta})
    
    def log_extra_val_info(self, log_data):
        if self.finding_best_beta:
            log_data.update({'best_beta': self.best_beta})

    @property
    def beta(self):
        if self.model.training:
            self.__beta = min(self.__beta + self.anneal_amount, self.anneal_cap)
        return self.__beta

    def calculate_loss(self, batch):
        input_x = torch.stack(batch)
        recon_x, mu, logvar = self.model(input_x)
        CE = -torch.mean(torch.sum(F.log_softmax(recon_x, 1) * input_x, -1))
        KLD = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))

        return CE + self.beta * KLD

    def calculate_metrics(self, batch):
        inputs, labels = batch
        logits, _, _ = self.model(inputs)
        logits[inputs!=0] = -float("Inf") # IMPORTANT: remove items that were in the input
        metrics = recalls_and_ndcgs_for_ks(logits, labels, self.metric_ks)

        # Annealing beta
        if self.finding_best_beta:
            if self.current_best_metric < metrics[self.best_metric]:
                self.current_best_metric = metrics[self.best_metric]
                self.best_beta = self.__beta

        return metrics
