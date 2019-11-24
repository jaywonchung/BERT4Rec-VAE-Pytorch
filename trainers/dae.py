from .base import AbstractTrainer
from .utils import recalls_and_ndcgs_for_ks

import torch
import torch.nn as nn
import torch.nn.functional as F


class DAETrainer(AbstractTrainer):
    def __init__(self, args, model, train_loader, val_loader, test_loader, export_root):
        super().__init__(args, model, train_loader, val_loader, test_loader, export_root)

    @classmethod
    def code(cls):
        return 'dae'

    def add_extra_loggers(self):
        pass

    def log_extra_train_info(self, log_data):
        pass

    def log_extra_val_info(self, log_data):
        pass

    def calculate_loss(self, batch):
        input_x = torch.stack(batch)
        recon_x = self.model(input_x)
        CE = -torch.mean(torch.sum(F.log_softmax(recon_x, 1) * input_x, -1))
        return CE

    def calculate_metrics(self, batch):
        inputs, labels = batch
        logits = self.model(inputs)
        logits[inputs!=0] = -float("Inf") # IMPORTANT: remove items that were in the input
        metrics = recalls_and_ndcgs_for_ks(logits, labels, self.metric_ks)
        return metrics
