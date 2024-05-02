import torch
import torch.nn as nn
import utils as ut
from torch.nn import functional as F


# contrastive loss
class CentroidsConLoss(torch.nn.Module):
    def __init__(self, temperature):
        super(CentroidsConLoss, self).__init__()
        self.temperature = temperature
        self.cross_entropy_loss = nn.CrossEntropyLoss()
    
    def forward(self, features, labels, centroids):
        features = F.normalize(features, dim=1)
        centroids = F.normalize(centroids, dim=1)

        logits = features @ centroids.T
        logits = logits / self.temperature

        loss = self.cross_entropy_loss(logits, labels)
        return loss


# contrastive loss with LRS
class WCentroidsConLoss(torch.nn.Module):
    def __init__(self, temperature):
        super(WCentroidsConLoss, self).__init__()
        self.temperature = temperature
        self.cross_entropy_loss = nn.CrossEntropyLoss(reduction='none')
    
    def forward(self, features, labels, centroids, w):
        features = F.normalize(features, dim=1)
        centroids = F.normalize(centroids, dim=1)

        logits = features @ centroids.T
        logits = logits / self.temperature

        loss = (w * self.cross_entropy_loss(logits, labels)).mean()
        
        return loss
    

# ent loss
class EntropyMinLoss(torch.nn.Module):
    def __init__(self, temperature):
        super(EntropyMinLoss, self).__init__()
        self.temperature = temperature 

    def forward(self, features, source_centroids, text_centroids):
        features = F.normalize(features, dim=1)
        source_centroids = F.normalize(source_centroids, dim=1)
        text_centroids = F.normalize(text_centroids, dim=1)
        
        logits1 = features @ source_centroids.T
        logits2 = features @ text_centroids.T

        logits1 = logits1 / self.temperature
        logits2 = logits2 / self.temperature

        loss1 = -torch.mean(torch.sum(F.log_softmax(logits1, dim=1) * F.softmax(logits1, dim=1), dim=1))
        loss2 = -torch.mean(torch.sum(F.log_softmax(logits2, dim=1) * F.softmax(logits2, dim=1), dim=1))

        loss = (loss1 + loss2) / 2
        
        return loss
    

# disc loss
class DiscriminationLoss(nn.Module):
    def __init__(self, temperature):
        super(DiscriminationLoss, self).__init__()
        # self.temperature = temperature
        self.temperature = 0.07

    def forward(self, source_centroids, text_centroids):
        source_centroids = F.normalize(source_centroids, dim=1)
        text_centroids = F.normalize(text_centroids, dim=-1)

        class_num = source_centroids.shape[0]
        labels = torch.arange(0, class_num).to(source_centroids.device)
        text_labels = labels.to(source_centroids.device)
        
        cl_centroids = torch.cat([source_centroids, text_centroids])
        cl_labels = torch.cat([labels, labels])
        
        batch_size = text_centroids.shape[0] 
        batch_size_N = cl_centroids.shape[0] 
        mask = torch.eq(text_labels.unsqueeze(1).expand(batch_size, batch_size_N), \
            cl_labels.unsqueeze(0).expand(batch_size,batch_size_N))
        
        logits = torch.div(torch.matmul(text_centroids, cl_centroids.T), self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits = logits - logits_max.detach() 
        exp_logits = torch.exp(logits) 
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True)) 
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        loss = - mean_log_prob_pos.mean()
        return loss
