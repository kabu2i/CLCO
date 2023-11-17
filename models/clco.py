import torch
import torch.nn as nn
import torch.nn.functional as F

import models


class CLCO_Model(nn.Module):
    def __init__(self, args):
        '''
        CLCO model.

        Args:
            args (dict): Program arguments/commandline arguments.
        '''
        super(CLCO_Model, self).__init__()

        self.args = args
        
        # Load model
        self.encoder = getattr(models, args.backbone)()

        # Add the mlp head
        args.in_channels = self.encoder.fc.in_features
        self.encoder.fc = models.mlphead(args)

    def clco_logits(self, f, labels):
        '''
        Compute the similarity logits between positive samples and 
        positve to all negatives in the memory.

        args:
            f (Tensor): Feature reprentations.

            labels (Tensor): pseudo-labels generated by clustering.

        returns:
            logits (Tensor): Positve and negative logits computed 
                as by InfoNCE loss.
            
            loss (Tensor): multilabel categorical crossentropy loss
                and supervised contrastive loss.

            labels (Tensor): Labels of the positve and negative logits 
                to be used in softmax cross entropy.
        '''

        # Normalize the feature representations
        f = nn.functional.normalize(f, dim=1)

        sim = torch.matmul(f, f.T)

        # randomly generate cluster pseudo-labels
        if self.args.random_pseudo_label:
            labels = torch.randint(1, 12, (sim.shape[0] // 2,)) * torch.randint(0, 2, (sim.shape[0] // 2,))
        
        labels = labels.to(self.args.device)
        if self.args.views == 2:
            labels = labels.repeat(2)
        
        # mask for clustering positives
        labels2 = labels.clone()
        labels2[labels2 == 0] = -1
        cluster_pos_mask = (labels.unsqueeze(0) == labels2.unsqueeze(1))
        # mask for removing the similarity of between sample itself
        sample_self_mask = torch.eye(sim.shape[0], dtype=torch.bool).to(self.args.device)
        # mask for positive samples on instance level
        instance_labels = torch.cat([torch.arange(sim.shape[0] // 2, dtype=torch.int64) for i in range(2)], dim=0).to(self.args.device)
        instance_mask = (instance_labels.unsqueeze(0) == instance_labels.unsqueeze(1))
        instance_mask = instance_mask[~sample_self_mask].view(sim.shape[0], -1)
        if self.args.views == 1:
            instance_mask = torch.zeros_like(instance_mask)
        cluster_pos_mask = cluster_pos_mask[~sample_self_mask].view(sim.shape[0], -1)
        sim = sim[~sample_self_mask].view(sim.shape[0], -1)

        if self.args.loss == "mpc":
            pos, neg = self.multipositive_crossentropy(sim, cluster_pos_mask, instance_mask)

            logits = torch.cat((pos, neg), dim=1)

            logits /= self.args.temperature

            # Create labels, first logit is postive, all others are negative
            labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.args.device)

            return logits, labels
        elif self.args.loss == "supcon":
            mask = cluster_pos_mask + instance_mask
            logits = sim / self.args.temperature
            # compute log_prob
            exp_logits = torch.exp(logits) * ~mask
            log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

            # compute mean of log-likelihood over positive
            mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
            loss = - mean_log_prob_pos.mean()
            return loss, None
        elif self.args.loss == "mcc":
            if self.args.select_positive == 'combine_inst_dis':
                mask = cluster_pos_mask + instance_mask
            if self.args.select_positive == 'only_inst_dis':
                mask = instance_mask
            loss = self.multilabel_categorical_crossentropy(sim, mask.float())
            return loss, None
        else:
            raise 'No such loss function!'
    
    def multipositive_crossentropy(self, sim, cluster_pos_mask, instance_mask):
        # count the total number of positives pairs in each batch
        num_pos = torch.sum(cluster_pos_mask, dim=1)
        if self.args.views == 1:
            num_pos2 = num_pos.clone()
        # if there are no positive samples pairs, set the number of positive pairs to 1
        num_pos[num_pos == 0] = 1
        if self.args.select_positive == 'cluster_positive':
            # find the mask with pseudo-labels 0/-1, which set as negtive sample out of 
            # clusters, and set the two views similarity to 1.
            non_cluster_mask = (instance_mask + cluster_pos_mask) ^ cluster_pos_mask
            sim[non_cluster_mask] = 1
        mask = instance_mask + cluster_pos_mask
        if self.args.select_positive == 'only_inst_dis':
            mask = instance_mask
        pos = sim[mask].view(-1, 1)
        neg = sim * (~mask)
        # use the recorded num_pos to copy the corresponding negative pairs for each positive pairs
        neg = torch.repeat_interleave(sim, num_pos, 0)
        # adjust the positive pairs
        if self.args.views == 1:
            front, index_front = 0, 0
            new_pos = torch.zeros(neg.shape[0]).reshape(-1,1).to(self.args.device)
            for i in num_pos2:
                if(i):
                    new_pos[front: front+i-1] = pos[index_front: index_front+i-1]
                    front += i
                    index_front += i
                else:
                    new_pos[front] = 1
                    front += 1
            pos = new_pos
        return pos, neg

    def multilabel_categorical_crossentropy(self, y_pred, y_true):
        y_pred = (1 - 2 * y_true) * y_pred  # -1 -> pos classes, 1 -> neg classes
        y_pred_neg = y_pred - y_true * 1e12  # mask the pred outputs of pos classes
        y_pred_pos = y_pred - (1 - y_true) * 1e12 # mask the pred outputs of neg classes
        zeros = torch.zeros_like(y_pred[..., :1])
        y_pred_neg = torch.cat([y_pred_neg, zeros], dim=-1)
        y_pred_pos = torch.cat([y_pred_pos, zeros], dim=-1)
        neg_loss = torch.logsumexp(y_pred_neg, dim=-1)
        pos_loss = torch.logsumexp(y_pred_pos, dim=-1)
        return (neg_loss + pos_loss).mean()

    def forward(self, x_q, x_k, labels):
        if self.args.views == 1:
            features = self.encoder(x_q)
        elif self.args.views == 2:
            x = torch.cat((x_q, x_k), dim=0)
            features = self.encoder(x)
        else:
            raise("No implementation on such view setting.")

        # Compute the logits/loss for the contrastive loss.
        logits, labels = self.clco_logits(features, labels)
        return logits, labels