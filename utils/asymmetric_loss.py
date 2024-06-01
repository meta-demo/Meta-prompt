import torch
import torch.nn as nn
from collections import defaultdict
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
from torch.distributions import Categorical
from sklearn.metrics import precision_score,f1_score,recall_score, accuracy_score, average_precision_score

def euclidean_dist(x, y):
    '''
    Compute euclidean distance between two tensors
    '''
    # x: N x D
    # y: M x D
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    if d != y.size(1):
        raise Exception

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    return torch.pow(x - y, 2).sum(2)


class ProtypicalLoss(nn.Module):

    def __init__(self, device_id, beta, gamma):
        super(ProtypicalLoss, self).__init__()
        self.count_lossfn = CrossEntropyLoss()
        self.device_id = device_id
        self.beta = beta
        self.gamma = gamma

    def forward(self, image_features, text_features, count_outputs, weights, support_size, target):
        support_features = image_features[:support_size]
        query_features = image_features[support_size:]
        query_ids = target[support_size:]
        

        # consider support samples may have multiple labels
        support_dict = defaultdict(list)
        for i in range(support_size):
            label = target[i]
            assert support_size == label.shape[0], f"support_size ({support_size}) must equal to label_size ({label.shape[0]})"
            for j in range(label.shape[0]):
                if label[j] == 1:
                    support_dict[j].append(image_features[i])
        support_prototypes = []
        
        for i in range(support_size):
            sample_embs = torch.stack(support_dict[i])
            support_prototypes.append(sample_embs.mean(dim=0))

        support_prototypes = torch.stack(support_prototypes).unsqueeze(dim=1)
        text_features = text_features.unsqueeze(dim=1)
        label_features = torch.cat((support_prototypes, text_features), dim=1)
        prototypes = label_features.mean(dim=1)

       

        dists = euclidean_dist(query_features, prototypes)
        log_p_y = F.log_softmax(-dists, dim=1) # num_query x num_class
        loss = - query_ids * log_p_y
        loss = loss.mean()
        
        dist = Categorical(weights)
        entrpy_loss = dist.entropy()
        entrpy_loss = entrpy_loss.mean()
       
    
        # count loss
        labels_count = target.sum(dim=1)-1
        four = torch.ones_like(labels_count)*3
        labels_count = torch.where(labels_count > 3, four, labels_count)
        count_loss = self.count_lossfn(count_outputs, labels_count)

        
        all_loss = loss + self.beta * count_loss + self.gamma * entrpy_loss
       

        # multi
        _, count_pred  = torch.max(count_outputs, 1, keepdim=True)
        labels_count = labels_count.cpu().detach()
        count_pred = count_pred.cpu().detach()
        c_acc = accuracy_score(labels_count, count_pred)
        query_count = count_pred[support_size:]

        
      

        
        sorts, indices = torch.sort(log_p_y, descending=True)  
        x = []
        for i, t in enumerate(query_count):
            x.append(log_p_y[i][indices[i][query_count[i][0]]])

        device = torch.device("cuda", self.device_id)
        x = torch.tensor(x).view(log_p_y.shape[0], 1).to(device)
        one = torch.ones_like(log_p_y)
        zero = torch.zeros_like(log_p_y)
        y_pred = torch.where(log_p_y >= x, one, log_p_y)
        y_pred = torch.where(y_pred < x, zero, y_pred)
        
        target_mode = 'macro'

        query_ids = query_ids.cpu().detach()
        y_pred = y_pred.cpu().detach()
        macro_p = precision_score(query_ids, y_pred, average=target_mode)
        macro_r = recall_score(query_ids, y_pred, average=target_mode)
        macro_f = f1_score(query_ids, y_pred, average=target_mode)
        acc = accuracy_score(query_ids, y_pred)

        micro_p = precision_score(query_ids, y_pred, average='micro')
        micro_r = recall_score(query_ids, y_pred, average='micro')
        micro_f = f1_score(query_ids, y_pred, average='micro')

        pred_logits = F.softmax(-dists, dim=1).cpu().detach()
        ma_ap = average_precision_score(query_ids.cpu().numpy(), pred_logits.cpu().numpy(), average='macro')
        mi_ap = average_precision_score(query_ids.cpu().numpy(), pred_logits.cpu().numpy(), average='micro')

        return all_loss, 100 * acc, 100 * c_acc, \
                100 * mi_ap, 100 * micro_p, 100 * micro_r, 100 * micro_f, \
                100 * ma_ap, 100 * macro_p, 100 * macro_r, 100 * macro_f


