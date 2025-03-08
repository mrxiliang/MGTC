import torch
import numpy as np
import torch.nn.functional as F

from random import sample


class NTXentLoss(torch.nn.Module):

    def __init__(self, device, batch_size, temperature, use_cosine_similarity):
        super(NTXentLoss, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.device = device
        self.softmax = torch.nn.Softmax(dim=-1)
        self.mask_samples_from_same_repr = self._get_correlated_mask().type(torch.bool)
        self.similarity_function = self._get_similarity_function(use_cosine_similarity)
        self.criterion = torch.nn.CrossEntropyLoss(reduction="sum")

    def _get_similarity_function(self, use_cosine_similarity):
        if use_cosine_similarity:
            self._cosine_similarity = torch.nn.CosineSimilarity(dim=-1)
            return self._cosine_simililarity
        else:
            return self._dot_simililarity

    def _get_correlated_mask(self):
        diag = np.eye(2 * self.batch_size)
        l1 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=-self.batch_size)
        l2 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=self.batch_size)
        mask = torch.from_numpy((diag + l1 + l2))
        mask = (1 - mask).type(torch.bool)
        return mask.to(self.device)

    @staticmethod
    def _dot_simililarity(x, y):
        v = torch.tensordot(x.unsqueeze(1), y.T.unsqueeze(0), dims=2)
        # x shape: (N, 1, C)
        # y shape: (1, C, 2N)
        # v shape: (N, 2N)
        return v

    def _cosine_simililarity(self, x, y):
        # x shape: (N, 1, C)
        # y shape: (1, 2N, C)
        # v shape: (N, 2N)
        v = self._cosine_similarity(x.unsqueeze(1), y.unsqueeze(0))
        return v

    def forward(self, zis, zjs):
        representations = torch.cat([zjs, zis], dim=0)

        similarity_matrix = self.similarity_function(representations, representations)

        # filter out the scores from the positive samples
        l_pos = torch.diag(similarity_matrix, self.batch_size)
        r_pos = torch.diag(similarity_matrix, -self.batch_size)
        positives = torch.cat([l_pos, r_pos]).view(2 * self.batch_size, 1)

        negatives = similarity_matrix[self.mask_samples_from_same_repr].view(2 * self.batch_size, -1)

        logits = torch.cat((positives, negatives), dim=1)
        logits /= self.temperature

        labels = torch.zeros(2 * self.batch_size).to(self.device).long()
        loss = self.criterion(logits, labels)

        return loss / (2 * self.batch_size)


def granularity_aware_contrastive_loss(args,
                z1,
                z2,
                x_t1_cluster_projection, x_t2_cluster_projection,
                bce_criterion,
                cluster_result, c, index, alpha=0.5, temporal_unit=0):
    loss = torch.tensor(0., device=z1.device)
    d = 0
    while z1.size(1) > 1:
        if alpha != 0:
            loss += alpha * instance_contrastive_loss(z1, z2)
        if d >= temporal_unit:
            if 1 - alpha != 0:
                loss += (1 - alpha) * temporal_contrastive_loss(z1, z2)
        d += 1
        z1 = F.max_pool1d(z1.transpose(1, 2), kernel_size=2).transpose(1, 2)
        z2 = F.max_pool1d(z2.transpose(1, 2), kernel_size=2).transpose(1, 2)
    if z1.size(1) == 1:
        if alpha != 0:
            loss += alpha * instance_contrastive_loss(z1, z2)
        d += 1

    loss /= d

    if alpha != 0:
        logits, labels = cluster_contrast(args, x_t1_cluster_projection, x_t2_cluster_projection, cluster_result, c, index)
        loss_cluster = bce_criterion(logits.cuda(), torch.tensor(labels).cuda())

        total_loss = loss + loss_cluster
    # return loss / d
    return total_loss


def instance_contrastive_loss(z1, z2):
    B, T = z1.size(0), z1.size(1)
    if B == 1:
        return z1.new_tensor(0.)
    z = torch.cat([z1, z2], dim=0)  # 2B x T x C
    z = z.transpose(0, 1)  # T x 2B x C
    # z:(T,2B,C) z.transpose(1,2):(T,C,2B)
    sim = torch.matmul(z, z.transpose(1, 2))  # T x 2B x 2B
    # torch,tril():返回一个下三角矩阵
    logits = torch.tril(sim, diagonal=-1)[:, :, :-1]  # T x 2B x (2B-1)
    logits += torch.triu(sim, diagonal=1)[:, :, 1:]  # logits对角线为0
    logits = -F.log_softmax(logits, dim=-1)  # logits:(2305,16,15)

    i = torch.arange(B, device=z1.device)
    loss = (logits[:, i, B + i - 1].mean() + logits[:, B + i, i].mean()) / 2
    return loss


def temporal_contrastive_loss(z1, z2):
    B, T = z1.size(0), z1.size(1)
    if T == 1:
        return z1.new_tensor(0.)
    z = torch.cat([z1, z2], dim=1)  # B x 2T x C
    sim = torch.matmul(z, z.transpose(1, 2))  # B x 2T x 2T
    logits = torch.tril(sim, diagonal=-1)[:, :, :-1]  # B x 2T x (2T-1)
    logits += torch.triu(sim, diagonal=1)[:, :, 1:]  # logits对角线为0
    logits = -F.log_softmax(logits, dim=-1)  # logits:(8,4610,4609)

    t = torch.arange(T, device=z1.device)
    loss = (logits[:, t, T + t - 1].mean() + logits[:, T + t, t].mean()) / 2
    return loss

def cluster_contrast(args,x_t1, x_t2, cluster_result,c,index):

    p0_label = {}  # dict (key:index, value:label)
    label_index = {}  # dict
    index_u = {}

    for u in range(0, index.shape[0]):
        index_u[index[u].item()] = u
        p0_label[index[u].item()] = cluster_result['im2cluster'][0][index[u]].item()
    # find keys(ids) with same value(cluster label) in dict p0_label
    for key, value in p0_label.items():
        label_index.setdefault(value, []).append(key)

    posid = {}
    negid = {}

    neg_instances = [[] for _ in range(len(p0_label))]
    pos_instances = [[] for _ in range(len(p0_label))]
    all_instances = [[] for _ in range(len(p0_label))]

    for i in p0_label:
        posid[i] = label_index[p0_label[i]].copy() #all candidate pos instances(if not enough, copy itself)
        if(len(posid[i])) < args.posi:
            for _ in range(0, args.posi - len(posid[i])):
                posid[i].append(i)
        negid[i] = [x for x in index.tolist() if x not in posid[i]]
        # print(f'negid[i]:{negid[i]}')
        if (len(posid[i])) > args.posi:
            posid[i] = sample(posid[i], args.posi) #if len = self.posi, preserve
        negid[i] = sample(negid[i], args.negi)

        for m in range(len(posid[i])):
            if posid[i][m] == i:
                pos_instances[index_u[i]].append(x_t2[index_u[posid[i][m]]])
                pos_instances[index_u[i]].append(x_t2[index_u[posid[i][m]]])  # all candidate pos instances(if not enough, copy itself)
            else:
                pos_instances[index_u[i]].append(x_t1[index_u[posid[i][m]]])
                pos_instances[index_u[i]].append(x_t2[index_u[posid[i][m]]])
        pos_instances[index_u[i]] = torch.stack(pos_instances[index_u[i]])

        for n in range(len(negid[i])):
            neg_instances[index_u[i]].append(x_t1[index_u[negid[i][n]]])
            neg_instances[index_u[i]].append(x_t2[index_u[negid[i][n]]])
        neg_instances[index_u[i]] = torch.stack(neg_instances[index_u[i]])

        all_instances[index_u[i]] = torch.cat([pos_instances[index_u[i]], neg_instances[index_u[i]]], dim=0)

    all_instances = torch.stack(all_instances)  # [batch_size, 2*posi+2*negi, dim]

    all_instances = torch.reshape(all_instances, (all_instances.shape[0], all_instances.shape[2], all_instances.shape[1]))

    new_x_t1 = x_t1.unsqueeze(1)
    logits = torch.einsum('nab,nbc->nac', [new_x_t1, all_instances])
    logits = logits.squeeze(1)

    # labels of instances
    temp_label = np.zeros(args.posi * 2 + args.negi * 2)
    temp_label[0: args.posi * 2] = 1
    labels = np.tile(temp_label, (x_t1.shape[0], 1))

    return logits, labels
