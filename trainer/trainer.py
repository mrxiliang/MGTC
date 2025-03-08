import sys
from sklearn.exceptions import UndefinedMetricWarning
from datetime import datetime
sys.path.append("..")
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
from models.loss import granularity_aware_contrastive_loss
from utils.utils import EarlyStopping
from models.clustering import hierarchical_clustering


from sklearn.metrics import roc_auc_score,average_precision_score,f1_score
np.seterr(divide='ignore',invalid='ignore')
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
def Trainer(model, model_optimizer, train_dl, valid_dl, test_dl, device, logger, config,args, ckpt_dir, training_mode,patience,patience_finetune,mask_ratio1,mask_ratio2):
    # Start training
    logger.debug("Training started ....")

    criterion = nn.CrossEntropyLoss()


    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(model_optimizer, 'min')

    bce_criterion = nn.BCEWithLogitsLoss(reduce=None)

    early_stopping = EarlyStopping(patience, verbose=True,
                                         checkpoint_pth=f'{ckpt_dir}/{mask_ratio1}_{mask_ratio2}_ckp_last.pt')
    early_stopping_finetune = EarlyStopping(patience_finetune, verbose=True,
                                   checkpoint_pth=f'{ckpt_dir}/{mask_ratio1}_{mask_ratio2}_ckp_last.pt')


    epoch_max = 0
    test_loss, test_acc, test_auc, test_prc, test_f1 = 0, 0, 0, 0, 0
    best_acc, best_auc, best_prc, best_f1 = 0, 0, 0, 0
    if training_mode == "self_supervised":
        num_epoch = config.num_epoch
    else:
        num_epoch = config.fine_tune_epoch


    for epoch in range(1, num_epoch + 1):
        if training_mode == "self_supervised":
            cluster_result = None
            features = compute_features(args, train_dl, model, training_mode)

            cluster_result = {'im2cluster': [], 'centroids': [], 'density': []}
            features = F.normalize(features, dim=1)
            features = features.numpy()

            """ perform hierarchical clustering """
            c, num_clust, partition_clustering, lowest_level_centroids, cluster_result = hierarchical_clustering(
                features, args, initial_rank=None, distance='euclidean',
                ensure_early_exit=True, verbose=True, ann_threshold=40000)
        else:
            cluster_result, c = None, None

        train_loss, train_acc,train_auc, train_prc, train_f1 = model_train(args,model, model_optimizer, criterion, bce_criterion, train_dl,device, training_mode, cluster_result, c,mask_ratio1,mask_ratio2)


        valid_loss, valid_acc,valid_auc, valid_prc, valid_f1 = model_evaluate(model, valid_dl, device, training_mode)


        if training_mode == 'self_supervised':
            early_stopping(-1*train_loss, model)
            if early_stopping.early_stop:
                print("Early stopping")
                break

        if training_mode != 'self_supervised':  # use scheduler in all other modes.
            scheduler.step(valid_loss)

        
        if training_mode != 'self_supervised':
            if valid_acc >= best_acc:
                best_acc = valid_acc
                epoch_max = epoch

                start_time = datetime.now()
                test_loss, test_acc, test_auc, test_prc, test_f1= model_evaluate(
                    model, test_dl,
                    device, training_mode)
                end_time = datetime.now()
                test_time = end_time - start_time
                logger.debug(f"MGTC test time: {test_time}")

            early_stopping_finetune(valid_acc, model)
            if early_stopping_finetune.early_stop:
                print("Early stopping")
                break



        logger.debug(f'\nEpoch : {epoch}\n'
                     f'Train: Loss     : {train_loss:.4f}\t | \tAccuracy     : {train_acc:2.4f}\t | AUROC     : {train_auc:2.4f}\t | Precision     : {train_prc:2.4f}\t | F1     : {train_f1:2.4f}\n'
                     f'Valid: Loss     : {valid_loss:.4f}\t | \tAccuracy     : {valid_acc:2.4f}\t | AUROC     : {valid_auc:2.4f}\t | Precision     : {valid_prc:2.4f}\t | F1     : {valid_f1:2.4f}\n'
                     f'Test : Loss      :{test_loss:0.4f}\t | \tAccuracy     : {test_acc:0.4f}\t  | AUROC     : {test_auc:2.4f}\t  | Precision     : {test_prc:2.4f}\t  | F1     : {test_f1:2.4f}')


    logger.debug("\n################## Training is Done! #########################")



    return test_acc, test_auc, test_prc, test_f1, epoch_max




def model_train(args, model, temp_cont_optimizer, criterion, bce_criterion,train_loader,  device, training_mode, cluster_result, c,mask_ratio1,mask_ratio2):
    total_loss = []
    total_acc = []
    total_auc = []
    total_prc = []
    total_f1 = []

    model.train()
    labels_numpy_all, pred_numpy_all = np.zeros(1), np.zeros(1)


    for batch_idx, (data, labels, _, _, index) in tqdm(enumerate(train_loader)):
        # send to device
        data, labels = data.float().to(device), labels.long().to(device)

        # optimizer
        temp_cont_optimizer.zero_grad()

        if training_mode == "self_supervised":


            features1, features2 = timeseries_mask(data,mask_ratio1,mask_ratio2)
            features1, features2 = features1.float(), features2.float()
            # normalize projection feature vectors

            features1 = F.normalize(features1, dim=1)
            features2 = F.normalize(features2, dim=1)

            loss_cross_f, feat_cross_f,loss_cross_c, feat_cross_c,z_f_clu, z_c_clu = model(features1, features2, training_mode)


            zis = feat_cross_f
            zjs = feat_cross_c
        else:
            # normalize projection feature vectors
            data = F.normalize(data, dim=1)
            output = model(data, data, training_mode)


        # compute loss
        if training_mode == "self_supervised":
            lambda1 = 1

            lambda2 = 0.7


            loss_granularity = granularity_aware_contrastive_loss(
                args,
                zis,
                zjs,
                z_f_clu, z_c_clu,
                bce_criterion,
                cluster_result = cluster_result, c = c, index = index,
                temporal_unit=0
            )

            loss = (loss_cross_f + loss_cross_c) * lambda1 +  loss_granularity * lambda2



        else:
            predictions, features = output
            loss = criterion(predictions, labels)
            total_acc.append(labels.eq(predictions.detach().argmax(dim=1)).float().mean())

            onehot_label = F.one_hot(labels,predictions.shape[1])
            pred_numpy = predictions.detach().cpu().numpy()
            labels_numpy = labels.detach().cpu().numpy()
            try:

                auc_bs = roc_auc_score(onehot_label.reshape(-1).detach().cpu().numpy(), pred_numpy.reshape(-1), average="macro",
                                       multi_class="ovr")
            except:
                auc_bs = 0.0

            try:
                prc_bs = average_precision_score(onehot_label.reshape(-1).detach().cpu().numpy(), pred_numpy.reshape(-1))

            except:
                prc_bs = 0.0

            pred_numpy = np.argmax(pred_numpy, axis=1)
            F1 = f1_score(labels_numpy, pred_numpy, average='macro',
                          zero_division="warn")  # labels=np.unique(ypred))
            if auc_bs != 0:
                total_auc.append(auc_bs)
            if prc_bs != 0:
                total_prc.append(prc_bs)
            total_f1.append(F1)

            labels_numpy_all = np.concatenate((labels_numpy_all, labels_numpy))
            pred_numpy_all = np.concatenate((pred_numpy_all, pred_numpy))


        total_loss.append(loss.item())

        loss.backward()
        temp_cont_optimizer.step()


    total_loss = torch.tensor(total_loss).mean()


    if training_mode == "self_supervised":
        total_acc = 0
        total_auc,total_prc,F1=0,0,0

    else:
        total_acc = torch.tensor(total_acc).mean()
        labels_numpy_all = labels_numpy_all[1:]
        pred_numpy_all = pred_numpy_all[1:]
        F1 = f1_score(labels_numpy_all, pred_numpy_all, average='macro', )
        total_auc = torch.tensor(total_auc).mean()  # average auc
        total_prc = torch.tensor(total_prc).mean()

    return total_loss, total_acc, total_auc,total_prc,F1





def timeseries_mask(data,mask_ratio1,mask_ratio2):
    data = data.permute(0, 2, 1)
    batch_size,ts_len,input_channel = data.size(0),data.size(1),data.size(2)

    original_sequence = data
    mask_1 = torch.from_numpy(np.random.binomial(1, mask_ratio1, size=(batch_size, ts_len, input_channel))).cuda()
    enhanced_sequence_1 = original_sequence * (1 - mask_1)

    mask_2 = torch.from_numpy(np.random.binomial(1, mask_ratio2, size=(batch_size, ts_len, input_channel))).cuda()
    enhanced_sequence_2 = original_sequence * (1 - mask_2)

    enhanced_sequence_1 = enhanced_sequence_1.permute(0,2,1)
    enhanced_sequence_2 = enhanced_sequence_2.permute(0,2,1)

    return enhanced_sequence_1,enhanced_sequence_2




def model_evaluate(model, test_dl, device, training_mode):

    model.eval()

    total_loss = []
    total_acc = []
    total_auc = []
    total_prc = []
    total_f1 = []

    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        labels_numpy_all, pred_numpy_all = np.zeros(1), np.zeros(1)

        for data, labels, _, _,_ in test_dl:
            data, labels = data.float().to(device), labels.long().to(device)

            if training_mode == "self_supervised":
                pass
            else:
                data = F.normalize(data, dim=1)
                output = model(data, data, training_mode)

            # compute loss
            if training_mode != "self_supervised":
                predictions, features = output
                loss = criterion(predictions, labels)
                total_acc.append(labels.eq(predictions.detach().argmax(dim=1)).float().mean())
                total_loss.append(loss.item())

                onehot_label = F.one_hot(labels,predictions.shape[1])
                pred_numpy = predictions.detach().cpu().numpy()
                labels_numpy = labels.detach().cpu().numpy()


                try:

                    auc_bs = roc_auc_score(onehot_label.reshape(-1).detach().cpu().numpy(), pred_numpy.reshape(-1), average="macro",
                                           multi_class="ovr")
                except:
                    auc_bs = 0.0

                try:
                    prc_bs = average_precision_score(onehot_label.reshape(-1).detach().cpu().numpy(), pred_numpy.reshape(-1))
                except:
                    prc_bs = 0.0

                pred_numpy = np.argmax(pred_numpy, axis=1)
                F1 = f1_score(labels_numpy, pred_numpy, average='macro',
                              zero_division="warn")  # labels=np.unique(ypred))

                if auc_bs != 0:
                    total_auc.append(auc_bs)
                if prc_bs != 0:
                    total_prc.append(prc_bs)
                total_f1.append(F1)

            if training_mode != "self_supervised":
                pred = predictions.max(1, keepdim=True)[1]  # get the index of the max log-probability

                labels_numpy_all = np.concatenate((labels_numpy_all, labels_numpy))
                pred_numpy_all = np.concatenate((pred_numpy_all, pred_numpy))
    if training_mode != "self_supervised":

        labels_numpy_all = labels_numpy_all[1:]
        pred_numpy_all = pred_numpy_all[1:]

        F1 = f1_score(labels_numpy_all, pred_numpy_all, average='macro', )

        total_loss = torch.tensor(total_loss).mean()  # average loss
        total_acc = torch.tensor(total_acc).mean()  # average acc
        total_auc = torch.tensor(total_auc).mean()  # average auc
        total_prc = torch.tensor(total_prc).mean()
    else:
        total_loss = 0
        total_acc = 0
        total_auc,total_prc,F1=0,0,0
        return total_loss, total_acc, total_auc,total_prc,F1

    return total_loss, total_acc, total_auc, total_prc,F1




def compute_features(args,data_loader,model,training_mode):
    model.eval()
    b,c,ts_len = data_loader.dataset.x_data.shape[0],data_loader.dataset.x_data.shape[1],data_loader.dataset.x_data.shape[2]

    features = torch.zeros((b,args.d_model)).cuda()

    for i, (data, target, _, _, index) in enumerate(data_loader):
        with torch.no_grad():

            data = data.cuda()
            feat = model(data,None,training_mode)

            features[index] = feat

    return features.cpu()



