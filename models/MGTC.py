import torch
import torch.nn as nn
import numpy as np
from random import sample

from .ts_transformer import TSTransformerEncoderClassiregressor
from dataloader.dataloader import padding_mask


class MGTC(nn.Module):
    def __init__(self, args, configs, device):
        super(MGTC, self).__init__()
        self.num_channels = configs.final_out_channels
        self.timestep = configs.timesteps
        self.t_samples = configs.t_samples
        self.num_classes = configs.num_classes
        self.input_channels = configs.input_channels
        self.device = device
        self.max_seq_len = configs.ts_len
        self.posi = args.posi
        self.negi = args.negi

        self.Wk = nn.ModuleList([nn.Linear(self.input_channels, self.input_channels) for i in range(self.timestep)])
        self.Wk1 = nn.ModuleList([nn.Linear(self.t_samples, 1) for i in range(self.timestep)])

        self.Wk2 = nn.ModuleList([nn.Linear(args.d_model, self.num_channels) for i in range(self.timestep)])
        self.Wk22 = nn.ModuleList([nn.Linear(self.t_samples, 1) for i in range(self.timestep)])


        self.lsoftmax = nn.LogSoftmax(dim=1)


        self.ts_transformer = TSTransformerEncoderClassiregressor(self.input_channels, self.max_seq_len,args.d_model,
                                                        args.num_heads,
                                                        args.num_layers, args.dim_feedforward,
                                                        num_classes=self.num_classes,
                                                        dropout=args.dropout, pos_encoding=args.pos_encoding,
                                                        activation=args.activation,

                                                        norm=args.normalization_layer, freeze=args.freeze)

        self.cluster_projection = torch.nn.Sequential(torch.nn.Linear(configs.ts_len * args.d_model, 256),
                                                      nn.BatchNorm1d(256),
                                                      nn.ReLU(inplace=True),
                                                      nn.Linear(256,args.d_model),
                                                      )

        self.logits = nn.Linear(self.max_seq_len * args.d_model, self.num_classes)


        self.projection_head = nn.Sequential(
            nn.Linear(args.d_model, configs.final_out_channels // 2),
            nn.BatchNorm1d(configs.final_out_channels // 2),
            nn.ReLU(inplace=True),
            nn.Linear(configs.final_out_channels // 2, configs.final_out_channels // 4),
        )


    def forward(self, x_f, x_c, training_mode):

        if training_mode == "self_supervised" and x_c != None and x_f != None:

            _, feature_f = self.ts_transformer_fw(x_f.transpose(1, 2), training_mode)
            _, feature_c = self.ts_transformer_fw(x_c.transpose(1, 2), training_mode)
            # feature_f = self.ts_transformer_fw(x_f.transpose(1, 2), training_mode)
            # feature_c = self.ts_transformer_fw(x_c.transpose(1, 2), training_mode)

            z_f_clu = self.cluster_projection(feature_f.reshape(feature_f.shape[0], -1))
            z_c_clu = self.cluster_projection(feature_c.reshape(feature_c.shape[0], -1))

            nce_f = self.prediction_contrast(x_f, x_c, training_mode)
            nce_c = self.prediction_contrast(x_c, x_f, training_mode)


            projection_f = self.projection_head(feature_f.reshape(-1, feature_f.shape[-1]))
            projection_f = projection_f.reshape(feature_f.shape[0], -1, projection_f.shape[-1])

            projection_c = self.projection_head(feature_c.reshape(-1, feature_c.shape[-1]))
            projection_c = projection_c.reshape(feature_c.shape[0], -1, projection_c.shape[-1])

            return nce_f, projection_f, nce_c, projection_c, z_f_clu, z_c_clu


        else:

            if x_f is None:
                forward_seq = x_c.permute(0, 2, 1)
            elif x_c is None:
                forward_seq = x_f.permute(0,2,1)
            else:
                forward_seq = x_f.permute(0,2,1)

            prediction, features = self.ts_transformer_fw(forward_seq, training_mode)

            if x_c is None:
                features = features.reshape(features.shape[0],-1)
                features = self.cluster_projection(features)
                return features

            return prediction, features

    def ts_transformer_fw(self, forward_seq,training_mode):
        if training_mode == "self_supervised":


            max_len = None
            lengths = [X.shape[0] for X in forward_seq]
            if max_len is None:
                max_len = max(lengths)
            padding_masks = padding_mask(torch.tensor(lengths, dtype=torch.int16),
                                         max_len=max_len).to(self.device)

            data_in = forward_seq
            data_out = self.ts_transformer(data_in,padding_masks)  #(batch_size, seq_len_out, feat_dim_out)

            return data_out
        else:
            max_len = None
            lengths = [X.shape[0] for X in forward_seq]
            if max_len is None:
                max_len = max(lengths)
            padding_masks = padding_mask(torch.tensor(lengths, dtype=torch.int16),
                                         max_len=max_len)
            padding_masks = padding_masks.to(self.device)


            output = self.ts_transformer(forward_seq, padding_masks)
            features = output[1]
            features = features * padding_masks.unsqueeze(-1)  # zero-out padding embeddings

            features_flat = features.reshape(features.shape[0], -1)

            logits = self.logits(features_flat)

            return logits, features

    def prediction_contrast(self,x_1,x_2,training_mode):
        x_1 = x_1.transpose(1, 2)
        x_2 = x_2.transpose(1, 2)

        batch = x_1.shape[0]
        nce = 0

        encode_samples = torch.empty((self.timestep, batch, self.input_channels)).float().to(self.device)
        for i in np.arange(0, self.timestep):
            encode_samples[i] = x_2[:, self.t_samples + i, :].view(batch, self.input_channels)

        forward_seq = x_1[:, :self.t_samples, :]

        _, z_1_forward = self.ts_transformer_fw(forward_seq, training_mode)
        _, z_2_backward = self.ts_transformer_fw(encode_samples.permute(1, 0, 2), training_mode)


        pred = torch.empty((self.timestep, batch, self.num_channels)).float().to(self.device)


        for i in np.arange(0, self.timestep):
            linear = self.Wk22[i]
            pred[i] = self.Wk2[i](linear(z_1_forward.permute(0, 2, 1)).squeeze(2))


        z_2_backward = z_2_backward.permute(1, 0, 2)


        for i in np.arange(0, self.timestep):
            total = torch.mm(z_2_backward[i], torch.transpose(pred[i], 0, 1))
            nce += torch.sum(torch.diag(self.lsoftmax(total)))
        nce /= -1. * batch * self.timestep


        nce_total = nce


        return nce_total



