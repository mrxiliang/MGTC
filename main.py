import pytz
import torch
import os
from datetime import datetime
import argparse
from utils.utils import _logger, init_dl_program
from dataloader.dataloader import data_generator
from trainer.trainer import Trainer
from models.MGTC import MGTC
torch.multiprocessing.set_sharing_strategy('file_system')

def parse_option():
    parser = argparse.ArgumentParser()

    home_dir = os.getcwd()
    parser.add_argument('--experiment_description', default='Exp1', type=str,
                        help='Experiment Description')
    parser.add_argument('--run_description', default='run1', type=str,
                        help='Experiment Description')
    parser.add_argument('--seed', default=0, type=int,
                        help='seed value')
    parser.add_argument('--gpu', type=int, default=0,
                        help='The gpu no. used for training and inference (defaults to 0)')
    parser.add_argument('--training_mode', default='self_supervised', type=str,
                        help='Modes of choice: self_supervised, fine_tune')
    parser.add_argument('--selected_dataset', default='LSST', type=str,
                        help='Dataset of choice: FingerMovements, LSST, InsectWingbeat, Plane, UWaveGestureLibrary, Wafer')
    parser.add_argument('--data_path', type=str, default='./data', help='data file')
    parser.add_argument('--logs_save_dir', default='experiments_logs', type=str,
                        help='saving directory')
    parser.add_argument('--device', default='cuda', type=str,
                        help='cpu or cuda')
    parser.add_argument('--home_path', default=home_dir, type=str,
                        help='Project home directory')
    parser.add_argument('--ckpt_dir', type=str, default='experiments_logs',
                        help='Data path for checkpoint.')
    parser.add_argument('--optimizer', choices={"Adam", "RAdam"}, default="Adam", help="Optimizer")

    # Model
    parser.add_argument('--d_model', type=int, default=128,
                        help='Internal dimension of transformer embeddings')
    parser.add_argument('--dim_feedforward', type=int, default=256,
                        help='Dimension of dense feedforward part of transformer layer')
    parser.add_argument('--num_heads', type=int, default=8,
                        help='Number of multi-headed attention heads')
    parser.add_argument('--num_layers', type=int, default=3,
                        help='Number of transformer/LSTM/GRU encoder layers (blocks)')
    parser.add_argument('--hidden_size', type=int, default=64,
                        help='Number of LSTM/GRU/CausalConv encoder features in the hidden state')
    parser.add_argument('--kernel_size', type=int, default=3,
                        help='Size of the convolving kernel')
    parser.add_argument('--stride', type=int, default=1,
                        help='Stride of the convolution')

    parser.add_argument('--out_channel', type=int, default=64,
                        help='Number of output channels for CausalConv')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout applied to most transformer encoder layers')
    parser.add_argument('--pos_encoding', choices={'fixed', 'learnable'}, default='fixed',
                        help='Internal dimension of transformer embeddings')
    parser.add_argument('--activation', choices={'relu', 'gelu'}, default='gelu',
                        help='Activation to be used in transformer encoder')
    parser.add_argument('--normalization_layer', choices={'BatchNorm', 'LayerNorm'}, default='BatchNorm',
                        help='Normalization layer to be used internally in transformer encoder')

    parser.add_argument('--freeze', action='store_true',
                        help='If set, freezes all layer parameters except for the output layer. Also removes dropout except before the output layer')
    parser.add_argument('--max-threads', type=int, default=None,
                        help='The maximum allowed number of threads used by this process')
    parser.add_argument('--patience', type=int, default=50,
                        help='self_spervised training patience')
    parser.add_argument('--patience_finetune', type=int, default=20,
                        help='finetune training patience')

    parser.add_argument('--layers', default=1, type=int,
                        help='save the results of bottom # layers')
    parser.add_argument('--posi', default=2, type=int,
                        help='number of positive instance pairs (default: 3)')
    parser.add_argument('--negi', default=3, type=int,
                        help='number of negative instance pairs(default: 4)')

    parser.add_argument('--mask_ratio1', default=0.2, type=float,
                        help='the num of ')
    parser.add_argument('--mask_ratio2', default=0.5, type=float,
                        help='the num of ')
    parser.add_argument('--past_ratio', type=float, default=0.6, help='past_ratio ratio')

    parser.add_argument('--lr', type=float, default=3e-4, help='learning rate')

    args = parser.parse_args()

    return args



args = parse_option()

start_time = datetime.now()
args.hidden_size = args.d_model


experiment_description = args.experiment_description
data_type = args.selected_dataset
method = 'MGTC'
training_mode = args.training_mode
run_description = args.run_description

logs_save_dir = args.logs_save_dir
os.makedirs(logs_save_dir, exist_ok=True)


d = {}
exec(f'from config_files.{data_type}_Configs import Config as Configs', globals(), d)
Configs = d['Configs']

configs = Configs()
configs.t_samples = int(args.past_ratio * configs.ts_len)
configs.timesteps = int((1 - args.past_ratio) * configs.ts_len)




mask_ratio1 = 0.3
mask_ratio2 = 0.5

#####################################################

experiment_log_dir = os.path.join(logs_save_dir, experiment_description, run_description, training_mode,
                                  data_type, "transformer")


os.makedirs(experiment_log_dir, exist_ok=True)

# loop through domains

# Logging
log_file_name = os.path.join(experiment_log_dir,
                             f"logs_{((datetime.now()).astimezone(pytz.timezone('Asia/Shanghai'))).strftime('%Y-%m-%d %H:%M:%S')}.log")


logger = _logger(log_file_name)
logger.debug(
    f"Start time is {(start_time.astimezone(pytz.timezone('Asia/Shanghai'))).strftime('%Y-%m-%d %H:%M:%S')}")

result_log_dir = os.path.join(experiment_log_dir, "result_log")
os.makedirs(result_log_dir, exist_ok=True)


seed = args.seed
ckpt_dir = '{}/{}/saved_models_lr'.format(experiment_log_dir, str(seed))
args.ckpt_dir = ckpt_dir
os.makedirs(args.ckpt_dir, exist_ok=True)
device = init_dl_program(args.gpu, seed=args.seed)
args.device = device

start_time_seed = datetime.now()

logger.debug("=" * 45)
logger.debug(f'Dataset: {data_type}')
logger.debug(f'Method:  {method}')
logger.debug(f'Mode:    {training_mode}')
logger.debug(
f"Seed:    {seed}, Start time: {(start_time_seed.astimezone(pytz.timezone('Asia/Shanghai'))).strftime('%Y-%m-%d %H:%M:%S')}")
logger.debug("=" * 45)


# Load datasets
data_path = f"./data"
args.data_path = data_path
train_dl, valid_dl, test_dl = data_generator(data_path, configs, training_mode,args)
logger.debug("Data loaded ...")
logger.debug(
f'Train_Size: {len(train_dl.dataset)} | Valid Size:{len(valid_dl.dataset)} | Test Size:{len(test_dl.dataset)}')
logger.debug(f'Batch_Size:  {configs.batch_size}  Batch_Size_finetune: {configs.batch_size_finetune} ')


# Load Model
logger.debug("=" * 45)
logger.debug(f'Dataset: {data_type}   Seed: {seed}')
logger.debug(f'mask_ratio1: {mask_ratio1}   mask_ratio2: {mask_ratio2}')
logger.debug("=" * 45)

model = MGTC(args, configs, device).to(device)


if training_mode == "fine_tune":
    load_from = os.path.join(
        os.path.join(logs_save_dir, experiment_description, run_description, f"self_supervised",
                     data_type, "transformer",
                     f'{seed}',
                     "saved_models_lr"))


    finetune_ckp_path = os.path.join(logs_save_dir, experiment_description, run_description, f"fine_tune",
                     data_type, "transformer",f'{seed}',
                     "saved_models_3")


    args.finetune_ckp_path = finetune_ckp_path
    chkpoint = torch.load(
        os.path.join(load_from, f'{mask_ratio1}_{mask_ratio2}_ckp_last.pt'),
        map_location=device)

    pretrained_dict = chkpoint["model_state_dict"]

    model_dict = model.state_dict()
    del_list = ['logits']
    pretrained_dict_copy = pretrained_dict.copy()
    for i in pretrained_dict_copy.keys():
        for j in del_list:
            if j in i:
                del pretrained_dict[i]
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

if training_mode == "self_supervised":
    model_optimizer = torch.optim.Adam(model.parameters(),
                                                lr=args.lr,
                                                betas=(configs.beta1, configs.beta2),
                                                weight_decay=3e-5)
else:
    model_optimizer = torch.optim.Adam(model.parameters(),
                                                lr=args.lr*(1e-1),
                                                betas=(configs.beta1, configs.beta2),
                                                weight_decay=3e-5)



patience_finetune = args.patience_finetune
patience = args.patience

# Trainer
test_acc, test_auc, test_prc, test_f1, epoch_max = Trainer(model,
                                                           model_optimizer, train_dl,
                                                           valid_dl, test_dl,
                                                           device,
                                                           logger, configs, args,
                                                           ckpt_dir,
                                                           training_mode, patience, patience_finetune,

                                                           mask_ratio1, mask_ratio2)






logger.debug(f"Training time is : {datetime.now() - start_time}")


