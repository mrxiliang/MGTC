class Config(object):
    def __init__(self):
        # model configs
        self.data_name = 'InsectWingbeat'

        self.input_channels = 200
        self.kernel_size = 8
        self.stride = 1
        self.final_out_channels = 128
        self.ts_len = 22

        self.num_classes = 10
        self.dropout = 0.35

        # training configs
        self.num_epoch = 200
        self.fine_tune_epoch = 100


        # optimizer parameters
        self.beta1 = 0.9
        self.beta2 = 0.99
        self.self_supervised_lr = 3e-4
        self.fine_tune_lr = 3e-5

        # data parameters
        self.drop_last = True
        self.batch_size = 512
        self.batch_size_finetune = 256

        self.augmentation = augmentations()


class augmentations(object):
    def __init__(self):
        self.jitter_scale_ratio = 1.1
        self.jitter_ratio = 0.8
        self.max_seg = 8



