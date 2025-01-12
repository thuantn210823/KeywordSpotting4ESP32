import yaml
import argparse

from KWS_helper.datasets import SpeechCommands
from KWS_helper.transforms import transform, Augmentation
from KWS_helper.optim import LinearDecay, CosineAnnealing
from KWS_helper import S4T as S
from TFResNet import *
from BCResNet import *
from EdgeCRNN import *
from MDTC import *

class GSC(S.SDataModule):
    def __init__(self, args):
        super().__init__()
        self.root = args.root
        self.batch_size = args.batch_size
        self.augmentation = args.augmentation
        if args.transform_type == 'MFCC':
            trans = transform(transform_type = 'MFCC',
                              sample_rate = args.sample_rate,
                              n_mfcc = args.n_mfcc,
                              melkwargs = {
                                  'f_min': args.f_min,
                                  'f_max': args.f_max,
                                  'n_mels': args.n_mels,
                                  'win_length': args.win_length,
                                  'hop_length': args.hop_length,
                                  'n_fft': args.n_fft})                             
        elif args.transform_type == 'logmel':
            trans = transform(transform_type = 'logmel',
                              sample_rate = args.sample_rate,
                              n_fft = args.n_fft,
                              win_length = args.win_length,
                              hop_length = args.hop_length,
                              n_mels = args.n_mels)
        elif args.transform_type == 'LFBE_Delta':
            trans = transform(transform_type = 'LFBE_Delta',
                              sample_rate = args.sample_rate,
                              n_mfcc = args.n_mfcc,
                              n_mels = args.n_mels,
                              melkwargs = {'n_fft': args.n_fft,
                                           'hop_length': args.hop_length,
                                           'f_min': args.f_min,
                                           'f_max': args.f_max})
        else:
            raise "The transform type doesn't exist!"
        if args.augmentation:
            self.train_transform = Augmentation(noise_dir = args.noise_dir,
                                                noise_prob = args.noise_prob,
                                                snr = args.snr,
                                                shift = args.shift,
                                                is_train = True,
                                                augment = True,
                                                transform = trans,
                                                spec_augment = args.spec_augment,
                                                n_times_masks = args.n_times_masks,
                                                time_mask_param = args.time_mask_param,
                                                n_freq_masks = args.n_freq_masks,
                                                freq_mask_param = args.freq_mask_param)
            self.test_transform = Augmentation(noise_dir = args.noise_dir,
                                               noise_prob = args.noise_prob,
                                               snr = args.snr,
                                               shift = args.shift,
                                               is_train = False,
                                               augment = False,
                                               transform = trans)
        else:
            self.train_transform = self.test_transform = trans

        self.train_dataset = SpeechCommands(args.root,
                                            download = args.download,
                                            version = args.version,
                                            subset = 'training',
                                            wanted_words = args.wanted_words)
        self.val_dataset = SpeechCommands(args.root,
                                          download = args.download,
                                          version = args.version,
                                          subset = 'validation',
                                          wanted_words = args.wanted_words)
        self.test_dataset = SpeechCommands(args.root,
                                           download = args.download,
                                           version = args.version,
                                           subset = 'testing',
                                           wanted_words = args.wanted_words)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset,
                                           batch_size = self.batch_size,
                                           shuffle = True,
                                           collate_fn = lambda x: self.collate_fn(x, True),
                                           num_workers = 1,
                                           prefetch_factor = 1,
                                           pin_memory = True,
                                           drop_last = False)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_dataset,
                                           batch_size = self.batch_size,
                                           shuffle = False,
                                           collate_fn = lambda x: self.collate_fn(x, False),
                                           num_workers = 1,
                                           prefetch_factor = 1,
                                           pin_memory = True,
                                           drop_last = False)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_dataset,
                                           batch_size = self.batch_size,
                                           shuffle = False,
                                           collate_fn = lambda x: self.collate_fn(x, False),
                                           num_workers = 1,
                                           prefetch_factor = 1,
                                           pin_memory = True,
                                           drop_last = False)
    
    def collate_fn(self, batch, train):
        spec_batch, label_batch = [], []
        for wav, label_idx in batch:
            if train:
                if self.augmentation:
                    spec = self.train_transform(wav, label_idx)
                else:
                    spec = self.train_transform(wav)
            else:
                if self.augmentation:
                    spec = self.test_transform(wav, label_idx)
                else:
                    spec = self.test_transform(wav)
            spec_batch.append(spec)
            label_batch.append(label_idx)
        return torch.stack(spec_batch), torch.tensor(label_batch).to(torch.long)

class BCResNet_training(S.SModule, BCResNet):
    def __init__(self,
                 lr,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lr = lr
    
    def loss(self, y_hat, y):
        return nn.functional.cross_entropy(y_hat, y, reduction = 'mean')
    
    def accuracy(self, Y_hat, Y, averaged = True):
        """
        Compute the number of correct predictions
        """
        Y_hat = Y_hat.reshape((-1, Y_hat.shape[-1]))
        preds = Y_hat.argmax(dim = 1).type(Y.dtype)
        compare = (preds == Y.reshape(-1)).type(torch.float32)
        return compare.mean() if averaged else compare
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.loss(y_hat, y)
        acc = self.accuracy(y_hat, y)

        values = {"train_loss": loss, "train_acc": acc}
        self.log_dict(values, pbar = True, train_logging = True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.loss(y_hat, y)
        acc = self.accuracy(y_hat, y)
        values = {"val_loss": loss, "val_acc": acc}
        self.log_dict(values, pbar = True, train_logging = False)

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr = 0, weight_decay = 0.00005)
        return optimizer
    
    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure):
        # update params
        optimizer.step(closure = optimizer_closure)

        try:
            lr = self.lr.calculate_lr(epoch, batch_idx)
        except:
            lr = self.lr

        for pg in optimizer.param_groups:
            pg['lr'] = lr
        self.log('lr', lr, pbar = True, train_logging = True)

class MDTC_training(S.SModule, MDTC):
    def __init__(self,
                 lr,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lr = lr

    def loss(self, y_hat, y):
        return nn.functional.cross_entropy(y_hat, y, reduction = 'mean')
    
    def accuracy(self, Y_hat, Y, averaged = True):
        """
        Compute the number of correct predictions
        """
        Y_hat = Y_hat.reshape((-1, Y_hat.shape[-1]))
        preds = Y_hat.argmax(dim = 1).type(Y.dtype)
        compare = (preds == Y.reshape(-1)).type(torch.float32)
        return compare.mean() if averaged else compare

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.loss(y_hat, y)
        acc = self.accuracy(y_hat, y)

        values = {"train_loss": loss, "train_acc": acc}
        self.log_dict(values, pbar = True, train_logging = True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.loss(y_hat, y)
        acc = self.accuracy(y_hat, y)
        values = {"val_loss": loss, "val_acc": acc}
        self.log_dict(values, pbar = True, train_logging = False)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), 0, weight_decay = 0.00005)
        return optimizer
    
    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure):
        # update params
        optimizer.step(closure = optimizer_closure)

        try:
            lr = self.lr.calculate_lr(epoch, batch_idx)
        except:
            lr = self.lr

        for pg in optimizer.param_groups:
            pg['lr'] = lr
        self.log('lr', lr, pbar = True, train_logging = True)
    
class EdgeCRNN_training(S.SModule, EdgeCRNN):
    def __init__(self,
                 lr,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lr = lr

    def loss(self, y_hat, y):
        return nn.functional.cross_entropy(y_hat, y, reduction = 'mean')
    
    def accuracy(self, Y_hat, Y, averaged = True):
        """
        Compute the number of correct predictions
        """
        Y_hat = Y_hat.reshape((-1, Y_hat.shape[-1]))
        preds = Y_hat.argmax(dim = 1).type(Y.dtype)
        compare = (preds == Y.reshape(-1)).type(torch.float32)
        return compare.mean() if averaged else compare

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.loss(y_hat, y)
        acc = self.accuracy(y_hat, y)

        values = {"train_loss": loss, "train_acc": acc}
        self.log_dict(values, pbar = True, train_logging = True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.loss(y_hat, y)
        acc = self.accuracy(y_hat, y)
        values = {"val_loss": loss, "val_acc": acc}
        self.log_dict(values, pbar = True, train_logging = False)
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), 0, weight_decay = 0.00005)
        return optimizer
    
    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure):
        # update params
        optimizer.step(closure = optimizer_closure)

        try:
            lr = self.lr.calculate_lr(epoch, batch_idx)
        except:
            lr = self.lr

        for pg in optimizer.param_groups:
            pg['lr'] = lr
        self.log('lr', lr, pbar = True, train_logging = True)

class TFResNet_training(S.SModule, TFResNet):
    def __init__(self,
                 lr,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lr = lr
    
    def loss(self, y_hat, y):
        return nn.functional.cross_entropy(y_hat, y, reduction = 'mean')
    
    def accuracy(self, Y_hat, Y, averaged = True):
        """
        Compute the number of correct predictions
        """
        Y_hat = Y_hat.reshape((-1, Y_hat.shape[-1]))
        preds = Y_hat.argmax(dim = 1).type(Y.dtype)
        compare = (preds == Y.reshape(-1)).type(torch.float32)
        return compare.mean() if averaged else compare
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.loss(y_hat, y)
        acc = self.accuracy(y_hat, y)

        values = {"train_loss": loss, "train_acc": acc}
        self.log_dict(values, pbar = True, train_logging = True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.loss(y_hat, y)
        acc = self.accuracy(y_hat, y)
        values = {"val_loss": loss, "val_acc": acc}
        self.log_dict(values, pbar = True, train_logging = False)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), 0, weight_decay = 0.00005)
        return optimizer
    
    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure):
        # update params
        optimizer.step(closure = optimizer_closure)

        try:
            lr = self.lr.calculate_lr(epoch, batch_idx)
        except:
            lr = self.lr

        for pg in optimizer.param_groups:
            pg['lr'] = lr
        self.log('lr', lr, pbar = True, train_logging = True)

def train(args):
    ### Setup data for training
    data = GSC(args)

    ### Configure model
    lr = args.lr
    if lr == 'linear_decay':
        lr = LinearDecay(lr_init = args.lr_init,
                         lr_end = args.lr_end,
                         max_epochs = args.max_epochs)
    elif lr == 'cosine':
        lr = CosineAnnealing(base_lr = args.base_lr,
                             warmup_epochs = args.warmup_epochs,
                             steps_per_epoch = len(data.train_dataloader())//args.batch_size + 1,
                             max_epochs = args.max_epochs)
    
    if args.model == 'MDTC':
        model = MDTC_training(in_channels = args.in_channels,
                              out_channels = args.out_channels,
                              kernel_size = args.kernel_size,
                              stack_num = args.stack_num,
                              stack_size = args.stack_size,
                              classification = True,
                              hidden_size = args.hidden_size,
                              num_classes = args.num_classes,
                              dropout = 0.5,
                              lr = lr)
    elif args.model == 'EdgeCRNN':
        model = EdgeCRNN_training(in_channels = args.in_channels,
                                  hidden_size = args.hidden_size,
                                  num_classes = args.num_classes,
                                  width_multiplier = args.width_multiplier,
                                  lr = lr)
    elif args.model == 'BCResNet':
        model = BCResNet_training(in_channels = args.in_channels,
                                  num_classes = args.num_classes,
                                  bias = False,
                                  mul_factor = args.mul_factor,
                                  lr = lr)
    elif args.model == 'TFResNet':
        model = TFResNet_training(in_channels = args.in_channels,
                                  num_classes = args.num_classes,
                                  bias = False,
                                  mul_factor = args.mul_factor,
                                  lr = lr)
    else:
        raise "The model doesn't exist!!!"
    
    ### Train
    torch.manual_seed(args.seed)
    checkpoint_callback = S.ModelCheckpoint(dirpath = args.ckpt_dir,
                                            save_top_k = 10, monitor = 'val_acc',
                                            mode = 'min',
                                            filename = f'{args.model}-epoch:%02d-val_acc:%.4f')
    trainer = S.Trainer(accelerator = args.device,
                        callbacks = [checkpoint_callback],
                        enable_checkpointing = True,
                        max_epochs = args.max_epochs,
                        gradient_clip_val = args.gradient_clip_val)
    history = trainer.fit(model, data)
    return history

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--config_yaml", type = str)
    args = parser.parse_args()
    args = yaml.load(open(args.config_yaml, 'rb'), Loader = yaml.SafeLoader)
    args = argparse.Namespace(**args)
    print(args)
    #train(args)