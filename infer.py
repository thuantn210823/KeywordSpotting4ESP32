import yaml
import argparse

import librosa

from KWS_helper.transforms import transform
from TFResNet import *
from BCResNet import *
from EdgeCRNN import *
from MDTC import *

def infer(args):
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
    
    if args.model == 'MDTC':
        model = MDTC(in_channels = args.in_channels,
                     out_channels = args.out_channels,
                     kernel_size = args.kernel_size,
                     stack_num = args.stack_num,
                     stack_size = args.stack_size,
                     classification = True,
                     hidden_size = args.hidden_size,
                     num_classes = args.num_classes,
                     dropout = 0.5)
    elif args.model == 'EdgeCRNN':
        model = EdgeCRNN(in_channels = args.in_channels,
                         hidden_size = args.hidden_size,
                         num_classes = args.num_classes,
                         width_multiplier = args.width_multiplier)
    elif args.model == 'BCResNet':
        model = BCResNet(in_channels = args.in_channels,
                         num_classes = args.num_classes,
                         bias = False,
                         mul_factor = args.mul_factor)
    elif args.model == 'TFResNet':
        model = TFResNet(in_channels = args.in_channels,
                         num_classes = args.num_classes,
                         bias = False,
                         mul_factor = args.mul_factor)
    else:
        raise "The model doesn't exist!!!"
    
    ckpt = torch.load(args.ckpt_path, map_location = 'cpu', weights_only = False)
    model.load_state_dict(ckpt['state_dict'])
    wav, sr = librosa.load(args.audio_path, sr = args.sample_rate)
    wav = torch.from_numpy(wav.copy()).unsqueeze(0)
    spec = trans(wav).unsqueeze(0)
    model.eval()
    preds = model(spec)
    pred_idx = preds.argmax(dim = -1).item()
    wanted_words = ['__silence__', '__unknown__'] + args.wanted_words 
    print(f"Detected: {wanted_words[pred_idx]} command!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--config_yaml", type = str)
    parser.add_argument("--audio_path", type = str)
    args = parser.parse_args()
    args_ = yaml.load(open(args.config_yaml, 'rb'), Loader = yaml.SafeLoader)
    args_ = argparse.Namespace(**args_)
    setattr(args_, "audio_path", args.audio_path)
    infer(args_)