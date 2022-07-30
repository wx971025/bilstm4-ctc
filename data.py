import os
from typing import List

import torch
from torch.utils.data import DataLoader, Dataset
import torchaudio
import torchaudio.transforms as T


import matplotlib.pyplot as plt

def get_w2i(data_path):
    res_dict = {}
    with open(data_path, 'r', encoding='utf8') as f:
        context_list = f.readlines()
        for context in context_list:
            k, v = context.strip().split(":")
            res_dict[k] = int(v)
    return res_dict


def data_w2i(data_list, w2i):
    new_data_list = []
    for content in data_list:
        idx_list = [w2i.get(el) for el in content]
        new_data_list.append(idx_list)
    return new_data_list


def load_audio(audio_path):
    sound, sample_rate = torchaudio.load(audio_path)
    sound = sound.numpy()
    if sound.shape[1] == 2:
        sound = sound.mean(axis=0)
    sound = sound.squeeze(0)
    return torch.from_numpy(sound)

def wav_feature(wav_path_list):
    wav_feature_list = []
    n_fft = 512
    win_length = 400
    hop_length = 200
    n_mels = 64
    n_mfcc = 32
    mfcc_feature = T.MFCC(sample_rate=16000,
                          n_mfcc=n_mfcc,
                          melkwargs={'n_fft': n_fft,
                                     'n_mels': n_mels,
                                     'hop_length': hop_length,
                                     'win_length': win_length})
    for path in wav_path_list:
        sound = load_audio(path)
        mfcc = mfcc_feature(sound).T
        wav_feature_list.append(mfcc)
    return wav_feature_list


def get_chr_dict():
    i2w_chr_list = ['_', ' ', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p',
                    'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
                    'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', ',', '.', '?',
                    '!', '\'', '-', ':', ';', '"']
    w2i_chr_dict = {}
    for i, c in enumerate(i2w_chr_list):
        w2i_chr_dict[c] = i

    return i2w_chr_list, w2i_chr_dict

def data_process(batch_data):
    def key(p):
        return p[0].size(0)

    batch_data = sorted(batch_data, key=key, reverse=True)
    longest_sample = batch_data[0][0]
    feat_size = longest_sample.size(1)
    max_length = longest_sample.size(0)
    batch_size = len(batch_data)

    inputs = torch.zeros(batch_size, max_length, feat_size)
    targets = []
    input_sizes = torch.IntTensor(batch_size)
    target_sizes = torch.IntTensor(batch_size)

    for x in range(batch_size):
        sample = batch_data[x]
        feature = sample[0]
        label = sample[1]
        seq_length = feature.size(0)
        inputs[x].narrow(0, 0, seq_length).copy_(feature)
        input_sizes[x] = seq_length
        target_sizes[x] = len(label)
        targets.extend(label)

    targets = torch.IntTensor(targets)
    return inputs, input_sizes, targets, target_sizes

class TimitDataSet(Dataset):
    def __init__(self, data_path, data_type="TRAIN"):
        self.data_path = data_path
        self.dataset_path = os.path.join(data_path, data_type)

        self.i2w, self.w2i = get_chr_dict()
        self.data_list = []
        self.wav_path_list = []
        self.get_target_and_wav(self.dataset_path, data_type="TXT")

        assert len(self.data_list) == len(self.wav_path_list)

        self.data_char_idx = data_w2i(self.data_list, self.w2i)
        self.data_wav_feature = wav_feature(self.wav_path_list)

        assert len(self.data_char_idx) == len(self.data_wav_feature)

    def __getitem__(self, item):
        return self.data_wav_feature[item], self.data_char_idx[item]

    def __len__(self):
        return len(self.data_char_idx)

    def get_target_and_wav(self, data_path, data_type="TXT"):
        for DR_name in os.listdir(data_path):
            DR_path = os.path.join(data_path, DR_name)
            for PR_name in os.listdir(DR_path):
                PR_path = os.path.join(DR_path, PR_name)
                for sound_name in os.listdir(PR_path):
                    if sound_name.split('.')[-1] == data_type:
                        temp_data_list = []
                        if data_type == "PHN":
                            with open(os.path.join(PR_path, sound_name), 'r', encoding='utf8') as rf:
                                contents = rf.readlines()
                                for content in contents:
                                    temp_data_list.append(content.strip().split()[-1])
                        elif data_type == "TXT":
                            with open(os.path.join(PR_path, sound_name), 'r', encoding='utf8') as rf:
                                contents = rf.readlines()
                                sentence = contents[0].strip().split(" ", 2)[-1]
                                for c in sentence:
                                    temp_data_list.append(c)
                        self.data_list.append(temp_data_list)
                        self.wav_path_list.append(os.path.join(PR_path, sound_name.split('.')[0] + 'WAV'))

def data_loader(data_path, data_type="TRAIN"):
    dataset = TimitDataSet(data_path, data_type=data_type)
    dataloader = DataLoader(dataset,
                            batch_size=16,
                            shuffle=False,
                            collate_fn=data_process)
    return dataloader, dataset.i2w

if __name__ == '__main__':
    pass


