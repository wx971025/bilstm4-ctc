import torch.cuda
import torch.nn as nn
from tqdm import tqdm
import os
import argparse
from typing import Dict

parse = argparse.ArgumentParser(description="BiLSTM+CTC")

parse.add_argument('--cuda', type=int, default=0)
parse.add_argument('--epoch', type=int, default=0)
parse.add_argument('--rnn_layer', type=int, default=4)

args = parse.parse_args()

device = f'cuda:{args.cuda}' if torch.cuda.is_available() else 'cpu'
device = torch.device(device)


# 获取音素字典
def get_w2i_phn_dict(path: str) -> Dict:
    w2i_dict = {}
    with open(path, 'w', encoding='utf-8') as rf:
        contents = rf.readlines()
        for content in contents:
            k, v = content.strip().split(":")
            w2i_dict[k] = int(v)
    return w2i_dict


def decoder(prob, input_size_list, i2w):
    _, prob_max = torch.max(prob, dim=2)
    prob_max = prob_max.cpu()
    strings_list = []
    for i in range(len(input_size_list)):
        temp_list = []
        k = prob_max[i].numpy().tolist()[:input_size_list[i]]
        for x in k:
            temp_list.append(i2w[x])
        strings_list.append(temp_list)
    return strings_list


# 将目标index解码成文字
def target_decoder(target_list, inputs_size_list, i2w):
    string_list = []
    s = 0
    for i in range(len(inputs_size_list)):
        temp_list = []
        k = target_list[s: s + inputs_size_list[i]]
        s += inputs_size_list[i]
        for x in k:
            temp_list.append(i2w[x])
        new_temp_list = "".join(temp_list).split()
        string_list.append(new_temp_list)
    return string_list


def merge_pron(decoded_strings):
    new_strings = []
    for string_list in decoded_strings:
        temp_string = [string_list[0]]  # 第一个字放进字符串中
        for idx, pron in enumerate(string_list[1:]):
            if pron == temp_string[len(temp_string) - 1]:   # 如果当前字符和上一个字符相等，就直接跳过
                continue
            else:
                temp_string.append(pron)    # 新字符 放入字符串中
        new_temp_string = [pron for pron in temp_string if pron != '_']
        new_strings.append("".join(new_temp_string).split())
    return new_strings


def min_distance(word1, word2) -> int:
    row = len(word1) + 1
    column = len(word2) + 1

    cache = [[0] * column for i in range(row)]

    for i in range(row):
        for j in range(column):
            if i == 0 and j == 0:
                cache[i][j] = 0
            elif i == 0 and j != 0:
                cache[i][j] = j
            elif j == 0 and i != 0:
                cache[i][j] = i
            else:
                if word1[i - 1] == word2[j - 1]:
                    cache[i][j] = cache[i - 1][j - 1]
                else:
                    replace = cache[i - 1][j - 1] + 1
                    insert = cache[i][j - 1] + 1
                    remove = cache[i - 1][j] + 1

                    cache[i][j] = min(replace, insert, remove)
    return cache[row - 1][column - 1]

def main():
    data_path = './TIMIT'
    epoch_num = args.epoch
    rnn_layer_num = args.rnn_layer

    train_data_loader, i2w = data_loader(data_path)
    dev_data_loader, _ = data_loader(data_path, data_type='TEST')

    model = CTC_Model(rnn_layer=rnn_layer_num, num_class=len(i2w))

    cuda_available = torch.cuda.is_available()

    model = model.to(device)

    loss_func = nn.CTCLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0002)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    print_loss = 0
    total_loss = 0
    print_every = 50

    n = 0.1

    for epoch in range(epoch_num):
        print(f"epoch_num is {epoch + 1}")

        model.train()
        e_wer = 0
        wer = 0

        t = tqdm(enumerate(train_data_loader))
        for i, contents in t:
            t.desc = f"now is {i + 1} epoch"

            inputs, inputs_size, targets, targets_size = contents

            inputs_size_list = inputs_size.numpy().tolist()

            batch_size = inputs.size(0)

            inputs = inputs.to(device)

            inputs = nn.utils.rnn.pack_padded_sequence(inputs, inputs_size_list, batch_first=True)

            prob, out = model(inputs)

            decoded_strings = decoder(prob, inputs_size_list, i2w)

            decoded_strings = merge_pron(decoded_strings)

            targets = targets.cpu()

            targets_string = targets.numpy().tolist()

            target_s = target_decoder(targets_string, targets_size, i2w)

            wer = 0
            for ss in range(len(target_s)):
                wer += min_distance(decoded_strings[ss], targets_string[ss])/len(target_s[ss])
            wer /= len(target_s)

            e_wer += wer

            targets = targets.to(device)

            out = out.transpose(0, 1)

            loss = loss_func(out, targets, inputs_size, targets_size)
            loss /= batch_size

            print_loss += loss.item()
            total_loss += loss.item()

            if (i+1) % print_every == 0:
                print(f'epoch={epoch+1}, batch={i+1}, loss={print_loss / print_every}, wer={wer*100}%')
                print(f'pre is {decoded_strings[0]}')
                print(f'tar is {target_s[0]}')
                print(f'pre is {decoded_strings[1]}')
                print(f'tar is {target_s[1]}')
                print(f'pre is {decoded_strings[2]}')
                print(f'tar is {target_s[2]}')
                print(f'pre is {decoded_strings[3]}')
                print(f'tar is {target_s[3]}')
                print(f'pre is {decoded_strings[4]}')
                print(f'tar is {target_s[4]}')

                print_loss = 0

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 400)
            optimizer.step()
            # scheduler.step()

            n = i
        average_loss = total_loss / n
        print(f"Epoch {epoch + 1} done, average train_loss: {average_loss}, e_wer: {e_wer/289*100}%")
        total_loss = 0

        dev_wer, dev_loss = dev(model, dev_data_loader, cuda_available, loss_func, i2w)
        print("loss on dev set is %.4f" % dev_loss)
        print(f"dev wer is {(dev_wer/105)*100}%")

        if epoch + 1 % 10 == 0:
            best_path = os.path.join(f'./data/dev_{epoch}' + str(wer) + '.pkl')
            torch.save(CTC_Model.save_package(model, optimizer=optimizer, epoch=epoch, loss_result=total_loss,
                                              dev_loss_results=dev_loss, dev_wer_results=dev_wer), best_path)

def dev(model, dev_loader, cuda_available, loss_fn, i2w):
    model.eval()
    if cuda_available:
        model = model.to(device)

    total_cer = 0
    total_tokens = 0
    total_loss = 0
    i = 0

    e_wer = 0

    for i, contents in enumerate(dev_loader):

        inputs, inputs_size, targets, targets_size = contents

        batch_size = inputs.size(0)

        if cuda_available:
            inputs = inputs.to(device)

        inputs = nn.utils.rnn.pack_padded_sequence(inputs, inputs_size_list, batch_first=True)

        prob, out = model(inputs)

        decoded_strings = decoder(prob, inputs_size, i2w)

        decoded_strings = merge_pron(decoded_strings)

        targets = targets.cpu()

        targets_string = targets.numpy().tolist()

        target_s = target_decoder(targets_string, targets_size, i2w)

        wer = 0
        for ss in range(len(target_s)):
            wer += min_distance(decoded_strings[ss], targets_string[ss]) / len(target_s[ss])
        wer /= len(target_s)

        e_wer += wer

        if cuda_available:
            targets = targets.to(device)

        out = out.transpose(0, 1)

        loss = loss_fn(out, targets, inputs_size, targets_size)
        loss /= batch_size

        total_loss += loss.item()
        target_sizes = targets_size.data
        total_tokens += sum(target_sizes)


        if (i + 1) % 20 == 0:
            print(f'epoch={epoch + 1}, batch={i + 1}, loss={loss}, wer={wer * 100}%')
            print(f'pre is {decoded_strings[0]}')
            print(f'tar is {target_s[0]}')
            print(f'pre is {decoded_strings[1]}')
            print(f'tar is {target_s[1]}')
            print(f'pre is {decoded_strings[2]}')
            print(f'tar is {target_s[2]}')
            print(f'pre is {decoded_strings[3]}')
            print(f'tar is {target_s[3]}')
            print(f'pre is {decoded_strings[4]}')
            print(f'tar is {target_s[4]}')

    average_loss = total_loss / i
    return e_wer, average_loss


if __name__ == '__main__':
    data_root_path = './TIMIT'
    epoch_num = args.epoch
    rnn_layer = args.rnn_layer

    main()
