import torch
import torch.distributed as dist
from torch.distributions.uniform import Uniform

import os
import re
import sys
import math
import logging
from copy import deepcopy
from collections import OrderedDict

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def randint(low, high):
    return int(torch.randint(low, high, (1, )))


def rand_uniform(low, high):
    return float(Uniform(low, high).sample())


def get_logger(out_dir):
    logger = logging.getLogger('Exp')
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")

    file_path = os.path.join(out_dir, "run.log")
    file_hdlr = logging.FileHandler(file_path)
    file_hdlr.setFormatter(formatter)

    strm_hdlr = logging.StreamHandler(sys.stdout)
    strm_hdlr.setFormatter(formatter)

    logger.addHandler(file_hdlr)
    logger.addHandler(strm_hdlr)
    return logger


def update_lr_cos(nb_iter, warm_up_iter, total_iter, max_lr, optimizer, min_lr=1e-7):

    if nb_iter < warm_up_iter:
        current_lr = max_lr * (nb_iter + 1) / (warm_up_iter + 1)
    else:
        current_lr = min_lr + (max_lr - min_lr) * 0.5 * (1. + math.cos(math.pi * nb_iter / (total_iter - warm_up_iter)))

    for param_group in optimizer.param_groups:
        param_group["lr"] = current_lr

    return optimizer, current_lr


class CTCLabelConverter(object):
    """CTC Loss 标签转换器"""
    def __init__(self, character):
        dict_character = list(character)
        self.dict = {}
        for i, char in enumerate(dict_character):
            self.dict[char] = i + 1
        if len(self.dict) == 87:     # '[' and ']' are not in the test set but in the training and validation sets.
            self.dict['['], self.dict[']'] = 88, 89
        self.character = ['[blank]'] + dict_character

    def encode(self, text):
        length = [len(s) for s in text]
        text = ''.join(text)
        text = [self.dict[char] for char in text]

        return (torch.IntTensor(text).to(device), torch.IntTensor(length).to(device))

    def decode(self, text_index, length):
        texts = []
        index = 0

        for l in length:
            t = text_index[index:index + l]
            char_list = []
            for i in range(l):
                if t[i] != 0 and (not (i > 0 and t[i - 1] == t[i])) and t[i]<len(self.character):
                    char_list.append(self.character[t[i]])
            text = ''.join(char_list)

            texts.append(text)
            index += l
        return texts


class AttnLabelConverter(object):
    """
    Attention-based Label Converter (for ABINet branch)
    
    与 CTC 不同，Attention 分支使用固定长度的输出，需要:
    - EOS token 标记序列结束
    - Padding 到固定长度
    
    字符映射:
    - Index 0: [PAD] - padding token
    - Index 1: [EOS] - end of sequence  
    - Index 2+: 实际字符
    
    Args:
        character: 字符列表或迭代器
        max_length: 最大序列长度 (包含 EOS)
    """
    def __init__(self, character, max_length=26):
        self.max_length = max_length
        dict_character = list(character)
        
        # Special tokens
        self.PAD_TOKEN = '[PAD]'
        self.EOS_TOKEN = '[EOS]'
        self.PAD_IDX = 0
        self.EOS_IDX = 1
        
        # Build character list: [PAD], [EOS], char1, char2, ...
        self.character = [self.PAD_TOKEN, self.EOS_TOKEN] + dict_character
        
        # Build encoding dict
        self.dict = {}
        for i, char in enumerate(dict_character):
            self.dict[char] = i + 2  # +2 for PAD and EOS
        
        # Handle special characters that might be missing
        if len(self.dict) == 87:
            if '[' not in self.dict:
                self.dict['['] = len(self.character)
                self.character.append('[')
            if ']' not in self.dict:
                self.dict[']'] = len(self.character)
                self.character.append(']')
        
        self.num_classes = len(self.character)

    def encode(self, text_list):
        """
        将文本列表转换为固定长度的 tensor
        
        Args:
            text_list: list of strings
            
        Returns:
            targets: (B, max_length) LongTensor, 每个位置是字符索引
                    序列末尾是 EOS，之后是 PAD
            lengths: (B,) 实际长度 (包含 EOS)
        """
        batch_size = len(text_list)
        targets = torch.full((batch_size, self.max_length), self.PAD_IDX, dtype=torch.long)
        lengths = []
        
        for i, text in enumerate(text_list):
            # Encode characters
            text_indices = []
            for char in text:
                if char in self.dict:
                    text_indices.append(self.dict[char])
                else:
                    # Skip unknown characters (or could map to UNK)
                    pass
            
            # Truncate if too long (leave room for EOS)
            if len(text_indices) > self.max_length - 1:
                text_indices = text_indices[:self.max_length - 1]
            
            # Add EOS
            text_indices.append(self.EOS_IDX)
            length = len(text_indices)
            lengths.append(length)
            
            # Fill tensor
            targets[i, :length] = torch.LongTensor(text_indices)
        
        return targets.to(device), torch.LongTensor(lengths).to(device)

    def decode(self, text_index, length=None):
        """
        将索引序列解码为字符串
        
        Args:
            text_index: (B, T) tensor of indices, or (T,) for single sample
            length: optional lengths tensor
            
        Returns:
            list of decoded strings
        """
        if text_index.dim() == 1:
            text_index = text_index.unsqueeze(0)
        
        batch_size = text_index.size(0)
        texts = []
        
        for i in range(batch_size):
            char_list = []
            for idx in text_index[i]:
                idx = idx.item()
                if idx == self.EOS_IDX:
                    break  # Stop at EOS
                if idx == self.PAD_IDX:
                    continue  # Skip PAD
                if idx < len(self.character):
                    char_list.append(self.character[idx])
            texts.append(''.join(char_list))
        
        return texts


class Averager(object):
    def __init__(self):
        self.reset()

    def add(self, v):
        count = v.data.numel()
        v = v.data.sum()
        self.n_count += count
        self.sum += v

    def reset(self):
        self.n_count = 0
        self.sum = 0

    def val(self):
        res = 0
        if self.n_count != 0:
            res = self.sum / float(self.n_count)
        return res


class Metric(object):
    def __init__(self, name=''):
        self.name = name
        self.sum = torch.tensor(0.).double()
        self.n = torch.tensor(0.)

    def update(self, val):
        rt = val.clone()
        dist.all_reduce(rt, op=dist.ReduceOp.SUM)
        rt /= dist.get_world_size()
        self.sum += rt.detach().cpu().double()
        self.n += 1

    @property
    def avg(self):
        return self.sum / self.n.double()


class ModelEma:
    def __init__(self, model, decay=0.9999, device='', resume=''):
        self.ema = deepcopy(model)
        self.ema.eval()
        self.decay = decay
        self.device = device
        if device:
            self.ema.to(device=device)
        self.ema_has_module = hasattr(self.ema, 'module')
        if resume:
            self._load_checkpoint(resume)
        for p in self.ema.parameters():
            p.requires_grad_(False)

    def _load_checkpoint(self, checkpoint_path, mapl=None):
        checkpoint = torch.load(checkpoint_path,map_location=mapl)
        assert isinstance(checkpoint, dict)
        if 'state_dict_ema' in checkpoint:
            new_state_dict = OrderedDict()
            for k, v in checkpoint['state_dict_ema'].items():
                if self.ema_has_module:
                    name = 'module.' + k if not k.startswith('module') else k
                else:
                    name = k
                new_state_dict[name] = v
            self.ema.load_state_dict(new_state_dict)
            print("=> Loaded state_dict_ema")
        else:
            print("=> Failed to find state_dict_ema, starting from loaded model weights")

    def update(self, model, num_updates=-1):
        needs_module = hasattr(model, 'module') and not self.ema_has_module
        if num_updates >= 0:
            _cdecay = min(self.decay, (1 + num_updates) / (10 + num_updates))
        else:
            _cdecay = self.decay

        with torch.no_grad():
            msd = model.state_dict()
            for k, ema_v in self.ema.state_dict().items():
                if needs_module:
                    k = 'module.' + k
                model_v = msd[k].detach()
                if self.device:
                    model_v = model_v.to(device=self.device)
                ema_v.copy_(ema_v * _cdecay + (1. - _cdecay) * model_v)


def format_string_for_wer(str):
    str = re.sub('([\[\]{}/\\()\"\'&+*=<>?.;:,!\-—_€#%°])', r' \1 ', str)
    str = re.sub('([ \n])+', " ", str).strip()
    return str