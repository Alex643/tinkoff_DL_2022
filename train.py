#!/usr/bin/env python
# coding: utf-8

# # 0. Preparation

# ## 0.0. Input params


# read input params
input_params = {}
with open('cfg.txt', encoding = 'utf-8', mode = 'r') as file:
    for line in file:
        line = line.replace('\n', '')
        (key, value) = line.split("::")
        input_params[(key)] = value
# input_params



# create model config
model_config = {
    'HID_DIM': 8,
    'DEC_LAYERS': 1,
    'DEC_HEADS': 2,
    'DEC_PF_DIM': 32,
    'DEC_DROPOUT': 0.1,
    'LEARNING_RATE': 0.005,
    'N_EPOCHS': 12, # 6
    
    'L2_reg': 1e-5,
    'clip': 1,
    'max_len': 15,
    'min_freq': 1,
    'batch_size': 512,
    'seed': 1001,
    
    'path_data_raw': input_params['input_dir'] + 'data_raw.parquet',
    'path_data': input_params['input_dir'],
    'model_path': input_params['input_dir'] + 'model.pkl',
    'tokenizer_path': input_params['input_dir'] + 'tokenizer.pt'
}

cut_batch_len_value = model_config['max_len']


# ## 0.1. Libs


import torch
import torch.nn as nn
import torch.optim as optim

from torchtext.legacy.data import Field, BucketIterator, TabularDataset # '0.9.1'

import pandas as pd
import re

import spacy
import numpy as np
import random
import time

import matplotlib.pyplot as plt

import tqdm.notebook as tq

import pickle

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # cuda if available


# ## 0.2. Funcs




##### Data preparation
def data_preparation(x):
    if any([i.isdigit() for i in x]): # идея в том, что так как мы имеем ограничение по объему обучения, следует избавиться от строк с числами, чтобы избежать переобучения;
        x = ''
    else:
        x = x.lower() # lowercase
        x = re.sub('[^A-Za-zА-Яа-я]+', ' ', x) # only A-z; А-я
        x = re.sub(' +', ' ', x) # удаление дабл сэйсов
        x = x[1:] if x[0] == ' ' else x
        x = x.split(' ') if len(x) > 3 else ''
    return(x)


def train_val_test_split(df, path_save,  seed = 1001):
    import random
    random.seed(seed)
    
    df = df.sample(frac = 1)
    
    l = len(df)
    len_tr = round(l * 0.7)
    len_val = round(l * 0.85)
    
    train = df[:len_tr]
    val = df[(len_tr + 1) : len_val]
    test = df[(len_val + 1) :]
    
    
    train.to_json(path_save + 'train.json', lines = True, orient = 'records')
    val.to_json(path_save + 'validation.json', lines = True, orient = 'records')
    test.to_json(path_save + 'test.json', lines = True, orient = 'records')
    
    return('Данные сохранены по указанному пути')

##### Tokenization
def tokenize_domains(text:list):
    return text

def cut_batch_by_max_length(): # looks at length ad if it is larger threashold - cuts it
    global cutted_batch
    def cutted_batch(batch, vocab, v = cut_batch_len_value):
        # 1. Pad TAU with zeros
        for idx, ex in enumerate(batch):
            if len(ex) > v:
                batch[idx] = ex[:v]
        # 2. return updated batch
        return batch
    return cutted_batch

##### Model management
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def initialize_weights(m):
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        nn.init.xavier_uniform_(m.weight.data)


# ## 0.3. Read data




data = pd.read_parquet(model_config['path_data_raw'])


# ## 0.4. Seed


random.seed(model_config['seed'])
np.random.seed(model_config['seed'])
torch.manual_seed(model_config['seed'])
torch.cuda.manual_seed(model_config['seed'])
torch.backends.cudnn.deterministic = True


# # 1. Data preparation

# ## 1.1. Text preproc



data['text'] = data['raw_text'].apply(lambda x: data_preparation(x))
data = data[~(data.text == ' ') | ~(data.text == '')].reset_index(drop=True)
data = data[data.text.apply(lambda x: len(x)) > 3].reset_index(drop=True)


# ## 1.2. Split data



train_val_test_split(data, path_save = model_config['path_data'])


# # 2. Токенизация и итератор



TRG = Field(
    tokenize = tokenize_domains, 
    init_token = '<sos>', 
    eos_token = '<eos>',
    # pad_token = '<pad>' by def
    lower = True, 
    batch_first = True, 
    postprocessing = cut_batch_by_max_length()
)

fields = {
    'text': ('text', TRG)
}

train_data, valid_data, test_data = TabularDataset.splits(
    path=input_params['input_dir'],
    train='train.json',
    validation='validation.json',
    test='test.json',
    format='json',
    fields = fields
)




train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
    (train_data, valid_data, test_data), 
    batch_size = model_config['batch_size'], 
    sort_key = lambda x: len(x.src), # сортировка: чтобы батчи были +- одного размера, т.к. их реальный размер в модели отличается из-за пэддинга
    sort_within_batch=False,
    device = device,
    # снижение потребления памяти
    shuffle=False,
    sort=False     
)




TRG.build_vocab(train_data, min_freq = model_config['min_freq'])
print(len(TRG.vocab.stoi))

##### Save tokenizer
with open(model_config['tokenizer_path'], 'wb') as fid:
    pickle.dump(TRG, fid)


# # 4. Инициализация модели




class Decoder(nn.Module):
    def __init__(self, 
                 output_dim, 
                 hid_dim, 
                 n_layers, 
                 n_heads, 
                 pf_dim, 
                 dropout, 
                 device,
                 max_length = model_config['max_len']):
        super().__init__()
        
        self.device = device
        
        # Layer for Input data to embedding
        self.tok_embedding = nn.Embedding(output_dim, hid_dim)
        # Layer for position to embedding
        self.pos_embedding = nn.Embedding(max_length, hid_dim)
        # Preprocess data (pos. encoder; embedding)
        self.layers = nn.ModuleList([DecoderLayer(hid_dim, 
                                                  n_heads, 
                                                  pf_dim, 
                                                  dropout, 
                                                  device)
                                     for _ in range(n_layers)])
        # from embedding to vocab size 
        self.fc_out = nn.Linear(hid_dim, output_dim)
        # Dropout
        self.dropout = nn.Dropout(dropout)
        # Sqrt of m-embedding size
        self.scale = torch.sqrt(torch.FloatTensor([hid_dim])).to(self.device)
        
    def forward(self, trg, trg_mask):
        # trg = [batch size, trg len]
        # trg_mask = [batch size, 1, trg len, trg len]
                
        batch_size = trg.shape[0]
        trg_len = trg.shape[1]
        
        pos = torch.arange(0, trg_len, device = self.device).unsqueeze(0).repeat(batch_size, 1) #.to(self.device) # pos = [batch size, trg len]
        # torch.arange(0, trg_len) - tensor from 0 to trg_len - 1
        # unsqueeze(0) - turn d to d+1 dimension
        # repeat(batch_size, 1) - repeat to b-times
        # to(self.device) - to cpu/gpu
        
        # scaling: in order to make embedding 'larger' compared to pos. encoder
        trg = self.dropout((self.tok_embedding(trg) * self.scale) + self.pos_embedding(pos)) # add TAU (1)! 
        # trg = [batch size, trg len, hid dim]
        
        for layer in self.layers:
            trg = layer(trg, trg_mask)
        # trg = [batch size, trg len, hid dim]
        
        output = self.fc_out(trg) # output = [batch size, trg len, output dim]
        return output




class DecoderLayer(nn.Module):
    def __init__(self, 
                 hid_dim, 
                 n_heads, 
                 pf_dim, 
                 dropout, 
                 device):
        super().__init__()
        
        self.self_attn_layer_norm = nn.LayerNorm(hid_dim) 
        self.enc_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.ff_layer_norm = nn.LayerNorm(hid_dim)
        self.self_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, device)
        self.positionwise_feedforward = PositionwiseFeedforwardLayer(hid_dim, 
                                                                     pf_dim, 
                                                                     dropout)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, trg, trg_mask):
        
        # trg = [batch size, trg len, hid dim]
        # trg_mask = [batch size, 1, trg len, trg len]
        
        #self attention
        _trg, _ = self.self_attention(trg, trg, trg, trg_mask)
        
        # dropout, residual connection and layer norm
        trg = self.self_attn_layer_norm(trg + self.dropout(_trg))
            
        # trg = [batch size, trg len, hid dim]
        
        # dropout, residual connection and layer norm
        trg = self.enc_attn_layer_norm(trg + self.dropout(_trg))
                    
        # trg = [batch size, trg len, hid dim]
        
        # positionwise feedforward
        _trg = self.positionwise_feedforward(trg)
        
        # dropout, residual and layer norm
        trg = self.ff_layer_norm(trg + self.dropout(_trg))
        
        # trg = [batch size, trg len, hid dim]
        return trg



class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, dropout, device):
        super().__init__()
        
        assert hid_dim % n_heads == 0
        
        self.hid_dim = hid_dim
        self.n_heads = n_heads
        self.head_dim = hid_dim // n_heads # Размер эмбеддинга головы 
        
        self.fc_q = nn.Linear(hid_dim, hid_dim)
        self.fc_k = nn.Linear(hid_dim, hid_dim)
        self.fc_v = nn.Linear(hid_dim, hid_dim)
        
        self.fc_o = nn.Linear(hid_dim, hid_dim)
        
        self.dropout = nn.Dropout(dropout)
        
        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim])).to(device)
        
    def forward(self, query, key, value, mask = None):
        
        batch_size = query.shape[0]
        
        # query = [batch size, query len, hid dim]
        # key = [batch size, key len, hid dim]
        # value = [batch size, value len, hid dim]
                
        Q = self.fc_q(query) # [batch size, query len, hid dim]
        K = self.fc_k(key) # [batch size, query len, hid dim]
        V = self.fc_v(value) # [batch size, query len, hid dim]
        
        # Q = [batch size, query len, hid dim]
        # K = [batch size, key len, hid dim]
        # V = [batch size, value len, hid dim]
                
        Q = Q.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3) # [batch size, n heads, query len, head dim]
        K = K.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3) # [batch size, n heads, query len, head dim]
        V = V.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3) # [batch size, n heads, query len, head dim]
        
        # Q = [batch size, n heads, key len, head dim]
        # K = [batch size, n heads, key len, head dim]
        # V = [batch size, n heads, value len, head dim]
        
        # Расчёт энергии (веса внимания)
        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale # [batch size, n heads, query len, key len] | q_len, key_len = seq len
        
        # Маскирование, если необходимо (да для декодера и нет для энкодера)
        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)
            
        # Софтмакс от весов внимания
        attention = torch.softmax(energy, dim = -1) # [batch size, n heads, query len, key len]
        
        # Расчёт оценки внимания
        x = torch.matmul(self.dropout(attention), V) # [batch size, n heads, query len, head dim]
        
        x = x.permute(0, 2, 1, 3).contiguous() # [batch size, query len, n heads, head dim]
        
        x = x.view(batch_size, -1, self.hid_dim) # [batch size, query len, hid dim] hid dim = h * k
        
        x = self.fc_o(x)
        return x, attention




class PositionwiseFeedforwardLayer(nn.Module):
    def __init__(self, hid_dim, pf_dim, dropout):
        super().__init__()
        
        self.fc_1 = nn.Linear(hid_dim, pf_dim)
        self.fc_2 = nn.Linear(pf_dim, hid_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        
        # x = [batch size, seq len, hid dim]
        # Apply RELU to f hidden layers
        x = self.dropout(torch.relu(self.fc_1(x))) # x = [batch size, seq len, pf dim]
        # Back to m size of layer
        x = self.fc_2(x) # x = [batch size, seq len, hid dim]
        return x



class model_transformer(nn.Module):
    def __init__(self,
                 decoder,
                 trg_pad_idx,
                 device):
        super().__init__()

        self.decoder = decoder
        self.trg_pad_idx = trg_pad_idx
        self.device = device


    def make_trg_mask(self, trg):

        # trg = [batch size, trg len]

        trg_pad_mask = (trg != self.trg_pad_idx).unsqueeze(1).unsqueeze(2) # trg_pad_mask = [batch size, 1, 1, trg len]

        trg_len = trg.shape[1]

        trg_sub_mask = torch.tril(torch.ones((trg_len, trg_len), device = self.device)).bool() # trg_sub_mask = [trg len, trg len]

        trg_mask = trg_pad_mask & trg_sub_mask # trg_mask = [batch size, 1, trg len, trg len]

        return trg_mask

    def forward(self, trg):
        # trg = [batch size, trg len] 
        trg_mask = self.make_trg_mask(trg) # trg_mask = [batch size, 1, trg len, trg len]
        output = self.decoder(trg, trg_mask)

        # output = [batch size, trg len, output dim]
        return output


# ### Создаём модель



model_config['INPUT_DIM'] = len(TRG.vocab)
model_config['OUTPUT_DIM'] = len(TRG.vocab)

dec = Decoder(
    model_config['INPUT_DIM'], 
    model_config['HID_DIM'], 
    model_config['DEC_LAYERS'], 
    model_config['DEC_HEADS'], 
    model_config['DEC_PF_DIM'], 
    model_config['DEC_DROPOUT'], 
    device,
    model_config['OUTPUT_DIM']
)
model_config['TRG_PAD_IDX'] = TRG.vocab.stoi[TRG.pad_token]

model = model_transformer(dec, model_config['TRG_PAD_IDX'], device).to(device)




# Инициализация весов
print(f'The model has {count_parameters(model):,} trainable parameters')
model.apply(initialize_weights);


# ### Оптимизатор 



optimizer = torch.optim.Adam(model.parameters(), lr = model_config['LEARNING_RATE'], weight_decay = model_config['L2_reg'])
criterion = nn.CrossEntropyLoss(ignore_index = model_config['TRG_PAD_IDX'])



class modeling(object):
    
    def __init__(self, model, model_config, train_iterator, valid_iterator, optimizer, criterion):
        self.model = model       
        self.model_config = model_config
        self.train_iterator = train_iterator
        self.valid_iterator = valid_iterator
        self.optimizer = optimizer
        self.criterion = criterion
        
    def train(self):
        self.model.train()
        epoch_loss = 0

        for i, batch in tq.tqdm(enumerate(self.train_iterator), total=(len(self.train_iterator))):
            src = batch.text
            self.optimizer.zero_grad()
            
            output = self.model(src[:,:-1])

            # output = [batch size, trg len - 1, output dim]
            # trg = [batch size, trg len]

            output_dim = output.shape[-1]

            output = output.contiguous().view(-1, output_dim)
            # print(output)
            # print(output.shape)
            src = src[:,1:].contiguous().view(-1)

            #output = [batch size * trg len - 1, output dim]
            #trg = [batch size * trg len - 1]

            loss = self.criterion(output, src)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.model_config['clip'])

            self.optimizer.step()

            epoch_loss += loss.item()

        return epoch_loss / len(self.train_iterator)
    
    
    def evaluate(self):
        self.model.eval()
        epoch_loss = 0

        with torch.no_grad():

            for i, batch in tq.tqdm(enumerate(self.valid_iterator), total=(len(self.valid_iterator))):

                src = batch.text

                output = self.model(src[:,:-1])

                # output = [batch size, trg len - 1, output dim]
                # trg = [batch size, trg len]

                output_dim = output.shape[-1]

                output = output.contiguous().view(-1, output_dim)
                src = src[:,1:].contiguous().view(-1)

                # output = [batch size * trg len - 1, output dim]
                # trg = [batch size * trg len - 1]

                loss = self.criterion(output, src)

                epoch_loss += loss.item()

        return epoch_loss / len(self.valid_iterator)
    
    
    
    def epoch_time(self, start_time, end_time):
        elapsed_time = end_time - start_time
        elapsed_mins = int(elapsed_time / 60)
        elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
        return elapsed_mins, elapsed_secs
    
    def print_losses(self, train_losses, valid_losses):
        plt.plot(train_losses,'-o')
        plt.plot(valid_losses,'-o')
        plt.xlabel('epoch')
        plt.ylabel('losses')
        plt.legend(['Train','Valid'])
        plt.title('Train vs Valid Losses')
        plt.show()

    def fit(self):
        train_losses = []
        valid_losses = []
        best_valid_loss = float('inf')

        for epoch in range(self.model_config['N_EPOCHS']):

            start_time = time.time()

            ##### 1. Training
            # model.decoder.training_features_param = True
            train_loss = self.train()
            train_losses.append(train_loss)

            ##### 2. Evaluation: turn off noising
            # model.decoder.training_features_param = False
            valid_loss = self.evaluate()
            valid_losses.append(valid_loss)

            end_time = time.time()

            epoch_mins, epoch_secs = self.epoch_time(start_time, end_time)

            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                torch.save([self.model_config, self.model.state_dict()], (self.model_config['model_path']))

            print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
            print(f'\tTrain Loss: {train_loss:.3f}')
            print(f'\t Val. Loss: {valid_loss:.3f} ')

        self.print_losses(train_losses, valid_losses)
        
    def generate(self, input_words):
        # 0. load tok & model 
        with open(self.model_config['tokenizer_path'], 'rb') as fid:
            trg_field = pickle.load(fid) 
        
        self.model.load_state_dict(torch.load(self.model_config['model_path'])[1])
        self.model.eval()
        
        # 1. tokens to seq; if seq is empty: generate random token
        if len(input_words) == 0:
            words = list([i[0] for i in trg_field.vocab.stoi.items() if i[1] != trg_field.vocab.stoi[trg_field.unk_token]])
            exclude = ['<unk>', '<pad>', '<sos>', '<eos>']
            list_randomize = set(words) - set(exclude)
            input_words = [random.choice(list(list_randomize))]
        
        tokens = [token.lower() for token in input_words]
        tokens = [trg_field.init_token] + tokens
        trg_indexes = [trg_field.vocab.stoi[token] for token in tokens]
        # 2. predict
        for i in range(model_config['max_len']):
            trg_tensor = torch.LongTensor(trg_indexes, device = device).unsqueeze(0) #.to(device)
            trg_mask = model.make_trg_mask(trg_tensor)

            with torch.no_grad():
                output = model.decoder(trg_tensor, trg_mask)

            pred_token = output.argmax(2)[:,-1].item()

            trg_indexes.append(pred_token)

            if pred_token == trg_field.vocab.stoi[trg_field.eos_token]:
                break
        if trg_field.vocab.stoi[trg_field.unk_token] in trg_indexes: trg_indexes.remove(trg_field.vocab.stoi[trg_field.unk_token])
        if trg_field.vocab.stoi[trg_field.eos_token] in trg_indexes: trg_indexes.remove(trg_field.vocab.stoi[trg_field.eos_token])
        trg_tokens = [trg_field.vocab.itos[i] for i in trg_indexes]

        return trg_tokens[1:]



model_text = modeling(model, model_config, train_iterator, valid_iterator, optimizer, criterion)


# # 5. Обучение



model_text.fit()

