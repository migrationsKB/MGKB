import torch
import torch.nn as nn
import torch.nn.functional as F

def global_max_pooling(tensor, dim, topk):
    """Global max pooling"""
    ret, _ = torch.topk(tensor, topk, dim)
    return ret



#### deep hate structure, sentiment, topic, semantic ########## 
class CombinedModel(nn.Module):
    def __init__(self, vocab_size, config, filter_size =100, window_sizes=(3,4,5), trainable=False, vec=None):
        super().__init__()
        self.hidden_dim = config['hidden_dim']
        self.batch_size = config['batch_size']
        self.emb_dim = config['emb_dim']
        self.gpu = config['gpu']
        self.config = config
        self.embedding = nn.Embedding(vocab_size, config['emb_dim'])
        if vec is not None:
            self.embedding.weight.data.copy_(vec) #load pretrained
            self.embedding.weight.requires_grad = trainable #non-trainable
            
        #### CNN ###########
        ## https://github.com/yongjincho/cnn-text-classification-pytorch/blob/master/model.py
        
        self.convs = nn.ModuleList([
            nn.Conv2d(1, filter_size, [window_size, 300], padding=(window_size - 1, 0))
            for window_size in window_sizes
        ])
        
        if config['rnn_layer'] == 'lstm':
            self.encoder = nn.LSTM(filter_size*len(window_sizes), config['hidden_dim'], 
                             num_layers= config['nlayers'], bidirectional=config['bidir'],
                             dropout=config['dropout'])
                
        if config['rnn_layer'] == 'gru':
            self.encoder = nn.GRU(filter_size* len(window_sizes), config['hidden_dim'], 
                                 num_layers= config['nlayers'], bidirectional=config['bidir'],
                                 dropout=config['dropout'])
        
        if config['has_attn']:
            self.fc = nn.Linear(config['hidden_dim'], config['out_dim'])
        else:
            self.fc = nn.Linear(2* config['hidden_dim'], config['out_dim'])
        self.dropout = nn.Dropout(config['dropout'])
        #self.hidden = nn.Parameters(self.batch_size, self.hidden_dim)
    
    def attnetwork(self, encoder_out, final_hidden):
        hidden = final_hidden.squeeze(0)
        
        #M = torch.tanh(encoder_out)
        # attention weights from the hidden layer of the encoder.
        attn_weights = torch.bmm(encoder_out, hidden.unsqueeze(2)).squeeze(2)
        soft_attn_weights = F.softmax(attn_weights, 1)
        new_hidden = torch.bmm(encoder_out.transpose(1,2), soft_attn_weights.unsqueeze(2)).squeeze(2)
        #print (wt.shape, new_hidden.shape)
        #new_hidden = torch.tanh(new_hidden)
        #print ('UP:', new_hidden, new_hidden.shape)
        
        return new_hidden
    
    def forward(self, sequence):
        emb_input = self.embedding(sequence)    
        inputx = self.dropout(emb_input)        
        
        ## inspired by CNN text classification, 2014.############# 
        
        inputx= inputx.permute(1,0,2)
        x = torch.unsqueeze(inputx, 1) # (1,1,16,300)
        
        xs = []
        for conv in self.convs:
            x2 = F.relu(conv(x))        # [B, F, T, 1]
            x2 = torch.squeeze(x2, -1)  # [B, F, T]
            x2 = F.max_pool1d(x2, x2.size(2))  # [B, F, 1]
            xs.append(x2)
        x = torch.cat(xs, 2)   
        x = x.view(x.size(0), -1)
        x= torch.unsqueeze(x, 0)
        ################################################
        
        if self.config['rnn_layer']=='lstm':
            output, (hn, cn) = self.encoder(x) # BILSTM
        if self.config['rnn_layer']=='gru':
            output, hn = self.encoder(x) # BIGRU        
        
        if self.config['has_attn']:
            fbout = output[:, :, :self.hidden_dim]+ output[:, :, self.hidden_dim:] # sum bidir outputs F+B
            fbout = fbout.permute(1,0,2)
            fbhn = (hn[-2,:,:]+hn[-1,:,:]).unsqueeze(0)
            attn_out = self.attnetwork(fbout, fbhn)  # attention layer
#             print(attn_out.shape) # 16,50
            
            logits = self.fc(attn_out)
#             print(logits.shape)
            
            return logits
        else:
            output = output.permute(1,0,2)
            output = torch.squeeze(output,1)
#             print(output.shape) # 16,100
            #attn1_out = self.attnetwork1(output, hn)
            
            logits = self.fc(output)
#             print(logits.shape)
            return logits




### cnn + RNN
class cnnRNN(nn.Module):
    def __init__(self, vocab_size, config, filter_size =100, window_sizes=(3,4,5), trainable=False, vec=None):
        super().__init__()
        self.hidden_dim = config['hidden_dim']
        self.batch_size = config['batch_size']
        self.emb_dim = config['emb_dim']
        self.gpu = config['gpu']
        self.config = config
        self.embedding = nn.Embedding(vocab_size, config['emb_dim'])
        if vec is not None:
            self.embedding.weight.data.copy_(vec) #load pretrained
            self.embedding.weight.requires_grad = trainable #non-trainable
            
        #### CNN ###########
        ## https://github.com/yongjincho/cnn-text-classification-pytorch/blob/master/model.py
        
        self.convs = nn.ModuleList([
            nn.Conv2d(1, filter_size, [window_size, 300], padding=(window_size - 1, 0))
            for window_size in window_sizes
        ])
        
        if config['rnn_layer'] == 'lstm':
            self.encoder = nn.LSTM(filter_size*len(window_sizes), config['hidden_dim'], 
                             num_layers= config['nlayers'], bidirectional=False,
                             dropout=config['dropout'])
                
        if config['rnn_layer'] == 'gru':
            self.encoder = nn.GRU(filter_size* len(window_sizes), config['hidden_dim'], 
                                 num_layers= config['nlayers'], bidirectional=False,
                                 dropout=config['dropout'])
        if config['has_attn']:
            half_hidden_dim = int(config['hidden_dim']/2)
            self.fc = nn.Linear(half_hidden_dim, config['out_dim'])
        else:
            self.fc = nn.Linear(config['hidden_dim'], config['out_dim'])
        self.dropout = nn.Dropout(config['dropout'])
        #self.hidden = nn.Parameters(self.batch_size, self.hidden_dim)
    
    def attnetwork(self, encoder_out, final_hidden):
        hidden = final_hidden.squeeze(0)
        
        #M = torch.tanh(encoder_out)
        # attention weights from the hidden layer of the encoder.
        attn_weights = torch.bmm(encoder_out, hidden.unsqueeze(2)).squeeze(2)
        soft_attn_weights = F.softmax(attn_weights, 1)
        new_hidden = torch.bmm(encoder_out.transpose(1,2), soft_attn_weights.unsqueeze(2)).squeeze(2)
        #print (wt.shape, new_hidden.shape)
        #new_hidden = torch.tanh(new_hidden)
        #print ('UP:', new_hidden, new_hidden.shape)
        
        return new_hidden
    
    def forward(self, sequence):
        emb_input = self.embedding(sequence)    
        inputx = self.dropout(emb_input)        
        
        ## inspired by CNN text classification, 2014.############# 
        
        inputx= inputx.permute(1,0,2)
        x = torch.unsqueeze(inputx, 1) # (1,1,16,300)
        
        xs = []
        for conv in self.convs:
            x2 = F.relu(conv(x))        # [B, F, T, 1]
            x2 = torch.squeeze(x2, -1)  # [B, F, T]
            x2 = F.max_pool1d(x2, x2.size(2))  # [B, F, 1]
            xs.append(x2)
        x = torch.cat(xs, 2)   
        x = x.view(x.size(0), -1)
        x= torch.unsqueeze(x, 0)
        ################################################
        
        if self.config['rnn_layer']=='lstm':
            output, (hn, cn) = self.encoder(x) # BILSTM
        if self.config['rnn_layer']=='gru':
            output, hn = self.encoder(x) # BIGRU        
#         print(output.shape)
#         print(hn.shape)
        
        if self.config['has_attn']:
            half_hidden_dim = int(self.hidden_dim/2)
            fbout = output[:, :, :half_hidden_dim]+ output[:, :, half_hidden_dim:] # sum bidir outputs F+B
#             print(fbout.shape)
            fbout = fbout.permute(1,0,2)
            fbhn = (hn[-2,:,:half_hidden_dim]+hn[-1,:,half_hidden_dim:]).unsqueeze(0)
            attn_out = self.attnetwork(fbout, fbhn)  # attention layer
#             print(attn_out.shape) # 16,50
            
            logits = self.fc(attn_out)
#             print(logits.shape)
            
            return logits
        else:
            output = output.permute(1,0,2)
            output = torch.squeeze(output,1)
#             print(output.shape) # 16,100
            #attn1_out = self.attnetwork1(output, hn)
            
            logits = self.fc(output)
#             print(logits.shape)
            return logits




#### cnn2d + GRU, with/without attention ########## 
class cnnBRNN(nn.Module):
    def __init__(self, vocab_size, config, filter_size =100, window_sizes=(3,4,5), trainable=False, vec=None):
        super().__init__()
        self.hidden_dim = config['hidden_dim']
        self.batch_size = config['batch_size']
        self.emb_dim = config['emb_dim']
        self.gpu = config['gpu']
        self.config = config
        self.embedding = nn.Embedding(vocab_size, config['emb_dim'])
        if vec is not None:
            self.embedding.weight.data.copy_(vec) #load pretrained
            self.embedding.weight.requires_grad = trainable #non-trainable
            
        #### CNN ###########
        ## https://github.com/yongjincho/cnn-text-classification-pytorch/blob/master/model.py
        
        self.convs = nn.ModuleList([
            nn.Conv2d(1, filter_size, [window_size, 300], padding=(window_size - 1, 0))
            for window_size in window_sizes
        ])
        
        if config['rnn_layer'] == 'lstm':
            self.encoder = nn.LSTM(filter_size*len(window_sizes), config['hidden_dim'], 
                             num_layers= config['nlayers'], bidirectional=config['bidir'],
                             dropout=config['dropout'])
                
        if config['rnn_layer'] == 'gru':
            self.encoder = nn.GRU(filter_size* len(window_sizes), config['hidden_dim'], 
                                 num_layers= config['nlayers'], bidirectional=config['bidir'],
                                 dropout=config['dropout'])
        if config['has_attn']:
            self.fc = nn.Linear(config['hidden_dim'], config['out_dim'])
        else:
            self.fc = nn.Linear(2* config['hidden_dim'], config['out_dim'])
        self.dropout = nn.Dropout(config['dropout'])
        #self.hidden = nn.Parameters(self.batch_size, self.hidden_dim)
    
    def attnetwork(self, encoder_out, final_hidden):
        hidden = final_hidden.squeeze(0)
        
        #M = torch.tanh(encoder_out)
        # attention weights from the hidden layer of the encoder.
        attn_weights = torch.bmm(encoder_out, hidden.unsqueeze(2)).squeeze(2)
        soft_attn_weights = F.softmax(attn_weights, 1)
        new_hidden = torch.bmm(encoder_out.transpose(1,2), soft_attn_weights.unsqueeze(2)).squeeze(2)
        #print (wt.shape, new_hidden.shape)
        #new_hidden = torch.tanh(new_hidden)
        #print ('UP:', new_hidden, new_hidden.shape)
        
        return new_hidden
    
    def forward(self, sequence):
        emb_input = self.embedding(sequence)    
        inputx = self.dropout(emb_input)        
        
        ## inspired by CNN text classification, 2014.############# 
        inputx= inputx.permute(1,0,2)
        x = torch.unsqueeze(inputx, 1) # (1,1,16,300)
        
        xs = []
        for conv in self.convs:
            x2 = F.relu(conv(x))        # [B, F, T, 1]
            x2 = torch.squeeze(x2, -1)  # [B, F, T]
            x2 = F.max_pool1d(x2, x2.size(2))  # [B, F, 1]
            xs.append(x2)
        x = torch.cat(xs, 2)   
        x = x.view(x.size(0), -1)
        x= torch.unsqueeze(x, 0)
        ################################################
        
        if self.config['rnn_layer']=='lstm':
            output, (hn, cn) = self.encoder(x) # BILSTM
        if self.config['rnn_layer']=='gru':
            output, hn = self.encoder(x) # BIGRU        
        
        if self.config['has_attn']:
            fbout = output[:, :, :self.hidden_dim]+ output[:, :, self.hidden_dim:] # sum bidir outputs F+B
            fbout = fbout.permute(1,0,2)
            fbhn = (hn[-2,:,:]+hn[-1,:,:]).unsqueeze(0)
            attn_out = self.attnetwork(fbout, fbhn)  # attention layer
#             print(attn_out.shape) # 16,50
            
            logits = self.fc(attn_out)
#             print(logits.shape)
            
            return logits
        else:
            output = output.permute(1,0,2)
            output = torch.squeeze(output,1)
#             print(output.shape) # 16,100
            #attn1_out = self.attnetwork1(output, hn)
            
            logits = self.fc(output)
#             print(logits.shape)
            return logits

    
    


###### BRNN+ with/without attention layer#########################
class BRNN(nn.Module):
    def __init__(self, vocab_size, config,trainable=False, vec=None):
        super().__init__()
        self.hidden_dim = config['hidden_dim']
        self.batch_size = config['batch_size']
        self.emb_dim = config['emb_dim']
        self.gpu = config['gpu']
        self.config = config
        self.embedding = nn.Embedding(vocab_size, config['emb_dim'])
        if vec is not None:
            self.embedding.weight.data.copy_(vec) #load pretrained
            self.embedding.weight.requires_grad = trainable #non-trainable
        
        if config['rnn_layer'] == 'lstm':
            # encoder with bidirectional LSTM
            self.encoder = nn.LSTM(config['emb_dim'], config['hidden_dim'], num_layers=config['nlayers'], bidirectional=config['bidir'], dropout=config['dropout'])
         
        if config['rnn_layer'] == 'gru':
            self.encoder = nn.GRU(config['emb_dim'], config['hidden_dim'], 
                             num_layers= config['nlayers'], bidirectional=config['bidir'],
                             dropout=config['dropout'])
            
        if config['has_attn']:
            self.fc = nn.Linear(config['hidden_dim'], config['out_dim'])
        else:
            self.fc = nn.Linear(2* config['hidden_dim'], config['hidden_dim'])
            self.fc1 = nn.Linear(config['hidden_dim'], config['out_dim'])
        
        self.dropout = nn.Dropout(config['dropout'])
        #self.hidden = nn.Parameters(self.batch_size, self.hidden_dim)
    
    def attnetwork(self, encoder_out, final_hidden):
        hidden = final_hidden.squeeze(0)
        
        #M = torch.tanh(encoder_out)
        attn_weights = torch.bmm(encoder_out, hidden.unsqueeze(2)).squeeze(2)
        soft_attn_weights = F.softmax(attn_weights, 1)
        new_hidden = torch.bmm(encoder_out.transpose(1,2), soft_attn_weights.unsqueeze(2)).squeeze(2)
        #print (wt.shape, new_hidden.shape)
        #new_hidden = torch.tanh(new_hidden)
        #print ('UP:', new_hidden, new_hidden.shape)
        
        return new_hidden
    
    def forward(self, sequence):
        emb_input = self.embedding(sequence)    
        inputx = self.dropout(emb_input)
        if self.config['rnn_layer']=='lstm':
            output, (hn, cn) = self.encoder(inputx) # BILSTM
        if self.config['rnn_layer']=='gru':
            output, hn = self.encoder(inputx) # BIGRU
            
        if self.config['has_attn']:
            fbout = output[:, :, :self.hidden_dim]+ output[:, :, self.hidden_dim:] # sum bidir outputs F+B
            fbout = fbout.permute(1,0,2)
            fbhn = (hn[-2,:,:]+hn[-1,:,:]).unsqueeze(0)
            attn_out = self.attnetwork(fbout, fbhn)  # attention layer
            logits = self.fc(attn_out)
            return logits
        else:
            output = output.permute(1,0,2)
            output = torch.squeeze(output,1)
#             print(output.shape) # 16, 48,100
            #attn1_out = self.attnetwork1(output, hn)
            output_1 = self.fc(output)
#             print(output_1.shape)
            output_2 = self.fc1(output_1)
            logits = output_2.view(output_2.size(0), -1)
#             print(logits.shape)  ## 16, n, 3
            return logits


