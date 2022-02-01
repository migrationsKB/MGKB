import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.weight_norm import weight_norm
import math


def global_max_pooling(tensor, dim, topk):
    """Global max pooling"""
    ret, _ = torch.topk(tensor, topk, dim)
    return ret


class Gate_Attention(nn.Module):
    def __init__(self, num_hidden_a, num_hidden_b, num_hidden):
        super(Gate_Attention, self).__init__()
        self.hidden = num_hidden
        self.w1 = nn.Parameter(torch.Tensor(num_hidden_a, num_hidden))
        self.w2 = nn.Parameter(torch.Tensor(num_hidden_b, num_hidden))
        self.bias = nn.Parameter(torch.Tensor(num_hidden))
        self.reset_parameter()

    def reset_parameter(self):
        stdv1 = 1. / math.sqrt(self.hidden)
        stdv2 = 1. / math.sqrt(self.hidden)
        stdv = (stdv1 + stdv2) / 2.
        self.w1.data.uniform_(-stdv1, stdv1)
        self.w2.data.uniform_(-stdv2, stdv2)
        self.bias.data.uniform_(-stdv, stdv)

    def forward(self, a, b):
        wa = torch.matmul(a, self.w1)
        wb = torch.matmul(b, self.w2)
        gated = wa + wb + self.bias
        gate = torch.sigmoid(gated)
        output = gate * a + (1 - gate) * b
        return output


class BaseModel(nn.Module):
    def __init__(self, vocab_size, config, trainable=True, vec=None):
        super(BaseModel, self).__init__()

        self.hidden_dim = config['hidden_dim']
        self.mid_dim = config['mid_dim']
        self.out_dim = config['out_dim']
        self.dropout = config['dropout']
        self.theta_dim = config['theta_dim']
        self.bert_dim = config['bert_dim']

        # the bottom model for semantic representation, cnn+biLSTM+attention
        self.cnnbrnn_model = cnnBRNN(vocab_size, config, trainable=True, vec=vec)

        ### gru layer on top of bert
        self.gru_layer = nn.GRU(self.bert_dim, self.hidden_dim, num_layers=2, dropout=self.dropout, bidirectional=True)

        self.gru_layer_topic = nn.GRU(self.hidden_dim * 2, self.hidden_dim, num_layers=2, dropout=self.dropout,
                                      bidirectional=True)

        # the latent fusion layer, 
        self.model_gct = Gate_combine_three(self.hidden_dim, self.mid_dim, self.dropout)

        # linear layers for converting the dimensions for senti/topic embeddings.
        self.fc_topic = nn.Linear(self.theta_dim, self.hidden_dim * 2)
        self.fc_senti = nn.Linear(self.bert_dim, self.hidden_dim)

        ## fcnet for output.
        self.fcnet = FCNet(self.hidden_dim, self.out_dim, self.dropout)

        # SimpleClassifier:
        self.simple_classifier = SimpleClassifier(self.hidden_dim, self.mid_dim, self.out_dim, self.dropout)

    #         self.fc = nn.Linear(self.last_dim, self.out_dim)

    def attnetwork(self, encoder_out, final_hidden):
        hidden = final_hidden.squeeze(0)

        # M = torch.tanh(encoder_out)
        # attention weights from the hidden layer of the encoder.
        attn_weights = torch.bmm(encoder_out, hidden.unsqueeze(2))
        attn_weights = attn_weights.permute((1, 0, 2))  # 16,290,1
        attn_weights = attn_weights.squeeze(2)  # 16,290
        soft_attn_weights = F.softmax(attn_weights, 1)  # 16,290
        encoder = encoder_out.permute(1, 2, 0)  # [16, 100, 290]
        soft_attn_weights = soft_attn_weights.unsqueeze(2)  # [16,290,1]
        new_hidden = torch.bmm(encoder, soft_attn_weights).squeeze(2)
        return new_hidden

    def forward(self, senti_repr, semantic_repr, topic_repr):
        ### semantic representation, trained from cnnbrnn.
        semantic_repr = self.cnnbrnn_model(semantic_repr)  # (16,100)
        ### topic representation, trained from ETM.

        #         topic_repr = self.fc_topic(topic_repr) # (16,100)

        #         topic_repr = topic_repr.unsqueeze(1)
        #         out_, hn_ = self.gru_layer_topic(topic_repr) # out: ([16, 290, 200])
        #         fbout_ = out_[:, :, :self.hidden_dim]+ out_[:, :, self.hidden_dim:]
        #         fbout_ = fbout_.permute(1,0,2)
        #         fbhn_ = (hn_[-2,:,:]+hn_[-1,:,:]).unsqueeze(0)
        #         attn_out_topic = self.attnetwork(fbout_, fbhn_) # 16,100

        ### sentiment representation, output from fine-tuned bert.
        ## (16,290,768)
        out, hn = self.gru_layer(senti_repr)  # out: ([16, 290, 200])

        fbout = out[:, :, :self.hidden_dim] + out[:, :, self.hidden_dim:]
        fbout = fbout.permute(1, 0, 2)
        fbhn = (hn[-2, :, :] + hn[-1, :, :]).unsqueeze(0)
        attn_out_senti = self.attnetwork(fbout, fbhn)  # 16,100

        t = semantic_repr + attn_out_senti

        #         t = semantic_repr + attn_out_topic+attn_out_senti
        #### f: senti_repr, a:proj_topic, q:semantic
        # fuse the representations in the gated attention layer
        #         fused_output = self.model_gct(attn_out_senti, semantic_repr, attn_out_topic )

        #         fused_output = self.model_gct(attn_out, semantic_repr, topic_repr)
        # fused_output = self.model_gct(semantic_repr, topic_repr, attn_out)
        ##output from fcn.

        output = self.simple_classifier(t)

        return output


class DeepModel(nn.Module):
    def __init__(self, vocab_size, config, trainable=True, vec=None):
        super(DeepModel, self).__init__()

        self.hidden_dim = config['hidden_dim']
        self.mid_dim = config['mid_dim']
        self.out_dim = config['out_dim']
        self.dropout = config['dropout']
        self.theta_dim = config['theta_dim']
        self.bert_dim = config['bert_dim']

        # the bottom model for semantic representation, cnn+biLSTM+attention
        self.cnnbrnn_model = cnnBRNN(vocab_size, config, trainable=True, vec=vec)

        ### gru layer on top of bert
        self.gru_layer = nn.GRU(self.bert_dim, self.hidden_dim, num_layers=2, dropout=self.dropout, bidirectional=True)

        self.gru_layer_topic = nn.GRU(self.hidden_dim * 2, self.hidden_dim, num_layers=2, dropout=self.dropout,
                                      bidirectional=True)

        # the latent fusion layer, 
        self.model_gct = Gate_combine_three(self.hidden_dim, self.mid_dim, self.dropout)

        # linear layers for converting the dimensions for senti/topic embeddings.
        self.fc_topic = nn.Linear(self.theta_dim, self.hidden_dim * 2)
        self.fc_senti = nn.Linear(3, self.hidden_dim * 2)

        ## fcnet for output.
        self.fcnet = FCNet(self.hidden_dim, self.out_dim, self.dropout)

        # SimpleClassifier:
        self.simple_classifier = SimpleClassifier(self.hidden_dim, self.mid_dim, self.out_dim, self.dropout)

    #         self.fc = nn.Linear(self.last_dim, self.out_dim)

    def attnetwork(self, encoder_out, final_hidden):
        hidden = final_hidden.squeeze(0)

        # M = torch.tanh(encoder_out)
        # attention weights from the hidden layer of the encoder.
        attn_weights = torch.bmm(encoder_out, hidden.unsqueeze(2))
        attn_weights = attn_weights.permute((1, 0, 2))  # 16,290,1
        attn_weights = attn_weights.squeeze(2)  # 16,290
        soft_attn_weights = F.softmax(attn_weights, 1)  # 16,290
        encoder = encoder_out.permute(1, 2, 0)  # [16, 100, 290]
        soft_attn_weights = soft_attn_weights.unsqueeze(2)  # [16,290,1]
        new_hidden = torch.bmm(encoder, soft_attn_weights).squeeze(2)
        return new_hidden

    def forward(self, senti_repr, semantic_repr, topic_repr):
        ### semantic representation, trained from cnnbrnn.
        semantic_repr = self.cnnbrnn_model(semantic_repr)  # (16,100)
        ### topic representation, trained from ETM.

        topic_repr = self.fc_topic(topic_repr)  # (16,100)
        topic_repr = topic_repr.unsqueeze(1)
        out_, hn_ = self.gru_layer_topic(topic_repr)  # out: ([16, 290, 200])
        fbout_ = out_[:, :, :self.hidden_dim] + out_[:, :, self.hidden_dim:]
        fbout_ = fbout_.permute(1, 0, 2)
        fbhn_ = (hn_[-2, :, :] + hn_[-1, :, :]).unsqueeze(0)
        attn_out_topic = self.attnetwork(fbout_, fbhn_)  # 16,100

        ### sentiment representation, output from fine-tuned bert.
        ## senti_repr:(16,3)
        senti_repr = self.fc_senti(senti_repr)  # out: ([16, 1, 3])
        senti_repr = senti_repr.unsqueeze(1)
        out, hn = self.gru_layer_topic(senti_repr)  # out: ([16, 290, 200])
        fbout = out[:, :, :self.hidden_dim] + out[:, :, self.hidden_dim:]
        fbout = fbout.permute(1, 0, 2)
        fbhn = (hn[-2, :, :] + hn[-1, :, :]).unsqueeze(0)
        attn_out_senti = self.attnetwork(fbout, fbhn)  # 16,100

        #         t = semantic_repr + attn_out_topic+attn_out_senti
        #### f: senti_repr, a:proj_topic, q:semantic
        # fuse the representations in the gated attention layer
        t = self.model_gct(attn_out_senti, attn_out_topic, semantic_repr)
        output = self.simple_classifier(t)

        return output


class SimpleClassifier(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, dropout):
        super(SimpleClassifier, self).__init__()
        layer = [
            weight_norm(nn.Linear(in_dim, hid_dim), dim=None),
            nn.ReLU(),
            nn.Dropout(dropout, inplace=True),
            weight_norm(nn.Linear(hid_dim, out_dim), dim=None)
        ]
        self.main = nn.Sequential(*layer)

    def forward(self, x):
        logits = self.main(x)
        return logits


class FCNet(nn.Module):
    def __init__(self, in_dim, out_dim, dropout):
        super(FCNet, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.relu = nn.ReLU()
        self.linear = weight_norm(nn.Linear(in_dim, out_dim), dim=None)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        logits = self.dropout(self.linear(x))
        return logits


class Gate_combine_three(nn.Module):
    def __init__(self, hidden, mid, dropout):
        super(Gate_combine_three, self).__init__()
        self.f_proj = nn.Linear(hidden, mid)
        self.a_proj = nn.Linear(hidden, mid)
        self.f_att = nn.Linear(mid, 1)
        self.a_att = nn.Linear(mid, 1)
        self.sig = nn.Sigmoid()
        self.dropout = nn.Dropout(dropout)

    def forward(self, f, a, q):
        # f: senti_repr, a:proj_topic, q:semantic
        f_proj = self.f_proj(f + q)
        f_proj = self.dropout(f_proj)
        f_g = self.sig(self.f_att(f_proj))

        a_proj = self.a_proj(a + q)
        a_proj = self.dropout(a_proj)
        a_g = self.sig(self.a_att(a_proj))

        fa_comb = f_g * f + a_g * a + q
        return fa_comb


### cnn + RNN
class cnnRNN(nn.Module):
    def __init__(self, vocab_size, config, filter_size=100, window_sizes=(3, 4, 5), trainable=False, vec=None):
        super().__init__()
        self.hidden_dim = config['hidden_dim']
        self.batch_size = config['batch_size']
        self.emb_dim = config['emb_dim']
        self.gpu = config['gpu']
        self.config = config
        self.embedding = nn.Embedding(vocab_size, config['emb_dim'])
        if vec is not None:
            self.embedding.weight.data.copy_(vec)  # load pretrained
            self.embedding.weight.requires_grad = trainable  # non-trainable

        #### CNN ###########
        ## https://github.com/yongjincho/cnn-text-classification-pytorch/blob/master/model.py

        self.convs = nn.ModuleList([
            nn.Conv2d(1, filter_size, [window_size, 300], padding=(window_size - 1, 0))
            for window_size in window_sizes
        ])

        if config['rnn_layer'] == 'lstm':
            self.encoder = nn.LSTM(filter_size * len(window_sizes), config['hidden_dim'],
                                   num_layers=config['nlayers'], bidirectional=False,
                                   dropout=config['dropout'])

        if config['rnn_layer'] == 'gru':
            self.encoder = nn.GRU(filter_size * len(window_sizes), config['hidden_dim'],
                                  num_layers=config['nlayers'], bidirectional=False,
                                  dropout=config['dropout'])
        if config['has_attn']:
            half_hidden_dim = int(config['hidden_dim'] / 2)
            self.fc = nn.Linear(half_hidden_dim, config['out_dim'])
        else:
            self.fc = nn.Linear(config['hidden_dim'], config['out_dim'])
        self.dropout = nn.Dropout(config['dropout'])
        # self.hidden = nn.Parameters(self.batch_size, self.hidden_dim)

    def attnetwork(self, encoder_out, final_hidden):
        hidden = final_hidden.squeeze(0)

        # M = torch.tanh(encoder_out)
        # attention weights from the hidden layer of the encoder.
        attn_weights = torch.bmm(encoder_out, hidden.unsqueeze(2)).squeeze(2)
        soft_attn_weights = F.softmax(attn_weights, 1)
        new_hidden = torch.bmm(encoder_out.transpose(1, 2), soft_attn_weights.unsqueeze(2)).squeeze(2)
        # print (wt.shape, new_hidden.shape)
        # new_hidden = torch.tanh(new_hidden)
        # print ('UP:', new_hidden, new_hidden.shape)

        return new_hidden

    def forward(self, sequence):
        emb_input = self.embedding(sequence)
        inputx = self.dropout(emb_input)

        ## inspired by CNN text classification, 2014.############# 

        inputx = inputx.permute(1, 0, 2)
        x = torch.unsqueeze(inputx, 1)  # (1,1,16,300)

        xs = []
        for conv in self.convs:
            x2 = F.relu(conv(x))  # [B, F, T, 1]
            x2 = torch.squeeze(x2, -1)  # [B, F, T]
            x2 = F.max_pool1d(x2, x2.size(2))  # [B, F, 1]
            xs.append(x2)
        x = torch.cat(xs, 2)
        x = x.view(x.size(0), -1)
        x = torch.unsqueeze(x, 0)
        ################################################

        if self.config['rnn_layer'] == 'lstm':
            output, (hn, cn) = self.encoder(x)  # BILSTM
        if self.config['rnn_layer'] == 'gru':
            output, hn = self.encoder(x)  # BIGRU

        if self.config['has_attn']:
            half_hidden_dim = int(self.hidden_dim / 2)
            fbout = output[:, :, :half_hidden_dim] + output[:, :, half_hidden_dim:]  # sum bidir outputs F+B
            #             print(fbout.shape)
            fbout = fbout.permute(1, 0, 2)
            fbhn = (hn[-2, :, :half_hidden_dim] + hn[-1, :, half_hidden_dim:]).unsqueeze(0)
            attn_out = self.attnetwork(fbout, fbhn)  # attention layer
            #             print(attn_out.shape) # 16,50
            lineared = self.fc(attn_out)
            return lineared

        else:
            output = output.permute(1, 0, 2)
            output = torch.squeeze(output, 1)
            #             print(output.shape) # 16,100
            #             attn1_out = self.attnetwork1(output, hn)
            lineared = self.fc(output)
            return lineared


#### cnn2d + GRU, with/without attention ########## 
class cnnBRNN(nn.Module):
    def __init__(self, vocab_size, config, filter_size=100, window_sizes=(3, 4, 5), trainable=False, vec=None):
        super().__init__()
        self.hidden_dim = config['hidden_dim']
        self.batch_size = config['batch_size']
        self.emb_dim = config['emb_dim']
        self.gpu = config['gpu']
        self.config = config
        self.embedding = nn.Embedding(vocab_size, config['emb_dim'])
        if vec is not None:
            self.embedding.weight.data.copy_(vec)  # load pretrained
            self.embedding.weight.requires_grad = trainable  # non-trainable

        #### CNN ###########
        ## https://github.com/yongjincho/cnn-text-classification-pytorch/blob/master/model.py

        self.convs = nn.ModuleList([
            nn.Conv2d(1, filter_size, [window_size, 300], padding=(window_size - 1, 0))
            for window_size in window_sizes
        ])

        if config['rnn_layer'] == 'lstm':
            self.encoder = nn.LSTM(filter_size * len(window_sizes), config['hidden_dim'],
                                   num_layers=config['nlayers'], bidirectional=config['bidir'],
                                   dropout=config['dropout'])

        if config['rnn_layer'] == 'gru':
            self.encoder = nn.GRU(filter_size * len(window_sizes), config['hidden_dim'],
                                  num_layers=config['nlayers'], bidirectional=config['bidir'],
                                  dropout=config['dropout'])
        if config['has_attn']:
            self.fc = nn.Linear(config['hidden_dim'], config['out_dim'])
        else:
            self.fc = nn.Linear(2 * config['hidden_dim'], config['out_dim'])
        self.dropout = nn.Dropout(config['dropout'])
        # self.hidden = nn.Parameters(self.batch_size, self.hidden_dim)

    def attnetwork(self, encoder_out, final_hidden):
        hidden = final_hidden.squeeze(0)

        # M = torch.tanh(encoder_out)
        # attention weights from the hidden layer of the encoder.
        attn_weights = torch.bmm(encoder_out, hidden.unsqueeze(2)).squeeze(2)
        soft_attn_weights = F.softmax(attn_weights, 1)
        new_hidden = torch.bmm(encoder_out.transpose(1, 2), soft_attn_weights.unsqueeze(2)).squeeze(2)
        # print (wt.shape, new_hidden.shape)
        # new_hidden = torch.tanh(new_hidden)
        # print ('UP:', new_hidden, new_hidden.shape)

        return new_hidden

    def forward(self, sequence):
        emb_input = self.embedding(sequence)
        inputx = self.dropout(emb_input)

        ## inspired by CNN text classification, 2014.############# 

        inputx = inputx.permute(1, 0, 2)
        x = torch.unsqueeze(inputx, 1)  # (1,1,16,300)

        xs = []
        for conv in self.convs:
            x2 = F.relu(conv(x))  # [B, F, T, 1]
            x2 = torch.squeeze(x2, -1)  # [B, F, T]
            x2 = F.max_pool1d(x2, x2.size(2))  # [B, F, 1]
            xs.append(x2)
        x = torch.cat(xs, 2)
        x = x.view(x.size(0), -1)
        x = torch.unsqueeze(x, 0)
        ################################################

        if self.config['rnn_layer'] == 'lstm':
            output, (hn, cn) = self.encoder(x)  # BILSTM
        if self.config['rnn_layer'] == 'gru':
            output, hn = self.encoder(x)  # BIGRU

        if self.config['has_attn']:
            fbout = output[:, :, :self.hidden_dim] + output[:, :, self.hidden_dim:]  # sum bidir outputs F+B
            fbout = fbout.permute(1, 0, 2)
            fbhn = (hn[-2, :, :] + hn[-1, :, :]).unsqueeze(0)
            attn_out = self.attnetwork(fbout, fbhn)  # attention layer
            #             lineared = self.fc(attn_out)
            return attn_out

        else:
            output = output.permute(1, 0, 2)
            output = torch.squeeze(output, 1)
            #             lineared = self.fc(output)
            return output


###### BRNN+ with/without attention layer#########################
class BRNN(nn.Module):
    def __init__(self, vocab_size, config, trainable=False, vec=None):
        super().__init__()
        self.hidden_dim = config['hidden_dim']
        self.batch_size = config['batch_size']
        self.emb_dim = config['emb_dim']
        self.gpu = config['gpu']
        self.config = config
        self.embedding = nn.Embedding(vocab_size, config['emb_dim'])
        if vec is not None:
            self.embedding.weight.data.copy_(vec)  # load pretrained
            self.embedding.weight.requires_grad = trainable  # non-trainable

        if config['rnn_layer'] == 'lstm':
            # encoder with bidirectional LSTM
            self.encoder = nn.LSTM(config['emb_dim'], config['hidden_dim'], num_layers=config['nlayers'],
                                   bidirectional=config['bidir'], dropout=config['dropout'])

        if config['rnn_layer'] == 'gru':
            self.encoder = nn.GRU(config['emb_dim'], config['hidden_dim'],
                                  num_layers=config['nlayers'], bidirectional=config['bidir'],
                                  dropout=config['dropout'])

        if config['has_attn']:
            self.fc = nn.Linear(config['hidden_dim'], config['out_dim'])
        else:
            self.fc = nn.Linear(2 * config['hidden_dim'], config['hidden_dim'])
            self.fc1 = nn.Linear(config['hidden_dim'], config['out_dim'])

        self.dropout = nn.Dropout(config['dropout'])
        # self.hidden = nn.Parameters(self.batch_size, self.hidden_dim)

    def attnetwork(self, encoder_out, final_hidden):
        # ecnoder_out: (290,16,100) , final_hidden: ([1, 290, 100])
        hidden = final_hidden.squeeze(0)

        # M = torch.tanh(encoder_out)
        attn_weights = torch.bmm(encoder_out, hidden.unsqueeze(2)).squeeze(2)
        soft_attn_weights = F.softmax(attn_weights, 1)
        new_hidden = torch.bmm(encoder_out.transpose(1, 2), soft_attn_weights.unsqueeze(2)).squeeze(2)
        # print (wt.shape, new_hidden.shape)
        # new_hidden = torch.tanh(new_hidden)
        # print ('UP:', new_hidden, new_hidden.shape)

        return new_hidden

    def forward(self, sequence):
        emb_input = self.embedding(sequence)
        inputx = self.dropout(emb_input)
        if self.config['rnn_layer'] == 'lstm':
            output, (hn, cn) = self.encoder(inputx)  # BILSTM
        if self.config['rnn_layer'] == 'gru':
            output, hn = self.encoder(inputx)  # BIGRU

        if self.config['has_attn']:
            fbout = output[:, :, :self.hidden_dim] + output[:, :, self.hidden_dim:]  # sum bidir outputs F+B
            fbout = fbout.permute(1, 0, 2)
            fbhn = (hn[-2, :, :] + hn[-1, :, :]).unsqueeze(0)
            attn_out = self.attnetwork(fbout, fbhn)  # attention layer
            lineared = self.fc(attn_out)
            return lineared
        else:
            output = output.permute(1, 0, 2)
            output = torch.squeeze(output, 1)
            output_1 = self.fc(output)
            output_2 = self.fc1(output_1)

            lineared = output_2.view(output_2.size(0), -1)
            return lineared
