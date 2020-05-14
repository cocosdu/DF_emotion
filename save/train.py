import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils
from sklearn.model_selection import train_test_split,StratifiedKFold
import time

import utils

with open("midSaveData/weights_100_word2vec_train_test_unsplit.pickle",'rb') as f:
    Data = pickle.load(f)

#Data = {"train":train_id,"label":y,"xtest":xtest_id,"ids":ids,
#        "pre_weights":pre_weights,"WORD2IDX":WORD2IDX,"train_seq":train_data,
#        "xtest_seq":test_data,'unlabel':unlabel_id}

train_data =np.array(Data['train'])
label =np.array(Data['label'])
#xtrain,xvalid,ytrain,yvalid = train_test_split(Data['train'],Data['label'],stratify=Data['label'],test_size=0.1,shuffle=True)
xtest =Data['xtest']
ids =Data['ids']

batch_size = 32
epoches = 1
learning_rate = 0.001
device = torch.device("cuda")
Vocab_size = len(Data['WORD2IDX'])
embedding_dim = len(Data['pre_weights'][0])
hidden_dim = 128
extra_input_dim = 0#300

#random_seed = 40
#np.random.seed(random_seed)

max_len = 0


class EmotionClassifier(nn.Module):
    def __init__(self,vocab_size,embedding_dim,extra_input_dim,hidden_dim,pre_weights=None,num_layers=1,bidirectional = True):
        super(EmotionClassifier,self).__init__()
        self.num_layers = num_layers
        self.num_directions = 2 if bidirectional else 1 
        self.embedding = nn.Embedding(vocab_size,embedding_dim)
        self.embedding.weight.data.copy_(pre_weights)
        self.dropout_layer = nn.Dropout(0.3)
        #self.pooling_layer = nn.Linear(max_len,1)

        self.lstm = nn.LSTM(embedding_dim,hidden_dim,num_layers=num_layers,bidirectional = bidirectional ,batch_first=True)
        #self.hidden2class = nn.Linear(num_layers*self.num_directions*hidden_dim,3)
        self.hidden2class = nn.Linear(num_layers*self.num_directions*hidden_dim+extra_input_dim,3)
    def forward(self,data,data_lens,extra_input):
        embs= self.embedding(data) #32*seq *100
        packed_embeds = rnn_utils.pack_padded_sequence(embs,data_lens,batch_first=True,enforce_sorted=True)
        #降序排列
        packed_out,packed_h_t_c_t = self.lstm(packed_embeds) #out batch,seq_len,num_directions*hidden_size
        #h_n ( batch,num_layers * num_directions, hidden_size)
        '''
        #用输出的最大值 
        out,_ = rnn_utils.pad_packed_sequence(packed_out,batch_first=True,total_length=max_len) #batch*seq_len
        att_out = self.pooling_layer(out.contiguous().view(out.size(0),out.size(2),out.size(1)))
        #att_out = torch.max(out,dim=1)[0] #batch, 128
        #att_out = torch.mean(out,dim=1)
        linear_out = self.hidden2class(self.dropout_layer(att_out.squeeze(2))) #batch*3
        '''
         #用隐藏层
        h_t = packed_h_t_c_t[0] #[2, 32, 128]
        h_t = h_t.transpose(0,1)
        h_t = h_t.contiguous().view(h_t.size(0),-1) #batch

        #h_t = torch.cat([h_t,extra_input],1)
        linear_out = self.hidden2class(self.dropout_layer(h_t)) #batch*3
        

        scores = F.log_softmax(linear_out,dim=1)
        return scores

def train_or_test(model,data,label,data_show,data_index,batch_size,train_type,optimizer,epoch,lossF=nn.NLLLoss(reduction='sum')):
    #F.nll_loss nn.NLLLoss(reduction='sum')
    start_time = time.time()
    if train_type=="train":
        np.random.shuffle(data_index)
        model.train()
    else:
        model.eval()

    losses = []
    preds =[]

    Y =[]
    test_probs =[]
    for i in range(0,len(data),batch_size):
        data_batch,label_batch,lens = utils.get_minibatch(data,label,data_index,i,batch_size,max_len)
        scores = model(data_batch,lens,None)
        if train_type!='test':
            loss = lossF(scores,label_batch)
            losses.append(loss.item())

        scores = scores.data.cpu().numpy()
        if train_type=='test':
            test_probs.extend(scores)
        pred = [np.argmax(s) for s in scores]
        preds.extend(pred)
        Y.extend(label_batch.data.cpu().numpy()) #valid 是label test是ids
        
        if train_type=="train":
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params,1)
            optimizer.step()

    if train_type=='valid':
        mean_loss = np.mean(losses)
        P,F1 = utils.getF1(preds,Y)
        #utils.show_result_txt(preds,Y,data_index,data_show,"valid_epoch%d"%(epoch))
        print("Valid epoch:%d time:%.4f loss:%.4f  F1:%.4f  P:%.4f"%(epoch,time.time()-start_time,mean_loss,F1,P))
    elif train_type=='train':
        mean_loss = np.mean(losses)
        P,F1 = utils.getF1(preds,Y)
        print("--Train epoch:%d time:%.4f loss:%.4f  F1:%.4f P:%.4f"%(epoch,time.time()-start_time,mean_loss,F1,P))
    else:
        print(np.shape(test_probs))
        #np.save("LSTM_test_scores_%d.npy"%(epoch),test_probs)
        #np.save("test_ids.npy",Y)
        utils.saveResult(Y,preds,'LSTM_Result_%d'%(epoch))
        return test_probs,Y
'''
pre_weights = torch.tensor(Data['pre_weights'],dtype =torch.long,device = device)
model = EmotionClassifier(Vocab_size,embedding_dim,extra_input_dim,hidden_dim,pre_weights)
if torch.cuda.is_available():
    print("cuda is avaliable")
    model.cuda()

params = filter(lambda p: p.requires_grad, model.parameters())
optimizer = optim.Adam(params, lr=learning_rate)

train_data_index = np.arange(len(xtrain))
valid_data_index = np.arange(len(xvalid))
test_data_index = np.arange(len(xtest))

for epoch in range(1,epoches+1):
    train_or_test(model,xtrain,ytrain,None,train_data_index,batch_size,'train',optimizer,epoch)
    with torch.no_grad():
        #train_or_test(model,Data['xtrain'],Data['ytrain'],Data['xtrain_seq'],train_data_index,batch_size,'valid',optimizer,epoch)
        train_or_test(model,xvalid,yvalid,None,valid_data_index,batch_size,'valid',optimizer,epoch)
        train_or_test(model,xtest,ids,None,test_data_index,batch_size,'test',None,epoch)
'''
skf = StratifiedKFold(n_splits = 5).split(train_data,label)

res =[]
for i,(train_index,valid_index) in enumerate(skf):
    xtrain,ytrain = train_data[train_index],label[train_index]
    xvalid,yvalid = train_data[valid_index],label[valid_index]
    train_data_index =np.arange(len(xtrain))
    valid_data_index = np.arange(len(xvalid))
    test_data_index = np.arange(len(xtest))

    pre_weights = torch.tensor(Data['pre_weights'],dtype =torch.long,device = device)
    model = EmotionClassifier(Vocab_size,embedding_dim,extra_input_dim,hidden_dim,pre_weights)
    model.cuda()

    params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.Adam(params, lr=learning_rate)
    for epoch in range(1,epoches+1):

        train_or_test(model,xtrain,ytrain,None,train_data_index,batch_size,'train',optimizer,epoch)
        with torch.no_grad():
            #train_or_test(model,Data['xtrain'],Data['ytrain'],Data['xtrain_seq'],train_data_index,batch_size,'valid',optimizer,epoch)
            train_or_test(model,xvalid,yvalid,None,valid_data_index,batch_size,'valid',optimizer,epoch)
            test_scores,Y = train_or_test(model,xtest,ids,None,test_data_index,batch_size,'test',None,epoch)
            print(Y[:5])
    res.append(test_scores)
    print("finish %d fold"%(i))
average = np.average(res,axis=0)
preds = np.argmax(average,axis=1)
utils.saveResult(Y,preds,"kfold/result_kfold")




