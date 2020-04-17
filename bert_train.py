import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils
import time
from transformers import BertTokenizer, BertModel
from transformers import AdamW, get_linear_schedule_with_warmup
import logging
import bert_utils

logger =logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s: - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S')
fh = logging.FileHandler('bert_log.txt')
fh.setLevel(logging.DEBUG)
fh.setFormatter(formatter)

ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
ch.setFormatter(formatter)
logger.addHandler(fh)
#logger.addHandler(ch)


batch_size = 4 #256
epoches = 3
learning_rate = 2e-5
device = torch.device("cuda")

random_seed = 42
np.random.seed(random_seed)

pretrained_model = BertModel.from_pretrained('bert-base-chinese')
# https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-chinese-pytorch_model.bin
tokenizer =BertTokenizer.from_pretrained('bert-base-chinese')
#pretrained_model = BertModel.from_pretrained('hfl/chinese-roberta-wwm-ext')
#tokenizer =BertTokenizer.from_pretrained('hfl/chinese-roberta-wwm-ext')
#pretrained_model = BertModel.from_pretrained('hfl/chinese-roberta-wwm-ext-large')
#tokenizer =BertTokenizer.from_pretrained('hfl/chinese-roberta-wwm-ext-large')

max_len = 150
xtrain,xvalid,ytrain,yvalid,test_data,ids = bert_utils.get_split_data(random_seed)
#train_data,label,test_data,ids = bert_utils.get_Bert_whole_data()


class BertEmotionClassifier(nn.Module):
    def __init__(self,pretrained_model,class_size):
        super(BertEmotionClassifier,self).__init__()
        self.class_size = class_size
        self.pretrained_model = pretrained_model
        self.hidden_dim = self.pretrained_model.config.hidden_size
        self.dropout_layer = nn.Dropout(0.15)#0.15
        self.pooling =nn.Linear(max_len,1)

        self.hidden2class = nn.Linear(self.hidden_dim,self.class_size)

        #self.init_weights()
    def init_weights(self,initrange=0.2):
        self.hidden2class.weight.data.uniform_(-initrange,initrange)
        self.hidden2class.bias.data.uniform_(-initrange,initrange)
    
    def forward(self,data):
        tokens,segments,mask = data['tokens'],data['segments'],data['mask']
        outputs = self.pretrained_model(tokens,token_type_ids=segments,attention_mask=mask)
        # last_hidden_state,pooler_output,hidden_states,attentions
        #output = outputs[1] #batch_size*hidden_size
        output = outputs[0] #batch_size*sequence_length*hidden_size
        #output = self.pooling(output.contiguous().view(output.size(0),output.size(2),output.size(1)))
        #output = torch.mean(output,1) #globalaveragepooling1D 
        output = torch.max(output,1)[0]
        #print(output.size())

        linear_out = self.hidden2class(self.dropout_layer(output))
        scores = F.log_softmax(linear_out,dim=1)
        return scores

def train_or_test(model,data,label,data_index,batch_size,train_type,optimizer,epoch,lossF=nn.NLLLoss(reduction='sum')):#F.nll_loss
    start_time = time.time()
    if train_type=="train":
        np.random.shuffle(data_index)
        model.train()
    else:
        model.eval()
    losses = []
    preds =[]
    Y =[]
    for i in range(0,len(data),batch_size):
        indexes = data_index[i:i+batch_size]
        data_batch,label_batch = bert_utils.get_minibatch(data,label,indexes,train_type)
        data_bert = bert_utils.prepare_inputs_for_bert(data_batch,max_len,tokenizer)
        scores = model(data_bert)
        if train_type!='test':
            loss = lossF(scores,label_batch)
            losses.append(loss.item())
            label_batch =label_batch.data.cpu().numpy()
        scores = scores.data.cpu().numpy()
        pred = np.argmax(scores,axis=1)  #[np.argmax(s) for s in scores]
        preds.extend(pred)
        Y.extend(label_batch) #valid 是label test是ids
        
        if train_type=="train":
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params,1)
            optimizer.step()
            scheduler.step()


    if train_type=='valid':
        mean_loss = np.mean(losses)
        P,F1 = bert_utils.getF1(preds,Y)
        print("Valid epoch:%d time:%.4f loss:%.4f  F1:%.4f  P:%.4f"%(epoch,time.time()-start_time,mean_loss,F1,P))
    elif train_type=='train':
        mean_loss = np.mean(losses)
        P,F1 = bert_utils.getF1(preds,Y)
        print("--Train epoch:%d time:%.4f loss:%.4f  F1:%.4f P:%.4f"%(epoch,time.time()-start_time,mean_loss,F1,P))
    else:
        bert_utils.saveResult(Y,preds,'Bert_Result_100_%d'%(epoch))

model = BertEmotionClassifier(pretrained_model,3)
if torch.cuda.is_available():
    print("cuda is avaliable")
    model.cuda()

params = filter(lambda p: p.requires_grad, model.parameters())
named_params = list(model.named_parameters())
no_decay =['bias','LayerNorm.weight']
group_parameters =[{'params':[p for n,p in named_params if not any(nd in n for nd in no_decay)],'weight_decay':0.01},
  {'params':[p for n,p in named_params if any(nd in n for nd in no_decay)],'weight_decay':0.0}]
#optimizer = optim.Adam(params, lr=learning_rate)
optimizer = AdamW(group_parameters,lr=learning_rate)
num_steps =len(xtrain)//batch_size*epoches
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.1*num_steps), num_training_steps=num_steps)  # PyTorch scheduler

'''
temp =1000
xtrain =xtrain[:temp]
xvalid =xvalid[:temp]
test_data =test_data[:temp]
'''


train_data_index = np.arange(len(xtrain))
valid_data_index = np.arange(len(xvalid))
test_data_index = np.arange(len(test_data))

for epoch in range(1,epoches+1):

    train_or_test(model,xtrain,ytrain,train_data_index,batch_size,'train',optimizer,epoch)
    with torch.no_grad():
        train_or_test(model,xvalid,yvalid,valid_data_index,batch_size,'valid',optimizer,epoch)
        train_or_test(model,test_data,ids,test_data_index,batch_size,'test',None,epoch)
