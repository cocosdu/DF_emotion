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
import argparse
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

parser = argparse.ArgumentParser()
parser.add_argument('--random_seed',type=int,default =42,help='for split train valid data')
parser.add_argument('--bert_model',type=str,default='base',help='bert-base-chinese:base  hfl/chinese-roberta-wwm-ext: wwm hfl/chinese-roberta-wwm-ext-large: large')
parser.add_argument('--test_function',type=bool,default=False,help='test with 1000 samples')
parser.add_argument('--batch_size',type=int,default=4)
parser.add_argument('--save_model',type=bool,default=False)
parser.add_argument('--dropout',type=float,default=0.15)
parser.add_argument('--clip_grad',type=int,default=1)
parser.add_argument('--maxlen',type=int,default=150)
parser.add_argument('--max_epoch',type=int,default=3)
parser.add_argument('--RNN_type',type=str,default=None)
parser.add_argument('--rnn_hidden_size',type=int,default=128)
parser.add_argument('--rnn_layers',type=int,default=1)
parser.add_argument('--Train_valid',type=bool,default=False)
parser.add_argument('--learning_rate',type=float,default=2e-5)
parser.add_argument('--N_batch_optimizer',type = int,default=1)
parser.add_argument('--Train_test',type=bool,default=False)
args = parser.parse_args()


batch_size = args.batch_size
epoches = args.max_epoch
learning_rate = args.learning_rate
device = torch.device("cuda")

random_seed = args.random_seed
np.random.seed(random_seed)

BERT_base = 'bert-base-chinese'
BERT_wwm = 'hfl/chinese-roberta-wwm-ext'
BERT_wwm_large = 'hfl/chinese-roberta-wwm-ext-large'

BERT_MODEL = None
tokenizer_path =None
if args.bert_model=='base':
    BERT_MODEL = BERT_base
    tokenizer_path =BERT_MODEL
elif args.bert_model=='wwm':
    BERT_MODEL = BERT_wwm
    tokenizer_path = BERT_MODEL
elif args.bert_model=='large':
    BERT_MODEL = '../data/model'
    tokenizer_path = '../data/model/vocab.txt'
else:
    print("input bert_model ERROR! it should be one of [base wwm large]")


pretrained_model = BertModel.from_pretrained(BERT_MODEL)
tokenizer =BertTokenizer.from_pretrained(tokenizer_path)


max_len = args.maxlen
xtrain,xvalid,ytrain,yvalid,test_data,ids = bert_utils.get_split_data(random_seed)
test_y = bert_utils.get_best_test_label()
#train_data,label,test_data,ids = bert_utils.get_Bert_whole_data()
Valid_F1 = 0

class BertEmotionClassifier(nn.Module):
    def __init__(self,pretrained_model,class_size):
        super(BertEmotionClassifier,self).__init__()
        self.class_size = class_size
        self.pretrained_model = pretrained_model
        self.hidden_dim = self.pretrained_model.config.hidden_size
        self.dropout_layer = nn.Dropout(args.dropout)#0.15
        #self.pooling =nn.Linear(max_len,1)

        if args.RNN_type=='LSTM':
            self.lstm = nn.LSTM(self.hidden_dim,args.rnn_hidden_size,num_layers=args.rnn_layers,bidirectional =True,batch_first=True)
            self.hidden_dim = args.rnn_hidden_size * args.rnn_layers * 2

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
        if args.RNN_type =='LSTM':
            out,h_t_c_t = self.lstm(output)
            #out batch,seq_len,num_directions*hidden_size h_n ( num_layers * num_directions, batch,hidden_size)
            h_t = h_t_c_t[0]
            h_t = h_t.transpose(0,1)
            output = h_t.contiguous().view(h_t.size(0),-1)
        else:
            #output = self.pooling(output.contiguous().view(output.size(0),output.size(2),output.size(1)))
            output = torch.mean(output,1) #globalaveragepooling1D 
            #output = torch.max(output,1)[0] #结果略差
            #print(output.size())
        linear_out = self.hidden2class(self.dropout_layer(output))
        scores = F.log_softmax(linear_out,dim=1)
        return scores

def train_or_test(model,data,label,data_index,batch_size,train_type,optimizer,epoch,lossF=nn.NLLLoss()):#F.nll_loss
    start_time = time.time()
    if train_type=="train":
        np.random.shuffle(data_index)
        model.train()
    else:
        model.eval()
    losses = []
    preds =[]
    Y =[]
    step = 0
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
        step +=1
        if train_type=="train":
            loss = loss/args.N_batch_optimizer
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params,args.clip_grad)
            if step%args.N_batch_optimizer == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

    if train_type=='valid':
        mean_loss = np.mean(losses)
        P,F1 = bert_utils.getF1(preds,Y)
        print("Valid epoch:%d time:%.4f loss:%.4f  F1:%.4f  P:%.4f"%(epoch,time.time()-start_time,mean_loss,F1,P))
        return F1
    elif train_type=='train':
        mean_loss = np.mean(losses)
        P,F1 = bert_utils.getF1(preds,Y)
        print("--Train epoch:%d time:%.4f loss:%.4f  F1:%.4f P:%.4f"%(epoch,time.time()-start_time,mean_loss,F1,P))
    else:
        bert_utils.saveResult(Y,preds,'%s_rs%d_ep%d_bc%d_%.4f_acc%d'%(args.bert_model,random_seed,epoch,batch_size,Valid_F1,args.N_batch_optimizer))

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

if args.test_function:
    temp =1000
    xtrain =xtrain[:temp]
    xvalid =xvalid[:temp]
    test_data =test_data[:temp]

if args.Train_valid:
    xtrain = np.concatenate((xtrain,xvalid))
    ytrain = np.concatenate((ytrain,yvalid))
    print("Train all data,len=%d"%(len(xtrain)))
if args.Train_test:
    xtrain = np.concatenate((xtrain,test_data))
    ytrain = np.concatenate((ytrain,test_y))
    print("Train test with best scores,len=%d"%(len(xtrain)))

num_steps =len(xtrain)//batch_size//args.N_batch_optimizer *epoches
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.1*num_steps), num_training_steps=num_steps)  # PyTorch scheduler

train_data_index = np.arange(len(xtrain))
valid_data_index = np.arange(len(xvalid))
test_data_index = np.arange(len(test_data))

for epoch in range(1,epoches+1):
    train_or_test(model,xtrain,ytrain,train_data_index,batch_size,'train',optimizer,epoch)
    with torch.no_grad():
        Valid_F1 = train_or_test(model,xvalid,yvalid,valid_data_index,batch_size,'valid',optimizer,epoch)
        train_or_test(model,test_data,ids,test_data_index,batch_size,'test',None,epoch)
    if args.save_model and  epoch == 2:
        torch.save(model,'ignore/%s_rs%d_ep%d_bc%d_%.4f_%s.model'%(args.bert_model,random_seed,epoch,batch_size,Valid_F1,args.RNN_type))
