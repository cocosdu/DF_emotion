from sklearn.model_selection import train_test_split, StratifiedKFold
import torch
import re
import pandas as pd
import numpy as np

train_label_path = "data/train_label.csv"
train_unlabel_path = "data/train_unlabel.csv"
test_path ="data/test.csv"



cls_token='[CLS]'
sep_token='[SEP]'
PAD = 0
sequence_a_segment_id=0
cls_token_segment_id=1


def get_data():
    train_data = pd.read_csv(train_label_path)
    #print('train_origin shape=',train_data.shape)
    test_data = pd.read_csv(test_path)
    train_data = train_data[train_data['情感倾向'].isin(['-1','0','1'])]  #去除标签错误的
    labels = np.asarray(train_data['情感倾向'].astype(int)+1)
    #print('train_shape=',train_data.shape)
    train_con = train_data['微博中文内容'].fillna("")
    test_con = test_data['微博中文内容'].fillna("")
    return train_con.values,labels,test_con.values,test_data['微博id'].values
    #df.values 转换为numpy

def get_split_data(random_state):
    train_data,label,test_data,ids = get_data()
    xtrain,xvalid,ytrain,yvalid = train_test_split(train_data,label,stratify=label,
                                                   random_state=random_state,test_size=0.1,shuffle=True)
    print("train_data:%d,valid_data:%d,test_data:%d"%(len(xtrain),len(xvalid),len(test_data)))
    return xtrain,xvalid,ytrain,yvalid,test_data,ids

def get_best_test_label():
    data = pd.read_csv('72.94.csv')
    return np.asarray(data['y']+1)

#max(train_lens)=241 # 99560  56637 <=100 93218  <=150 

def get_minibatch(inputs,labels,indexes,train_type,device=torch.device("cuda")):
    data_batch =[inputs[idx] for idx in indexes]
    label_batch =[labels[idx] for idx in indexes]

    if train_type!='test':
        label_batch = torch.tensor(label_batch,dtype =torch.long,device = device)
    return data_batch,label_batch  #label 到device

def prepare_inputs_for_bert(sentences,maxlen,tokenizer,device=torch.device("cuda"),lens=None):
    tokens =[]
    segments =[]
    for ws in sentences:
        #ts = [tokenizer.tokenize(w) for w in ws] #[['你']]
        ts = tokenizer.tokenize(ws)
        ts = ts[:maxlen-2] #分完词之后再取maxlen
        ts = [cls_token] +ts
        ts += [sep_token]
        si = [0]*len(ts)
        si[0] = 1
        tokens.append(ts)
        segments.append(si)
    lens =[len(ts) for ts in tokens]
    max_l =  maxlen
    input_mask =[[1]*l+[0]*(max_l-l)  for l in lens]
    input_tokens = [tokenizer.convert_tokens_to_ids(ts)+[PAD]*(max_l-lens[idx]) for idx,ts in enumerate(tokens)]
    segments = [si +[PAD]*(max_l-lens[idx]) for idx,si in enumerate(segments)]
    
    input_mask = torch.tensor(input_mask, dtype=torch.long, device=device)
    input_tokens = torch.tensor(input_tokens, dtype=torch.long, device=device)
    segments = torch.tensor(segments, dtype=torch.long, device=device)
    return {'tokens':input_tokens,'segments':segments,'mask':input_mask}


def getF1(pred,y):
    n = len(pred)
    TP = np.zeros(3)
    FP = np.zeros(3)
    FN = np.zeros(3)
    for i in range(n):
        if pred[i]==y[i]:
            TP[y[i]] +=1
        else:
            FN[y[i]] +=1
            FP[pred[i]] +=1
    #print(TP,FP,FN)
    P = TP*1.0/(TP+FP)
    R = TP*1.0/(TP+FN)
    F1 = 2*P*R/(P+R)
    return P.mean(),F1.mean()

def saveResult(ids,test_pred,name='Result'):
    dic =[-1,0,1]
    result = [dic[i] for i in test_pred]
    test = pd.DataFrame({'id':ids,'y':result})
    #test=pd.DataFrame(result,index=ids,columns=['y'])
    test.to_csv(name+'.csv',index=False)
