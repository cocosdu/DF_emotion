from sklearn.model_selection import train_test_split
import torch
import re

train_label_path = "data/nCoV_100k_train.labled.csv"
train_unlabel_path = "data/nCoV_900k_train.unlabled.csv"
test_path ="data/nCov_10k_test.csv"

Label_dict ={'-1':0,'0':1,'1':2}

stop_word = {}
#{'我','们','的','了','你','他','她','它','现','在','自','己','爸','妈','师','第','一','天','二','三','您','李','今',
#'天','这','那','从','妹','哥','姐','弟','娘'}


cls_token='[CLS]'
sep_token='[SEP]'
PAD = 0
sequence_a_segment_id=0
cls_token_segment_id=1

re_han_default = re.compile("([\u4E00-\u9FD5]+)",re.U)

def read_File_Bert(path,NeedId,NeedLabel,DelNotext):
    no_text = 0
    with open(path,'r',encoding='gbk',errors='ignore') as f:
        line = f.readline().strip()
        print(line)
        unlabel = 0
        label =[]
        ids =[]
        data =[]
        lens =[]
        for line in f:
            seg = line.strip().split(',')
            sentence = seg[3].strip()
            #sentence = ''.join(re_han_default.findall(sentence)) #包括全部符号
            if len(sentence)==0:
                no_text +=1
                if DelNotext:
                    continue
                sentence =''
            if NeedLabel:
                if seg[-1] not in Label_dict:
                    unlabel +=1
                    continue
                label.append(Label_dict[seg[-1]])
            if NeedId:
                ids.append(seg[0].strip())
            #lens.append(len(sentence))
            #if len(sentence)>100:
            #sentence =''.join([w for w in sentence if w not in stop_word])
            data.append(sentence)
            '''
            if NeedLabel:
                if len(sentence)>75:
                    data.append(sentence[50:100])
                    label.append(Label_dict[seg[-1]])
                elif len(sentence)>125:
                    data.append(sentence[100:150])
                    label.append(Label_dict[seg[-1]])
            '''
    print("Finish read File len:%d,unlabel:%d no_text：%d"%(len(data),unlabel,no_text))
    return data,label,ids,lens

def get_Bert_data():
    train_data,label,_,train_lens = read_File_Bert(train_label_path,False,True,True)
    test_data,_,ids,test_lens = read_File_Bert(test_path,True,False,False)
    xtrain,xvalid,ytrain,yvalid = train_test_split(train_data,label,stratify=label,
                                                   random_state=42,test_size=0.1,shuffle=True)
    print("train_data:%d,valid_data:%d,test_data:%d"%(len(xtrain),len(xvalid),len(test_data)))
    return xtrain,xvalid,ytrain,yvalid,test_data,ids
def get_Bert_whole_data():
    train_data,label,_,train_lens = read_File_Bert(train_label_path,False,True,True)
    test_data,_,ids,test_lens = read_File_Bert(test_path,True,False,False)
    return train_data,label,test_data,ids

'''
#max(train_lens) #241 # 99560  56637 <=100 93218  <=150 max
#去除标点符号和数字，217  59283<=100
#去除英文，164  #需要将英文作为一个整体进行
#max(test_lens) #226 # 10000  5679 <=100 9352 <=150
temp = [lens for lens in test_lens if lens<=100]
len(temp)
print(xtrain[:5])
print(ytrain[:10])
'''
def get_minibatch(inputs,labels,data_index,i,batch_size,train_type,device=torch.device("cuda")):
    list_i = data_index[i:i+batch_size]
    data_batch =[inputs[idx] for idx in list_i]
    label_batch =[labels[idx] for idx in list_i]
    
    data_mb = list(zip(data_batch,label_batch))
    data_mb.sort(key=lambda x:len(x[0]),reverse=True)
    
    data_batch =[seq for seq,_ in data_mb]
    label_batch =[l for _,l in data_mb]
    lens =[len(seq) for seq,_ in data_mb]
    if train_type!='test':
        label_batch = torch.tensor(label_batch,dtype =torch.long,device = device)
    return data_batch,label_batch,lens  #label 到device

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
