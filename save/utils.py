import pandas as pd
import numpy as np
import jieba
import re
import pickle
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD
from gensim.models import Word2Vec
from gensim import corpora
from gensim.models.word2vec import LineSentence


import torch

train_label_path = "data/nCoV_100k_train.labled.csv"
train_unlabel_path = "data/nCoV_900k_train.unlabled.csv"
test_path ="data/nCov_10k_test.csv"
submit ="data/submit_example.csv"

seg_data_path = "seg_file_train_test_unlabel_nochar.txt"
#word2vec_path = "train_test_unlabel_word2vec_100.model"


STOPWORDS =[' ','我们','的','了','我','你','他','她','它','他们','因为','所以']
re_han_default = re.compile("([\u4E00-\u9FD5a-zA-Z0-9]+)",re.U)
PAD_ID = 0

def preprocess(sentence,isStr):
    #sentences = re.findall('\w+',sentence)
    sentences = re_han_default.findall(sentence.lower())  #131172有字母65.67  122567无字母65.39
    #sentences = re_han_default.split(sentence.lower()) #差不多  #151253,unknown:63877
    #print(sentences)
    words =[]
    for s in sentences:
        seg =jieba.cut(s)  # s  单字 len(weights):8281,unknown:946
        temp = [w for w in seg if w not in STOPWORDS]
        words.extend(temp)
    if isStr:
    	words = ' '.join(words) # TfidfVectorizer() 是用空格连接的句子  get_seg_File也需要这个
    return words #gensim是[['我'，'是'],]
def readFile(path,NeedId,NeedLabel,NeedData,writeFile):
	no_text = 0
	with open(path,'r',encoding='gbk',errors='ignore') as f:
		line = f.readline()
		print(line)
		Labels =['1','0','-1']
		unlabel = 0
		label= []
		ids =[]
		data = []
		for line in f:
			seg = line.strip().split(',')
			sentence = preprocess(seg[3],True)
			if len(sentence)==0:
				no_text +=1
			else:
				if NeedLabel:
					if seg[-1] not in Labels:
						unlabel +=1
						continue
					else:
						label.append(seg[-1])
				if NeedId:
					ids.append(seg[0])
				if NeedData:
					data.append(sentence)
				if writeFile:
					writeFile.write(sentence+'\n')
	print("Finish read File len:%d,unlabel:%d no_text：%d"%(len(data),unlabel,no_text))
	return data,label,ids
def get_seg_File(): #暂时只用于训练word2vec
	with open(seg_data_path,'w') as file:
		train_data,label,_ = readFile(train_label_path,False,True,True,file)
		test_data,_,ids = readFile(test_path,True,False,True,file)
		readFile(train_unlabel_path,False,True,False,file) #unlabel data暂时只需要输出到文件中，不需要返回

def get_word2Vec_model(word_dim=100):
	#get_seg_File() #可以单独运行，保存好seg_data_path文件
	model = Word2Vec(LineSentence(seg_data_path),min_count=2,size=word_dim,workers=4)
	model.save("train_test_unlabel_word2vec_%d_min2.model"%(word_dim))


def getData():
	label =['1','0','-1']
	unlabel = 0
	no_text = 0
	with open(train_label_path,'r',encoding='gbk',errors='ignore') as f:
		line = f.readline().strip()
		print(line)
		train_data = []
		train_label = []
		for line in f:
			seg = line.strip().split(',')
			if seg[-1] not in label:
				#print(line)
				unlabel +=1
			else:
				sentence = preprocess(seg[3],False)
				if len(sentence)==0:
					#print("len(line)==0 :%s"%(line))
					no_text +=1
				else:
					train_data.append(sentence)
					train_label.append(seg[-1])
	print("finish train data len:%d,unlabel:%d,no_text:%d"%(len(train_data),unlabel,no_text))


	with open(test_path,'r',encoding='gbk',errors='ignore') as f:
		line = f.readline().strip()
		print(line)
		test_data =[]
		ids =[]
		no_text =0
		for line in f:
			seg = line.strip().split(',')
			sentence = preprocess(seg[3],False)
			if len(sentence)==0:
				no_text +=1
				#print("test len(line)==0 :%s"%(line))
				sentence =['我','是']
				#sentence = #['UNK'] 
			ids.append(int(seg[0]))
			test_data.append(sentence)
	print("finish test data len:%d no_text:%d"%(len(test_data),no_text))
	'''
	with open(train_unlabel_path,'r',encoding='gbk',errors='ignore') as f:
		line =f.readline().strip()
		print(line)
		unlabel_data=[]
		no_text = 0
		for line in f:
			seg = line.strip().split(',')
			sentence =preprocess(seg[3])
			if len(sentence)==0:
				no_text +=1
			else:
				unlabel_data.append(sentence)
	print("finish unlabel data len:%d no_text:%d"%(len(unlabel_data),no_text))
	'''
	return train_data,train_label,test_data,ids,None

def show_result_txt(pred,y,index,data_show,path_str):
    file_right = open(path_str+'_right.txt','w')
    file_wrong = open(path_str+'_wrong.txt','w')
    data_right =[]
    data_wrong =[]
    for i in range(len(pred)):
    	seqstr = ' '.join(data_show[index[i]])
    	if pred[i]==y[i]:
    		sentence = seqstr +' %d \n'%(y[i])
    		file_right.write(sentence)
    	else:
    		sentence =seqstr +'label:%d pred: %d\n'%(y[i],pred[i])
    		file_wrong.write(sentence)
    file_right.close()
    file_wrong.close()
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
    test=pd.DataFrame(result,index=ids,columns=['y'])
    test.to_csv(name+'.csv')


def get_weights(Words,pre_train_path):
    with open(pre_train_path,'rb') as f:
        Data = pickle.load(f)
    WORD2IDX = Data['WORD2IDX']
    weights = Data['weights']
    
    dim = len(weights[0])
    pre_weights = []
    UNK = 0
    for word in Words:
        if word in WORD2IDX:
            w = weights[WORD2IDX[word]]
        else:
            w = [0]*dim
            UNK += 1
        pre_weights.append(w)
    print("finish get_weights len(weights):%d,unknown:%d"%(len(pre_weights),UNK))
    return pre_weights

def get_word2vec_weight(Words,path,word_dim):
	model = Word2Vec.load(path)
	WORD2IDX ={'UNK':0}
	ID2WORD =['UNK']
	weights = []
	weights.append([0]*word_dim)
	unknown =[]
	UNK = 0
	for word in Words:
		if word in model:
			WORD2IDX[word] =len(WORD2IDX)
			ID2WORD.append(word)
			weights.append(model[word])
		else:
			unknown.append(word)
			UNK +=1
	print("finish get_weights len(weights):%d,unknown:%d"%(len(weights),UNK))
	#print("unknown: "+' '.join(unknown))
	with open('unknown_word.txt','w') as f:
		f.write(' '.join(unknown))
	return weights,WORD2IDX,ID2WORD

def getSVDData(xtrain,xvalid,test_data):
	xtrain_str =[' '.join(l) for l in xtrain]
	xvalid_str =[' '.join(l) for l in xvalid]
	test_data_str =[' '.join(l) for l in test_data]
	tfv = TfidfVectorizer()
	tfv.fit(xtrain_str) #+xvalid
	xtrain_tfv = tfv.transform(xtrain_str)
	xvalid_tfv = tfv.transform(xvalid_str)
	#tfv.get_feature_names()
	#print(xtrain_tfv.shape)
	xtest_tfv = tfv.transform(test_data_str)

	svd = TruncatedSVD(n_components=300)
	svd.fit(xtrain_tfv)
	xtrain_svd = svd.transform(xtrain_tfv)
	xvalid_svd = svd.transform(xvalid_tfv)
	xtest_svd = svd.transform(xtest_tfv)
	return xtrain_svd,xvalid_svd,xtest_svd

def get_word2idx(data,WORD2IDX):
	return [[WORD2IDX[w] if w in WORD2IDX else 0 for w in sen] for sen in data]

def saveLSTMData(word_dim):
	train_data,train_label,test_data,ids,unlabel_data = getData()

	lbl_enc = preprocessing.LabelEncoder()
	y = lbl_enc.fit_transform(train_label)
	#print(lbl_enc.classes_) #['-1' '0' '1']
	'''
	xtrain,xvalid,ytrain,yvalid = train_test_split(train_data,y,stratify=y,
		test_size=0.1,shuffle=True)
	print(len(xtrain))
	print(len(xvalid))
	"xtrain,xvalid,ytrain,yvalid  test_data ids"
    '''
	dictionary = corpora.Dictionary(train_data)

	#xtrain_id = [dictionary.doc2idx(text) for text in xtrain]

	#pre_train_path ="../QA/large_weights_636013.pickle"
	#pre_weights = get_weights(dictionary.token2id,pre_train_path)
	pre_weights,WORD2IDX,ID2WORD = get_word2vec_weight(dictionary.token2id,"midSaveData/train_test_unlabel_word2vec_%d_min2.model"%(word_dim),word_dim)
	train_id = get_word2idx(train_data,WORD2IDX)
	#xvalid_id = get_word2idx(xvalid,WORD2IDX)
	xtest_id = get_word2idx(test_data,WORD2IDX)
	unlabel_id = None #get_word2idx(unlabel_data,WORD2IDX)

	#xtrain_svd,xvalid_svd,xtest_svd = getSVDData(xtrain,xvalid,test_data)

	#xtrain_id ytrain xvalid_id yvalid xtest_id ids pre_weights dictionary.token2id
	Data = {"train":train_id,"label":y,"xtest":xtest_id,"ids":ids,
		"pre_weights":pre_weights,"WORD2IDX":WORD2IDX,"train_seq":train_data,
		"xtest_seq":test_data,'unlabel':unlabel_id}
	with open("midSaveData/weights_%d_word2vec_train_test_unsplit.pickle"%(word_dim),'wb') as f:
		pickle.dump(Data,f)

def get_minibatch(data,label,data_index,i,batch_size,max_len,device=torch.device("cuda")):
    list_i = data_index[i:i+batch_size]
    if max_len !=0:
        data_batch = [data[idx][:max_len] for idx in list_i]
    else:
    	data_batch = [data[idx] for idx in list_i]

    label_batch =[label[idx] for idx in list_i]
    #extra_batch =[extra_input[idx] for idx in list_i]
    
    data_mb = list(zip(data_batch,label_batch))
    data_mb.sort(key=lambda x:len(x[0]),reverse=True)
    
    lens = [len(seq) for seq,_ in data_mb]
    if max_len == 0:
    	max_len =max(lens)

    data_batch = [seq+[PAD_ID]*(max_len-len(seq)) for seq,_ in data_mb]
    label_batch =[l for _,l in data_mb]
    
    data_batch = torch.tensor(data_batch,dtype =torch.long,device = device)
    label_batch = torch.tensor(label_batch,dtype =torch.long,device = device)
    return data_batch,label_batch,lens