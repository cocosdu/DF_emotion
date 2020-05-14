import argparse
import pandas as pd
import jieba 
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--resave_file',type=bool,default =False,help='First Time ')

args = parser.parse_args()

train_label_path ='data/train_label.csv'
train_unlabel_path ='data/train_unlabel.csv'
test_path ='data/test.csv'
Label =['-1','0','1']

"只运行一次，为了去掉格式不对的行，可以直接用pandas读取分析数据"
def save_utf8_file():
	train_label_path_gbk = "data/nCoV_100k_train.labled.csv"
	train_unlabel_path_gbk = "data/nCoV_900k_train.unlabled.csv"
	test_path_gbk ="data/nCov_10k_test.csv"

	def resave_file(gbk_path,utf8_path):
		with open(gbk_path,'r',encoding='gbk',errors='ignore') as f:
			with open(utf8_path,'w',encoding='utf-8') as wf:
				for line in f:
					wf.write(line)

	resave_file(train_label_path_gbk,train_label_path)
	resave_file(train_unlabel_path_gbk,train_unlabel_path)
	resave_file(test_path_gbk,test_path)
	print("finish save gbk file to utf8, can read with pandas")

def seg_data(data,IsStr,re_han_default,STOPWORDS):
	contents =[]
	for i, content in enumerate(data):
		if pd.isnull(content):
			content=''
		if re_han_default:
			content = ' '.join(re_han_default.findall(content))
		seg =jieba.cut(content)
		if STOPWORDS:
			seg = [w for w in seg if w not in STOPWORDS]
		else:
			seg =[w for w in seg]
		if IsStr:
			seg = ' '.join(seg)
		contents.append(seg)
	return contents

def get_data(IsStr,IsSeg,removeInvalid=False,setDefault='-1',re_han_default=None,STOPWORDS=[]): #gensim是[['我'，'是'],]
	# TfidfVectorizer() 是用空格连接的句子   ML不需要去掉空白的数据，LSTM需要去掉 BERT不确定
	train_data = pd.read_csv(train_label_path)
	test_data = pd.read_csv(test_path)
	train_data = train_data[train_data['情感倾向'].isin(['-1','0','1'])]
	if removeInvalid:
		train_data = train_data.dropna(axis=0,subset=['微博中文内容'])

	train_con = train_data['微博中文内容'].fillna(setDefault)
	test_con = test_data['微博中文内容'].fillna(setDefault)

	if IsSeg:
		train_con = seg_data(train_con,IsStr,re_han_default,STOPWORDS)
		test_con = seg_data(test_con,IsStr,re_han_default,STOPWORDS)
	train_con = np.array(train_con)
	test_con = np.array(test_con)

	assert len(train_con)==len(train_data['情感倾向'])
	assert len(test_con)==len(test_data['微博id'])
	print("Finish train data %d test_data:%d"%(len(train_con),len(test_con)))
	return train_con,train_data['情感倾向'],test_con,test_data['微博id']



if __name__ == '__main__':
	pass
	#save_utf8_file()
