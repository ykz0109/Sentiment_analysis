import torch
import jieba
from torch import nn
from gensim.models import Word2Vec
import numpy as np
 
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
 
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        output = self.fc(lstm_out[:, -1, :])  # 取序列的最后一个输出
        return output
 
# 数据读取
def load_txt(path):
    with open(path,'r',encoding='utf-8') as f:
        data=[[line.strip()] for line in f.readlines()]
        return data
 
#去停用词
def drop_stopword(datas):
    # 假设你有一个函数用于预处理文本数据
    with open('./hit_stopwords.txt', 'r', encoding='UTF8') as f:
        stop_words = [word.strip() for word in f.readlines()]
    datas=[x for x in datas if x not in stop_words]
    return datas
 
def preprocess_text(text):
    text=list(jieba.cut(text))
    text=drop_stopword(text)
    return text
 
# 将文本转换为Word2Vec向量表示
def text_to_vector(text):
    train_x = load_txt('train.txt')
    test_x = load_txt('test.txt')
    train = train_x + test_x
    X_all = [i for x in train for i in x]
    # 训练Word2Vec模型
    word2vec_model = Word2Vec(sentences=X_all, vector_size=100, window=5, min_count=1, workers=4)
    vector = [word2vec_model.wv[word] for word in text if word in word2vec_model.wv]
    return sum(vector) / len(vector) if vector else [0] * word2vec_model.vector_size
 
if __name__ == '__main__':
    input_text = input("请输入你想识别的文字：")
    label = {1: "正面情绪", 0: "负面情绪"}
    model = torch.load('model.pth')
    # 预处理输入数据
    input_data = preprocess_text(input_text)
    # 确保输入词向量与模型维度和数据类型相同
    input_data=[[text_to_vector(input_data)]]
    input_arry= np.array(input_data, dtype=np.float32)
    input_tensor = torch.Tensor(input_arry)
    # 将输入数据传入模型
    with torch.no_grad():
        output = model(input_tensor)
    # 这里只一个简单的示例
    predicted_class = label[torch.argmax(output).item()]
    print(f"predicted_text:{input_text}")
    print(f"模型预测的类别: {predicted_class}")