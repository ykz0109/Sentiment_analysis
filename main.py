import pandas as pd
import torch
from torch import nn
import jieba
from gensim.models import Word2Vec
import numpy as np
from data_set import load_tsv
from torch.utils.data import DataLoader, TensorDataset
 
 
# 数据读取
def load_txt(path):
    with open(path,'r',encoding='utf-8') as f:
        data=[[line.strip()] for line in f.readlines()]
        return data
 
train_x=load_txt('train.txt')
test_x=load_txt('test.txt')
train=train_x+test_x
X_all=[i for x in train for i in x]
 
_, train_y = load_tsv("./train.tsv")
_, test_y = load_tsv("./test.tsv")
# 训练Word2Vec模型
word2vec_model = Word2Vec(sentences=X_all, vector_size=100, window=5, min_count=1, workers=4)
 
# 将文本转换为Word2Vec向量表示
def text_to_vector(text):
    vector = [word2vec_model.wv[word] for word in text if word in word2vec_model.wv]
    return sum(vector) / len(vector) if vector else [0] * word2vec_model.vector_size
 
X_train_w2v = [[text_to_vector(text)] for line in train_x for text in line]
X_test_w2v = [[text_to_vector(text)] for line in test_x for text in line]
 
# 将词向量转换为PyTorch张量
X_train_array = np.array(X_train_w2v, dtype=np.float32)
X_train_tensor = torch.Tensor(X_train_array)
X_test_array = np.array(X_test_w2v, dtype=np.float32)
X_test_tensor = torch.Tensor(X_test_array)
#使用DataLoader打包文件
train_dataset = TensorDataset(X_train_tensor, torch.LongTensor(train_y))
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_dataset = TensorDataset(X_test_tensor,torch.LongTensor(test_y))
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)
# 定义LSTM模型
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
 
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        output = self.fc(lstm_out[:, -1, :])  # 取序列的最后一个输出
        return output
 
# 定义模型
input_size = word2vec_model.vector_size
hidden_size = 50  # 你可以根据需要调整隐藏层大小
output_size = 2  # 输出的大小，根据你的任务而定
 
model = LSTMModel(input_size, hidden_size, output_size)
# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()  # 交叉熵损失函数
optimizer = torch.optim.Adam(model.parameters(), lr=0.0002)
 
if __name__ == "__main__":
    # 训练模型
    num_epochs = 10
    log_interval = 100  # 每隔100个批次输出一次日志
    loss_min=100
    for epoch in range(num_epochs):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            outputs = model(data)
            loss = criterion(outputs, target)
 
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
 
            if batch_idx % log_interval == 0:
                print('Epoch [{}/{}], Batch [{}/{}], Loss: {:.4f}'.format(
                    epoch + 1, num_epochs, batch_idx, len(train_loader), loss.item()))
            # 保存最佳模型
            if loss.item()<loss_min:
                loss_min=loss.item()
                torch.save(model, 'model.pth')
 
    # 模型评估
    with torch.no_grad():
        model.eval()
        correct = 0
        total = 0
        for data, target in test_loader:
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
 
        accuracy = correct / total
        print('Test Accuracy: {:.2%}'.format(accuracy))
