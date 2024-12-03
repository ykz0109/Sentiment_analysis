<img width="453" alt="image" src="https://github.com/user-attachments/assets/630aeb50-17d4-4110-91fb-9a04dbd9ae2e"># Sentiment_analysis
 计算机开源项目实践课程小组作业
本研究旨在设计和实现一个基于LSTM的文本情感分析系统，以帮助企业和机构更好地理解和分析社交媒体上的用户情感。通过分析文本数据（如社交媒体、评论、评价等），确定其中的情感倾向。研究首先介绍了情感分析的背景和意义，回顾了情感分析技术的发展历程，从早期的基于情感词汇表的规则型方法到现代的基于深度学习的方法。然后，详细描述了系统的整体架构和关键技术，包括数据预处理、词向量表示、LSTM模型构建和训练、模型评估等步骤。最后，实验结果表明，该系统在情感分类任务上具有较高的准确率，能够有效地识别和分类文本中的情感倾向。

运行条件:
安装依赖库：
pip install pandas torch jieba gensim numpy
虚拟环境：
Python 3.9.18(‘pytorch’)

技术架构：
project_root/
│
├── data/
│   ├── train.tsv
│   ├── test.tsv
│   └── train.txt
│   └── test.txt
│
├── models/
│   └── model.pth
│
├── dataset.py
├── main.py
└── test.py
<img width="453" alt="image" src="https://github.com/user-attachments/assets/630aeb50-17d4-4110-91fb-9a04dbd9ae2e">

运行说明：
① 首先，通过（data_set.py）文件，对训练集、测试集数据进行预处理，基于LSTM的文本情感分析项目的数据预处理阶段，包括数据读取、停用词过滤、分词以及数据保存等步骤。
② 然后，通过（main.py)对模型进行训练，代码主要包括数据加载、词向量转换、模型定义、训练和评估等步骤。

③ 最后，通过（test.py)对LSTM模型进行样例测试，用户可手动输入一段文字，系统会对其进行相应判断（1：积极情绪:2：消极情绪）。

协作者：
于凯正 2022201629
王明浩 2022201614
晏  菱 2022201622




