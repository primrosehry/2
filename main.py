import torch
import random
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import confusion_matrix, matthews_corrcoef, roc_auc_score, recall_score, f1_score

import model
import utils

import warnings
warnings.filterwarnings("ignore")


# 加载数据，并划分训练和验证集
path = './graphs/573.pt'
data = torch.load(path)
# print(data)   # Data(x=[73, 2560], edge_index=[2, 572], y=[1, 73])
# 计算要选择的元素数量
num_train = int(0.7 * len(data))
# 随机选择70%的元素
train_data = random.sample(data, num_train)
# 剩余的元素
val_data = [x for x in data if x not in train_data]


in_features = 2560
out_features = 64
dropout = 0.4

input_dim = 64
hidden_dim1 = 32
hidden_dim2 = 16
out_dim = 2

Model = model.Sage_En(in_features, out_features, dropout, input_dim, hidden_dim1, hidden_dim2, out_dim)
# out=sage_En(feature,adj)
# print(out)

crit = utils.FocalLoss(alpha=0.2, gamma=5)
# 传入优化器
optimizer = torch.optim.Adam(Model.parameters(), lr=0.01)

new_train_data = utils.process_train_dataset(train_data)

enc = OneHotEncoder(sparse=False)
for epoch in range(250):
    loss_all = 0
    Model.train()
    for data in new_train_data:
        optimizer.zero_grad()
        feature = data.x
        adj = data.edge_index
        output = Model(feature,adj)
        # print(output)

        label = data.y.unsqueeze(1)
        label = torch.tensor(enc.fit_transform(label), dtype=torch.float32)
        # print(type(output),type(label))
        # print(output, label)
        loss = crit(output, label)
        loss_all += loss.item()
        loss.backward()
        optimizer.step()

    print('loss_all:',loss_all)

torch.save(Model.state_dict(), './weights/573_181.pt')

Model.eval()

new_val_data = utils.process_val_dataset(val_data)

real_label = []
predict_label = []
probabilities = []
val_loss = 0

for data in new_val_data:
    feature = data.x
    adj = data.edge_index
    output = Model(feature, adj)
    # print(type(output),output.shape)

    # 获取预测概率
    positive_class_probabilities = output[:, 1].cpu().detach().numpy().tolist()
    probabilities.extend(positive_class_probabilities)

    label = data.y.unsqueeze(1)
    onehot_label = torch.tensor(enc.fit_transform(label), dtype=torch.float32)

    loss = crit(output, onehot_label)
    val_loss += loss.item()
    label = label.view(-1).tolist()
    real_label += label

    # print(output)
    output = output.detach().numpy()
    pred_label = np.argmax(output, axis=1).tolist()
    predict_label += pred_label


TN, FP, FN, TP = confusion_matrix(real_label, predict_label).ravel()
spe = TN / (TN + FP)

# 计算召回率（Recall）
rec = recall_score(real_label, predict_label)

pre = TP / (TP + FP)
# 计算F1分数
f1 = f1_score(real_label, predict_label)
mcc = matthews_corrcoef(real_label, predict_label)

# 计算AUC
auc = roc_auc_score(real_label, probabilities)

print('Test Set Spe: {:.4f}'.format(spe))
print('Test Set Rec: {:.4f}'.format(rec))
print('Test Set Pre: {:.4f}'.format(pre))
print('Test Set F1: {:.4f}'.format(f1))
print('Test Set MCC: {:.4f}'.format(mcc))
print('Test Set AUC: {:.4f}'.format(auc))
