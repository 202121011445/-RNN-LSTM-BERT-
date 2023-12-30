from torch import nn
from models.bert import *


class SentimentModel(nn.Module):
    def __init__(self, config):
        super(SentimentModel, self).__init__()
        self.bert = BertModel(config)
        self.dense = nn.Linear(config.hidden_size * 2, config.hidden_size)  # 构建全连接层，参数分别为输入特征大小，输出特征的大小
        self.final_dense = nn.Linear(config.hidden_size, 1)
        self.activation = nn.Sigmoid()

    def compute_loss(self, predictions, labels):
        # 将预测和标记的维度展平
        predictions = predictions.view(-1)
        labels = labels.float().view(-1)
        epsilon = 1e-8
        # 交叉熵
        loss = - labels * torch.log(predictions + epsilon) - (torch.tensor(1.0) - labels) * torch.log(
            torch.tensor(1.0) - predictions + epsilon)
        loss = torch.mean(loss)
        return loss

    def forward(self, text_input, positional_enc, labels=None):
        encoded_layers, pooled_output = self.bert(text_input, positional_enc, output_all_encoded_layers=True)
        sequence_output = encoded_layers[2] # 维度是[batch_size, seq_len, embed_dim]
        avg_pooled = sequence_output.mean(1)
        max_pooled = torch.max(sequence_output, dim=1)
        pooled = torch.cat((avg_pooled, max_pooled[0]), dim=1)
        pooled = self.dense(pooled)
        # 下面是[batch_size, hidden_dim * 2] 到 [batch_size, 1]的映射
        predictions = self.final_dense(pooled)
        # 用sigmoid函数做激活, 返回0-1之间的值
        predictions = self.activation(predictions)
        if labels is not None:
            loss = self.compute_loss(predictions, labels)
            return predictions, loss
        else:
            return predictions
