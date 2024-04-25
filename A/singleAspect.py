import torch
import time
import numpy as np
import pandas as pd
import torch.nn as nn
from tqdm import tqdm
from transformers import BertModel
from transformers import BertTokenizer
from sklearn.model_selection import train_test_split
from transformers import AdamW, get_linear_schedule_with_warmup
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler


def preprocessing_for_bert(data):

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

    # 空列表来储存信息
    input_ids = []
    attention_masks = []

    # 每个句子循环一次
    for sent in data:
        encoded_sent = tokenizer.encode_plus(
            text=sent,  # 预处理语句
            add_special_tokens=True,  # 加 [CLS] 和 [SEP]
            truncation_strategy = "longest_first",
            padding='max_length',  # 填充为最大长度，这里的padding在之间可以直接用pad_to_max但是版本更新之后弃用了，老版本什么都没有，可以尝试用extend方法
            return_attention_mask=True  # 返回 attention mask
        )

        # 把输出加到列表里面
        input_ids.append(encoded_sent.get('input_ids'))
        attention_masks.append(encoded_sent.get('attention_mask'))

    # 把list转换为tensor
    input_ids = torch.tensor(input_ids)
    attention_masks = torch.tensor(attention_masks)

    return input_ids, attention_masks



class BertClassifier(nn.Module):
    def __init__(self, ):
        """
        freeze_bert (bool): 设置是否进行微调，0就是不，1就是调
        """
        super(BertClassifier, self).__init__()
        # 输入维度(hidden size of Bert)默认768，分类器隐藏维度，输出维度(label)
        D_in, H, D_out = 768, 100, 5

        # 实体化Bert模型
        self.bert = BertModel.from_pretrained('bert-base-uncased')

        # 实体化一个单层前馈分类器，说白了就是最后要输出的时候搞个全连接层
        self.classifier = nn.Sequential(
            nn.Linear(D_in, H),  # 全连接
            nn.ReLU(),  # 激活函数
            nn.Linear(H, D_out)  # 全连接
        )

    def forward(self, input_ids, attention_mask):
        # 开始搭建整个网络了
        # 输入
        outputs = self.bert(input_ids=input_ids,
                            attention_mask=attention_mask)
        # 为分类任务提取标记[CLS]的最后隐藏状态，因为要连接传到全连接层去
        last_hidden_state_cls = outputs[0][:, 0, :]
        # 全连接，计算，输出label
        logits = self.classifier(last_hidden_state_cls)

        return logits


# 注意这个地方的logits是全连接的返回， 两个output就是01二分类，我们这里用的是ouput为3，就是老师所需要的三分类问题


"""
然后就是深度学习的老一套定义优化器还有学习率等
"""



def initialize_model(epochs=2):
    """
    初始化我们的bert，优化器还有学习率，epochs就是训练次数
    """
    # 初始化我们的Bert分类器
    bert_classifier = BertClassifier()
    # 用GPU运算
    bert_classifier.to(device)
    # 创建优化器
    optimizer = AdamW(bert_classifier.parameters(),
                      lr=5e-5,  # 默认学习率
                      eps=1e-8  # 默认精度
                      )
    # 训练的总步数
    total_steps = len(train_dataloader) * epochs
    # 学习率预热
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0,  # Default value
                                                num_training_steps=total_steps)
    return bert_classifier, optimizer, scheduler


# 实体化loss function
loss_fn = nn.CrossEntropyLoss()  # 交叉熵



def train(model, train_dataloader, test_dataloader=None, epochs=2, evaluation=False):
    # 开始训练循环
    for epoch_i in range(epochs):
        # =======================================
        #               Training
        # =======================================
        # 表头
        print(f"{'Epoch':^7} | {'每40个Batch':^9} | {'训练集 Loss':^12} | {'测试集 Loss':^10} | {'测试集准确率':^9} | {'时间':^9}")
        print("-" * 80)

        # 测量每个epoch经过的时间
        t0_epoch = time.time()

        # 在每个epoch开始时重置跟踪变量
        total_loss = 0

        # 把model放到训练模式
        model.train()

        # 分batch训练
        with tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc=f"Epoch {epoch_i+1}/{epochs}", unit="batch") as t:
            for step, batch in t:
                # 把batch加载到GPU
                b_input_ids, b_attn_mask, b_labels = tuple(t.to(device) for t in batch)
                # print(b_labels.shape)
                # print(b_labels)
                # 归零导数
                model.zero_grad()
                # 真正的训练
                logits = model(b_input_ids, b_attn_mask)
                # print(logits.shape)
                # print(logits)
                # 计算loss并且累加

                loss = loss_fn(logits, b_labels)

                total_loss += loss.item()
                # 反向传播
                loss.backward()
                # 归一化，防止梯度爆炸
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                # 更新参数和学习率
                optimizer.step()
                scheduler.step()

                # Print每40个batch的loss和time
                if (step % 40 == 0 and step != 0) or (step == len(train_dataloader) - 1):
                    # Print训练结果
                    t.set_postfix_str(f"Loss: {total_loss / (step + 1):.4f}")
        
        # 计算平均loss 这个是训练集的loss
        avg_train_loss = total_loss / len(train_dataloader)

        print("-" * 80)
        # =======================================
        #               Evaluation
        # =======================================
        if evaluation:  # 这个evalution是我们自己给的，用来判断是否需要我们汇总评估
            # 每个epoch之后评估一下性能
            # 在我们的验证集/测试集上.
            test_loss, test_accuracy = evaluate(model, test_dataloader)
            # Print 整个训练集的耗时
            time_elapsed = time.time() - t0_epoch

            print(
                f"{epoch_i + 1:^7} | {'-':^10} | {avg_train_loss:^14.6f} | {test_loss:^12.6f} | {test_accuracy:^12.2f}% | {time_elapsed:^9.2f}")
            print("-" * 80)
        print("\n")



# 在测试集上面来看看我们的训练效果
def evaluate(model, test_dataloader):
    """
    在每个epoch后验证集上评估model性能
    """
    # model放入评估模式
    model.eval()

    # 准确率和误差
    test_accuracy = []
    test_loss = []

    # 验证集上的每个batch
    for batch in test_dataloader:
        # 放到GPU上
        b_input_ids, b_attn_mask, b_labels = tuple(t.to(device) for t in batch)

        # 计算结果，不计算梯度
        with torch.no_grad():
            logits = model(b_input_ids, b_attn_mask)  # 放到model里面去跑，返回验证集的ouput就是一行三列的
            # label向量可能性，这个时候还没有归一化所以还不能说是可能性，反正归一化之后最大的就是了

        # 计算误差
        loss = loss_fn(logits, b_labels.long())
        test_loss.append(loss.item())

        # get预测结果，这里就是求每行最大的索引咯，然后用flatten打平成一维
        preds = torch.argmax(logits, dim=1).flatten()  # 返回一行中最大值的序号

        # 计算准确率，这个就是俩比较，返回相同的个数, .cpu().numpy()就是把tensor从显卡上取出来然后转化为numpy类型的举证好用方法
        # 最后mean因为直接bool形了，也就是如果预测和label一样那就返回1，正好是正确的个数，求平均就是准确率了
        accuracy = (preds == b_labels).cpu().numpy().mean() * 100
        test_accuracy.append(accuracy)

    # 计算整体的平均正确率和loss
    val_loss = np.mean(test_loss)
    val_accuracy = np.mean(test_accuracy)

    return val_loss, val_accuracy