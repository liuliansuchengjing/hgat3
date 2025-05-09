import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class DKT(nn.Module):

    def __init__(self, emb_dim, hidden_dim, num_skills, dropout=0.2, bias=True):
        super(DKT, self).__init__()
        self.emb_dim = emb_dim  # 嵌入维度
        self.hidden_dim = hidden_dim
        self.bias = bias
        self.correct_embed = nn.Embedding(2, emb_dim)  # 答案结果嵌入（正确、错误）
        self.rnn = nn.LSTM(emb_dim * 2, hidden_dim, bias=bias, dropout=dropout, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_skills, bias=bias)

    def forward(self, dynamic_skill_embeds, questions, correct_seq):
        """
                Parameters:
                    dynamic_skill_embeds: 动态生成的题目嵌入 [num_skills, emb_dim]
                    questions: 题目ID序列 [batch_size, seq_len]
                    correct_seq: 答题结果序列 [batch_size, seq_len]（0错误/1正确）
                Returns:
                    pred: 下一题正确概率预测 [batch_size, seq_len-1]
                """
        mask = (questions[:, 1:] >= 2).float()

        # --- 生成每个时间步的输入特征 ---
        # 根据题目ID获取动态嵌入 [batch_size, seq_len, emb_dim]
        skill_embeds = dynamic_skill_embeds[questions]  # 索引操作

        # 生成答题结果嵌入 [batch_size, seq_len, emb_dim]
        correct_embeds = self.correct_embed(correct_seq.long().to('cuda'))

        # 拼接题目嵌入和答题结果嵌入 [batch_size, seq_len, emb_dim*2]
        lstm_input = torch.cat([skill_embeds, correct_embeds], dim=-1)
        output, (hn, cn) = self.rnn(lstm_input)

        # --- 预测下一题正确概率 ---
        yt = torch.sigmoid(self.fc(output))  # [batch, seq_len, num_skills]
        yt_all = yt
        ht = yt[:, :-1, :]
        yt = yt[:, :-1, :]  # 对齐下一题预测 [batch, seq_len-1, num_skills]

        # --- 提取目标题概率 ---
        next_skill_ids = questions[:, 1:]  # 下一题的skill_id [batch, seq_len-1]
        pred = torch.gather(yt, dim=2, index=next_skill_ids.unsqueeze(-1).to('cuda')).squeeze(-1)

        return pred, mask, yt, yt_all, ht  # [batch, seq_len-1]


class lstmKT(nn.Module):
    def __init__(self, feature_dim, hidden_dim, output_dim, dropout=0., bias=True):
        super(lstmKT, self).__init__()
        self.feature_dim = feature_dim  # 特征数量，等于 n_node * 2
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim    # 输出维度，等于 n_node + 1
        self.bias = bias
        self.rnn = nn.LSTM(feature_dim, hidden_dim, bias=bias, dropout=dropout, batch_first=True)
        self.f_out = nn.Linear(hidden_dim, output_dim, bias=bias)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
            elif isinstance(m, (nn.LSTM)):
                for i, weight in enumerate(m.parameters()):
                    if i < 2:
                        nn.init.orthogonal_(weight)

    def _get_next_pred(self, yt, questions):
        """
        Parameters:
            yt: predicted correct probability of all concepts at the next timestamp
            questions: question index matrix
        Shape:
            yt: [batch_size, seq_len - 1, output_dim]
            questions: [batch_size, seq_len]
            pred: [batch_size, seq_len - 1]
        Return:
            pred: predicted correct probability of the question answered at the next timestamp
        """
        one_hot = torch.eye(self.output_dim, device=yt.device)
        padding_idx = self.output_dim  # 使用 output_dim 作为填充值的索引
        one_hot = torch.cat((one_hot, torch.zeros(1, self.output_dim, device=yt.device)), dim=0)  # [output_dim + 1, output_dim]

        next_qt = questions[:, 1:]  # [batch_size, seq_len - 1]
        # 过滤并验证索引（填充值为 0）
        valid_next_qt = next_qt[next_qt != 0]  # 忽略填充值 0
        if valid_next_qt.numel() > 0:
            if (valid_next_qt < 0).any() or (valid_next_qt >= padding_idx).any():
                raise ValueError(
                    f"next_qt contains invalid indices: min {valid_next_qt.min()}, max {valid_next_qt.max()}, "
                    f"expected [0, {padding_idx-1}]")

        # 将填充值 0 替换为 padding_idx
        next_qt = torch.where(next_qt != 0, next_qt, padding_idx * torch.ones_like(next_qt, device=yt.device))
        one_hot_qt = F.embedding(next_qt, one_hot)  # [batch_size, seq_len - 1, output_dim]
        pred = (yt * one_hot_qt).sum(dim=-1)  # [batch_size, seq_len - 1]
        return pred

    def forward(self, features, questions):
        """
        Parameters:
            features: input one-hot matrix
            questions: question index matrix
        seq_len dimension needs padding, because different students may have learning sequences with different lengths.
        Shape:
            features: [batch_size, seq_len]
            questions: [batch_size, seq_len]
            pred_res: [batch_size, seq_len - 1]
        Return:
            pred_res: the correct probability of questions answered at the next timestamp
        """
        device = next(self.parameters()).to('cuda')  # 获取模型的设备
        features = features.to('cuda')
        questions = questions.to('cuda')
        # 创建 one-hot 编码矩阵，包含填充值 0 的映射
        feat_one_hot = torch.eye(self.feature_dim, device=features.device)  # [feature_dim, feature_dim]
        padding_idx = self.feature_dim  # 使用 feature_dim 作为填充值的索引
        feat_one_hot = torch.cat((feat_one_hot, torch.zeros(1, self.feature_dim, device=features.device)), dim=0)  # [feature_dim + 1, feature_dim]

        # # 调试：打印输入和最大值
        # print("features:", features)
        # print("features.max():", features.max())
        # print("questions:", questions)
        # print("questions.max():", questions.max())
        # print("self.feature_dim:", self.feature_dim)
        # print("self.output_dim:", self.output_dim)

        # 将填充值 0 替换为 padding_idx，并验证索引
        feat = torch.where(features != 0, features, padding_idx * torch.ones_like(features, device=features.device))
        if feat.numel() > 0:  # 避免空张量
            valid_feat = feat[feat != padding_idx]  # 忽略填充值
            if (valid_feat < 0).any() or (valid_feat >= padding_idx).any():
                raise ValueError(
                    f"features contains invalid indices: min {valid_feat.min()}, max {valid_feat.max()}, "
                    f"expected [0, {padding_idx-1}]")

        # 应用 one-hot 编码
        features = F.embedding(feat, feat_one_hot)  # [batch_size, seq_len, feature_dim]

        # 计算序列长度（使用 0 作为填充值）
        feature_lens = torch.ne(questions, 0).sum(dim=1)

        # LSTM 前向传播
        output, _ = self.rnn(features)  # [batch, seq_len, hidden_dim]
        yt = self.f_out(output)  # [batch, seq_len, output_dim]
        yt = torch.sigmoid(yt)
        yt = yt[:, :-1, :]  # [batch, seq_len - 1, output_dim]

        # 获取下一题的预测
        pred_res = self._get_next_pred(yt, questions)  # [batch, seq_len - 1]
        return pred_res
