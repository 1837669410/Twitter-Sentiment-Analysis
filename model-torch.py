import torch
from torch import nn
from data import load_data
from utils import one_hot

class TextCNN(nn.Module):

    def __init__(self, vocab_num, out_dim, units, dropout_rate, l_rate, decay):
        super(TextCNN, self).__init__()

        # embedding
        self.embedding = nn.Embedding(num_embeddings=vocab_num, embedding_dim=out_dim, padding_idx=0)
        # conv
        self.conv11 = nn.Conv2d(in_channels=1, out_channels=units, kernel_size=(2,out_dim), stride=1, padding=0)
        self.conv12 = nn.Conv2d(in_channels=1, out_channels=units, kernel_size=(3,out_dim), stride=1, padding=0)
        self.conv13 = nn.Conv2d(in_channels=1, out_channels=units, kernel_size=(4,out_dim), stride=1, padding=0)
        self.conv21 = nn.Conv2d(in_channels=1, out_channels=units, kernel_size=(2,out_dim), stride=1, padding=0)
        self.conv22 = nn.Conv2d(in_channels=1, out_channels=units, kernel_size=(3,out_dim), stride=1, padding=0)
        self.conv23 = nn.Conv2d(in_channels=1, out_channels=units, kernel_size=(4,out_dim), stride=1, padding=0)
        # pool
        self.pool1 = [nn.MaxPool2d(kernel_size=(i,1), stride=1, padding=0) for i in [19,18,17]]
        self.pool2 = [nn.MaxPool2d(kernel_size=(i,1), stride=1, padding=0) for i in [19,18,17]]
        # dropout
        self.dropout = nn.Dropout(dropout_rate)
        # linear
        self.l1 = nn.Linear(in_features=600, out_features=100)
        self.l2 = nn.Linear(in_features=100, out_features=4)
        # relu
        self.relu = nn.ReLU()

        # loss_func
        self.loss_func = nn.CrossEntropyLoss()
        # opt
        self.opt = torch.optim.Adam(self.parameters(), lr=l_rate, weight_decay=decay)

    def forward(self, x):
        # embedding [None step] -> [None step out_dim] -> [None 1 step out_dim]
        emb = self.embedding(x)
        emb = torch.unsqueeze(emb, dim=1)
        # conv
        out11, out12, out13 = self.relu(self.conv11(emb)), self.relu(self.conv12(emb)), self.relu(self.conv13(emb))
        out21, out22, out23 = self.relu(self.conv21(emb)), self.relu(self.conv22(emb)), self.relu(self.conv23(emb))
        # pool
        out11, out12, out13 = self.pool1[0](out11), self.pool1[1](out12), self.pool1[2](out13)
        out21, out22, out23 = self.pool2[0](out21), self.pool2[1](out22), self.pool2[2](out23)
        # concat
        out = torch.concat((out11, out12, out13, out21, out22, out23), dim=1)
        out = torch.squeeze(out)
        # linear
        out = self.l1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.l2(out)
        return out

def train():
    epoch = 15
    db_train, db_val, vocab_num = load_data("twitter_training.csv", "twitter_validation.csv", max_text_length=20, framework="torch")
    device = torch.device("cuda")
    model = TextCNN(vocab_num=vocab_num, out_dim=64, units=100, dropout_rate=0.5, l_rate=0.0003, decay=0.001)
    model.to(device)
    for e in range(epoch):
        for step, (x,y) in enumerate(db_train):
            y = one_hot(y, depth=4)
            x, y = x.to(device), y.to(device)
            out = model.forward(x)
            loss = model.loss_func(out, y)
            model.opt.zero_grad()
            loss.backward()
            model.opt.step()
            if step % 100 == 0:
                print("epoch: %d | step: %d | epoch:%.3f"%(e,step,loss.item()))

        total_acc = 0
        total_num = 0
        for step, (x,y) in enumerate(db_val):
            x, y = x.to(device), y.to(device)
            out = model.forward(x)
            pred = out.softmax(dim=1).argmax(dim=1)
            acc = torch.eq(pred, y).float().sum().item()
            total_acc += acc
            total_num += x.shape[0]
        print("epoch: %d | acc:%.3f" % (e, total_acc / total_num))

if __name__ == "__main__":
    train()