import model
import config
import mydatasets
import torch
import torch.nn as nn
#import train

config = config.Config()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

train_iter, dev_iter, test_iter, src_token_num, trg_token_num, \
PAD_TOKEN, SOS_TOKEN = mydatasets.get_iter(config, device)

loss_func = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)
model = model.Seq2Seq(config, src_token_num, trg_token_num)
model.to(device)

for name, param in model.named_parameters():
    print(name)


def init_weight(m):
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        nn.init.xavier_uniform_(m.weight.data)


model.apply(init_weight)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


print(f'The model has {count_parameters(model):,} trainable parameters')

for item in train_iter:
    result = model(item.src, item.trg, PAD_TOKEN, device)
    print(result)
    print(result.shape)

# optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-9)

# choice = input("1-训练, 2-测试.请输入： ")
# choice = int(choice)
# if choice == 1:
#     train.train(model, loss_func, optimizer, train_iter, dev_iter, PAD_TOKEN, SOS_TOKEN, device)
# elif choice == 2:
#     #  把最好的存起来，然后去加载过来
#     train.eval(model, loss_func, test_iter, PAD_TOKEN, SOS_TOKEN, device)
