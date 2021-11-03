import torch

pretrained_weights = torch.load("./checkpoint.pth")

num_class = 5 + 1
pretrained_weights["model"]["class_embed.weight"].resize_(num_class+1,256)
pretrained_weights["model"]["class_embed.bias"].resize_(num_class+1)

torch.save(pretrained_weights,'detr_r50_%d.pth'%num_class)