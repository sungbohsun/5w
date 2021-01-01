import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from tqdm import tqdm 
from torch import optim
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import f1_score,confusion_matrix
from utils import CRNN,data_pre,adjusting_learning_rate


X_train,y_train,X_test,y_test = data_pre(augm=False)
    

os.environ["CUDA_VISIBLE_DEVICES"]="0"
print('use',torch.cuda.get_device_name(0))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
with torch.no_grad():
    torch.cuda.empty_cache()    
    
writer = SummaryWriter()
model = CRNN().to(device)
loss_fn = nn.NLLLoss()
learning_rate = 1e-4
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
score_train,score_val = [],[]
best_f1 = 0
early_stop = 0
epoch = 0
while early_stop <=50 :
    epoch += 1
    pre,tru = [],[]
    for batch_num,(x,y) in enumerate(zip(X_train,y_train)):
        model.train()
        output = model(x.to(device))
        y=torch.tensor(y, dtype=torch.long) 
        loss = loss_fn(output, y.to(device))
        if torch.flatten(torch.isnan(loss)).any():
            print('!!!!!pass')
            continue
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        pre.extend([f.argmax() for f in output.to('cpu')])
        tru.extend([f for f in y])    
    score = f1_score(pre,tru,average='micro')
    score_train.append(score)
    print('epoch',epoch,'train:',round(score*100,3),end=' ')
    writer.add_scalar("F1/train", score, epoch)
    
    for batch_num,(x,y) in enumerate(zip(X_test,y_test)):
        model.eval()
        with torch.no_grad():
            output = model(x.to(device))
            pre.extend([f.argmax() for f in output.to('cpu')])
            tru.extend([f for f in y])
    
    score_ = f1_score(pre,tru,average='micro')
    score_val.append(score_)
    print('val:',round(score_*100,3))
    writer.add_scalar("F1/val", score_, epoch)
    
    if score > best_f1: 
        best_f1 = score
        state_dict = {'model':model.state_dict(),
                      'optimizer': optimizer.state_dict(),
                      'epoch': epoch}
        torch.save(state_dict,'model/best_model'+str(epoch)+'.pt')
        early_stop = 0
    else : 
        #adjusting_learning_rate(optimizer=optimizer, factor=0.95, min_lr=5e-6)
        early_stop += 1

print('stop at epoch:',epoch)   
dic = {'score_train':score_train,'score_val':score_val}
torch.save(dic,'score_'+str(epoch)+'.pt')
#writer.add_graph(model.to('cpu'), images)
writer.close()