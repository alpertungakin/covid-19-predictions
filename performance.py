import json
import numpy as np 
import pandas as pd
import torch
from torch import nn
from torch.autograd import Variable
import sklearn.metrics as sm
import requests

torch.cuda.set_device('cuda:0')
api = requests.get("http://services.mapisso.com/covid19.services/api/v1/timeline")
api_in = api.text
data = json.loads(api_in)

df = pd.DataFrame(data["result"][150]["timeline"])
#row_df = pd.Series([27069, '2020-4-4',501,786])
#row_df = pd.DataFrame([row_df],index=[len(df)+1])
#row_df = row_df.rename(columns={0: "cases", 1: "date",2:"deaths",3:"recovered"})
#df = df.append(row_df)
first = df["confirmed"].to_numpy().nonzero()[0][0]
df = df.loc[first:,:]
#print(df.loc[first]['date'])
#df = df.tail(30)
#
df1 = df.iloc[:int(len(df)*0.8),:]
df2 = df.iloc[int(len(df)*0.8):,:]

def init_weights(m):
    if type(m) == nn.Linear:
        m.weight.data.normal_(0.0, 0.1)
        m.bias.data.fill_(0.01)
#df = pd.DataFrame(data["result"][156]["timeline"])
#row_df = pd.Series([312245, '2020-4-4',8503,15021])
#row_df = pd.DataFrame([row_df],index=[len(df)+1])
#row_df = row_df.rename(columns={0: "cases", 1: "date",2:"deaths",3:"recovered"})
#df = df.append(row_df)
#first = df["cases"].to_numpy().nonzero()[0][0]
#df = df.loc[first:,:]

days = np.arange(len(df1)).reshape((len(df1),1))+1
days = days.astype(np.float64)
valid = np.arange(np.max(days)+1,(np.max(days)+len(df2))+1).reshape((len(df2),1))
valid = valid.astype(np.float64)
days0 = np.vstack((days, valid))
#days0 = (days0 - np.min(days0))/(np.max(days0)-np.min(days0))
days0 = days0/np.max(days0)
ndays = days0[:len(df1)]
nvalid = days0[len(df1):]


target = np.array(df1["confirmed"]).reshape((len(df1),1))
target = target.astype(np.float64)
#ntarget = (target - np.min(target))/(np.max(target)-np.min(target))
ntarget = target/np.max(target)
#    train = torch.unsqueeze(torch.tensor(ndays),dim=1)
train = torch.from_numpy(ndays)
#    target0 = torch.unsqueeze(torch.tensor(ntarget),dim=1)
target0 = torch.from_numpy(ntarget)
#    intest = torch.unsqueeze(torch.tensor(ntest),dim=1)
invalid = torch.from_numpy(nvalid)
invalid = invalid.cuda()
train = train.cuda()
target0 = target0.cuda()

train, target0, invalid = Variable(train), Variable(target0), Variable(invalid)

model = nn.Sequential(
    nn.Linear(1, len(days)),
    nn.LeakyReLU(),
    nn.Linear(len(days), len(days)*2),
    nn.Sigmoid(),    
    nn.Linear(len(days)*2, 1)
)
model.apply(init_weights)
model.to('cuda:0')
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
#    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)
#optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss_func = nn.MSELoss()
steps = 0
stop = False
while stop == False:
    steps = steps + 1
    prediction = model(train.float())     # input x and predict based on x

    optimizer.zero_grad()   # clear gradients for next train
    loss = loss_func(prediction, target0.float())     # must be (1. nn output, 2. target)

    loss.backward()         # backpropagation, compute gradients
    optimizer.step()  
#        scheduler.step()
    print(loss)
    if loss<=0.0001 or steps>=200000:
        stop=True
#        elif steps>10000 and loss>0.0005:
#            stop = True

pred = model(invalid.float())
pred = pred.cpu()
#    pred = pred.detach().numpy()[:,:,0]
pred = pred.detach().numpy()
#denormal_pred = abs((pred*(np.max(target)-np.min(target))) + np.min(target))
denormal_pred = pred*np.max(target)
validation = np.array(df2["confirmed"]).reshape((len(df2),1))
meanse = sm.mean_squared_error(abs(denormal_pred), validation)
meanabse = sm.mean_absolute_error(denormal_pred, validation)
maxe = sm.max_error(denormal_pred, validation)
var_sc = sm.explained_variance_score(denormal_pred, validation)
r2_coeff = sm.r2_score(denormal_pred, validation)
adjr2_coeff = 1-(1-r2_coeff)*((2*len(validation)-1)/(2*len(validation)-1-1))
print("Mean Squared Error: {:.4f} cases".format(meanse))
print("Mean Absolute Error: {:.4f} cases".format(meanabse))
print("Maximum Error: {:.4f} cases".format(maxe))
print("Explained Variance Score: {:.4f}".format(var_sc))
print("R^2 Coefficient: {:.4f}".format(r2_coeff))
print("Adjusted R^2 Coefficient: {:.4f}".format(adjr2_coeff))

