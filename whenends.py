import json
import numpy as np 
import pandas as pd
import torch
from torch import nn
from torch.autograd import Variable
import sklearn.metrics as sm
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import requests
import datetime
from scipy.stats import spearmanr, pearsonr, chisquare
def init_weights(m):
    if type(m) == nn.Linear:
        m.weight.data.normal_(0.0, 0.01)
        m.bias.data.fill_(0.1)
        
torch.cuda.set_device("cuda:0")
api = requests.get("http://services.mapisso.com/covid19.services/api/v1/timeline")
before22 = pd.read_csv('before22Jan.csv', header = None)
before22 = before22.to_numpy()
api_in = api.text
data = json.loads(api_in)
ch_fin = datetime.date(2020, 3, 29)
stabil = datetime.date.today() - ch_fin
stabil = stabil.days
china = pd.DataFrame(data["result"][32]["timeline"])
turkey = pd.DataFrame(data["result"][150]["timeline"])
#first_china = china["cases"].to_numpy().nonzero()[0][0]
#china = china.loc[first_china:,:]
first_turkey = turkey["confirmed"].to_numpy().nonzero()[0][0]
turkey = turkey.loc[first_turkey:,:]
stabildays = np.zeros((stabil,1))
turkey = np.array(turkey["confirmed"]).reshape((len(turkey),1))
china = np.array(china["confirmed"]).reshape((len(china),1))
china = np.vstack((before22, china))
days = np.flip(np.arange(0,len(china)-stabil).reshape((len(china)-stabil,1)),axis = 0)
days = np.vstack((days, stabildays))
ndays = days/100
que_chi = np.ones_like(china)
que_chi[:len(before22),0] = 0
que_tr = np.zeros_like(turkey)
que_tr[31,0] = 1
que_tr[32,0] = 1
que_tr[37,0] = 1
que_tr[38,0] = 1
que_tr[39,0] = 1
que_tr[43,0] = 1
que_tr[44,0] = 1
que_tr[45,0] = 1
que_tr[46,0] = 1
que_tr[51,0] = 1
que_tr[52,0] = 1
que_tr[53,0] = 1
que_tr[59,0] = 1
que_tr[60,0] = 1
que_tr[66,0] = 1
que_tr[67,0] = 1
que_tr[68,0] = 1
que_tr[69,0] = 1
que_tr[73,0] = 1
que_tr[74,0] = 1
que_tr[75,0] = 1

#stra_disTr = que_tr*121/100
#stra_disCh = que_chi*118/100
#acq_disTr = que_tr*91/100
#acq_disCh = que_chi*89/100
#clo_disTr = que_tr*59/100
#clo_disCh = que_chi*58/100
china = china/float(1393000000)
turkey = turkey/float(82000000)
nturkey = (turkey - np.min(turkey))/(np.max(turkey) - np.min(turkey))
nchina = (china - np.min(china))/(np.max(china) - np.min(china))
###############################################################################
#test = np.hstack((nturkey, stra_disTr, acq_disTr, clo_disTr))
#train = np.hstack((nchina, stra_disCh, acq_disCh, clo_disCh))
test = np.hstack((nturkey, que_tr))
train = np.hstack((nchina, que_chi))
test = torch.unsqueeze(torch.tensor(test),dim=1)
train = torch.unsqueeze(torch.tensor(train),dim=1)
target = torch.unsqueeze(torch.tensor(ndays),dim=1)
train = train.cuda()
test = test.cuda()
target = target.cuda()
test, target, train = Variable(test), Variable(target), Variable(train)
# torch.cuda.set_device('cuda:0')
model = nn.Sequential(
    nn.Linear(2, len(days)),
    nn.ReLU(),
    nn.Linear(len(days), len(days)*2),
    nn.Sigmoid(),    
    nn.Linear(len(days)*2, 1),
    nn.ReLU()
)
model.apply(init_weights)
model.to("cuda:0")
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
#optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss_func = nn.MSELoss()
steps = 0
stop = False
while stop == False:
    steps = steps + 1
    prediction = model(train.float())     # input x and predict based on x

    optimizer.zero_grad()   # clear gradients for next train

    loss = loss_func(prediction, target.float())     # must be (1. nn output, 2. target)

    loss.backward()         # backpropagation, compute gradients
    optimizer.step()  
    print(loss)
    if loss<=0.01 or (loss<=0.01 and steps>25000):
        stop=True
    

output = model(train.float())
output = output.cpu()
#    output = output.detach().numpy()[:,:,-1]
output = output.detach().numpy()
#denormal_output = abs((output*(np.max(target)-np.min(target))) + np.min(target))
denormal_output = output*100
pred = model(test.float())
pred = pred.cpu()
#    pred = pred.detach().numpy()[:,:,0]
pred = pred.detach().numpy()
#denormal_pred = abs((pred*(np.max(target)-np.min(target))) + np.min(target))
denormal_pred = pred*100

meanse = sm.mean_squared_error(denormal_output[:,:,0], days)
meanabse = sm.mean_absolute_error(denormal_output[:,:,0], days)
maxe = sm.max_error(denormal_output[:,:,0], days)
var_sc = sm.explained_variance_score(denormal_output[:,:,0], days)
r2_coeff = sm.r2_score(denormal_output[:,:,0], days)
adjr2_coeff = 1-(1-r2_coeff)*((len(china)-1)/(len(china)-2-1))
out = denormal_output[:,:,0]
out = out.reshape((len(out),))
out = list(out)
days = days.reshape((len(days),))
days = list(days)
stat1, p1 = pearsonr(out, days)
stat2, p2 = spearmanr(out, days)
for i in range(len(days)):
    if out[i] == 0:
        out[i] = 1.0
    if days[i] == 0:
        days[i] = 1.0
stat3, p3 = chisquare(out, days, ddof = len(days)-3)
print("                                           ")
print("-------------------------------------------")
print("Mean Squared Error: {:.4f} days".format(meanse))
print("Mean Absolute Error: {:.4f} days".format(meanabse))
print("Maximum Error: {:.4f} days".format(maxe))
print("Explained Variance Score: {:.4f}".format(var_sc))
print("R^2 Coefficient: {:.4f}".format(r2_coeff))
print("Adjusted R^2 Coefficient: {:.4f}".format(adjr2_coeff))
del i
outputs = list()
outputs0 = list()
for i in range(len(denormal_pred)):
    outputs.append(float(denormal_pred[:,:][i]))
for j in range(len(outputs)):
    if outputs[j]>2*maxe:
        outputs0.append(outputs[j])

date_index = outputs.index(outputs0[-1])
xlabel = np.arange(0,len(denormal_pred[:,:]),1)
plt.plot(xlabel, denormal_pred[:,:,0], color = 'black') 
plt.title('Date', size=16)
plt.ylabel('Number of days to the end', size = 16)
ax = plt.gca()
ax.set_aspect('equal')
ax.axes.xaxis.set_ticklabels([])
#ax.axes.xaxis.set_major_locator(ticker.MultipleLocator(1))
plt.grid(True)
plt.annotate("{}".format(data["result"][150]["timeline"][-1]["date"]), (xlabel[-1],denormal_pred[:,:,0][-1]), size = 14, bbox=dict(boxstyle="round", fc="w"))
plt.show()
#%%



















