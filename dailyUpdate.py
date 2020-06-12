import json
import requests
import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
import datetime

api = requests.get("http://services.mapisso.com/covid19.services/api/v1/timeline")
api_in = api.text
data = json.loads(api_in)
torch.cuda.set_device('cuda:0')
week = datetime.datetime.today() + datetime.timedelta(days = 4)
week_str = str(week.day) + '-' + str(week.month) + '-' + str(week.year)
two_week = datetime.datetime.today() + datetime.timedelta(days = 9)
two_week_str = str(two_week.day) + '-' + str(two_week.month) + '-' + str(two_week.year)
result = []
def init_weights(m):
    if type(m) == nn.Linear:
        m.weight.data.normal_(0.0, 0.1)
        m.bias.data.fill_(0.01)
for i in range(len(data["result"])):
    result.append({"country":(data["result"][i]["country"]),"{}".format(week_str):(), "{}".format(two_week_str):()})

counter = 0
for every in data["result"]:
    frame = pd.DataFrame(every["timeline"])
    first = frame["confirmed"].to_numpy().nonzero()[0][0]
    frame = frame.loc[first+10:,:]
    
    if every['country'] == 'China':
        result[counter]["{}".format(week_str)]=0
        result[counter]["{}".format(two_week_str)]=0               
    elif len(frame)>=14:
        target = np.array(frame["confirmed"]).reshape((len(frame),1))
        days = np.arange(len(frame)).reshape((len(frame),1))+1
        days = days.astype(np.float64)
        test = np.arange(np.max(days)+1,(np.max(days)*2)+1).reshape((len(frame),1))
        test = test.astype(np.float64)
        days0 = np.vstack((days, test))
        days0 = days0/np.max(days0)
        ndays = days0[:len(frame)]
        ntest = days0[len(frame):]
        ntarget = target/np.max(target)
        target = target.astype(np.float64)
        #ntarget = (target - np.min(target))/(np.max(target)-np.min(target))
        ntarget = target/np.max(target)
    #    train = torch.unsqueeze(torch.tensor(ndays),dim=1)
        train = torch.from_numpy(ndays)
    #    target0 = torch.unsqueeze(torch.tensor(ntarget),dim=1)
        target0 = torch.from_numpy(ntarget)
    #    intest = torch.unsqueeze(torch.tensor(ntest),dim=1)
        intest = torch.from_numpy(ntest)
        intest = intest.cuda()
        train = train.cuda()
        target0 = target0.cuda()
        
        train, target0, intest = Variable(train), Variable(target0), Variable(intest)
        model = nn.Sequential(
            nn.Linear(1, len(days)),
            nn.LeakyReLU(),
            nn.Linear(len(days), len(days)*2),
            nn.Sigmoid(),    
            nn.Linear(len(days)*2, 1)
        )
        model.apply(init_weights)
        model.to('cuda:0')
#        optimizer = torch.optim.SGD(model.parameters(), lr = 0.01, momentum = 0.9)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        loss_func = nn.MSELoss()
        steps = 0
        stop = False
        while stop == False:
            steps = steps + 1
            prediction = model(train.float())     # input x and predict based on x
        
            optimizer.zero_grad() 
            loss = loss_func(prediction, target0.float())     # must be (1. nn output, 2. target)
        
              # clear gradients for next train
            loss.backward()         # backpropagation, compute gradients
            optimizer.step()  
            if loss<=0.00001:
                stop=True
                print("{} is completed".format(every["country"]))
                pred = model(intest.float())
                pred = pred.cpu()
                pred = pred.detach().numpy()
                denormal_pred = pred*np.max(target)
#                diff = np.abs(denormal_pred - target[-1,0])
#                day_index = diff.argmin() + 2
                result[counter]["{}".format(week_str)]=int(denormal_pred[4][0])
                result[counter]["{}".format(two_week_str)]=int(denormal_pred[9][0])
                
            elif steps>200000 and loss>0.00001:
                stop = True
                print("{} is completed".format(every["country"]))
                pred = model(intest.float())
                pred = pred.cpu()
                pred = pred.detach().numpy()
                denormal_pred = pred*np.max(target)#                diff = np.abs(denormal_pred - target[-1,0])
#                day_index = diff.argmin() + 2
                result[counter]["{}".format(week_str)]=int(denormal_pred[4][0])
                result[counter]["{}".format(two_week_str)]=int(denormal_pred[9][0])

    else:
        result[counter]["{}".format(week_str)]=0
        result[counter]["{}".format(two_week_str)]=0       
    counter = counter + 1
result = json.dumps(result, sort_keys = True, indent = 4)
with open('result.json', 'w') as f:
    f.write(result)

    
    
    
    
    
    
    
    

    
