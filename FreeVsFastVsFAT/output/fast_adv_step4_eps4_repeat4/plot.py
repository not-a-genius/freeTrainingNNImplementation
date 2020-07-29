import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import matplotlib.ticker as plticker


def set_pretty_epoch(x):
    print(x)
    integ = int(x.split("]")[0].strip("["))
    dec = x.split("]")[1].strip("[")
    dec = float(dec.split("/")[0]) / float(dec.split("/")[1])
    dec = float("{:.2f}".format(dec))
    print(str(integ + dec))
    return (int(integ + dec))

data = pd.read_csv('logfast.txt', skiprows=32, sep=" ", header=None)
#Not considering avg
data = data.iloc[:-1]
data = data.dropna()
data = data.drop(columns=[0,1,4,5,6,8,10,12])
data = data.rename(columns={2:"epoch",3:"time",7:"loss",9:"prec1",11:"prec5"})
data['epoch'] = data['epoch'].apply(lambda x: str(x))
data['epoch'] = data['epoch'].apply(lambda x: x.split("\t")[0])
data['epoch'] = data['epoch'].apply(lambda x: set_pretty_epoch(x))

data = data.groupby(by="epoch").mean()
data = data.reset_index()
data = data.sort_values(by=['epoch'])

loss = data["loss"]
acc = data["prec1"]
acc5 = data["prec5"]
epochs = data["epoch"]

fig, ax = plt.subplots()
# fig = plt.figure(figsize=(20, 10))
plt.plot(epochs, loss, 'b', label='loss')
plt.plot(epochs, acc, 'g', label='acc')
plt.plot(epochs, acc5, 'y', label='acc5')

plt.title('Training loss and acc')
plt.xlabel('Epochs')
plt.ylabel('Value')

plt.axis([0, 101, 0, 101])
loc = plticker.MultipleLocator(base=10.0) # this locator puts ticks at regular intervals
ax.xaxis.set_major_locator(loc)

plt.legend()
plt.show()

