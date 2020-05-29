import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms as transforms

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device: ",device)

# Mini batch size
mb_size =  64 

# Transform data to pytorch format
transform = transforms.Compose([transforms.ToTensor()])

# Download and transform dataset
trainset = torchvision.datasets.MNIST(root = './NewData', download = True, train = True, transform=transform)

# Load one minibatch at a time the dataset 
trainloader = torch.utils.data.DataLoader(trainset,shuffle=True,batch_size=mb_size)

#It is just an iterator
data_iter = iter(trainloader)

images, labels = data_iter.next()
#Flattening matrix
test = images.view(images.size(0),-1) 
print(images.size(),test.size())

# Global vars
Z_dim = 100  
X_dim = test.size(1)
h_dim = 128
# Learning rate for Adam
lr = 1e-3

def imshow(img):
    im = torchvision.utils.make_grid(img)
    # From tensor to numpy
    npimg = im.numpy()
    print(npimg.shape)
    plt.figure(figsize=(8,8))
    #transpose to make sure you get 28x28x1 
    transposed_im = np.transpose(npimg, (1,2,0))
    plt.imshow(transposed_im)
    plt.xticks([])
    plt.yticks([])

    plt.show()

imshow(images)


def init_weigth(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0)

class Gen(nn.Module):
    def __init__(self):
        super(Gen,self).__init__()
        self.model = nn.Sequential(
            nn.Linear(Z_dim,h_dim),
            nn.ReLU(),
            nn.Linear(h_dim,X_dim),
            nn.Sigmoid()
            )
        self.model.apply(init_weigth)
    def forward(self, input):
        return self.model(input)

class Dis(nn.Module):
    def __init__(self):
        super(Dis, self).__init__()
        self.model = nn.Sequential(
            nn.Linear( X_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim,1),
            nn.Sigmoid()
        )
        self.model.apply(init_weigth)
    
    def forward(self, input):
        return self.model(input)

generator = Gen().to(device)

discriminator = Dis()#.to(device)

g_solver = optim.Adam(generator.parameters(), lr = lr)

d_solver = optim.Adam(discriminator.parameters(), lr = lr)

def train_loop(num_epochs=10):
    g_loss_run = 0.0
    d_loss_run = 0.0
    
    for epoch in range(num_epochs):
        for i,data in enumerate(trainloader):
            X,labels = data # Labels not used (no classification)
            mb_size = X.size(0)
            X = X.view(X.size(0), -1)

            # 1 |=> real, 0 |=> fake
            one_labels = torch.ones(mb_size, 1)
            zero_labels = torch.zeros(mb_size,1)

            z = torch.randn(mb_size, Z_dim) # This is my "noise"
            g_sample = generator(z)
            d_fake = discriminator(g_sample) # output fakes estimation 
            d_real = discriminator(X) # output real estimation
            d_fake_loss = F.binary_cross_entropy(d_fake,zero_labels)
            d_real_loss = F.binary_cross_entropy(d_real,one_labels) 

            d_loss = d_fake_loss + d_real_loss
            d_solver.zero_grad() # make the gradient zero to start fresh
            d_loss.backward() # ?
            d_solver.step()

            # These are fresh not memorized (just to show not cheating here) 
            z = torch.randn(mb_size, Z_dim)
            g_sample = generator(z)
            d_fake = discriminator(g_sample)

            g_loss = F.binary_cross_entropy(d_fake, one_labels)
            g_solver.zero_grad()
            g_loss.backward()
            g_solver.step()

        print("Epoch: {}, G_loss: {}, D_loss: {}".format(epoch,g_loss_run/(i+1), g_loss_run/(i+1)))
        samples = generator(z).detach()
        samples = samples.view(mb_size, 1, 28, 28) #TODO replace hardcoding
        imshow(samples)


#train_loop()

exp = Experiment(save_dir=os.getcwd())

trainer = pl.Trainer(experiment=exp, gpus=[0], max_nb_epochs=200)
trainer.fit(generator)
trainer.fit(discriminator)

