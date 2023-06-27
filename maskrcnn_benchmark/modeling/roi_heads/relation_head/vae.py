import  torch
from torch.utils.data import DataLoader
from torchvision import transforms,datasets
from torch import nn,optim
 


import numpy as np
class VAE(nn.Module):
    def __init__(self,input_dim=1024,output_dim=1024):
        super(VAE, self).__init__()

 
        self.encoder1=nn.Sequential(
            nn.Linear(input_dim,512),
            nn.ReLU(),
            nn.Linear(512,512),
            nn.ReLU()
        )
        self.encoder2=nn.Sequential(
            nn.Linear(256,512),
            nn.ReLU(),
            nn.Linear(512,512),
            nn.ReLU()
        )

        self.decoder1=nn.Sequential(
            nn.Linear(256,512),
            nn.ReLU(),
            nn.Linear(512,256),
        )
        self.decoder2=nn.Sequential(
            nn.Linear(256,512),
            nn.ReLU(),
            nn.Linear(512,512),
            nn.ReLU(),
            nn.Linear(512,output_dim),
        )
        self.skip1=nn.Linear(1024,256)
        self.skip2=nn.Linear(1024,256)
    def forward(self,x):
        kld = 0
        batch_size=x.size(0)
        h1=self.encoder1(x)
        mu,sigma=h1.chunk(2,dim=1) 
        z1 =mu+sigma*torch.rand_like(sigma) 
        kld += (-0.5 * (sigma + 1 - mu**2 - torch.exp(sigma))).sum(1).mean()
        
        h2=self.encoder2(z1)
        mu,sigma=h2.chunk(2,dim=1) 
        z2 =mu+sigma*torch.rand_like(sigma) 
        kld += (-0.5 * (sigma + 1 - mu**2 - torch.exp(sigma))).sum(1).mean()
        h3=self.encoder2(z2)
        mu,sigma=h3.chunk(2,dim=1) 
        z3 =mu+sigma*torch.rand_like(sigma) 
        kld += (-0.5 * (sigma + 1 - mu**2 - torch.exp(sigma))).sum(1).mean()


        # decoder
        z2_hat=self.decoder1(z3)+self.skip1(x)
        z1_hat=self.decoder1(z2_hat)+self.skip2(x)
        x_hat=self.decoder2(z1_hat)
      
        criteon = nn.MSELoss()
        mse = criteon(x_hat,x)
        loss=mse + 1.0*kld

        return x_hat,loss
 
def main1():
    mnist_train=datasets.MNIST('mnist',True,transform=transforms.Compose([
        transforms.ToTensor()
    ]),download=True)
    mnist_train=DataLoader(mnist_train,batch_size=32,shuffle=True)
 
    mnist_test = datasets.MNIST('mnist', False, transform=transforms.Compose([
        transforms.ToTensor()
    ]), download=True)
    mnist_test = DataLoader(mnist_test, batch_size=32, shuffle=True)
 
    x,_=iter(mnist_train).next()
    print('x:',x.shape)
 
    model=VAE()
 
    criteon=nn.MSELoss()
    optimizer=optim.Adam(model.parameters(),lr=1e-3)
    print(model)
 
    viz=visdom.Visdom()
 
    for epoch in range(100):
        for batch_size,(x,_) in enumerate(mnist_train):
            x_hat,kld=model(x)
            loss=criteon(x_hat,x)
            if kld is not None:
                elbo=-loss-1.0*kld
                loss=-elbo
            #backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print('epoch:',epoch,'loss:',loss.item(),'kld:',kld.item())
        x,_=iter(mnist_test).next() 
        with torch.no_grad():
            x_hat,kld=model(x)
        viz.images(x,nrow=8,win='x',opts=dict(title='x'))
        viz.images(x_hat,nrow=8,win='x_hat',opts=dict(title='x_hat'))


def main():
    model=VAE()
    x=torch.rand(16,1024)*2

if __name__ == '__main__':
    main()

















# import  torch
# from torch.utils.data import DataLoader
# from torchvision import transforms,datasets
# from torch import nn,optim


# class VAE(nn.Module):
#     def __init__(self,input_dim=1024,output_dim=1024):
#         super(VAE, self).__init__()

 
#         #[b,784]=>[b,20]
#         #u:[b,10]
#         #sigma:[b,10]
#         self.encoder=nn.Sequential(
#             nn.Linear(input_dim,512),
#             nn.ReLU(),
#             nn.Linear(512,256),
#             nn.ReLU(),
#             nn.Linear(256,256),
#             nn.ReLU()
#         )
#         #[b,20]=>[b,784]
#         self.decoder=nn.Sequential(
#             nn.Linear(128,256),
#             nn.ReLU(),
#             nn.Linear(256,512),
#             nn.ReLU(),
#             nn.Linear(512,output_dim),
#             nn.Sigmoid()
#         )
#     def forward(self,x):
#         batch_size=x.size(0)
#         #flatten
#         # x=x.view(batch_size,784)
#         #encoder
#         #[b,20],including mean and sigma
#         h_=self.encoder(x)
#         #[b,20]=>[b,10] and [b,10]
#         mu,sigma=h_.chunk(2,dim=1) #chunk是拆分，dim=1是在一维上拆分
#         #reparametrize trick,解决不能sample的问题 ，epison~N(0,1)
#         h=mu+sigma*torch.rand_like(sigma) #torch.rand_like(sigma) 就是正态分布
 
 
#         # decoder
#         x_hat=self.decoder(h)
#         #reshape
#         # x_hat=x_hat.view(batch_size,1,28,28)
 
#         # 下面是KL散度
#         # kl divergence
#         kld = (-0.5 * (sigma + 1 - mu**2 - torch.exp(sigma))).sum(1).mean()
#         criteon = nn.MSELoss()
#         mse = criteon(x_hat,x)
#         loss=mse + 1.0*kld

#         return x_hat,loss
 




# if __name__ == '__main__':
#     model=VAE()
#     x=torch.rand(16,1024)
#     print(model(x)[0].shape)