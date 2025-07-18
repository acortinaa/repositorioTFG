from common import *




## block ## -------------------------------------

class Linear_Bn(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Linear_Bn, self).__init__()
        self.linear = nn.Linear(in_channels, out_channels,bias=False)
        self.bn   = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        x = self.linear(x)
        x = self.bn(x)
        return x


## net  ## -------------------------------------
class TripletNet(nn.Module):
    def __init__(self):
        super(TripletNet, self).__init__()
        num_points = 3


        self.feature = nn.Sequential(
            Linear_Bn(3*num_points, 64),
            nn.ReLU(inplace=True),
            #nn.Dropout(0.3),
            Linear_Bn(64,  128),
            nn.ReLU(inplace=True),
            #nn.Dropout(0.3),
            Linear_Bn(128,  256),
            nn.ReLU(inplace=True),
            #nn.Dropout(0.3),
            Linear_Bn(256,  512),
            nn.ReLU(inplace=True),
            #nn.Dropout(0.3),
            Linear_Bn(512,  1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            Linear_Bn(1024, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),

            Linear_Bn(2048, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            Linear_Bn(1024, 512),
            nn.ReLU(inplace=True),
            #nn.Dropout(0.3),
            Linear_Bn(512,  256),
            nn.ReLU(inplace=True),
            #nn.Dropout(0.3),
            Linear_Bn(256,  128),
            nn.ReLU(inplace=True),
            #nn.Dropout(0.3),
            Linear_Bn(128,   64),
            nn.ReLU(inplace=True),
        )

        self.logit = nn.Sequential(
            nn.Linear(64, 1)
        )
        # self.target = nn.Sequential(
        #     nn.Linear(64, 3)
        # )


    def forward(self, x):

        batch_size  = x.size(0)
        x = x.view(batch_size,-1)

        x      = self.feature(x)
        logit  = self.logit(x).view(-1)

        return logit


    def set_mode(self, mode ):
        self.mode = mode
        if mode in ['eval', 'valid', 'test']:
            self.eval()
        elif mode in ['train']:
            self.train()
        else:
            raise NotImplementedError


### run ##############################################################################


def run_check_net():

    #create dummy data
    batch_size  = 5
    num_points  = 100
    tracklet = torch.randn((batch_size,num_points,3))
    tracklet = tracklet.cuda()

    net = TripletNet().cuda()
    logit = net(tracklet)

    # print(type(net))
    # print(net,'\n')

    print(logit,'\n')
    print(logit.size(),'\n')


    print('')





########################################################################################
if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))

    run_check_net()

    print( 'sucessful!')


class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        num_points = 3
        self.net = nn.Sequential(
            Linear_Bn(3*num_points, 64),
            nn.ReLU(inplace=True),
            Linear_Bn(64,  128),
            nn.ReLU(inplace=True),
            Linear_Bn(128, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            Linear_Bn(256, 128),
            nn.ReLU(inplace=True),
            Linear_Bn(128,64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1)  
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.net(x).squeeze(1) 

