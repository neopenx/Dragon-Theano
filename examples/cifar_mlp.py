from layer.core import *
from algorithm.SGD import Mini_Batch
from data.process import loadData, loadScaleData
from layer.model import Model
if __name__ == '__main__':
    dataSet=loadScaleData('data.pkl')
    cifar=Model(batch_size=100,lr=0.01,dataSet=dataSet,weight_decay=0.0)
    neure=[1000,1000,1000]
    batch_size=100
    cifar.add(DataLayer(batch_size,32*32*3))
    cifar.add(FullyConnectedLayer(32*32*3,neure[0],'relu','Gaussian',0.1))
    cifar.add(DropoutLayer(0.2))
    cifar.add(FullyConnectedLayer(neure[0],neure[1],'relu','Gaussian',0.1))
    cifar.add(DropoutLayer(0.2))
    cifar.add(FullyConnectedLayer(neure[1],neure[2],'relu','Gaussian',0.1))
    cifar.add(DropoutLayer(0.2))     
    cifar.add(SoftmaxLayer(neure[2],10))
    cifar.pretrain()
    cifar.build_train_fn()
    cifar.build_vaild_fn()
    algorithm=Mini_Batch(model=cifar,n_epochs=100,load_param='mlp_params.pkl',save_param='mlp_params.pkl')
    algorithm.run()
    