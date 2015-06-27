from layer.core import *
from algorithm.SGD import Mini_Batch
from data.process import loadData
from layer.model import Model
if __name__ == '__main__':
    dataSet=loadData()
    cifar=Model(batch_size=100,lr=0.0001,dataSet=dataSet,weight_decay=0.004)
    neure=[32,32,64,64]
    batch_size=100
    cifar.add(DataLayer(batch_size,(32,32,3)))
    cifar.add(ConvolutionLayer((batch_size,3,32,32),(neure[0],3,3,3),'relu','Gaussian',0.0001))
    cifar.add(PoolingLayer())
    cifar.add(ConvolutionLayer((batch_size,neure[0],15,15),(neure[1],neure[0],4,4),'relu','Gaussian',0.01))
    cifar.add(PoolingLayer())
    cifar.add(ConvolutionLayer((batch_size,neure[1],6,6),(neure[2],neure[1],5,5),'relu','Gaussian',0.01))
    cifar.add(PoolingLayer())
    cifar.add(FullyConnectedLayer(neure[2]*1*1,neure[3],'relu','Gaussian',0.1))
    cifar.add(DropoutLayer(0.5))
    cifar.add(SoftmaxLayer(neure[3],5,'Gaussian',0.1))
    cifar.build_train_fn()
    cifar.build_vaild_fn()
    algorithm=Mini_Batch(model=cifar,n_epochs=100,load_param='cnn_params.pkl',save_param='cnn_params.pkl')
    algorithm.run()
    