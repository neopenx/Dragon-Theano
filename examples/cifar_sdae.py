from layer.core import *
from algorithm.SGD import Mini_Batch
from data.process import loadData, loadScaleData
from layer.model import Model
if __name__ == '__main__':
    dataSet=loadScaleData('data.pkl')
    cifar=Model(batch_size=100,lr=0.005,dataSet=dataSet,weight_decay=0.0)
    neure=[1000,1000,1000]
    batch_size=100
    cifar.add(DataLayer(batch_size,32*32*3))
    cifar.add(AutoEncodeLayer(32*32*3,neure[0],'relu','softplus',cost='squre',weight_init='Gaussian',gauss_std=0.1,level=0.3))
    cifar.add(DropoutLayer(0.2))
    cifar.add(SoftmaxLayer(neure[0],10))
    cifar.pretrain(batch_size=20,n_epoches=15)
    cifar.build_train_fn()
    cifar.build_vaild_fn()
    algorithm=Mini_Batch(model=cifar,n_epochs=100,load_param='',save_param='mlp_params.pkl')
    algorithm.run()
    