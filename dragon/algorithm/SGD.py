from layer import *
import numpy
class Pre_Training(object):
    def __init__(self,fn,n_epoches=15,train_batch=None):
        self.fn=fn
        self.n_epoches=n_epoches
        self.train_batch=train_batch
    def run(self):
        epoch=0
        while (epoch<self.n_epoches):
            epoch=epoch+1
            batch_cost=[]
            for batch_index in xrange(self.train_batch):
                    batch_cost+=[self.fn(batch_index)]
            print ("epoch:%i cost:%f\n")%(epoch,numpy.mean(batch_cost))
class Mini_Batch(object):
    def __init__(self,model,n_epochs=100,load_param=None,save_param=None,vaild_interval=None):
        self.train_fn=model.train_fn
        self.vaild_fn=model.vaild_fn
        self.test_fn=None if model.train_fn==None else model.test_fn
        self.model=model
        self.n_epochs=n_epochs
        if vaild_interval is None:self.vaild_interval=model.train_batch
        self.load_param=load_param
        self.save_param=save_param
    def run(self):
        print "Now training model."
        patience=5000
        patienceIncrease=2
        threshold=0.995
        epoch=0
        done=False
        self.model.load_params(self.load_param)
        while (epoch<self.n_epochs) and (not done):
            epoch=epoch+1
            trainCost=[];trainLoss=[]
            for train_index in xrange(self.model.train_batch):
                Cost,Loss=self.train_fn(train_index)
                trainCost+=[Cost];trainLoss+=[Loss]
                iter=(epoch-1)*self.model.train_batch+train_index
                if (iter+1)%self.vaild_interval ==0:
                    vaildLoss=[]
                    for vaild_index in xrange(self.model.vaild_batch):
                        loss=self.vaild_fn(vaild_index)
                        vaildLoss+=[loss]
                    vaildLoss=numpy.mean(vaildLoss)
                    print "epoch:%i with %d vaild_batches error rate %f%%\n" %\
                        (epoch,self.model.vaild_batch,vaildLoss*100),
            print ("        with %d train_batches NLL: %f error rate %f%%\n")%(self.model.train_batch,numpy.mean(trainCost),numpy.mean(trainLoss)*100)
            self.model.save_params(self.save_param)