from layer.core import *
import cPickle
import os
from algorithm.SGD import Pre_Training
class Model(object):
    def __init__(self,batch_size,lr,dataSet,momentum=0.9,weight_decay=0.004):
        self.layers=[]
        self.params=[]
        self.pretrain_cost=[]
        self.batch_size=batch_size
        self.lr=lr
        self.momentum=momentum
        self.x=T.matrix('x')
        self.index=T.lscalar()
        if dataSet is None:return
        if(len(dataSet)>1):
            self.trainSet_X,self.trainSet_Y=dataSet[0]
            self.vaildSet_X,self.vaildSet_Y=dataSet[1]
        else:self.vaildSet_X,self.vaildSet_Y=dataSet[0]
        self.weight_decay=weight_decay
        self.train_batch=self.trainSet_X.get_value(borrow=True).shape[0]/batch_size
        self.vaild_batch=self.vaildSet_X.get_value(borrow=True).shape[0]/batch_size
        self.train_fn=self.vaild_fn=self.test_fn=None
    def add(self,layer):
        self.layers.append(layer)
        if layer.params is not None:
            self.params.extend(layer.params)
    def pretrain(self,batch_size=20,n_epoches=15):
        cnt=0
        x=T.matrix('x');y=T.ivector('y');index=T.lscalar()
        for layer in self.layers:
            if type(layer) is DataLayer:
                input=x.reshape((batch_size,layer.size))
                continue
            if type(layer) is not AutoEncodeLayer:continue #ignore Dropout Layer
            cnt=cnt+1
            layer.input=input
            cost=layer.reconstruct()
            params=layer.params
            grads = T.grad(cost, params)
            updates = [(param_i, param_i - self.lr* grad_i) for param_i, grad_i in zip(params, grads)]
            pre_fn=theano.function(inputs=[index],outputs=cost,updates=updates,givens=
                                   {x:self.trainSet_X[index*batch_size:(index+1)*batch_size]})
            print "Pre-Training Layer %d \n"%cnt
            batches=self.batch_size/batch_size*self.train_batch
            algorithm=Pre_Training(pre_fn,n_epoches,batches)
            algorithm.run()
    def build_train_fn(self):
        cost=None;index=T.lscalar();
        x=T.matrix('x');y=T.ivector('y')
        L2=0.0
        for layer in self.layers:
            if type(layer) is DataLayer:
                layer.input=x
                input=layer.get_output()
                continue
            ####Model Input
            layer.input=input
            ####Model Output
            if layer.params is not None:
                L2+=self.weight_decay*(layer.W**2).sum()
            if type(layer) is SoftmaxLayer:
                cost=layer.NLL(y)
                loss=layer.error(y)
                continue
            if type(layer) is FullyConnectedLayer:
                input=input.flatten(2)
                layer.input=input
                input=layer.get_output()
            if type(layer) is DropoutLayer:
                input=layer.get_output(mode='train')
            else:
                input=layer.get_output()
        params=self.params
        cost=cost+L2
        updates=[]
        for param in params:
            param_update = theano.shared(param.get_value()*0.,broadcastable = param.broadcastable)
            updates.append((param,param - T.cast(self.lr,dtype=theano.config.floatX)*param_update))
            updates.append((param_update , self.momentum*param_update+(1.-self.momentum)*T.grad(cost,param)))
        train_model=theano.function(inputs=[index],outputs=[cost,loss],updates=updates,givens={
                                      x:self.trainSet_X[index*self.batch_size:(index+1)*self.batch_size],
                                      y:self.trainSet_Y[index*self.batch_size:(index+1)*self.batch_size]}
                              )
        #self.train_model=train_model
        self.train_fn=train_model
    def build_vaild_fn(self):
        error=None;index=T.lscalar();
        x=T.matrix('x');y=T.ivector('y')
        for layer in self.layers:
            if type(layer) is DataLayer:
                layer.input=x
                input=layer.get_output()
                continue
            ####Model Input
            layer.input=input
            ####Model Output
            if type(layer) is SoftmaxLayer:
                error=layer.error(y)
                continue
            if type(layer) is FullyConnectedLayer:
                input=input.flatten(2)
                layer.input=input
                input=layer.get_output()
            if type(layer) is DropoutLayer:
                input=layer.get_output(mode='test')
            else:
                input=layer.get_output()
        test_model=theano.function(inputs=[index],outputs=error,givens={
                                      x:self.vaildSet_X[index*self.batch_size:(index+1)*self.batch_size],
                                      y:self.vaildSet_Y[index*self.batch_size:(index+1)*self.batch_size]}
                              )
        self.vaild_fn=test_model
    def build_test_fn(self):
        pred=None;
        for layer in self.layers:
            if type(layer) is DataLayer:
                layer.input=self.x
                input=layer.get_output()
                continue
            ####Model Input
            layer.input=input
            ####Model Output
            if type(layer) is SoftmaxLayer:
                pred=layer.pred()
                continue
            if type(layer) is FullyConnectedLayer:
                input=input.flatten(2)
                layer.input=input
                input=layer.get_output()
            if type(layer) is DropoutLayer:
                input=layer.get_output(mode='test')
            else:
                input=layer.get_output()
        self.test_pred=pred
    def save_params(self,filename='params.pkl'):
        f=open(filename,"wb")
        cPickle.dump(self.params,f,1);
        f.close()
    def load_params(self,filename='params.pkl'):
        if not os.path.exists(filename):
            print "Check: No previous params to load.\n",
            return
        print "Check: Load previous params.\n",
        f=open(filename,"rb")
        Params=cPickle.load(f);f.close()
        Updates = [(params_i, Params_i) for params_i,Params_i in zip(self.params,Params)]
        updateModel=theano.function(inputs=[],updates=Updates)
        updateModel()