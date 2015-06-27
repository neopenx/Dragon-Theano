import theano 
import theano.tensor as T
import numpy
from theano.tensor.shared_randomstreams import RandomStreams
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv
def Gauss(std,size,rng):
    return numpy.asarray(rng.normal(0,std,size),dtype=theano.config.floatX)
class DataLayer(object):
    def __init__(self,batch_size,size):
        self.batch_size=batch_size
        self.size=size
        self.input=None
        self.params=None
    def get_output(self):
        output=self.input
        if type(self.size) is tuple: #Mode: 2D
            output=output.reshape((self.batch_size,self.size[2],self.size[1],self.size[0]))
        else: #Mode: 1D
            output=output.reshape((self.batch_size,self.size))
        return output
    
class SoftmaxLayer(object):
    def __init__(self,n_in,n_out,weight_init='None',gauss_std=None):
        rng=numpy.random.RandomState(23455)
        if weight_init == 'Gaussian':
            assert gauss_std!=None,"Gaussian Distribution must have a std(Standard Deviation)"
            self.W=theano.shared(Gauss(gauss_std,(n_in,n_out),rng),borrow=True)
            print "Softmax layer created  input:%d  output:%d  Gauss_Std:%f\n"%(n_in,n_out,gauss_std),
        else:
            self.W=theano.shared(value=numpy.zeros((n_in,n_out),dtype=theano.config.floatX),
                             name='b',borrow=True)
            print "Softmax layer created  input:%d  output:%d\n"%(n_in,n_out),
        self.b=theano.shared(value=numpy.zeros((n_out,),dtype=theano.config.floatX),name='b',borrow=True)
        self.input=T.matrix('x')
        self.params=[self.W,self.b]
        self.theano_rng=RandomStreams(rng.randint(2**30))
    def NLL(self,y):
        p_y_x=T.nnet.softmax(T.dot(self.input,self.W)+self.b)
        return -T.mean(T.log(p_y_x)[T.arange(y.shape[0]),y])
    def error(self,y):
        p_y_x=T.nnet.softmax(T.dot(self.input,self.W)+self.b)
        y_pred=T.argmax(p_y_x,axis=1)
        return T.mean(T.neq(y_pred,y))
    def pred(self):
        p_y_x=T.nnet.softmax(T.dot(self.input,self.W)+self.b)
        y_pred=T.argmax(p_y_x,axis=1)
        return y_pred

class FullyConnectedLayer(object):
    def __init__(self,n_in,n_out,activation='none',weight_init='Guassian',gauss_std=None):
        rng=numpy.random.RandomState(23455)
        if weight_init == 'Gaussian':
            assert gauss_std!=None,"Gaussian Distribution must have a std(Standard Deviation)"
            self.W=theano.shared(Gauss(gauss_std,(n_in,n_out),rng),borrow=True)
            print "FC layer created  input:%d  output:%d  activation:%s  Gauss_Std:%f\n"%(n_in,n_out,activation,gauss_std),
        elif weight_init == 'Xavier':
            bound=numpy.sqrt(6. / (n_in + n_out))
            if activation == 'logistic':bound=bound*4
            value = numpy.asarray(rng.uniform(low=-bound,high=bound,size=(n_in, n_out)),dtype=theano.config.floatX)
            self.W=theano.shared(value,borrow=True)
            print "FC layer created  input:%d  output:%d  activation:%s\n"%(n_in,n_out,weight_init),
        self.b=theano.shared(numpy.zeros((n_out,),dtype=theano.config.floatX),borrow=True)
        self.params=[self.W,self.b]
        self.input=T.matrix('x')
        self.activation=activation
    def get_output(self):
        output=T.dot(self.input,self.W)+self.b
        if self.activation == 'none':return output
        if self.activation == 'tanh':return T.tanh(output)
        if self.activation == 'logistic':return T.nnet.sigmoid(output)
        if self.activation == 'relu':return T.maximum(0,output)
        
class ConvolutionLayer(object):
    def __init__(self,image_shape,filter_shape,activation,weight_init='Guassian',gauss_std=None):
        rng=numpy.random.RandomState(23455)
        self.image_shape=image_shape
        self.filter_shape=filter_shape
        if weight_init == 'Gaussian':
            assert gauss_std!=None,"Gaussian Distribution must have a std(Standard Deviation)"
            self.W=theano.shared(Gauss(gauss_std,filter_shape,rng),borrow=True)
            print "ConvolutionLayer layer created  input:%d  output:%d  activation:%s Gauss_Std:%f\n"%(filter_shape[1],filter_shape[0],activation,gauss_std),
        elif weight_init == 'Xavier':
            n_in = numpy.prod(filter_shape[1:])
            n_out=(filter_shape[0] * numpy.prod(filter_shape[2:]) /numpy.prod(pool_size))
            bound=numpy.sqrt(6. / (n_in + n_out))
            if activation == 'logistic':bound=bound*4
            value = numpy.asarray(rng.uniform(low=-bound,high=bound,size=(n_in, n_out)),dtype=theano.config.floatX)
            self.W=theano.shared(value,borrow=True)
        self.b=theano.shared(numpy.zeros(filter_shape[0],dtype=theano.config.floatX),borrow=True)
        self.params=[self.W,self.b]
        self.activation=activation
    def get_output(self,activation='none'):
        output=conv.conv2d(input=self.input,filters=self.W,filter_shape=self.filter_shape,image_shape=self.image_shape)+self.b.dimshuffle('x',0,'x','x')
        if self.activation == 'none':return output
        if self.activation == 'tanh':return T.tanh(output)
        if self.activation == 'logistic':return T.nnet.sigmoid(output)
        if self.activation == 'relu':return T.maximum(0,output)
        
class PoolingLayer(object):
    def __init__(self,pool_size=(2,2),type='Max'):
        self.input=T.matrix('x')
        self.pool_size=pool_size
        self.params=None
    def get_output(self):
        output=downsample.max_pool_2d(self.input,self.pool_size,True)
        return output
    
class AutoEncodeLayer(object):
    def __init__(self,n_in,n_out,hid_activation='none',vis_activation='none',cost='squre',weight_init='Guassian',gauss_std=None,level=0.2):
        rng=numpy.random.RandomState(23455)
        if weight_init == 'Gaussian':
            assert gauss_std!=None,"Gaussian Distribution must have a std(Standard Deviation)"
            self.W=theano.shared(Gauss(gauss_std,(n_in,n_out),rng),borrow=True)
            self.WT=self.W.T
            print "AutoEncoder layer for FC created  input:%d  output:%d  activation:%s  Gauss_Std:%f\n"%(n_in,n_out,hid_activation,gauss_std),
        elif weight_init == 'Xavier':
            bound=numpy.sqrt(6. / (n_in + n_out))
            if hid_activation == 'logistic':bound=bound*4
            value = numpy.asarray(rng.uniform(low=-bound,high=bound,size=(n_in, n_out)),dtype=theano.config.floatX)
            self.W=theano.shared(value,borrow=True)
            self.WT=self.W.T
            print "AutoEncoder layer for FC created  input:%d  output:%d  activation:%s\n"%(n_in,n_out,weight_init),
        self.bhid=theano.shared(numpy.zeros((n_out,),dtype=theano.config.floatX),borrow=True)
        self.bvis=theano.shared(numpy.zeros((n_in,),dtype=theano.config.floatX),borrow=True)
        self.params=[self.W,self.bhid]
        self.pre_params=[self.W,self.bhid,self.bvis]
        self.input=T.matrix('x')
        self.hid_activation=hid_activation
        self.vis_activation=vis_activation
        self.level=level
        self.theano_rng=RandomStreams(rng.randint(2**30))
        self.cost=cost
    def get_output(self):
        output=T.dot(self.input,self.W)+self.bhid
        if self.hid_activation == 'none':return output
        if self.hid_activation == 'tanh':return T.tanh(output)
        if self.hid_activation == 'logistic':return T.nnet.sigmoid(output)
        if self.hid_activation == 'relu':return T.maximum(0,output)
    def corrupte(self,level):
        if(level>0.0):return self.theano_rng.binomial(size=self.input.shape,n=1,p=1-self.level,dtype=theano.config.floatX)*self.input
        else:return self.input
    def reconstruct(self):
        corrupted_x=self.corrupte(self.level)
        hidden=T.dot(corrupted_x,self.W)+self.bhid
        ####  input --> hidden ####
        if self.hid_activation=='relu':y=T.maximum(0,hidden)
        elif self.hid_activation=='logistic':y=T.nnet.sigmoid(hidden)
        elif self.hid_activation=='tanh':y=T.tanh(hidden)
        elif self.hid_activation=='softplus':y=T.nnet.softplus(hidden)
        ####  hideen --> visble ####
        visble=T.dot(y,self.WT)+self.bvis
        if self.vis_activation=='softplus':z=T.nnet.softplus(visble)
        elif self.vis_activation=='logistic':z=T.nnet.sigmoid(visble)
        elif self.vis_activation=='tanh':z=T.tanh(visble)
        elif self.vis_activation=='relu':z=T.maximum(0,visble)
        #### reconstruction ####
        
        if self.cost=='squre':L=T.sum((self.input-z)**2,axis=1)    #quadratic cost
        else:L=-T.sum(self.input * T.log(z) + (1 - self.input) * T.log(1 - z), axis=1) #cross entropy
        cost=T.mean(L)
        return cost
        
class DropoutLayer(object):
    def __init__(self,prob=0.5):
        rng=numpy.random.RandomState(23455)
        self.theano_rng=RandomStreams(rng.randint(2**30))
        self.params=None
        self.prob=prob
    def get_output(self,mode='train'):
        if mode == 'train':
            self.output=self.theano_rng.binomial(size=self.input.shape,n=1,p=1-self.prob,dtype=theano.config.floatX)*self.input
        else:self.output=self.input*(1-self.prob)
        return self.output
    