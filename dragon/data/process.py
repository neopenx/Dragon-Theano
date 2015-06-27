import theano
import numpy
import theano.tensor as T
import cPickle
import os
import socket
dim=dim=32*32*3
def build_data(filename="data.pkl"):
    train_file=open("train.bin","rb")
    vaild_file=open("vaild.bin","rb")
    data_file=open(filename,"wb");mean_file=open("mean.pkl","wb")
    train_data=train_file.read();vaild_data=vaild_file.read()
    train_data=numpy.fromstring(train_data,numpy.uint8);vaild_data=numpy.fromstring(vaild_data,numpy.uint8);
    train_num=train_data.shape[0]/(dim+1);vaild_num=vaild_data.shape[0]/(dim+1)
    train_data=train_data.tolist();vaild_data=vaild_data.tolist()
    print train_num,vaild_num
    trainX=[];trainY=[]
    vaildX=[];vaildY=[]
    start=0
    for i in xrange(train_num):
        trainY.append(train_data[start])
        trainX.extend(train_data[start+1:start+dim+1])
        start=start+dim+1
    start=0
    for i in xrange(vaild_num):
        vaildY.append(vaild_data[start])
        vaildX.extend(vaild_data[start+1:start+dim+1])
        start=start+dim+1
    #print trainY[0:20]
    trainX=numpy.asarray(trainX,dtype=numpy.uint8);vaildX=numpy.asarray(vaildX,dtype=numpy.uint8)
    trainY=numpy.asarray(trainY,dtype=numpy.uint8);vaildY=numpy.asarray(vaildY,dtype=numpy.uint8)
    trainX=trainX.reshape((train_num,dim)); vaildX=vaildX.reshape((vaild_num,dim));
    #calc mean.pkl
    xMean=numpy.mean(trainX,axis=0)
    xMean=xMean.reshape((1,xMean.shape[0]))
    #to pkl
    cPickle.dump(xMean,mean_file,1) #Mean file
    dataSet=((trainX,trainY),(vaildX,vaildY))
    cPickle.dump(dataSet,data_file,1)
def loadScaleData(filename):
    print "loading data..."
    file=open(filename,"rb")
    trainSet,vaildSet=cPickle.load(file)
    file.close();
    def sharedDataSet(data,borrow=True): 
        dataX,dataY=data
        dataX=numpy.asarray(dataX,theano.config.floatX)
        sharedX=theano.shared(numpy.asarray(dataX/255.0,dtype=theano.config.floatX),borrow=True)
        sharedY=theano.shared(numpy.asarray(dataY,dtype=theano.config.floatX),borrow=True)
        return sharedX,T.cast(sharedY,'int32')
    trainSet_X,trainSet_Y=sharedDataSet(trainSet)
    vaildSet_X,vaildSet_Y=sharedDataSet(vaildSet)
    rval=[(trainSet_X,trainSet_Y),(vaildSet_X,vaildSet_Y)]
    return rval
def loadData():
    print "loading data..."
    file=open("data.pkl","rb")
    mfile=open("mean.pkl","rb")
    trainSet,vaildSet=cPickle.load(file)
    xMean=cPickle.load(mfile)
    file.close();mfile.close()
    def sharedDataSet(data,borrow=True): 
        dataX,dataY=data
        dataX=numpy.asarray(dataX,theano.config.floatX)
        #print dataY[0:50]
        sharedX=theano.shared(numpy.asarray(dataX-xMean,dtype=theano.config.floatX),borrow=True)
        sharedY=theano.shared(numpy.asarray(dataY,dtype=theano.config.floatX),borrow=True)
        return sharedX,T.cast(sharedY,'int32')
    trainSet_X,trainSet_Y=sharedDataSet(trainSet)
    vaildSet_X,vaildSet_Y=sharedDataSet(vaildSet)
    rval=[(trainSet_X,trainSet_Y),(vaildSet_X,vaildSet_Y)]
    return rval
def loadTestData(filename):
    print "loading data..."
    file=open("bin/"+filename,"rb")
    mfile=open("mean.pkl","rb")
    xMean=cPickle.load(mfile)
    xMean=numpy.asarray(xMean)
    aList=file.read()
    a=numpy.fromstring(aList,numpy.uint8)
    aSize=a.shape[0]/dim
    a=a.reshape((aSize,dim))
    a=a-xMean
    a=numpy.asarray(a,dtype=theano.config.floatX)
    a=theano.shared(a,borrow=True)
    return aSize,a
def readParams():
    if not os.path.exists("params.pkl"):return False
    f=open("params.pkl","rb")
    params=cPickle.load(f);f.close()
    return params
def writeParams(params):
    f=open("params.pkl","wb")
    cPickle.dump(params,f,1);f.close()
def transfer(data,filename):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect(('localhost', 2334))
    text=filename
    for i in data:text=text+"$"+str(i)
    print text
    sock.send(text)
    sock.close()
#build_data()