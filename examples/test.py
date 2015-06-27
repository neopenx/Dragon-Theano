from layer.core import *
from algorithm.SGD import Mini_Batch
from data.process import loadData, loadTestData,transfer
from layer.model import Model
from SocketServer import ThreadingTCPServer, StreamRequestHandler
import traceback
class MyStreamRequestHandlerr(StreamRequestHandler):
    def handle(self):
        while True:
            try:
                data = self.rfile.readline().strip()
                print "receive from (%r):%r" % (self.client_address, data)
                test(data)
                self.wfile.write(data.upper())
            except:
                traceback.print_exc()
                break
def test(filename):
    size,dataSet=loadTestData(filename)
    test_fn=theano.function(inputs=[index],outputs=test_pred,givens={cifar.x:dataSet[index:index+1]})
    print "testing model....\n"
    ans=[]
    for i in xrange(size):
        ans+=[test_fn(i)[0]]
    transfer(ans,filename)
if __name__ == '__main__':
    cifar=Model(batch_size=1,lr=0.01,dataSet=None)
    neure=[32,32,64,64]
    batch_size=1
    x=T.matrix('x')
    index=T.lscalar()
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
    cifar.build_test_fn()
    cifar.load_params('cnn_params.pkl')
    test_pred=cifar.test_pred
    #### Muti-Thread Sevrer ####
    host = "localhost"
    port = 2335    
    addr = (host, port)
    server = ThreadingTCPServer(addr, MyStreamRequestHandlerr)
    print("now listening")
    server.serve_forever()
    
    