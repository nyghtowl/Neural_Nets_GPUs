from theano import function, config, shared, sandbox, tensor
import theano.tensor as T
import numpy
import time

'''
Sample code to test Theano on GPUs

reference: http://deeplearning.net/software/theano/tutorial/using_gpu.html

$ THEANO_FLAGS=mode=FAST_RUN,device=cpu,floatX=float32 python test_gpu.py 
- real = 0m5.17sec
$ THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python test_gpu.py
- real = 0m1.57sec
$ THEANO_FLAGS=mode=FAST_RUN,device=cuda0,floatX=float32 python test_gpu.py
- real = 0m1.57sec (with gpuarray and it says its running on cpu)
'''

vlen = 10 * 30 * 768  # 10 x #cores x # threads per core
iters = 1000

rng = numpy.random.RandomState(22)
x = shared(numpy.asarray(rng.rand(vlen), config.floatX))
# f = function([], T.exp(x))
f = function([], sandbox.gpuarray.basic_ops.gpu_from_host(tensor.exp(x)))
print f.maker.fgraph.toposort()

t0 = time.time()
for i in xrange(iters):
    r = f()
t1 = time.time()

print 'Looping %d times took' % iters, t1 - t0, 'seconds'
# print 'Result is', r
print 'Result is', numpy.asarray(r)

if numpy.any([isinstance(x.op, T.Elemwise) for x in f.maker.fgraph.toposort()]):
    print 'Used the cpu'
else:
    print 'Used the gpu'