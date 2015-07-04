How to Run Neural Nets on GPUs
==================

Repository to develop the example for an upcoming conference talk at Strange Loop in September 2015.

Mac 15inch Powerbook
	GeForce GT 750M
	CUDA 7.0
	926MHz clock rate
	2508Mhz memory clock rate
	2GB RAM2048 threads / multiprocessor
	384 Cores


DL4J
--------

Change pom.xml file to include jcublas backend

        <dependency>
            <groupId>org.nd4j</groupId>
            <artifactId>nd4j-jcublas-*.0</artifactId>
            <version>${nd4j.version}</version>
        </dependency>

Setup:
...

Theano
--------

Reference at this [link](http://deeplearning.net/software/theano/tutorial/using_gpu.html)

Setup:
	See reference and be sure to install libgpuarray if using gpuarray
	Be sure to homebrew install cmake if you don't alread have it for installation of libgpuarray


Caffe
--------

MNIST example at this [link](http://caffe.berkeleyvision.org/gathered/examples/mnist.html) but the code is a little different. Best to stick to the code in the actuall repo...
   
   
Setup:   
Explanation at this link (http://caffe.berkeleyvision.org/installation.html)
Mac specific at this link (http://caffe.berkeleyvision.org/install_osx.html)

When setting up on Mac, there are a couple key points:

- be sure to *brew update*
- incompatible with boost 1.58.0 so use the following in setup:
	- $ brew uninstall boost boost-python
	- $ cd $(brew --prefix)
	- $ git checkout ab47508 Library/Formula/boost.rb
	- $ git checkout 3141234 Library/Formula/boost-python.rb
	- $ brew install --build-from-source --fresh -vd boost boost-python
	- $ brew pin boost
	- $ brew pin boost-python
- *git clone the Caffe repo*
- despite Makefile, copy the example and remove example from the name (Makefile.config.example)
- if issues with installation *make clean* before trying again
- nvidia-smi doesn't work on Mac but istats (gui platform) gives insights
- use *make pycaffe* if planning to use python
- 	- before *make*, install requirements file with

		$ for req in $(cat requirements.txt); do pip install $req; done 	

	- enusre all numpy files can be found under */usr/lib/python2.7/dist-packages/numpy/core/include/numpy/*
	- setup python path to point to PYTHONPATH=/path/to/caffe/python

Check that cuda is working. 
		
	kextstat | grep -i cuda

If not then restart

	sudo kextload /System/Library/Extensions/CUDA.kext
   

Tricky bits
- Makefile.config needed to be setup to point to homebrew python vs. System. They provide the option for Anaconda but I went with Homebrew and just had to make sure the right files were loaded. Also needed to cleanup path variables and ensure homebrew came first.

 - Make sure to pip install lmdb (light weight backend) and load data. Otherwise you get an error...

I0628 16:27:53.495116 2084766464 layer_factory.hpp:74] Creating layer mnist
I0628 16:27:53.495136 2084766464 net.cpp:90] Creating Layer mnist
I0628 16:27:53.495143 2084766464 net.cpp:368] mnist -> data
I0628 16:27:53.495162 2084766464 net.cpp:368] mnist -> label
I0628 16:27:53.495170 2084766464 net.cpp:120] Setting up mnist
F0628 16:27:53.495229 2084766464 db_lmdb.hpp:13] Check failed: mdb_status == 0 (2 vs. 0) No such file or directory
*** Check failure stack trace: ***
    @        0x10c656076  google::LogMessage::Fail()
    @        0x10c655757  google::LogMessage::SendToLog()
    @        0x10c655cc5  google::LogMessage::Flush()
    @        0x10c659015  google::LogMessageFatal::~LogMessageFatal()
    @        0x10c656363  google::LogMessageFatal::~LogMessageFatal()
    @        0x107941dda  caffe::db::LMDB::Open()
    @        0x1078c2990  caffe::DataLayer<>::DataLayerSetUp()
    @        0x1078b2cfe  caffe::BasePrefetchingDataLayer<>::LayerSetUp()
    @        0x10791d3b4  caffe::Net<>::Init()
    @        0x10791c337  caffe::Net<>::Net()
    @        0x10792fb4e  caffe::Solver<>::InitTrainNet()
    @        0x10792f56c  caffe::Solver<>::Init()
    @        0x10792f3bd  caffe::Solver<>::Solver()
    @        0x107817c62  caffe::SGDSolver<>::SGDSolver()
    @        0x107815b85  caffe::GetSolver<>()
    @        0x107813479  train()
    @        0x10781571f  main
    @     0x7fff9390a5c9  start
   
[General Reference](http://tutorial.caffe.berkeleyvision.org/)   

CPU time: 8min 19 sec? or 24
GPU time: 2min 13sec
Accuracy 0.9911

Need to setup caffee alias in bash_profile to simplify call
bash file not working right now


Most setup references assume python and pip installed. Check documentation for other options especially if setting up on GPUs. 

If you want to add to this repo, send me a PR.



TO DO:
Look up the number of cores on my GPU
Figure out how many threads per block
	- do a thread per neuron if on one layer


