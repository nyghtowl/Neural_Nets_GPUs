How to Run Neural Nets on GPUs
==================

Repository to develop the example for an upcoming conference talk at Strange Loop in September 2015.


## Performance on Mnist

| **Package** | **Real CPU** | **Real GPU** | **Accuracy** |
|-------------|--------------|--------------|--------------|
| DL4J        | 0m45.84s     | 1m13.08      | 0.42         | 
| Theano      | 0m13.85s     | 0m48.02s     | 0.93         | - printing is slowing it down
| Caffe       | 7m50.48s     | 2m13.64s     | 0.99         |


Using bash function time which reports real, user and sys results. Real is elapsed time from start to finish of hte call. User is hte cpu time spent in user-mode code and includes outside the kernel. Sys is the amount of cpu time sepnt in the kernel.

CIFAR-10

Image training dataset collected by Alex Krizhevsky, Vinod Nair, and Geoffrey Hinton. Find more information at this [link](http://www.cs.toronto.edu/~kriz/cifar.html). Example of CIFAR real-time analysis show at this [link](https://cs.stanford.edu/people/karpathy/convnetjs/demo/cifar10.html)

DL4J
--------
Open source neural net package built for scale and speed and written in Java. Usess ND4J for core calculations.

Setup:
- Explanation at this [link](http://nd4j.org/getstarted.html)

How to Apply GPU:

Change pom.xml file to include jcublas backend

        <dependency>
            <groupId>org.nd4j</groupId>
            <artifactId>nd4j-jcublas-*.0</artifactId>
            <version>${nd4j.version}</version>
        </dependency>


Run:
    
    $ mvn clean install

    $ java -cp dl4j_examples/target/org.deeplearning-1.0-SNAPSHOT.jar org.deeplearning4j.gpu.examples.MnistExample


Theano
--------
Open source neural net package in Python and based on NumPy for calculations.

Setup:
- Explanation at this [link](http://deeplearning.net/software/theano/install.html)
- Install libgpuarray if using gpuarray
- Homebrew install cmake if you don't alread have it for installation of libgpuarray

How to Apply GPU:

    Reference at this [link](http://deeplearning.net/software/theano/tutorial/using_gpu.html)

Run:

    $ THEANO_FLAGS=mode=FAST_RUN,device=cpu,floatX=float32 python theano_examples/theano_example.py 
    $ THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python theano_examples/theano_example.py 


Caffe
--------
Open source neural net package built for scale and speed and written in C.

- MNIST example at this [link](http://caffe.berkeleyvision.org/gathered/examples/mnist.html) but the code is a little different. Best to stick to the code in the actuall repo...    
- CIFAR-10 example at this [link](http://caffe.berkeleyvision.org/gathered/examples/cifar10.html) 

   
Setup:   
- Explanation at this [link](http://caffe.berkeleyvision.org/installation.html)
- Mac specific at this [link](http://caffe.berkeleyvision.org/install_osx.html)

- When setting up on Mac, there are a couple key points:
	- Update brew:

		$ brew update

	- Setup correct boost version because incompatible with boost 1.58.0. Use the following as of June 2015:
	
		$ brew uninstall boost boost-python
		$ cd $(brew --prefix)
		$ git checkout ab47508 Library/Formula/boost.rb
		$ git checkout 3141234 Library/Formula/boost-python.rb
		$ brew install --build-from-source --fresh -vd boost boost-python
		$ brew pin boost
		$ brew pin boost-python

	- Setup CUDA (see below for tips)
	- Setup caffe:
		- git clone the caffe repo
		- despite existing Makefile, copy the example (Makefile.config.example) and remove example from the name 
			- Add path to homebrew python vs System. Anaconda is an alternative approach. Example on my computer but will be a little different on yours: */usr/local/Cellar/python/2.7.10/Frameworks/Python.framework/Versions/2.7/include/python2.7*
			- find and setup libpythonX.X.so or dylib path. Example on my computer but may be different on yours: */usr/local/Cellar/python/2.7.10/Frameworks/Python.framework/Versions/2.7/lib/*
			- cleanup path variables and ensure homebrew library comes first for python reference.
				- For example, I setup a syslink to Homebrew's python which is at: /usr/local/Cellar/python/2.7.10/bin/python on my computer to /usr/local/bin/python and then I put /usr/local/bin at the beginning of my python path
		- if issues with installation be sure to use the following before trying again:
		
			$ make clean

 		- before *make*, install requirements file with:

			$ for req in $(cat requirements.txt); do pip install $req; done 	

		- load data with the commands from the setup site
		- setup caffee alias in bash_profile to simplify exec call

	- Setup pycaffe:

 		- install lmdb (light weight backend) for python bindings; otherwise an error is thrown
 		 
 			pip install lmdb

		- use *make pycaffe* if planning to use python
		- setup python path to point to PYTHONPATH=/path/to/caffe/python
		

How to Apply GPU:

    Change in prototxt solver file associated to "solver_mode:"
    Checkout lenet_solver for an example

Run:

	// inside mnist_examples folder
	$ cd caffe_examples/mnist_example/ && bash train_lenet.sh
	OR
    $ cd caffe_examples/mnist_example/ && caffe train --solver=lenet_solver.prototxt 

	// inside cifar_examples folder
    $ cd caffe_examples/mnist_example/ && caffe train --solver=cifar10_quick_solver.prototxt 

To change between CPU & GPU, change the configuration in lenet_solver.protxt

Cuda Setup
--------
Software driver to enable running neural nets on Nvidia GPUs. An alternative is OpenCL which enables running NNs on Nvidia and AMD. CUDNN is Nvidia library that adds additional optimization to run NNs on Nvidia chips.

Add path in .bash_profile or .bashrc

    export CUDA_PATH="/usr/local/cuda"
    export PATH=$CUDA_PATH/lib:$PATH

Check that cuda is working. 
        
    kextstat | grep -i cuda

If not then restart

    sudo kextload /System/Library/Extensions/CUDA.kext

Check on Performance:
	- nvidia-smi doesn't work on Mac but istats (gui platform) gives insights 
	- use ./deviceQuery and setup deviceQuery alias in bash_profile 


PyCuda 
--------
Python wrapper for CUDA driver - allows meta-programing inside python code. It compiles source code and uploads to card as well as handles moving data to and from card as needed and allocating space. It infers cleanup automatically.

Use in python script:
	
	import pycuda

Installation:
	http://wiki.tiker.net/PyCuda/Installation

Reference materials at this [link](http://documen.tician.de/pycuda/)
Tutorial at this [link](http://documen.tician.de/pycuda/tutorial.html#transferring-data)

Run Remotely
--------

For all examples, a remote server will need to be setup and configured to run the needed software. Be sure to login to the remote server to follow these steps.

- DL4J: after configuration
	- git clone the example you setup on the remote server
	- compile the file on the server by running this command inside the repo root $ mvn clean install
	- run the file $ java -cp <jar file path> <class path>

- Caffe
	- scp or git clone the code you wrote onto the remote server 
	- run the file $ caffe train --solver=<solver prototxt file>

- Theano
	- scp or git clone the code you wrote onto the remote server 
	- run the file $ THEANO_FLAGS=mode=FAST_RUN,device=cpu,floatX=float32 python <file name>


System Configuration
--------

Currently running the code locally on the following setup:

    Mac 15inch Powerbook
    GeForce GT 750M
    CUDA 7.0
    926MHz clock rate
    2508Mhz memory clock rate
    2GB RAM
    2048 threads / multiprocessor
    384 Cores

For one layer can have up to 2K neurons / block running at the that same time


