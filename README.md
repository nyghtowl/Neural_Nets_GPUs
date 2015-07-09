How to Run Neural Nets on GPUs
==================

Repository to develop the example for an upcoming conference talk at Strange Loop in September 2015.


## Performance

| **Package** | **Real CPU** | **Real GPU** | **Accuracy** |
|-------------|--------------|--------------|--------------|
| DL4J        | 0m45.84s     | 1m13.08      | 0.42         | 
| Theano      | 0m13.85s     | 0m48.02s     | 0.93         | - printing is slowing it down
| Caffe       | 7m50.48s     | 2m13.64s     | 0.99         |


Using bash function time which reports real, user and sys results. Real is elapsed time from start to finish of hte call. User is hte cpu time spent in user-mode code and includes outside the kernel. Sys is the amount of cpu time sepnt in the kernel.


DL4J
--------


Setup:
Checkout this [link](http://nd4j.org/getstarted.html) to setup DL4J.

How to Apply GPU:

Change pom.xml file to include jcublas backend

        <dependency>
            <groupId>org.nd4j</groupId>
            <artifactId>nd4j-jcublas-*.0</artifactId>
            <version>${nd4j.version}</version>
        </dependency>


Run:
    
    $ mvn clean install

    $ java -cp ./target/org.deeplearning-1.0-SNAPSHOT.jar org.deeplearning4j.gpu.examples.MnistExample


Theano
--------

Setup:

	See reference and be sure to install libgpuarray if using gpuarray
	Be sure to homebrew install cmake if you don't alread have it for installation of libgpuarray

How to Apply GPU:

    Reference at this [link](http://deeplearning.net/software/theano/tutorial/using_gpu.html)

Run:

    $ THEANO_FLAGS=mode=FAST_RUN,device=cpu,floatX=float32 python theano_example.py 
    $ THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python theano_example.py 


Caffe
--------

MNIST example at this [link](http://caffe.berkeleyvision.org/gathered/examples/mnist.html) but the code is a little different. Best to stick to the code in the actuall repo...    
   
   
Setup:   
- Explanation at this [link](http://caffe.berkeleyvision.org/installation.html)
- Mac specific at this [link](http://caffe.berkeleyvision.org/install_osx.html)

- When setting up on Mac, there are a couple key points:
	- be sure to *brew update*
	- note: incompatible with boost 1.58.0 so use the following in setup for pre requirements:
	
		$ brew uninstall boost boost-python
		$ cd $(brew --prefix)
		$ git checkout ab47508 Library/Formula/boost.rb
		$ git checkout 3141234 Library/Formula/boost-python.rb
		$ brew install --build-from-source --fresh -vd boost boost-python
		$ brew pin boost
		$ brew pin boost-python

	- Setup CUDA (see below)
	- Set up caffe:
		- *git clone the Caffe repo*
		- despite Makefile, copy the example and remove example from the name (Makefile.config.example)
			- - Makefile.config needed to be setup to point to homebrew python vs. System. Annaconda is an alternative.
			- leanup path variables and ensure homebrew came first.
			- enusre all numpy files can be found under the file path you specify
		- if issues with installation *make clean* before trying again
		- use *make pycaffe* if planning to use python
		- setup python path to point to PYTHONPATH=/path/to/caffe/python
 		- before *make*, install requirements file with:

			$ for req in $(cat requirements.txt); do pip install $req; done 	

 		- Make sure to pip install lmdb (light weight backend) and load data. Otherwise you get an error...
		- Need to setup caffee alias in bash_profile to simplify call

How to Apply GPU:

    Change in prototxt solver file associated to "solver_mode:"
    Checkout lenet_solver for an example

Run:

    $ caffe train --solver=lenet_solver.prototxt

To change between CPU & GPU, change the configuration in lenet_solver.protxt

Cuda Setup
--------
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


PyCuda Setup
--------


Run Remotely
--------


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
