How to Run Neural Nets on GPUs
==================

Repository to develop the example for an upcoming conference talk at Strange Loop in September 2015.


DL4J
--------

.....


Theano
--------

...


Caffe
--------

MNIST example at this [link](?)    
   
   
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
   
[General Reference](http://tutorial.caffe.berkeleyvision.org/)   

- The tricky bits came down to the fact that the Makefile.config needed to be setup to point to homebrew python vs. System. They provide the option for Anaconda but I went with Homebrew and just had to make sure the right files were loaded. Also needed to cleanup path variables and ensure homebrew came first.
--------

Most setup references assume python and pip installed. Check documentation for other options especially if setting up on GPUs. 

If you want to add to this repo, send me a PR.
