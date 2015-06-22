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
	- enusre all numpy files can be found under */usr/lib/python2.7/dist-packages/numpy/core/include/numpy/*


   
[General Reference](http://tutorial.caffe.berkeleyvision.org/)   


--------

Most setup references assume python and pip installed. Check documentation for other options especially if setting up on GPUs. 

If you want to add to this repo, send me a PR.
