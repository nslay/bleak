# Disclaimer
This project serves as both an intellectual exercise to understand the very low level details of deep learning as well as a sandbox to test crazy ideas that might be harder to test in more mainstream toolkits! You should probably look at more mainstream toolkits like [tensorflow](https://github.com/tensorflow/tensorflow) or [pytorch](https://github.com/pytorch/pytorch).

# Introduction
This project started as a framework to implement Random Hinge Forest which is detailed in this arXiv draft

https://arxiv.org/abs/1802.03882

For benchmark experiments in this repository, Random Hinge Forest serves as both a standalone learning machine as well as a non-linearity for consecutive layers. So you will not find a conventional activation function in this *neural network* toolkit (at least as of this revision, but it wouldn't be hard to add!).

# Supported Environments
Bleak has been developed and/or tested in the following environments
- Windows 10, Visual Studio 2017, OpenBLAS 0.3.6, CUDA 10.2, cuDNN 7.6.5, LMDB 0.9.70, ITK 4.13
  - Uses Windows Subsystem for Linux for experiments.
  - GeForce GTX 980
- FreeBSD 12.1-STABLE, clang-10.0.0, OpenBLAS 0.3.9, LMDB 0.9.70, ITK 4.13
  - No GPU support on this Unix-like operating system. I don't have a spare computer to test on Linux!
  
# Compiling from Source
TODO
  
# Graphs and Vertices
In bleak, neural network computation is implemented as a directed graph. Vertices implement the forward/backward operations and have names, properties, and named inputs and outputs. This enables searching for vertices by name, assigning values to named properties as well as querying inputs and outputs by name. Edges serve to store tensor inputs and outputs and their gradients. Vertices uniquely own Edges for their outputs while being assigned Edges for their inputs. Graphs in bleak can be constructed/modified in C++ or can be read from a .sad file.

# Basic Graph Syntax (.sad)
A .sad file follows this general format. Sections denoted with [] are optional.

1. [Variable Declarations]
2. [Subgraph Declarations]
3. Vertex Declarations
4. [Connection Declarations]

Whitespace is ignored and all declarations are terminated with a semicolon (;) (except for includes). A file can be included at any time with an 'include' statement. For example
```
include "Config.sad"
```
This included file is treated as if its content were copied and pasted in place of the include. The included file by itself need not be a valid graph.

Comments are preceded by the octothorpe symbol (#). For example
```
# This is a comment.
```
They may occur anywhere outside of a string value.

## Variable Declarations
Variables are declared as a key value pair. For example
```
batchSize = 16;
learningRateMultiplier=1.0;
imageList = "alcoholicTrainList.txt";
```
And they may be overwritten by subsequent declarations. For example
```
include "Config.sad"
batchSize=32; # Override config file
```

Variables in .sad files support a small collection of basic types:
- integer
- float
- boolean (true/false)
- string ("value")
- integer vector ([8, 3, 256, 256])
- float vector ([0.5, 1.0])

Many of these are implicitly convertible to each other. Any type is convertible to a string and any string is (possibly) convertible to any type. Other implicit conversions are provided below.
- integer -> float
- integer -> boolean
- integer -> integer vector
- integer -> float vector
- float -> boolean
- float -> float vector
- boolean -> integer
- boolean -> float
- boolean -> integer vector
- boolean -> float vector
- integer vector -> integer (only if the vector has 1 component)
- integer vector -> float (only if the vector has 1 component)
- float vector -> float (only if the vector has 1 component)

### Expressions with Variables
TODO

## Subgraph Declarations
TODO

## Vertex Declarations
TODO

## Connection Declarations
TODO 

# Subgraphs
TODO 

# Bleak C++ API
TODO 

# Implementing your own Vertex in C++
TODO

# Creating a new Module
TODO 
 
