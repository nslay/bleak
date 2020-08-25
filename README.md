# Disclaimer
This project serves as both an intellectual exercise to understand the very low level details of deep learning as well as a sandbox to test crazy ideas that might be harder to test in more mainstream toolkits! You should probably look at more mainstream toolkits like [tensorflow](https://github.com/tensorflow/tensorflow) or [pytorch](https://github.com/pytorch/pytorch).

# Introduction
This project started as a framework to implement Random Hinge Forest which is detailed in this arXiv draft

https://arxiv.org/abs/1802.03882

For benchmark experiments in this repository, Random Hinge Forest serves as both a standalone learning machine as well as a non-linearity for consecutive layers. So you will not find a conventional activation function in this *neural network* toolkit (at least as of this revision, but it wouldn't be hard to add!).

# Tested Environments
Bleak has been developed and/or tested in the following environments
- Windows 10, Visual Studio 2017, OpenBLAS 0.3.6, CUDA 10.2, cuDNN 7.6.5, LMDB 0.9.70, ITK 4.13
  - Uses Windows Subsystem for Linux for experiments.
  - GeForce GTX 980
- FreeBSD 12.1-STABLE, clang-10.0.0, OpenBLAS 0.3.9, LMDB 0.9.70, ITK 4.13
  - No GPU support on this Unix-like operating system. I don't have a spare computer to test on Linux!
  
# Compiling from Source
To build bleak, you will need the following dependencies
- A C++14 compiler (GCC, Clang or Visual Studio 2017 or later)
- [cmake](https://cmake.org/) 3.10 or later (ccmake recommended on Unix-like systems)

First clone this repository and its submodules
```shell
git clone https://github.com/nslay/bleak
cd bleak
git submodule init
git submodule update
```
Create a separate empty folder (call it build) and
## Unix-like Systems
```shell
mkdir build
cd build
ccmake /path/to/bleak
```

Press 'c' to configure, select desired build options and modules (press 'c' again for any changes) and then finally press 'g' to generate the Makefiles to build bleak.

**NOTE**: Bleak should build and run on Unix-like systems (I occassionally compile and run it on FreeBSD). That said, the experiment shell scripts were written for Windows Subsystem for Linux. So some script modification is likely needed to run experiments on actual Unix-like systems.

## Windows
Run `cmake-gui` and set the source code and build folders. For example `C:/Work/Source/bleak` and `C:/Work/Build/bleak` respectively.

Press "Configure", select the desired build options and modules (press "Configure" for any changes) and then finally press "Generate". You can also press "Open Project" to launch Visual Studio automatically.

**NOTE**: Make sure to select the "Release" build mode in Visual Studio.

## Some General Options
- bleakUseOpenMP -- Try to enable OpenMP support in the compiler (if available).
- bleakUseCUDA -- Try to enable CUDA support (if available).
- bleakBLASType -- "slowblas" (default, built-in to bleak and very slow!) or "openblas" ([OpenBLAS](https://www.openblas.net/)).

## Modules
- bleakCommon -- A required module that is essentially the glue of all of bleak (Graph, Vertex, Array, BLAS wrappers, parsers, databases, etc...) and some optimizers (SGD, AdaGrad, Adam) and some basic Vertices (InnerProduct, BatchNormalization, SoftmaxLoss, etc...).
- bleakImage -- Gemm-based convolution and pooling.
- bleakTrees -- Random hinge forest, ferns, covnolutional Hinge Trees and Ferns, Feature Selection and Annealing.
- bleakITK -- [ITK](https://itk.org/) 1D/2D/3D image loader Vertex (supports PNG/JPEG, DICOM, MetaIO, Nifti, etc...). Requires ITK 4+.
- bleakCudnn -- cuDNN-based convolution and pooling. Requires cuDNN.
  
# Graphs and Vertices
In bleak, neural network computation is implemented as a directed graph. Vertices implement the forward/backward operations and have names, properties, and named inputs and outputs. This enables searching for vertices by name, assigning values to named properties as well as querying inputs and outputs by name. Edges serve to store tensor inputs and outputs and their gradients. Vertices uniquely own Edges for their outputs while being assigned Edges for their inputs. Graphs in bleak can be constructed/modified in C++ or can be read from a .sad file.

# Basic Graph Syntax
A .sad file follows this general format. Sections denoted with [] are optional.

1. [Variable Declarations]
2. [Subgraph Declarations]
3. Vertex Declarations
4. [Connection Declarations]

Whitespace is ignored and all declarations are terminated with a semicolon (;) (except for includes). A file can be included at any time with an "include" statement. For example
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

### Expressions with Variables
Variables can be referenced in a synonymous fashion as shell variables (with '$') and may be used in simple mathematical expressions if they are float or integer types. The mathematical operators available include +, -, \*, /, % (modulo), ^ (exponentiation) and \*\* (exponentiation). Resulting types follow the behavior of the C/C++ programming languages. For example, 1/2 results in 0 while 1.0/2 results in 0.5. The addition operator (+) may also be used to concatenate strings. Here are some examples
```
# This expression results in an integer (features3Width is an integer)
pool1Width = ($features3Width - 2)/2 + 1; 

# This concatenates two strings
imageList=$dataRoot + "/SMNI_CMI_TRAIN/alcoholicTrainList.txt"; 

# Variables and expressions can even be used inside of vectors
size = [ $numTrees, 2^$treeDepth - 1 ]; 
```
There are currently no built-in functions like min/max/exp or any syntax to reference vector components.

## Subgraph Declarations
Subgraphs are declared immediately after variables (if any). They recursively define graphs which follow the structure mentioned [above](#basic-graph-syntax) with some additional mechanisms to facilitate communicating properties and setting up connections. This topic will be covered in detail in section [Subgraphs](#subgraphs) after vertex declarations and connection declarations are covered.

## Vertex Declarations
After variables and subgraphs are declared (if any), then vertices are declared. Vertices have a type name, named properties and a unique name that refers to that instance of the vertex. They are declared in a manner as follows
```
VertexType {
  propertyName=propertValue;
  propertyName2=propertyValue2;
  # And so forth...
} uniqueVertexName;
```
If a vertex requires no properties, one may simply declare
```
VertexType uniqueVertexName; 
```
Vertex types are either provided by modules (compiled into bleak) or are instances of subgraphs (discussed in [Subgraphs](#subgraphs)). Some examples of vertices will be described later.

Vertex properties are used to communicate runtime settings to the Vertex. This may be information about the size of a convolution kernel or the stride or dilation of a convolution operation. Importantly, Vertex properties are *not* variables. They may not reference themselves and cannot be declared unexpected. Variables and expressions may be used in Vertex properties (which is the whole intention of variables!). For example
```
numTrees=100;
treeDepth = 7;
applyWeightDecay = false;

Parameters {
  size = [ $numTrees, 2^$treeDepth - 1 ];
  learnable=true;
  initType="uniform";
  applyWeightDecay=$applyWeightDecay;
  b = 3;
  a = -$b; # ERROR: Properties are not variables.
  giraffe = "Not a property"; # ERROR: giraffe is not a Parameters property.
} thresholds;
```
Vertex properties afford a bit of flexibility in value types. Many types of values are implicitly convertible. For example
```
Parameters {
  size = 10; # Integer convertible to one component integer vector [ 10 ].
  learnable = 1; # Integer convertible to boolean.
  a="3.0"; # String representation of a float is convertible to a float.
  b=[ -3 ]; # One component integer vector is convertible to a float.
} tensor;
```
Any type is convertible to a string and any string is (possibly) convertible to any type. Other implicit conversions are provided below.
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

How vertices are compiled into bleak and given named properties and named inputs/outputs will be discussed [Implementing your own Vertex in C++](#implementing-your-own-vertex-in-c).

### Some Common Vertices
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
 
