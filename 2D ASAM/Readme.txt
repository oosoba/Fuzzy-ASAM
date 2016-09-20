Name: 2D-ASAM
Author: Osonde Osoba
Date: 05-May-2014
Purpose: Trains three SAMs to approximate a user-specified 2-dimensional function using samples of the objective function.

How to run:
===========
1) - The program takes function samples from a file "FxnGen2D.dat". You supply this file in the same directory as the program. The samples are multiple lines in the form:
x	y	f(x,y)
for example:
5.56565657	6.14141414	0.0929060987	
I have included the sample "FxnGen2D.dat" file for the 2D sin function for you to compare.

2) - The program outputs 3 folders: Gauss, Sinc, and Cauchy. These contain the ASAM approximation details for the three different set functions. You should tune the number of rules, number of iterations, learning rates, and initializations to get better performance on your approximations. It also outputs the "Errors.dat" file. This is a log of MSEs for all the ASAMs at different points in the adaptation. "OutFxn.dat" is just a sanity-check procedure. 


Compilation Requirements: (Windows) 
===================================
1) Eigen library: You need to download the Eigen matrix manipulation library from http://eigen.tuxfamily.org/dox/
Your compiler needs to know where you put the library. You do this in Visual Studio by changing the "project properties->VC++ Directories->Include Directories". *Add* the eigen library folder path to the include directories.

2)  compile the project (source files: SAM2D.h, SAM2D.cpp, ASAM-Main.cpp)


Compilation Requirements: (Linux) 
===================================
1) Eigen library: You need to download the Eigen matrix manipulation library from http://eigen.tuxfamily.org/dox/
Your compiler needs to know where you put the library. The easy solution is to copy the whole Eigen library directory into the source folder.

2) Use the following command to run the g++ compiler in the same directory as the source files: 
"g++ -o test2d *.[ch]*"


Caveats:
========
1) - This 2D ASAM library differs from the previous version. I am using large arrays instead of matrices to store and manipulate f(x,y). You may need to do some further heavy lifting if you want to train on >1000 samples. 
2) - The program erases outputs from previous runs every time you run it. It is currently set to use 39 rules and stop after 60000 iterations.