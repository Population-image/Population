/*------------------------------------------------------------------------
#
#  File        : Population_documentation.h
#
#  Description : Extra documentation file for the Population Library.
#                
#               
#
#  Copyright   : Vincent Tariel
#                
#
#
-------------------------------------------------------------------------*/

/*-----------------------------------

  Main reference documentation page

  -------------------------------------*/

/**
   \mainpage

Population library is a scientific open source library dedicated to the processing, analysis, modeling and visualization of 2D/3D images (see http://www.population-image.fr).
Population is a versatile framework to answer the diversity of developer expectations:
- a generic library to implement optimized algorithms acting on versatile data-structures for contributor, 
- a ready-to-use library to allows its utilization for practitioner,
 <!-- - a dictionary of caméléon language to prototype and calibrate a data-process with drag and drop in real time for everybody. -->

Around a community project, its objective is to structure a reproducible and shared science in the image field.\n
If something wrong occurs or if you wan to join the community whatever your skills, contact me at vincent.tariel@polytechnique.edu .    
 
\section Download Download

The zip with sourceforge https://sourceforge.net/projects/population/ .
The Git repository:
\code
git clone git://69007hpv111117.ikoula.com/Population.git
\endcode
In linux, for opengl visualization, you need glut (for ubuntu distribution sudo apt-get install freeglut3-dev libx11-dev libxmu-dev libxi-dev)
You find this \ref pagedirectoryorganisation .  

\section pageC  Install C++
Select your build process: 
- \ref pageqmake 
- \ref pagecmake

\section Python  Install Python 
- \ref pagetechnicalpython

\section DocTuto  Documentation
The documentation is on the <a href="modules.html">module page</a>.

\section Tutorials  Tutorials

For these tutorials, the C++/python codes are here $${PopulationPath}/other/tutorial/ (with cmake-gui BUILD_TUTORIAL=ON).
The more important tutorial is this one  \ref pagefirststep to start coding. Then, you have : 
- Matrix data-structure
	- \ref pageimagebasic
	- \ref pageinout
	- \ref pagefastprotyping
- Segmentation 
	- \ref pagesegmentation
	- \ref pagesegmentation2
	- \ref pagesegmentation3
- Visualization
	- \ref pagevisu2d
	- \ref pagevisu3d
- Code your own algorithms
	- \ref pagearchitecture
	- \ref pagetemplateprogramming
	- \ref pageiteratormatrix

\section Theory  Theoritical aspects

A book is freely available <a href="http://www.population-image.fr/PopulationLibrary.pdf">here</a> under a creative Commons license.
  
**/

/*!
*  \defgroup Program Algorithms + Data-structures = Programs
*/

/*! \ingroup Program
*  \defgroup Data Data-structures 
*/
/*! \ingroup Program
*  \defgroup Algorithm Algorithm
*/
/*! \namespace pop
 *
 * namespace of Population library
 */
/*! \namespace std
 *
 * namespace of the standard template library  where I overload some functions
 */
