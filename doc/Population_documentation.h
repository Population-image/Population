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
If something wrong occurs, whatever your skills, if you wan to join the community, contact me at vincent.tariel@polytechnique.edu .    
 
\section Download Download

In sourceforge https://sourceforge.net/projects/population/ , you download  population source with this \ref pagedirectoryorganisation .  

\section pageC  Install C++
Select your integrated development environment: 
- \ref pageqmake 
- \ref pagecmake

\section Python  Install Python 
- \ref pagetechnicalpython
\section DocTuto  Documentation

The documentation is on the <a href="modules.html">module page</a>.

\section Tutorials  Tutorials

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

You can find some C++/python codes at the end of these following tutorials or here $${PopulationPath}/other/tutorial/.
  
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
