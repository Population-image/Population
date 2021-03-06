/******************************************************************************\
|*       Population library for C++ X.X.X     *|
|*----------------------------------------------------------------------------*|
The Population License is similar to the MIT license in adding this clause:
for any writing public or private that has resulted from the use of the
software population, the reference of this book "Population library, 2012,
Vincent Tariel" shall be included in it.

So, the terms of the Population License are:

Copyright © 2012-2015, Tariel Vincent

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to
deal in the Software without restriction, including without limitation the
rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
sell copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software and for any writing
public or private that has resulted from the use of the software population,
the reference of this book "Population library, 2012, Vincent Tariel" shall
be included in it.

The Software is provided "as is", without warranty of any kind, express or
implied, including but not limited to the warranties of merchantability,
fitness for a particular purpose and noninfringement. In no event shall the
authors or copyright holders be liable for any claim, damages or other
liability, whether in an action of contract, tort or otherwise, arising
from, out of or in connection with the software or the use or other dealings
in the Software.
\***************************************************************************/
#ifndef MatN_HPP
#define MatN_HPP
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <cmath>
#include <string>
#include <algorithm>
#include <numeric>

#include"PopulationConfig.h"
#include"data/typeF/TypeTraitsF.h"
#include"data/typeF/RGB.h"
#include"data/vec/VecN.h"
#include"data/mat/MatNBoundaryCondition.h"
#include"data/mat/MatNIteratorE.h"
#include"data/functor/FunctorF.h"
#include"algorithm/ForEachFunctor.h"
#include"data/utility/BasicUtility.h"

namespace pop
{
template<typename PixelType, int SIZEI, int SIZEJ>
class  Mat2x;
/*! \ingroup Data
* \defgroup Matrix  Matrix
* \brief n-dimensional matrices as dense array
*/


/*! \ingroup Matrix
* \defgroup MatN Mat{2,3}{UI8,RGBUI8}
* \brief template class for n-dimensional matrices which fixed type
*/

template<int Dim, typename PixelType >
class POP_EXPORTS MatN //: public Vec<PixelType>
{
public:

    /*!
    \class pop::MatN
    \ingroup MatN
    \brief template class for matrix (or Image)
    \author Tariel Vincent
    \tparam Dim Space dimension
    \tparam PixelType Pixel/Voxel type


    A  matrix is a regular tessellation of a domain of the n-dimensional euclidean space where
    each cell is a n-square (a square in 2-dimension called pixel, a cube in 3-dimension called
    voxel). A cell is located with n integers (2 integers for a pixel and 3 integers for a voxel).
    In 2d, the horizontal and vertical lines in a matrix are called rows and columns, respectively.
    To access a cell, we use the matrix notation f(i,j) refering to the i-th row and the j-th column.
    Each cell contains an information. For a gray-level matrix, this information is a grey-level
    value. This value can be coded in one byte representing the range (0, 28 −1) = (0, 255) where
    the value 0 represents the black color, 255 the white color, and any value between (0, 255) a
    fraction between both colors. This class is a model of Function concept.

    \image html grid2.png Left: 2D  matrix with 6 rows and 5 columns with a pixel type coded in one byte, right: grey-level representation

    \section PixelType Pixel(Voxel) Type

    This class is written in generic programming with template parameters. The first one is the dimension, the second one the pixel/voxel type.
    To facilite its utilisation, we use some typedef declarations to define the usual types to allow coding in C-style as these ones:
    - Mat2UI8: a 2d matrix with a pixel type coded in 1 byte for an unsigned integer between 0 and 255,
    - Mat2RGBUI8: a 2d matrix with a pixel type coded with a RGB color with 1 byte per channel,
    - Mat2F32: a 2d matrix with a pixel type coded in float type,
    - Mat2UI32: a 2d matrix with a pixel type coded in unsigned integer in 4 bytes,
    - Mat3UI8: a 3d matrix with a pixel type coded in 1 byte for an unsigned integer between 0 and 255.

    \section Structure Structure

    The cell values are stored in a %vector container, a class template of STL, which works like a dynamic array in a strict linear sequence. The position of the
    cell can be located by an index or a point VecN<Dim,int> as n integers (in 2D II32=(i,j), in 3D Vec3I32=(i,j,k)) as explained in the below figure:
    \image html vector.png the value f(i,j) corresponds to the element  v[j+i*ColSize] of the %vector container.
    We make the choice to have a single Vec and not a Vec of Vec as the classical ways because the matrix can be nD and also for optimization purpose in the case of iteration.\n
    As explained in my book, this data-structure is a model of the Function concept represented that
    \f[ \begin{matrix}  f\colon   \mathcal{D}\subset E & \to& F \\ x & \mapsto& y = f(x) \end{matrix} \f]
     such that the input quantity completely determines the output quantity. The input quantity belongs to a space \f$E\f$ and the output to the space \f$F\f$
     and  \f$\mathcal{D}\f$ a subset of \f$E\f$ that is the domain of definition of the function. In this model, \f$E\f$ is VecN<Dim,int>, \f$F\f$ is pixel/voxel type and \f$\mathcal{D}\f$ is VecN<Dim,int>.


    \section Constructor Constructor

    The construction of an object requires the domain of the definition that is the number of rows  and columns in 2D MatN(int sizei,int sizej) or MatN(const VecN<DIM,int> & domain). For instance,
    this code:
    \code
    Mat2UI8 img(3,2);//construct an matrix with 3 rows and 2 columns
    Vec2I32 domain(3,2);
    Mat2UI8 img(domain);//construct an matrix with 3 rows and 2 columns
    \endcode

    \section ObjectFunction Object function to access element

    The matrix access of the pixel/voxel value, f(i,j) or f(x=(i,j)), is called as if it were an ordinary function operator ()(int i, int j) or operator ()(const E & x). The first one is the simplest one
    but less generic. For instance, this code
    \code
    Mat2UI8 img(2,3);//construct an matrix with 3 columns and 2 rows
    img(1,2)=255;//set the pixel value at 255
    std::cout<<img;
    \endcode
    produces this output:\n
    0 0 0\n
    0 0 255\n
    In similarly way, this code:
    \code
    Vec2I32 x(2,3);
    Mat2UI8 img(x);//construct an matrix with 3 columns and 2 rows
    x(0)=1;
    x(1)=2;
    img(x)=255;//set the pixel value at 255
    std::cout<<img;
    \endcode
    produces the same output.

    \section Iteration Iteration

    We have two major categories of iterative loops: through the domain and through the neighborhood.

    \subsection Domain Iterared over all points of the domain
    This class provides four ways for the iterative loops on the domain: \f$ \forall x \in \mathcal{D}\f$. Each time, we illustrate with a code generating an matrix with a constant value:
    -  \a for: utilisation of the for statement.
    \code
    Mat2UI8 img(256,256);
    for(int i =0;i<img.sizeI();i++)
        for(int j =0;j<img.sizeJ();j++)
            img(i,j)=150;
    img.display();
    \endcode
    -  \a IteratorE: utilisation of the IteratorEDomain concept. This class exposes an IteratorEDomain model with its definition type, \a MatN::IteratorEDomain, and its defintion domain, \a getIteratorEDomain().
    For the IteratorEDomain object, the member \a next() advances to the next element in returning a boolean to indicate if the end of the collection is reached and the member \a x() returns the current element
    \code
    Mat2UI8 img(256,256);
    Mat2UI8::IteratorEDomain it(img.getIteratorEDomain());
    while(it.next())
        img(it.x())=150;
    img.display();
    \endcode
    -  \a ForEachDomain idiom: implicit iteration defined by a preprocessor directive. Note we do have to set the type of the point
    \code
    Mat2UI8 img(256,256);
    ForEachDomain2D(x,img)
        img(x)=150;
    img.display();
    \endcode
    - \a  dereferencing: utilisation of the iterator of the STL vector.
    \code
    void constant (unsigned char & v) {
        v= 150;
    }
    Mat2UI8 img(256,256);
    for_each(img.begin(),img.end(),constant);
    img.display();
    \endcode
    - \a std::vector accesor:  utilisation of the accesor of the STL vector.
    \code
    Mat2UI8 img(256,256);
    for(int i =0;i<img.size();i++)
        img(i)=150;
    img.display();
    \endcode

    The first one is the simplest one but less generic than the second one independant of the space. The last one is more optimized but you lost the point position that can be required for the subsequent process as in the erosion algorithm.

    \subsection Neighborhood Iterared over all points on a neighborhood of a point
    This class provides two ways for the iterative loops on a point neighborhood: \f$ \forall x' \in N(x)\f$.
    -  \a for: utilisation of the for statement. For instance, the erosion code is:
    \code
    Mat2UI8 img;
    img.load((std::string(POP_PROJECT_SOURCE_DIR)+"/image/Lena.bmp").c_str());//load lena matrix
    Mat2UI8 erosion(img.sizeI(),img.sizeJ());//construct an matrix with the same domain
    int radius=4;
    //Domain loop
    for(int i =0;i<img.sizeI();i++){
        for(int j =0;j<img.sizeJ();j++){
            UI8 v =255; //equivalent to unsigned char v=255
             //Neighborhood loop
            for(int m= i-radius;m<i+radius;m++){
                for(int n= j-radius;n<j+radius;n++){
                    if(img.isValid(m,n)){//test if (i,j) belongs to the domain of definition of the matrix
                        v = std::min(img(m,n),v);
                    }
                }
            }
            erosion(i,j)=v;
       }
    }
    erosion.display();
    \endcode
    -  \a IteratorE: utilisation of the IteratorENeighborhood concept. This class exposes an IteratorENeighborhood model
    with its definition type, \a MatN::IteratorENeighborhood, and its defintion domain, \a MatN::getIteratorENeighborhood(F32 radius=1,int norm=1).
    For the IteratorENeighborhood object, the member \a next() advances to the next element in returning a boolean to indicate if the end of the collection is reached,
    the member \a x() returns the current element and the member \a init(const E & x)  initializes the neighborhood on the VecN x. For instance, the erosion code is:
    \code
    Mat2UI8 img;
    img.load((std::string(POP_PROJECT_SOURCE_DIR)+"/image/Lena.bmp").c_str());
    Mat2UI8 erosion(img.getDomain());//construct an matrix with the same domain
    Mat2UI8::IteratorEDomain itdomain(img.getIteratorEDomain());//Domain IteratorE
    Mat2UI8::IteratorENeighborhood itneigh (img.getIteratorENeighborhood(4,2));//Neighborhood IteratorE with the norm euclidean and the radius 4
    while(itdomain.next()){
        UI8 v =255;
        itneigh.init(itdomain.x());
        while(itneigh.next()){
            v = std::min(v,img(itneigh.x()));
        }
        erosion(itdomain.x())=v;
    }
    erosion.display();
    \endcode



    The abstraction with IteratorENeighborhood concept provides an efficient and generic ways to iterate through a neighborhood (see this tutorial \ref pagetemplateprogramming ).

    \section Load Load/Save

    You have many information in this tutorial \ref  pageinout .

    The implementation of these methods are Included in the hedader MatNInOut.h. So do not
    forget to Include "#include"data/mat/MatNInOut.h" to avoid a such error message "undefined reference to "'pop::MatN<2, unsigned char>::save(char const*) const'".
    You can load/save various matrix formats, png, pgm, jpg, bmp. , the naturel format is pgm. To load/save a stack of matrices, you
    can use MatN::loadFromDirectory or MatN::saveFromDirectory.
    However, I extend the pgm format to directly save a 3D matrix in a single file.


    \section Display Display

    You have many information in these tutorials \ref  pagevisu2d \ref  pagevisu3d.

    In good architectur design, the data and its representation are separeted in two classes. However for conveniency, you provide the member display() in this data class for simple display and
    you provide the MatNDisplay class for extended display.

      \sa VecN RGB Complex MatNDisplay
    */

protected:
    PixelType * _data;
    bool _is_owner_data;
    VecN<Dim,int> _domain;
    VecN<Dim,int> _stride;
public:

    /*!
    \typedef F
    * Pixel/voxel type
    */
    typedef PixelType F;
    /*! \var DIM
     * Space dimension
     */
    enum {DIM=Dim};
    /*!
    \typedef E
    * E=VecN<Dim,int> for the pixel/voxel localization
    */
    typedef VecN<Dim,I32> E;
    /*!
    \typedef Domain
    * Domain type that is VecN<Dim,int>
    */
    typedef VecN<Dim,int> Domain;
    /*!
    \typedef IteratorEDomain
    This iterator allows the iteration through all elements of E in order to pick up one
    time and only one time each element of E without any order. The corresponding mathematical
    object is : \f$\forall x \in \mathcal{D} \f$. This iterator does not impose more requirements than IteratorE concept
    pattern. Its construction requires a domain of definition as argument given by the member getIteratorEDomain() of this class.
    Its member next() advances to the next element in returning a boolean to indicate if
    the end of the collection is reached. Its member x() returns the current element x.

    For intance, to define an ouput matrix as follows:\n
    \f$ \forall x \in \mathcal{D} : h(x) = f(x)+v\f$\n
    we can write this code:
    \code
    Mat2UI8 img;
    img.load((std::string(POP_PROJECT_SOURCE_DIR)+"/image/Lena.bmp").c_str());
    Mat2UI8::IteratorEDomain it(img.getIteratorEDomain());
    Mat2UI8::F v =100;
    FunctorF::FunctorAdditionF2<UI8,UI8,UI8> op;
    while(it.next()){
    img(it.x()) = op(img(it.x()),v);
    }
    img.save("lenaaddconst.pgm");
    \endcode
    produce this matrix
    \image html lenaadd.jpg
    \sa getIteratorEDomain
    */
    typedef MatNIteratorEDomain<E>  IteratorEDomain;
    /*!
    \typedef IteratorEROI
    This iterator allows the iteration through all elements of subset E called a Region Of Interest (ROI) in order to pick up one
    time and only one time each element of this subset without any order. The corresponding mathematical
    object is : \f$\forall x \in \mathcal{D}' \f$. This iterator does not impose more requirements than IteratorE concept
    pattern. Its construction requires a domain of definition as argument given by the member getIteratorEROI() of this class.
    Its member next() advances to the next element in returning a boolean to indicate if
    the end of the collection is reached. Its member x() returns the current element x.

    For intance, to define an ouput matrix as follows:\n
    \f$ \forall x \in \mathcal{D}'=\{x\in \mathcal{D}:mask(x)\neq 0\} : h(x) = f(x)+v\f$\n
    we can write this code:
    \code
    Mat2UI8 img;
    img.load((std::string(POP_PROJECT_SOURCE_DIR)+"/image/Lena.bmp").c_str());
    Mat2UI8 mask;
    mask.load("mask.pgm");
    Mat2UI8::IteratorEROI it(mask.getIteratorEROI());
    Mat2UI8::F addvalue =100;
    FunctorF::FunctorAdditionF2<UI8,UI8,UI8> op;
    while(it.next()){
    img(it.x()) = op(img(it.x()),addvalue);
    }
    img.save("lenaaddconstROI.pgm");
    \endcode
    The mask matrix is as follows:
    \image html roi.jpg
    and the lenaaddconstROI matrix is as follows:
    \image html lenaroi.jpg
    \sa getIteratorEROI
    */
    typedef MatNIteratorEROI<MatN<Dim, PixelType> >  IteratorEROI;
    /*!
    \typedef IteratorENeighborhood

    This iterator allows the iteration through the neighborhood of an element
    x of E without any order. The corresponding mathematical object is : \f$\forall x' \in N(x)\f$. This
    iterator plays an important role in algorithms of mathematical morphology and region growing.
    Its construction requires a domain of definition as argument given by the member getIteratorENeighborhood(F32 radius,F32 norm).
    \f$N(x) =\{\forall x'    \in \mathcal{D} : \|x'-x\|_{norm} \leq radius \}\f$, for instance for norm=1, radius=1 and dim=2
    N(x=(i,j)) is equal to {(i,j),(i-1,j),(i+1,j),(i,j-1),(i,j+1)} that is the 4-connectivity. Its member next() advances to the next element
    in returning a boolean to indicate if the end of the collection is reached. Its member x() returns the current element x.
    Its member init(const E & x) initializes the iteration  on the neighborhood of the
    VecN x.

    For intance, the erosion is:\n
    \f$ \forall x \in \mathcal{D} : h(x) = \min_{\forall x'\in N(x) }f(x')\f$\n
    we can write this code:
    \code
    Mat2RGBUI8 img;
    img.load((std::string(POP_PROJECT_SOURCE_DIR)+"/image/Lena.bmp").c_str());
    Mat2RGBUI8 img_erosion(img.getDomain());
    Mat2RGBUI8::IteratorEDomain it_total(img.getIteratorEDomain());
    F32 norm =2;
    F32 radius =4.5;
    Mat2RGBUI8::IteratorENeighborhood it_neigh(img.getIteratorENeighborhood(radius,norm));
    while(it_total.next()){
      Mat2RGBUI8::F mini(255);
      it_neigh.init(it_total.x());MatN
    while(it_neigh.next()){
      mini = min(mini, img(it_neigh.x()));
    }
    img_erosion(it_total.x()) = mini;
    }
    img_erosion.save("/home/vincent/Desktop/_.bmp");
    \endcode
    produce this matrix
    \image html lenaero.jpg
    \sa RGB getIteratorENeighborhood
    */

    typedef MatNIteratorENeighborhood<E,MatNBoundaryConditionBounded> IteratorENeighborhood;// neighborhood iteration with bounded condition
    typedef MatNIteratorENeighborhood<E,MatNBoundaryConditionMirror> IteratorENeighborhoodMirror;// neighborhood iteration with mirror condition
    typedef MatNIteratorENeighborhood<E,MatNBoundaryConditionPeriodic> IteratorENeighborhoodPeriodic;// neighborhood iteration with periodic condition
    typedef MatNIteratorENeighborhoodAmoebas<MatN> IteratorENeighborhoodAmoebas;
    typedef MatNIteratorEOrder<E> IteratorEOrder;
    typedef MatNBoundaryCondition BoundaryCondition;

    /*!
    \typedef IteratorERectangle
    Iterate in rectangle

     \code
    Mat2UI8 img2;
    img2.load((std::string(POP_PROJECT_SOURCE_DIR)+"/image/Lena.bmp").c_str());
    Mat2UI8::IteratorERectangle itrec = img2.getIteratorERectangle(Vec2I32(50,200),Vec2I32(500,300));
    Mat2UI8::IteratorENeighborhood itloc =img2.getIteratorENeighborhood(3,2);
    img2  = maximum(img2,ProcessingAdvanced::dilation(img2,itrec,itloc));
    img2.display();
        \endcode
    \image html lenadilarec.png
    \sa  getIteratorERectangle
    */

    typedef MatNIteratorERectangle<E> IteratorERectangle;


    /*!
    \typedef iterator



     \sa  begin() end()
    */
    typedef PixelType * iterator;
    typedef const PixelType * const_iterator;


    //-------------------------------------
    //
    //! \name Constructor
    //@{
    //-------------------------------------
    /*!
    * default constructor
    */
    MatN();

    /*!
    * destructor
    */
    ~MatN();
    /*!
    \param domain domain of definition
    \param v init pixel/voxel value
    *
    * construct an matrix of size domain(0),domain(1) for 2D matrix and  domain(0),domain(1),domain(2) for 3D matrix
   where each pixel/voxel value is set at 0.\n
    *   This code: \n
    \code
    Vec2I32 x;
    x(0)=2;
    x(1)=4;
    Mat2UI8 img(x);
    x(0)=1;x(1)=1;
    img(x)=255;
    std::cout<<img;
    \endcode
    produce this output\n
    \code
    0 0 0 0
    0 255 0 0
    \endcode
    */
    explicit MatN(const VecN<Dim,int>& domain,PixelType v=PixelType());
    /*!
    \param sizei number of  columns
    \param sizej number of  rows
    *
    * construct an matrix of size i,j where each pixel/voxel value is set at 0\n
    *   This code: \n
    \code
    Mat2UI8 img(2,4);
    Vec2I32 x;
    x(0)=1;x(1)=1;
    img(x)=255;
    std::cout<<img;
    \endcode
    produce this output\n
    \code
    0 0 0 0
    0 255 0 0
    \endcode
    */

    explicit MatN(unsigned int sizei,unsigned int sizej);
    /*!
    \param sizei number of  columns
    \param sizej number of  rows
    \param sizek number of  depths
    *
    * construct an matrix of size i,j,k where each pixel/voxel value is set at 0\n
    */
    explicit MatN(unsigned int sizei, unsigned int sizej,unsigned int sizek);
    /*!
    * \brief reference copy
    * \param x domain size of the matrix
    * \param v_value affection values for the matrix elements
    *
    * You copy only the pointer to the data structure and the destructor does not desallocate the data
    *
    \code
    UI8 _A[]={1,1,1,1,1,
             1,0,0,0,1,
             1,0,0,0,1,
             1,1,1,1,1,
             1,0,0,0,1,
             1,0,0,0,1,
             1,0,0,0,1
            };
    Mat2UI8 LetterA(Vec2I32(7,5),_A);
    cout<<LetterA<<endl;
    \endcode
    */
    explicit MatN(const VecN<Dim,int> & domain, PixelType* v_value );
    /*!
    \param img object to copy
    *
    * copy construct\n
    *   This code: \n
    \code
    Mat2UI8::Domain x;
    x(0)=2;
    x(1)=4;
    Vec<Mat2UI8::F> v;
    v.push_back(0);v.push_back(1);v.push_back(2);v.push_back(3);
    v.push_back(3);v.push_back(2);v.push_back(1);v.push_back(0);
    Mat2UI8 img1(x,v);
    Mat2UI8 img2(img1);
    std::cout<<img1;
    \endcode
    produce this output\n
    \code
    0 1 2 3
    3 2 1 0
    \endcode
    */
    template<class T1>
    MatN(const MatN<Dim, T1> & img );


    /*!
    \param img object to copy
    *
    * copy construct\n
    *   This code: \n
    \code
    Mat2UI8::Domain x;
    x(0)=2;
    x(1)=4;
    Vec<Mat2UI8::F> v;
    v.push_back(0);v.push_back(1);v.push_back(2);v.push_back(3);
    v.push_back(3);v.push_back(2);v.push_back(1);v.push_back(0);
    Mat2UI8 img1(x,v);
    Mat2UI8 img2(img1);
    std::cout<<img1;
    \endcode
    produce this output\n
    \code
    0 1 2 3
    3 2 1 0
    \endcode
    */
#ifndef HAVE_SWIG
    MatN(const MatN & img );
#endif
    /*!
      * \param m small 2d matrix of size (2,2)
      *
      * type conversion
    */
    MatN(const Mat2x<PixelType,2,2> m);
    /*!
      * \param m small 2d matrix of size (3,3)
      *
      * type conversion
    */
    MatN(const Mat2x<PixelType,3,3> m);

    template<int SIZEI,int SIZEJ>
    MatN(const Mat2x<PixelType,SIZEI,SIZEJ> m);

    /*!
    \param filepath path of the matrix
    *
    *  construct the matrix from an matrix file
    *
    \code
    Mat2UI8 img((POP_PROJECT_SOURCE_DIR+std::string("/image/Lena.bmp")).c_str());
    img.display();
    \endcode
    */
    MatN(const char * filepath );
    /*!
    \param img bigger matrix
    \param xmin inclusive lower bound
    \param xmax exclusive upper bound
    *
    *  construct  a matrix for a part of the bigger matrix
    *
    in 2D
    \code
    Mat3UI8 img2;
    img2.load((std::string(POP_PROJECT_SOURCE_DIR)+"/image/Lena.bmp").c_str());
    Mat2UI8 img3 (img2,Vec2I32(300,200),Vec2I32(400,400));
    img3.display();
    \endcode
    in 3D
    \code
    Mat3UI8 img2;
    img2.load("/home/vincent/Desktop/work/Population/image/rock3d.pgm");
    Mat3UI8 smallcube(img2,Vec3I32(29,67,20),Vec3I32(159,167,200));
    smallcube.display();
    \endcode

    */
    MatN(const MatN & img, const VecN<Dim,int>& xmin, const VecN<Dim,int> & xmax  );
    //@}
    //-------------------------------------
    //
    //! \name Domain
    //@{
    //-------------------------------------

    /*!
    \return Domain domain of definition
    *
    * return domain of definition of the matrix
    * \sa VecN
    */
    Domain  getDomain()const;

    /*!
    \return  number of rows
    *
    * return the number of rows
    */
    unsigned int sizeI()const;
    /*!
    \return number of rows
    *
    * return the number of rows
    */
    unsigned int rows()const;
    /*!
    \return number of columns
    *
    * return the number of columns
    */
    unsigned int sizeJ()const;
    /*!
    \return number of columns
    *
    * return the number of columns
    */
    unsigned int columns()const;
    /*!
    \return int sizek
    *
    * return the number of depths
    */
    unsigned int sizeK()const;
    /*!
    \return number of depths
    *
    * return the number of depths
    */
    unsigned int depth()const;
    /*!
    \param x VecN
    \return boolean
    *
    * return true if the VecN belongs to the domain, false otherwise
    */
    bool isValid(const E & x)const;
    /*!
    \param i i coordinate of the VecN
    \param j j coordinate of the VecN
    \return boolean
    *
    * return true if the VecN (i,j) belongs to the domain, false otherwise
    */
    bool isValid(int i,int j)const;
    /*!
    \param i i coordinate of the VecN
    \param j j coordinate of the VecN
    \param k k coordinate of the VecN
    \return boolean
    *
    * return true if the VecN (i,j,k) belongs to the domain, false otherwise
    */
    bool isValid(int i,int j,int k)const;
    /*!
    \param sizei  row size
    \param sizej coloumn size
    *
    * resize the matrix in loosing the data information
    */
    void resize(unsigned int sizei,unsigned int sizej);
    /*!
    \param sizei  row size
    \param sizej  col size
    \param sizek depth size
    *
    * resize the matrix in loosing the data information
    */
    void resize(unsigned int sizei,unsigned int sizej,unsigned int sizek);
    /*!
    \param d  domain =Vec2(i,j) in 2d  and domain =Vec3(i,j,k) in 3d
    *
    * resize the matrix in loosing the data information
    */
    void resize(const VecN<Dim,int> & d);
    /*!
    \param sizei  row size
    \param sizej coloumn size
    *
    * resize the matrix in keeping the data information
    */
    void resizeInformation(unsigned int sizei,unsigned int sizej);
    /*!

    \param sizei  row size
    \param sizej  colo size
    \param sizek depth size
    *
    * resize the matrix in keeping the data information
    */
    void resizeInformation(unsigned int sizei,unsigned int sizej,unsigned int sizek);
    /*!
    \param d  domain =Vec2(i,j) in 2d  and domain =Vec3(i,j,k) in 3d
    *
    * resize the matrix in keeping the data information
    */
    void resizeInformation(const VecN<Dim,int>& d);
    /*!
    \return true if matrix is empty
    *
    * return true if the the matrix empty
    */
    bool isEmpty()const;
    /*!
    *
    * clear the content of the matrix
    */
    void clear();
    //@}

    //-------------------------------------
    //
    //! \name Accessor cell or sub-matrix
    //@{
    //-------------------------------------

    /*!
    \param x pixel/voxel position
    \return pixel/voxel value
    *
    * access the reference of the pixel/voxel value at the given position
    * \code
    Mat2UI8 img;
    img.load((std::string(POP_PROJECT_SOURCE_DIR)+"/image/Lena.bmp").c_str());
    Mat2UI8::IteratorEDomain it(img.getIteratorEDomain());
    Distribution d(0,20,"NORMAL");
    FunctorF::FunctorAdditionF2<Mat2UI8::F,F32,Mat2UI8::F> op;
    while(it.next()){
    img(it.x())=op(img(it.x()),d.randomVariable());//access a VecN, add a random variable and set it
    }
    img.display();
    \endcode
    * \sa VecN
    */
    inline F & operator ()(const VecN<Dim,int> & x);

    /*!
    \param x pixel/voxel position
    \return pixel/voxel value
    *
    * access the reference of the pixel/voxel value at the given position
    * \code
    Mat2UI8 img;
    img.load((std::string(POP_PROJECT_SOURCE_DIR)+"/image/Lena.bmp").c_str());
    Mat2UI8::IteratorEDomain it(img.getIteratorEDomain());
    Distribution d(0,20,"NORMAL");
    FunctorF::FunctorAdditionF2<Mat2UI8::F,F32,Mat2UI8::F> op;
    while(it.next()){
    img(it.x())=op(img(it.x()),d.randomVariable());//access a VecN, add a random variable and set it
    }
    img.display();
    \endcode
    * \sa VecN
    */
    inline const F & operator ()( const VecN<Dim,int>& x)const;
    /*!
    \param i i coordinate (row)
    \param j j coordinate (column)
    \return pixel/voxel value
    *
    * access the reference of the pixel/voxel value at the position (i,j) for a 2D matrix
    */
    inline PixelType & operator ()(unsigned int i,unsigned int j);
    /*!
    \param i i coordinate (row)
    \param j j coordinate (column)
    \return pixel/voxel value
    *
    * access the reference of the pixel/voxel value at the position (i,j) for a 2D matrix
    */
    inline const PixelType & operator ()(unsigned int i,unsigned int j)const;
    /*!
    \param i i coordinate (row)
    \param j j coordinate (column)
    \param k k coordinate (depth)
    \return pixel/voxel value
    *
    * access the reference of the pixel/voxel value at the given position (i,j,k) for a 3D matrix
    */
    inline PixelType & operator ()(unsigned int i,unsigned int j,unsigned int k);
    /*!
    \param i i coordinate (row)
    \param j j coordinate (column)
    \param k k coordinate (depth)
    \return pixel/voxel value
    *
    * access the reference of the pixel/voxel value at the given position (i,j,k) for a 3D matrix
    */
    inline const PixelType & operator ()(unsigned int i,unsigned int j,unsigned int k)const;
    /*!
    \param xmin inclusive lower bound
    \param xmax exclusive upper bound
    *
    *  extracts a rectangular sub-matrix
    *
    \code
    Mat2UI8 img;
    img.load("/usr/share/doc/opencv-doc/examples/c/lena.jpg");
    img = img(Vec2I32(100,100),Vec2I32(400,300));
    img.display();
    \endcode
    */
    MatN operator()(const VecN<Dim,int> & xmin, const VecN<Dim,int> & xmax) const;
    /*!
    \param index vector index
    \return pixel/voxel value
    *
    * access the reference of the pixel/voxel value at the vector index (Vec contains pixel values)
    */
    PixelType & operator ()(unsigned int index);

    /*!
    \param index vector index
    \return pixel/voxel value
    *
    * access the reference of the pixel/voxel value at the vector index (Vec contains pixel values)
    */
    PixelType &operator[](int index) ;
    const PixelType &operator[](int index) const ;
    const PixelType & operator ()(unsigned int index)const;
    /*!
    \param xf vector position in float value
    \return pixel/voxel value
    *
    * access the interpolated pixel/voxel value at the float position
    */
    PixelType interpolationBilinear(const VecN<DIM,F32> xf)const;

    /*!
    * Return a ptr to the first pixel value
    *
    *Exception
    * direct access to the matrix data that can be usefull for optimized purposes
    */
    PixelType *  data();
    /*!
    * Return a ptr to the first pixel value
    *
    *
    * direct access to the matrix data that can be usefull for optimized purposes
    */
    const PixelType *  data()const;
    //@}

    //-------------------------------------
    //
    //! \name In-out facility
    //@{
    //-------------------------------------

    /*!
    * \param pathdir directory path
    * \param basefilename filename base by default "toto"
    * \param extension by default ".pgm"
    *
    *
    * The loadFromDirectory attempts to load all files as 2d slices of the  3D matrix in the  directory pathdir. If the extension is set,
    * we filter all filter all files with the extension. It is the same for basename.\n
    * For instance, this code produces:
    \code
    Mat3UI8 img;
    img.loadFromDirectory("/home/vincent/Desktop/WorkSegmentation/lavoux/","in","tiff");
    img.display();
    img.save("lavoux3d.pgm");
    \endcode

    */
    void loadFromDirectory(const char * pathdir,const char * basefilename="",const char * extension="");
    /*!
    * \param file input file
    * \return true in case of success
    *
    * The loader attempts to read the matrix using the specified format. Natively, this library support the pgm, png, jpg, bmp formats. However thanks to the CImg library, this library can
    read various matrix formats http://cimg.sourceforge.net/reference/group__cimg__files__io.html if you install Image Magick http://www.imagemagick.org/script/binary-releases.php.
    */
    bool load(const char * file);
    /*!
    * \param file input file
    * \return true in case of success
    *
    * \sa MatN::load(const char * file)
    */
    bool load(const std::string file) ;
    /*!
    * \param file input file
    * \param d  domain of definition of the image
    * \return true in case of success
    *
    * The loader attempts to read the 3d raw matrix. The voxel type of the matrix must be the same as in raw file and, in more, you need to
    * give the domain (size) of the image in the raw file. \n
    * For instance, if the voxel is coded in 1 byte, you write the following code
    * \code
    * Mat3UI8 menisque;
    * Vec3I32 d(1300,1500,401);//matriciel notation, so 1300 raw, 1500 columns and 401 depth
    * menisque.loadRaw("/home/vincent/Downloads/top_cap_1300_1500_401_1b.raw",d);//load the 3d raw matrix
    * \endcode
    *
    */
    bool loadRaw(const char * file,const Domain & d);
    /*!
    * \param pathdir directory path
    * \param basefilename filename base by default "toto"
    * \param extension by default ".pgm"
    *
    * The saveFromdirectory attempts to save Save all slices of the  3D matrix f in the  directory pathdir with the given basefilename and the extenion,\n
    * for instance pathdir="/home/vincent/Project/ENPC/ROCK/Seg/"  basefilename="seg" and extension=".bmp", will save the slices as follows \n
    * "/home/vincent/Project/ENPC/ROCK/Seg/seg0000.bmp", \n
    * "/home/vincent/Project/ENPC/ROCK/Seg/seg0001.bmp",\n
    * "/home/vincent/Project/ENPC/ROCK/Seg/seg0002.bmp"\n
    *  "and so one.
    */
    void saveFromDirectory(const char * pathdir,const char * basefilename="toto",const char * extension=".pgm")const ;

    /*!
    * \param file input file
    *
    * The saver attempts to write the matrix using the specified format.  Natively, this library support the pgm, png, jpg, bmp format. However thanks to the CImg library, this library can
    save various matrix formats http://cimg.sourceforge.net/reference/group__cimg__files__io.html .
    */
    void save(const char * file)const ;

    /*!
    * \param file input file
    *
    * \sa MatN::save(const char * file)
    */
    void save(const std::string file)const ;
    /*!
    * \param file input file
    *
    * save the data in raw format without header
    */
    void saveRaw(const char * file)const ;
    /*!
    * \param file input file
    * \param header header of the file
    *
    * save the data in ascii format without header
    */
    void saveAscii(const char * file,std::string header="")const ;
    /*!
    * \param title windows title
    * \param stoprocess for stoprocess=true, stop the process until the windows is closed, otherwise the process is still running
    * \param automaticresize for automaticresize=true, you scale the matrix before the display, we do nothing otherwise
    *
    * Display the matrix using the CIMG facility.
    * \code
    * Mat2UI8 img;
    * img.load((std::string(POP_PROJECT_SOURCE_DIR)+"/image/Lena.bmp").c_str());
    * img.display();
    * Mat2F32 gradx(img);
    * gradx = pop::Processing::gradientDeriche(gradx,0,0.5);
    * gradx = pop::Processing::greylevelRange(gradx,0,255);//to display the matrix with a float type, the good thing is to translate the grey-level range between [0-255] before
    * gradx.display();
    * \endcode
    *
    * display the matrix
    */

    void display(const char * title="",bool stoprocess=true, bool automaticresize=true)const ;

    //@}

    //-------------------------------------
    //
    //! \name Iterators
    //@{
    //-------------------------------------

    /*!
    \fn typename IteratorEDomain getIteratorEDomain()const
    \return total iterator
    *
    * return the total iterator of the matrix that will iterate through the domain + x\n
    *
    */
    IteratorEDomain getIteratorEDomain()const;
    /*!
    \fn typename IteratorEROI getIteratorEROI()const
    \return ROI iterator
    *
    * return the ROI iterator  of the matrix where the iteration is done on
    * pixel/voxel values different to 0.
    *

    */
    IteratorEROI getIteratorEROI()const;
    /*!
    \param radius ball radius
    \param norm ball norm
    \return Neighborhood iterator
    *
    * The neighborhood is defined using the iterative stuctural element,\f$S^n\f$,\n
    * The initial stuctural element \f$S= \{x :\mbox{structural}(x-center)\neq 0\}\f$ with center the center domain of the matrix.\n
    * For instance, with structural=\f$\begin{pmatrix} 0 & 0 & 0\\0 & 255 & 255\\0 & 255 & 0\end{pmatrix}\f$, we have \f$ S=\{(0,0),(1,0),(0,1)\}\f$.\n
    * The iterative stuctural element, \f$S^n\f$ is n times the mean by itselt : \f$ S\oplus S\ldots \oplus S\f$ n times. For instance,
    * return the Neighborhood iterator domain of the matrix as argument of the IteratorENeighborhood constructor with the given norm and radiu
    *
    \sa IteratorENeighborhood
    */
    IteratorENeighborhood getIteratorENeighborhood(F32 radius=1 ,int norm=1 )const;
    /*!
    * \param structural_element structural element
    * \param dilate number of dilation of the structural element
    \return Neighborhood iterator
    *
    * \code
    * Mat2UI8 img;
    * img.load((std::string(POP_PROJECT_SOURCE_DIR)+"/image/Lena.bmp").c_str());
    * Mat2UI8 S(3,3);
    * S(1,1)=255;S(2,2)=255;
    * Mat2UI8::IteratorENeighborhood itn(img.getIteratorENeighborhood(S,20));
    * Mat2UI8::IteratorEDomain itg(img.getIteratorEDomain());
    * Mat2UI8 ero(img.getDomain());
    * while(itg.next()){
    *     UI8 value = 255;
    *     itn.init(itg.x());
    *     while(itn.next()){
    *     value = min(value, img(itn.x()));
    *     }
    *     ero(itg.x())=value;
    * }
    * ero.display();
    * \endcode
    \sa IteratorENeighborhood
    */
    template<typename Type1>
    IteratorENeighborhood getIteratorENeighborhood(const MatN<Dim,Type1> & structural_element,int dilate=1 )const;
    /*!
    * \param coordinatelastloop coordinate of the last loop of iteratation
    * \param direction 1=0 to N , otherwise N to 0
    * \return order iterator
    *
    * Iteration through to the domain of definition such the last loop of iteration is given by the coordinate
    * and the way of iteration by the direction.\n
    * For instance in 2D,
    * \code
    *  Mat2UI8 m(512,256);
    *  Mat2UI8::IteratorEOrder it (0,-1);
    *  while(it.next()){
    *   // do something
    *  }
    * // this code os equivalent to
    *  for(unsigned int j=0;j<m.sizeJ();j++)
    *    for(unsigned int i=m.sizeJ()-1;i>=0;i++){// last loop is the 0-cooridnate in reverse ways
    *       // do something
    *  }
    * \endcode
    *
    *
    */
    IteratorEOrder getIteratorEOrder(int coordinatelastloop=0,int direction=1)const;
    /*!
    * \param xmin top left corner
    * \param xmax buttom right corner
    * \return Rectangle iterator
    *
    * Iteration through to the rectangle define by these two points [xmin,xmax].\n
    * For instance in 2D,
    * \code
    * Mat2UI8 m(1024,512);
    * Mat2UI8::IteratorERectangle it(m.getIteratorERectangle(Vec2I32(100,200),Vec2I32(102,201)));
    * while(it.next()){
    *     std::cout<<it.x()<<std::endl;
    * }
    * \endcode
    * produce this output
    * 100<P>200<P> \n
    * 101<P>200<P> \n
    * 102<P>200<P> \n
    * 100<P>201<P> \n
    * 101<P>201<P> \n
    * 102<P>201<P>
    *
    *
    */
    IteratorERectangle getIteratorERectangle(const E & xmin,const E & xmax )const;
    /*!
    * \brief Amoeabas kernel
    * \param distance_max maximum distance
    * \param lambda_param parameter of ameaba distance
    * \return Neighborhood iterator
    *
    * R. Lerallut, E. Decenciere, and F. Meyer. Image filtering using morphological amoebas. Image and Vision Computing, 25(4), 395–404 (2007)
    *
    * \code
    * Mat2UI8 m;
    * m.load("../doc/image/plate.jpg");
    * m.display("init",false);
    * Mat2UI8::IteratorENeighborhoodAmoebas  it_local = m.getIteratorENeighborhoodAmoebas(6,0.01);
    * Mat2UI8::IteratorEDomain it_global = m.getIteratorEDomain();
    * Mat2UI8 m_median = ProcessingAdvanced::median(m,it_global,it_local);
    * m_median.display();
    * \endcode
    * \image html plate.jpg "initial image"
    * \image html plate_median_classic.jpg "median filter with fixed kernel"
    * \image html plate_median_amoeba.jpg "median filter with ameaba kernel"
    */
    IteratorENeighborhoodAmoebas getIteratorENeighborhoodAmoebas(F32 distance_max=4,F32 lambda_param = 0.01 )const;


    iterator begin(){ return this->_data; }

    /**
     *  Returns a read-only (constant) iterator that points to the
     *  first element in the %vector.  Iteration is done in ordinary
     *  element order.
     */
    const_iterator begin() const { return this->_data; }

    /**
     *  Returns a read/write iterator that 1points one past the last
     *  element in the %vector.  Iteration is done in ordinary
     *  element order.
     */
    iterator  end() { return this->_data+_domain.multCoordinate(); }

    /**
     *  Returns a read-only (constant) iterator that points one past
     *  the last element in the %vector.  Iteration is done in
     *  ordinary element order.
     */
    const_iterator end() const{ return this->_data+_domain.multCoordinate(); }

    /**
     *  Returns true if the %vector is empty.  (Thus begin() would
     *  equal end().)
     */
    bool empty() const { return begin() == end(); }
    /**  Returns the number of elements  */
    unsigned int size() const { return this->getDomain().multCoordinate(); }

    //@}

    //-------------------------------------
    //
    //! \name Arithmetics
    //@{
    //-------------------------------------

    /*!
    * \param img other matrix
    * \return this matrix
    *
    * Basic assignement of this matrix by \a other
    */
    template<class T1>
    MatN& operator =(const MatN<Dim, T1> & img );
    /*!
    * \param img other matrix
    * \return this matrix
    *
    * Basic assignement of this matrix by \a other
    */
    MatN& operator =(const MatN & img );
    /*!
    * \param value value
    * \return this matrix
    *
    * Basic assignement of all pixel/voxel values by \a value
    */
    MatN<Dim, PixelType>&  operator=(PixelType value);
    /*!
    * \param value value
    * \return this matrix
    *
    * Basic assignement of all pixel/voxel values by \a value
    */
    MatN<Dim, PixelType>&  fill(PixelType value);
    /*!
    * \param mode mode by default 0
    * \return opposite matrix
    *
    * opposite of the matrix  h(x)=max(f::F)-f(x) with max(f::F) is the maximum value of the range defined by the pixel/voxel type for mode =0,\n
    * or h(x)=max(f)-f(x) with max(f) is the maximum value of the field for mode =1
    */
    MatN<Dim, PixelType>  opposite(int mode=0)const;
    /*!
    \param f input matrix
    \return boolean
    *
    * Equal operator true for all x in E f(x)=(*this)(x), false otherwise
    */
    bool operator==(const MatN<Dim, PixelType>& f)const;
    /*!
    \param f input matrix
    \return boolean
    *
    * Difference operator true for at least on x in E f(x)!=(*this)(x), false otherwise
    */
    bool operator!=(const MatN<Dim, PixelType>& f)const;
    /*!
    \param f input matrix
    \return object reference
    *
    * Addition assignment h(x)+=f(x)
    */
    MatN<Dim, PixelType>&  operator+=(const MatN<Dim, PixelType>& f);
    /*!
    * \param f input matrix
    * \return object
    *
    *  Addition h(x)= (*this)(x)+f(x)
    */
    MatN<Dim, PixelType>  operator+(const MatN<Dim, PixelType>& f)const;
    /*!
    * \param value input value
    * \return object reference
    *
    * Addition assignment h(x)+=value
    */
    MatN<Dim, PixelType>& operator+=(PixelType value);
    /*!
    \param value input value
    \return object
    *
    * Addition h(x)= (*this)(x)+value
    */
    MatN<Dim, PixelType>  operator+(PixelType value)const;
    /*!
    \param f input matrix
    \return object reference
    *
    * Subtraction assignment h(x)-=f(x)
    */
    MatN<Dim, PixelType>&  operator-=(const MatN<Dim, PixelType>& f);
    /*!
    \param value input value
    \return object reference
    *
    * Subtraction assignment h(x)-=value
    */
    MatN<Dim, PixelType>&  operator-=(PixelType value);
    /*!
    * \param f input matrix
    * \return output matrix
    *
    *  Subtraction h(x)= (*this)(x)-f(x)
    */
    MatN<Dim, PixelType>  operator-(const MatN<Dim, PixelType>& f)const;
    /*!
    * \return output matrix
    *
    *  opposite   h(x)= -this(x)
    */
    MatN<Dim, PixelType>  operator-()const;
    /*!
    * \param value input value
    * \return output matrix
    *
    * Subtraction h(x)= (*this)(x)-value
    */
    MatN<Dim, PixelType>  operator-(PixelType value)const;

    /*!
    * \param m  other matrix
    * \return output matrix
    *
    *  matrix multiplication see http://en.wikipedia.org/wiki/Matrix_multiplication
    *
    *  \code
    Mat2F32 m1(2,3);
    m1(0,0)=1; m1(0,1)=2; m1(0,2)=0;
    m1(1,0)=4; m1(1,1)=3; m1(1,2)=-1;

    Mat2F32 m2(3,2);
    m2(0,0)=5; m2(0,1)=1;
    m2(1,0)=2; m2(1,1)=3;
    m2(2,0)=3; m2(2,1)=4;
    Mat2F32 m3 = m1*m2;
    std::cout<<m3<<std::endl;
    *  \endcode
    *
    */
    MatN  operator*(const MatN &m)const;
    /*!
    * \param m  other matrix
    * \return output matrix
    *
    *  matrix multiplication see http://en.wikipedia.org/wiki/Matrix_multiplication
    */
    MatN & operator*=(const MatN &m);
    /*!
    \param v  vector
    \return output vector
    *
    *  matrix vector  multiplication
    */
    Vec<PixelType>  operator*(const Vec<PixelType> & v)const;
    /*!
    \param f  matrix
    \return output matrix
    *
    *  multTermByTerm h(x)= (*this)(x)*f(x) (to avoid the the confusion with the matrix multiplication, we use this signature)
    */
    MatN  multTermByTerm(const MatN& f)const;
    /*!
    \param value input value
    \return object reference
    *
    * Multiplication assignment h(x)*=value
    */
    MatN<Dim, PixelType>&  operator*=(PixelType  value);
    /*!
    \param value input value
    \return object
    *
    * Multiplication h(x)= (*this)(x)*value
    */
    MatN<Dim, PixelType>  operator*(PixelType value)const;
    /*!
    \param f  matrix
    \return output matrix
    *
    *  division term by term h(x)= (*this)(x)/f(x) (to avoid the the confusion with the matrix division, we use this signature)
    */
    MatN<Dim, PixelType>  divTermByTerm(const MatN& f);
    /*!
    \param value input value
    \return object reference
    *
    * Division assignment h(x)/=value
    */
    MatN<Dim, PixelType>&  operator/=(PixelType value);
    /*!
    \param value input value
    \return object
    *
    * Division h(x)= (*this)(x)/value
    */
    MatN<Dim, PixelType>  operator/(PixelType value)const;
    //@}
    //-------------------------------------
    //
    //! \name Linear algebra facilities
    //@{
    //-------------------------------------
    /*!
    * \param i  row entry
    *
    * delete the row of index i
    */
    MatN deleteRow(unsigned int i)const;
    /*!
    * \param j  column entry
    *
    * delete the column of index j
    */
    MatN deleteCol(unsigned int j)const;
    /*!
    * \param i  row entry
    * \return the row in a Vec
    *
    * the output Vec contained the row at the given index i
    * \sa Vec
    */
    Vec<F> getRow(unsigned int i)const;
    /*!
    * \param j  column entry
    * \return the column in a Vec
    *
    * the output Vec contained the column at the given index j
    * \sa Vec
    */
    Vec<F> getCol(unsigned int j)const;
    /*!
    * \param i  row entry
    * \param v  Vec
    *
    * set the row at the given row entry with the given Vec of size equal to number of column
    * \sa Vec
    */
    void setRow(unsigned int i,const Vec<F>& v);
    /*!
    * \param j  column entry
    * \param v  Vec
    *
    * set the column at the given column entry with the given Vec of size equal to number of row
    * \sa Vec
    */
    void setCol(unsigned int j,const Vec<F>& v);
    /*!
    * \param i_0  row entry
    * \param i_1  row entry
    *
    * swap the rows
    */
    void swapRow(unsigned int i_0,unsigned int i_1);
    /*!
    * \param j_0  column entry
    * \param j_1  column entry
    *
    * swap the columns
    */
    void swapCol(unsigned int j_0,unsigned int j_1);
    /*!
    * \param i  row entry
    * \param j  column entry
    *
    * the  minor of a matrix A is the determinant of the smaller square matrix, cut down from A by removing the i row and the j column.
    */
    F minorDet(unsigned int i, unsigned int j)const;
    /*!
    * \param i  row entry
    * \param j  column entry
    *
    * the cofactor of a matrix minor A is the minor determinant multiply by \f$(-1)^{i+j}\f$
    * \sa minorDet(int i, int j)const
    */
    F cofactor(unsigned int i,unsigned int j)const;
    /*!
    *
    * the matrix of cofactors  is the matrix whose (i,j) entry is the cofactor C_{i,j} of A
    * \sa cofactor(int i, int j)const
    */
    MatN cofactor()const;
    /*!
    *
    * the ith row, jth column element of transpose matrix is the jth row, ith column element of matrix:
    */
    MatN transpose()const;
    /*!
    *
    * the determinant is a value associated with a square matrix f <a href=http://en.wikipedia.org/wiki/Determinant>wiki</a>
    */
    F determinant()const;
    /*!
    * \return trace
    *
    * the trace of an n-by-n square matrix A is defined to be the sum of the elements on the main diagonal
    \code
    Mat2F32 m(3,3);
    m(0,0)=1;m(0,1)=1;m(0,2)=2;
    m(1,0)=2;m(1,1)=1;m(1,2)=2;
    m(2,0)=1;m(2,1)=3;m(2,2)=3;
    std::cout<<m.trace()<<std::endl;
    \endcode
    */
    F trace()const ;

    /*!
      *\return matrix reference
    *
    *  the inverse of the matrix <a href=http://en.wikipedia.org/wiki/Invertible_matrix>wiki</a>
    \code
    Mat2F32 m(3,3);
    m(0,0)=1;m(0,1)=1;m(0,2)=2;
    m(1,0)=2;m(1,1)=1;m(1,2)=2;
    m(2,0)=1;m(2,1)=3;m(2,2)=3;

    Mat2F32 minverse;
    minverse = m.inverse();
    std::cout<<minverse<<std::endl;
    std::cout<<m*minverse<<std::endl;
    \endcode
    For large matrix, you should use LinearAlgebra::inverseGaussianElimination()
    */
    MatN inverse()const;

    /*! \brief  \f$I_n = \begin{bmatrix}1 & 0 & \cdots & 0 \\0 & 1 & \cdots & 0 \\\vdots & \vdots & \ddots & \vdots \\0 & 0 & \cdots & 1 \end{bmatrix}\f$
     * \param size_mat size of the output matrix
     * \return  Identity matrix
     *
     *  Generate the identity matrix or unit matrix of square matrix with the given size for size!=0 or this matrix size with ones on the main diagonal and zeros elsewhere
    */
    MatN identity(int size_mat=0)const;

    //@}

#ifdef HAVE_SWIG
    MatN(const MatN<Dim,UI8> &img)
        :_data(new PixelType[img.getDomain().multCoordinate()]),_is_owner_data(true),_domain(img.getDomain())
    {
        _initStride();
        std::transform(img.begin(),img.end(),this->begin(),ArithmeticsSaturation<PixelType,UI8>::Range);
    }
    MatN(const MatN<Dim,UI16> &img)
        :_data(new PixelType[img.getDomain().multCoordinate()]),_is_owner_data(true),_domain(img.getDomain())
    {
        _initStride();
        std::transform(img.begin(),img.end(),this->begin(),ArithmeticsSaturation<PixelType,UI16>::Range);
    }
    MatN(const MatN<Dim,UI32> &img)
        :_data(new PixelType[img.getDomain().multCoordinate()]),_is_owner_data(true),_domain(img.getDomain())
    {
        _initStride();
        std::transform(img.begin(),img.end(),this->begin(),ArithmeticsSaturation<PixelType,UI32>::Range);
    }
    MatN(const MatN<Dim,F32> &img)
        :_data(new PixelType[img.getDomain().multCoordinate()]),_is_owner_data(true),_domain(img.getDomain())
    {
        _initStride();
        std::transform(img.begin(),img.end(),this->begin(),ArithmeticsSaturation<PixelType,F32>::Range);
    }
    MatN(const MatN<Dim,RGBUI8> &img)
        :_data(new PixelType[img.getDomain().multCoordinate()]),_is_owner_data(true),_domain(img.getDomain())
    {
        _initStride();
        std::transform(img.begin(),img.end(),this->begin(),ArithmeticsSaturation<PixelType,RGBUI8>::Range);
    }
    MatN(const MatN<Dim,RGBF32> &img)
        :_data(new PixelType[img.getDomain().multCoordinate()]),_is_owner_data(true),_domain(img.getDomain())
    {
        _initStride();
        std::transform(img.begin(),img.end(),this->begin(),ArithmeticsSaturation<PixelType,RGBF32>::Range);
    }
    MatN(const MatN<Dim,ComplexF32> &img)
        :_data(new PixelType[img.getDomain().multCoordinate()]),_is_owner_data(true),_domain(img.getDomain())
    {
        _initStride();
        std::transform(img.begin(),img.end(),this->begin(),ArithmeticsSaturation<PixelType,ComplexF32>::Range);
    }
    PixelType getValue(int i, int j)const{
        return  this->operator[](j+i*_domain(1));
    }
    PixelType getValue(int i, int j, int k )const{
        return  this->operator[](j+i*_domain(1)+k*_domain(0)*_domain(1));
    }
    PixelType getValue(const E & x )const{
        return  this->operator[](VecNIndice<Dim>::VecN2Indice(_stride,x));
    }
    void setValue(int i, int j , PixelType value){
        this->operator[](j+i*_domain(1)) =value;
    }
    void setValue(int i, int j , int k, PixelType value){
        this->operator[](j+i*_domain(1)+k*_domain(0)*_domain(1)) =value;
    }
    void setValue(const E & x, PixelType value){
        this->operator[](VecNIndice<Dim>::VecN2Indice(_stride,x)) =value;
    }

#endif


    const VecN<Dim,int>& stride()const{
        return _stride;
    }
    VecN<Dim,int>& stride(){
        return _stride;
    }
    bool isOwnerData()const{
        return _is_owner_data;
    }
    MatN selectColumn(int index_column){
        MatN m(VecN<Dim,int>(_domain(0),1),this->data()+_stride[1]*index_column);
        m._stride(0)=this->_stride(0);
        m._stride(1)=1;
        return m;
    }

    MatN selectRow(int index_row){
        MatN m(VecN<Dim,int>(1,_domain(1)),this->data()+_stride[0]*index_row);
        m._stride(1)=this->_stride(1);
        m._stride(0)=1;
        return m;
    }

    MatN copyData()const{
        MatN m(this->getDomain());
        std::copy(this->begin(),this->end(),m.begin());
        return m;
    }

private:
    void _initStride(){
        _stride[1]=1;
        _stride[0]=_domain[1];
        for(unsigned int i=2;i<DIM;i++){
            if(i==2)
                _stride[2]=_domain[1]*_domain[0];
            else
                _stride[i]=_domain[i-1]*_stride[i-1];
        }
    }
};

typedef MatN<2,UI8> Mat2UI8;
typedef MatN<2,UI16> Mat2UI16;
typedef MatN<2,UI32> Mat2UI32;
typedef MatN<2,F32> Mat2F32;

typedef MatN<2,RGBUI8> Mat2RGBUI8;
typedef MatN<2,RGBF32> Mat2RGBF32;
typedef MatN<2,ComplexF32> Mat2ComplexF32;
typedef MatN<2,Vec2F32 >  Mat2Vec2F32;


typedef MatN<3,UI8> Mat3UI8;
typedef MatN<3,UI16> Mat3UI16;
typedef MatN<3,UI32> Mat3UI32;
typedef MatN<3,F32> Mat3F32;

typedef MatN<3,RGBUI8> Mat3RGBUI8;
typedef MatN<3,RGBF32> Mat3RGBF32;
typedef MatN<3,ComplexF32> Mat3ComplexF32;
typedef MatN<3,VecN<3,F32> >  Mat3Vec3F32;



template<int Dim, typename PixelType>
MatN<Dim,PixelType>::MatN()
    :_data(NULL),_is_owner_data(true)
{
    _domain=0;
    _initStride();
}
template<int Dim, typename PixelType>
MatN<Dim,PixelType>::~MatN()
{
    if(_is_owner_data==true&& _data!=NULL)
        delete[] _data;
}
template<int Dim, typename PixelType>
MatN<Dim,PixelType>::MatN(const VecN<Dim,int>& domain,PixelType v)
    :_data(new PixelType[domain.multCoordinate()]),_is_owner_data(true),_domain(domain)
{
    std::fill(this->begin(), this->end(), v);
    _initStride();
}


template<int Dim, typename PixelType>
MatN<Dim,PixelType>::MatN(unsigned int sizei,unsigned int sizej)
    :_data(new PixelType[sizei*sizej]),_is_owner_data(true),_domain(sizei,sizej)
{
    std::fill(this->begin(), this->end(), PixelType(0));
    _initStride();
    POP_DbgAssertMessage(Dim==2,"In MatN::MatN(int i, int j), your matrix must be 2D");

}
template<int Dim, typename PixelType>
MatN<Dim,PixelType>::MatN(unsigned int sizei, unsigned int sizej,unsigned int sizek)
    :_data(new PixelType[sizei*sizej*sizek]),_is_owner_data(true),_domain(sizei,sizej,sizek)
{
    std::fill(this->begin(), this->end(), PixelType(0));
    _initStride();
    POP_DbgAssertMessage(Dim==3,"In MatN::MatN(int sizei, int sizej,int sizek), your matrix must be 3D");
}
//template<int Dim, typename PixelType>
//MatN<Dim,PixelType>::MatN(const VecN<Dim,int> & x,const Vec<PixelType>& data_values )
//    :_data(new PixelType[x.multCoordinate()]),_is_owner_data(true),_domain(x)
//{
//    _initStride();
//    POP_DbgAssertMessage((int)data_values.size()==_domain.multCoordinate(),"In MatN::MatN(const VecN<Dim,int> & x,const Vec<PixelType>& data ), the size of input Vec data must be equal to the number of pixel/voxel");
//}

template<int Dim, typename PixelType>
MatN<Dim,PixelType>::MatN(const VecN<Dim,int> & domain, PixelType* v_value )
    :_data(v_value),_is_owner_data(false),_domain(domain)
{
    _initStride();
}

template<int Dim, typename PixelType>
template<typename T1>
MatN<Dim,PixelType>::MatN(const MatN<Dim, T1> & img )
    :_data(new PixelType[img.getDomain().multCoordinate()]),_is_owner_data(true),_domain(img.getDomain())
{
    _initStride();
    std::transform(img.begin(),img.end(),this->begin(),ArithmeticsSaturation<PixelType,T1>::Range);
}


#ifndef HAVE_SWIG
template<int Dim, typename PixelType>
MatN<Dim,PixelType>::MatN(const MatN<Dim,PixelType> & img )
{
    if(img._is_owner_data==false){
        this->_is_owner_data = img._is_owner_data;
        this->_data  = img._data;
        this->_domain  = img._domain;


    }else{
        this->_data= new PixelType[img.getDomain().multCoordinate()];
        this->_is_owner_data = img._is_owner_data;
        this->_domain  = img._domain;
        std::copy(img.begin(),img.end(),this->begin());
    }
    _initStride();
}
#endif

template<int Dim, typename PixelType>
MatN<Dim,PixelType>::MatN(const char * filepath )
    :_data( NULL),_is_owner_data(true),_domain(0)
{
    if(filepath!=0)
        load(filepath);
    _initStride();
}
template<int Dim, typename PixelType>
MatN<Dim,PixelType>::MatN(const MatN<Dim,PixelType> & img, const VecN<Dim,int>& xmin, const VecN<Dim,int> & xmax  )
    :_data( new PixelType[(xmax-xmin).multCoordinate()]),_is_owner_data(true),_domain(xmax-xmin)
{
    POP_DbgAssertMessage(xmin.allSuperiorEqual(0),"xmin must be superior or equal to 0");
    POP_DbgAssertMessage(xmax.allSuperior(xmin),"xmax must be superior to xmin");
    POP_DbgAssertMessage(xmax.allInferior(img.getDomain()+1),"xmax must be superior or equal to xmin");
    _initStride();
    if(  DIM==2 ){
        if(_domain(1)==img.getDomain()(1)){
            if(_domain(0)==img.getDomain()(0))
                std::copy(img.begin(),img.end(),this->begin());
            else
                std::copy(img.begin()+ xmin(0)*img._domain(1),img.begin()+xmax(0)*img._domain(1),this->begin());
        }else{

            typename MatN<Dim,PixelType>::const_iterator itb = img.begin() + xmin(1)+xmin(0)*img._domain(1);
            typename MatN<Dim,PixelType>::const_iterator ite = img.begin() + xmax(1)+xmin(0)*img._domain(1);
            typename MatN<Dim,PixelType>::iterator it = this->begin();
            for( int i=xmin(0);i<xmax(0);i++){
                std::copy(itb,ite,it);
                itb+=img._domain(1);
                ite+=img._domain(1);
                it+=this->_domain(1);
            }
        }
    }
    else if(DIM==3){

        if(_domain(1)==img.getDomain()(1)){
            if(_domain(0)==img.getDomain()(0))
                std::copy(img.begin()+xmin(2)*img._domain(1)*img._domain(0),img.begin()+xmax(2)*img._domain(1)*img._domain(0),this->begin());
            else{
                int intra_slice_add                = img._domain(1)*img._domain(0);
                int intra_slice_add_this           = this->_domain(1)*this->_domain(0);
                typename MatN<Dim,PixelType>::const_iterator itb = img.begin() + xmin(0)*img._domain(1) + xmin(2)*intra_slice_add;
                typename MatN<Dim,PixelType>::const_iterator ite = img.begin() + xmax(0)*img._domain(1) + xmin(2)*intra_slice_add;
                typename MatN<Dim,PixelType>::iterator it        = this->begin();

                for( int k=xmin(2);k<xmax(2);k++){
                    std::copy(itb,ite,it);
                    itb+=intra_slice_add;
                    ite+=intra_slice_add;
                    it +=intra_slice_add_this;
                }
            }
        }else{

            unsigned int indexmini = xmin(0);
            unsigned int indexmaxi = xmax(0);
            int intra_slice_add      = img._domain(1)*img._domain(0);
            int intra_slice_add_this = this->_domain(1)*this->_domain(0);
            typename MatN<Dim,PixelType>::const_iterator itb = img.begin() + xmin(1) +indexmini*img._domain(1) + xmin(2)*intra_slice_add;
            typename MatN<Dim,PixelType>::const_iterator ite = img.begin() + xmax(1) +indexmini*img._domain(1) + xmin(2)*intra_slice_add;
            typename MatN<Dim,PixelType>::iterator it        = this->begin();
            unsigned int indexmin = xmin(2);
            unsigned int indexmax = xmax(2);
            for(unsigned int i=indexmin;i<indexmax;i++){
                typename MatN<Dim,PixelType>::const_iterator itbb = itb;
                typename MatN<Dim,PixelType>::const_iterator itee = ite;
                typename MatN<Dim,PixelType>::iterator itt =it;
                for(unsigned int j=indexmini;j<indexmaxi;j++){
                    std::copy(itbb ,itee,itt);
                    itbb+=img._domain(1);
                    itee+=img._domain(1);
                    itt+=this->_domain(1);
                }
                itb+=intra_slice_add;
                ite+=intra_slice_add;
                it +=intra_slice_add_this;
            }
        }
    }
    _initStride();
}
template<int Dim, typename PixelType>
typename MatN<Dim,PixelType>::Domain  MatN<Dim,PixelType>::getDomain()
const
{
    return _domain;
}
template<int Dim, typename PixelType>
unsigned int MatN<Dim,PixelType>::sizeI()const{
    return this->getDomain()[0];
}
template<int Dim, typename PixelType>
unsigned int MatN<Dim,PixelType>::rows()const{
    return this->getDomain()[0];
}
template<int Dim, typename PixelType>
unsigned int MatN<Dim,PixelType>::sizeJ()const{
    return this->getDomain()[1];
}
template<int Dim, typename PixelType>
unsigned int MatN<Dim,PixelType>::columns()const{
    return this->getDomain()[1];
}
template<int Dim, typename PixelType>
unsigned int MatN<Dim,PixelType>::sizeK()const{
    POP_DbgAssert(Dim==3);
    return this->getDomain()[2];
}
template<int Dim, typename PixelType>
unsigned int MatN<Dim,PixelType>::depth()const{
    POP_DbgAssert(Dim==3);
    return this->getDomain()[2];
}
template<int Dim, typename PixelType>
bool MatN<Dim,PixelType>::isValid(const E & x)const{
    if(x.allSuperiorEqual(E(0)) && x.allInferior(this->getDomain()))
        return true;
    else
        return false;
}
template<int Dim, typename PixelType>
bool MatN<Dim,PixelType>::isValid(int i,int j)const{
    if(i>=0&&j>=0 && i<static_cast<int>(sizeI())&& j<static_cast<int>(sizeJ()))
        return true;
    else
        return false;
}
template<int Dim, typename PixelType>
bool MatN<Dim,PixelType>::isValid(int i,int j,int k)const{
    if(i>=0&&j>=0&&k>=0 && i<static_cast<int>(sizeI())&& j<static_cast<int>(sizeJ())&&k<static_cast<int>(sizeK()))
        return true;
    else
        return false;
}
template<int Dim, typename PixelType>
void MatN<Dim,PixelType>::resize(unsigned int sizei,unsigned int sizej){
    VecN<Dim,int> d(sizei,sizej);
    resize(d);
}
template<int Dim, typename PixelType>
void MatN<Dim,PixelType>::resize(unsigned int sizei,unsigned int sizej,unsigned int sizek){
    VecN<Dim,int> d(sizei,sizej,sizek);
    resize(d);
}
template<int Dim, typename PixelType>
void MatN<Dim,PixelType>::resize(const VecN<Dim,int> & d){

    if(_is_owner_data==true){
        if(_data!=NULL)
            delete[] _data;
        _domain=d;
        _initStride();
        _data = new PixelType[_domain.multCoordinate()];
    }else{
        std::cerr<<"[ERROR] in MatN::resize, reference structure, you cannot allocate data"<<std::endl;
    }
}
template<int Dim, typename PixelType>
void MatN<Dim,PixelType>::resizeInformation(unsigned int sizei,unsigned int sizej){
    Domain d;
    d(0)=sizei;
    d(1)=sizej;
    resizeInformation(d);
}
template<int Dim, typename PixelType>
void MatN<Dim,PixelType>::resizeInformation(unsigned int sizei,unsigned int sizej,unsigned int sizek){
    Domain d;
    d(0)=sizei;
    d(1)=sizej;
    d(2)=sizek;
    resizeInformation(d);
}
template<int Dim, typename PixelType>
void MatN<Dim,PixelType>::resizeInformation(const VecN<Dim,int>& d){
    if(_is_owner_data==true){
        MatN<Dim,PixelType> temp(*this);
        _domain=d;
        _initStride();
        if(_data!=NULL)
            delete[] _data;
        _data = new PixelType[_domain.multCoordinate()];
        IteratorEDomain it(this->getIteratorEDomain());
        while(it.next()){
            if(temp.isValid(it.x())){
                this->operator ()(it.x())=temp(it.x());
            }else{
                this->operator ()(it.x())=0;
            }
        }
    }else{
        std::cerr<<"[ERROR] in MatN::resizeInformation, reference structure, you cannot allocate data"<<std::endl;
    }
}
template<int Dim, typename PixelType>
bool MatN<Dim,PixelType>::isEmpty()const{
    if(_domain.multCoordinate()==0)
        return true;
    else
        return false;
}
template<int Dim, typename PixelType>
void MatN<Dim,PixelType>::clear(){
    _domain=0;
    if(_is_owner_data==true){
        if(_data!=NULL)
            delete[] _data;
        _data = NULL;
    }
}

template<int Dim, typename PixelType>
PixelType & MatN<Dim,PixelType>::operator ()(const VecN<Dim,int> & x)
{
    POP_DbgAssert( x.allSuperiorEqual( E(0))&&x.allInferior(getDomain()));
    return  this->_data[VecNIndice<Dim>::VecN2Indice(_stride,x)];
}

template<int Dim, typename PixelType>
const PixelType & MatN<Dim,PixelType>::operator ()( const VecN<Dim,int>& x)
const
{
    POP_DbgAssert( x.allSuperiorEqual(E(0))&&x.allInferior(getDomain()));
    return  this->_data[VecNIndice<Dim>::VecN2Indice(_stride,x)];
}
template<int Dim, typename PixelType>
PixelType & MatN<Dim,PixelType>::operator ()(unsigned int i,unsigned int j)
{
    POP_DbgAssert( i<(sizeI())&&j<(sizeJ()));
    return  this->_data[i*_stride[0]+j*_stride[1]];
}


template<int Dim, typename PixelType>
const PixelType & MatN<Dim,PixelType>::operator ()(unsigned int i,unsigned int j)const
{
    POP_DbgAssert(i>=0&&j>=0&&i<(sizeI())&&j<(sizeJ()));
    return  this->_data[i*_stride[0]+j*_stride[1]];
}
template<int Dim, typename PixelType>
PixelType & MatN<Dim,PixelType>::operator ()(unsigned int i,unsigned int j,unsigned int k)
{
    POP_DbgAssert(i>=0&&j>=0&&i<(sizeI())&&j<(sizeJ())&&k<(sizeK()));
    return  this->_data[i*_stride[0]+j*_stride[1]+k*_stride[2]];
}

template<int Dim, typename PixelType>
const PixelType & MatN<Dim,PixelType>::operator ()(unsigned int i,unsigned int j,unsigned int k)const
{
    POP_DbgAssert(i>=0&&j>=0&&i<(sizeI())&&j<(sizeJ())&&k<(sizeK()));
    return  this->_data[i*_stride[0]+j*_stride[1]+k*_stride[2]];
}

template<int Dim, typename PixelType>
MatN<Dim,PixelType> MatN<Dim,PixelType>::operator()(const VecN<Dim,int> & xmin, const VecN<Dim,int> & xmax) const{
    return MatN(*this,xmin,xmax);
}
template<int Dim, typename PixelType>
PixelType &MatN<Dim,PixelType>::operator[](int index)
{ return this->_data[index]; }

template<int Dim, typename PixelType>
const PixelType &MatN<Dim,PixelType>::operator[](int index) const
{ return this->_data[index]; }

template<int Dim, typename PixelType>
PixelType & MatN<Dim,PixelType>::operator ()(unsigned int index)
{
    POP_DbgAssert( index<this->size());
    return this->operator[](index);
}
template<int Dim, typename PixelType>
const PixelType & MatN<Dim,PixelType>::operator ()(unsigned int index)const
{
    POP_DbgAssert( index<this->size());
    return this->operator[](index);
}
template<int Dim, typename PixelType>
PixelType MatN<Dim,PixelType>::interpolationBilinear(const VecN<DIM,F32> xf)const
{

    return MatNInterpolationBiliniear::apply(*this,xf);
}
template<int Dim, typename PixelType>
PixelType *  MatN<Dim,PixelType>::data()
{
    return &(*this->begin());
}
template<int Dim, typename PixelType>
const PixelType *  MatN<Dim,PixelType>::data()
const
{
    return &(*this->begin());
}
template<int Dim, typename PixelType>
bool MatN<Dim,PixelType>::load(const std::string file) {
    return this->load(file.c_str());
}
template<int Dim, typename PixelType>
void MatN<Dim,PixelType>::save(const std::string file)const {
    save(file.c_str());
}
template<int Dim, typename PixelType>
typename MatN<Dim,PixelType>::IteratorEDomain MatN<Dim,PixelType>::getIteratorEDomain()const
{
    return IteratorEDomain(getDomain());
}
template<int Dim, typename PixelType>
typename MatN<Dim,PixelType>::IteratorEROI MatN<Dim,PixelType>::getIteratorEROI()const
{
    return IteratorEROI(*this);
}
template<int Dim, typename PixelType>
typename MatN<Dim,PixelType>::IteratorENeighborhood MatN<Dim,PixelType>::getIteratorENeighborhood(F32 radius ,int norm )const
{
    return IteratorENeighborhood(getDomain(),radius , norm);
}
template<int Dim, typename PixelType>
template<typename Type1>
typename MatN<Dim,PixelType>::IteratorENeighborhood MatN<Dim,PixelType>::getIteratorENeighborhood(const MatN<Dim,Type1> & structural_element,int dilate )const
{
    Vec<E> _tab;
    typename MatN<Dim,Type1>::IteratorEDomain it(structural_element.getDomain());
    typename MatN<Dim,Type1>::E center = VecN<Dim,F32>(structural_element.getDomain()-1)*0.5;
    while(it.next()){
        if(normValue(structural_element(it.x()))!=0){
            _tab.push_back(it.x()-center);
        }
    }
    if(dilate<=1)
        return IteratorENeighborhood(std::make_pair(getDomain(),_tab));
    else{
        IteratorENeighborhood itinit(std::make_pair(getDomain(),_tab));
        IteratorENeighborhood ititerative(std::make_pair(getDomain(),_tab));
        for(int i =1;i<dilate;i++){
            ititerative.dilate(itinit);
        }
        return IteratorENeighborhood(ititerative.getDomain());
    }
}
template<int Dim, typename PixelType>
typename MatN<Dim,PixelType>::IteratorEOrder MatN<Dim,PixelType>::getIteratorEOrder(int coordinatelastloop,int direction)const
{
    return IteratorEOrder(getDomain(),coordinatelastloop,direction);
}
template<int Dim, typename PixelType>
typename MatN<Dim,PixelType>::IteratorERectangle MatN<Dim,PixelType>::getIteratorERectangle(const E & xmin,const E & xmax )const
{
    return IteratorERectangle(std::make_pair(xmin,xmax));
}
template<int Dim, typename PixelType>
typename MatN<Dim,PixelType>::IteratorENeighborhoodAmoebas MatN<Dim,PixelType>::getIteratorENeighborhoodAmoebas(F32 distance_max,F32 lambda_param )const
{
    return IteratorENeighborhoodAmoebas(*this,distance_max,lambda_param );
}

template<int Dim, typename PixelType>
template<class T1>
MatN<Dim,PixelType> & MatN<Dim,PixelType>::operator =(const MatN<Dim, T1> & img ){

    if(img.isOwnerData()==false){
        std::cerr<<"[ERROR] MatN::operator=, cannot copy MatN if PixelType are different and img is not the data owner"<<std::endl;
    }else{
        this->resize(img.getDomain());
        std::transform(img.begin(),img.end(),this->begin(),ArithmeticsSaturation<PixelType,T1>::Range);
    }



    return *this;
}
template<int Dim, typename PixelType>
MatN<Dim,PixelType> & MatN<Dim,PixelType>::operator =(const MatN<Dim,PixelType> & img ){
    if(img.isOwnerData()==false){
        if(this->_is_owner_data ==true&&_data!=NULL)
            delete[] _data;
        this->_is_owner_data = img.isOwnerData();
        this->_data  = const_cast<PixelType*>(img.data());
        this->_domain  = img.getDomain();
        this->_stride = img.stride();
    }else{
        this->resize(img.getDomain());
        std::copy(img.begin(),img.end(),this->begin());
    }
    return *this;
}
template<int Dim, typename PixelType>
MatN<Dim, PixelType>&  MatN<Dim,PixelType>::operator=(PixelType value)
{
    std::fill (this->begin(),this->end(),value);
    return *this;
}
template<int Dim, typename PixelType>
MatN<Dim, PixelType>&  MatN<Dim,PixelType>::fill(PixelType value)
{
    std::fill (this->begin(),this->end(),value);
    return *this;
}
template<int Dim, typename PixelType>
MatN<Dim, PixelType>  MatN<Dim,PixelType>::opposite(int mode)const
{
    MatN<Dim, PixelType> temp;
    PixelType maxi;
    if(mode==0)
        maxi=NumericLimits<PixelType>::maximumRange();
    else{
        FunctorF::FunctorAccumulatorMax<PixelType > func;
        func = std::for_each (this->begin(), this->end(), func);
        maxi=func.getValue();
    }
    temp=maxi-*this;
    return temp;
}

template<int Dim, typename PixelType>
bool MatN<Dim,PixelType>::operator==(const MatN<Dim, PixelType>& f)const
{
    FunctionAssert(f,*this,"In MatN::operator==");
    return std::equal (f.begin(), f.end(), this->begin());
}
template<int Dim, typename PixelType>
bool MatN<Dim,PixelType>::operator!=(const MatN<Dim, PixelType>& f)const
{
    FunctionAssert(f,*this,"In MatN::operator==");
    return !std::equal (f.begin(), f.end(), this->begin());
}
template<int Dim, typename PixelType>
MatN<Dim, PixelType>&  MatN<Dim,PixelType>::operator+=(const MatN<Dim, PixelType>& f)
{

    FunctionAssert(f,*this,"In MatN::operator+=");
    FunctorF::FunctorAdditionF2<PixelType,PixelType,PixelType> op;
    std::transform (this->begin(), this->end(), f.begin(),this->begin(),  op);
    return *this;
}
template<int Dim, typename PixelType>
MatN<Dim, PixelType>  MatN<Dim,PixelType>::operator+(const MatN<Dim, PixelType>& f)const{
    MatN<Dim, PixelType> h(*this);
    h +=f;
    return h;
}
template<int Dim, typename PixelType>
MatN<Dim, PixelType>& MatN<Dim,PixelType>::operator+=(PixelType value)
{
    FunctorF::FunctorArithmeticConstantValueAfter<PixelType,PixelType,PixelType,FunctorF::FunctorAdditionF2<PixelType,PixelType,PixelType> > op(value);
    std::transform (this->begin(), this->end(), this->begin(),  op);
    return *this;
}
template<int Dim, typename PixelType>
MatN<Dim, PixelType>  MatN<Dim,PixelType>::operator+(PixelType value)const{
    MatN<Dim, PixelType> h(*this);
    h +=value;
    return h;
}
template<int Dim, typename PixelType>
MatN<Dim, PixelType>&  MatN<Dim,PixelType>::operator-=(const MatN<Dim, PixelType>& f)
{
    FunctionAssert(f,*this,"In MatN::operator-=");
    FunctorF::FunctorSubtractionF2<PixelType,PixelType,PixelType> op;
    std::transform (this->begin(), this->end(), f.begin(),this->begin(),  op);
    return *this;
}
template<int Dim, typename PixelType>
MatN<Dim, PixelType>&  MatN<Dim,PixelType>::operator-=(PixelType value)
{
    FunctorF::FunctorArithmeticConstantValueAfter<PixelType,PixelType,PixelType,FunctorF::FunctorSubtractionF2<PixelType,PixelType,PixelType> > op(value);
    std::transform (this->begin(), this->end(), this->begin(),  op);
    return *this;
}

template<int Dim, typename PixelType>
MatN<Dim, PixelType>  MatN<Dim,PixelType>::operator-(const MatN<Dim, PixelType>& f)const{
    MatN<Dim, PixelType> h(*this);
    h -=f;
    return h;
}
template<int Dim, typename PixelType>
MatN<Dim, PixelType>  MatN<Dim,PixelType>::operator-()const{
    MatN<Dim, PixelType> h(this->getDomain(),PixelType(0));
    h -=*this;
    return h;
}
template<int Dim, typename PixelType>
MatN<Dim, PixelType>  MatN<Dim,PixelType>::operator-(PixelType value)const{
    MatN<Dim, PixelType> h(*this);
    h -=value;
    return h;
}

template<int Dim, typename PixelType>
MatN<Dim,PixelType>  MatN<Dim,PixelType>::operator*(const MatN<Dim,PixelType> &m)const
{
    POP_DbgAssertMessage(DIM==2&&this->sizeJ()==m.sizeI() ,"In Matrix::operator*, Not compatible size for the operator * of the class Matrix (A_{n,k}*B_{k,p})");
    MatN<Dim,PixelType> mtrans = m.transpose();
    MatN<Dim,PixelType> mout(this->sizeI(),m.sizeJ());
    for( int i=0;i<static_cast<int>(this->sizeI());i++){
        for(unsigned  j=0;j<(m.sizeJ());j++){
            PixelType sum = 0;
            typename MatN<Dim,PixelType>::const_iterator this_it  = this->begin() +  i*this->sizeJ();
            typename MatN<Dim,PixelType>::const_iterator mtrans_it= mtrans.begin() + j*mtrans.sizeJ();
            for(unsigned int k=0;k<this->sizeJ();k++){
                sum+=(* this_it) * (* mtrans_it);
                this_it++;
                mtrans_it++;
            }
            mout(i,j)=sum;
        }
    }
    return mout;
}
template<int Dim, typename PixelType>
MatN<Dim,PixelType> & MatN<Dim,PixelType>::operator*=(const MatN<Dim,PixelType> &m)
{
    *this = this->operator *(m);
    return *this;
}
template<int Dim, typename PixelType>
Vec<PixelType>  MatN<Dim,PixelType>::operator*(const Vec<PixelType> & v)const{
    POP_DbgAssertMessage(DIM==2&&this->sizeJ()==v.size() ,"In Matrix::operator*, Not compatible size for the operator *=(Vec) of the class Matrix (A_{n,k}*v_{k})");
    Vec<PixelType> temp(this->sizeI());
    for(unsigned int i=0;i<this->sizeI();i++){
        PixelType sum = 0;
        typename MatN::const_iterator this_it  = this->begin() +  i*this->sizeJ();
        typename Vec<PixelType>::const_iterator mtrans_it= v.begin();
        for(;mtrans_it!=v.end();this_it++,mtrans_it++){
            sum+=(* this_it) * (* mtrans_it);
        }
        temp(i)=sum;
    }
    return temp;
}

template<int Dim, typename PixelType>
MatN<Dim,PixelType>  MatN<Dim,PixelType>::multTermByTerm(const MatN& f)const{
    FunctionAssert(f,*this,"In MatN::operator*=");
    FunctorF::FunctorMultiplicationF2<PixelType,PixelType,PixelType> op;
    MatN<Dim,PixelType> out(*this);
    std::transform (out.begin(), out.end(), f.begin(),out.begin(),  op);
    return out;
}
template<int Dim, typename PixelType>
MatN<Dim, PixelType>&  MatN<Dim,PixelType>::operator*=(PixelType  value)
{
    FunctorF::FunctorArithmeticConstantValueAfter<PixelType,PixelType,PixelType,FunctorF::FunctorMultiplicationF2<PixelType,PixelType,PixelType> > op(value);
    std::transform (this->begin(), this->end(), this->begin(),  op);
    return *this;
}
template<int Dim, typename PixelType>
MatN<Dim, PixelType>  MatN<Dim,PixelType>::operator*(PixelType value)const{
    MatN<Dim, PixelType> h(*this);
    h *=value;
    return h;
}
template<int Dim, typename PixelType>
MatN<Dim, PixelType>  MatN<Dim,PixelType>::divTermByTerm(const MatN& f){
    FunctionAssert(f,*this,"In MatN::divTermByTerm");
    FunctorF::FunctorDivisionF2<PixelType,PixelType,PixelType> op;
    std::transform (this->begin(), this->end(), f.begin(),this->begin(),  op);
    return *this;
}
template<int Dim, typename PixelType>
MatN<Dim, PixelType>&  MatN<Dim,PixelType>::operator/=(PixelType value)
{
    FunctorF::FunctorArithmeticConstantValueAfter<PixelType,PixelType,PixelType,FunctorF::FunctorDivisionF2<PixelType,PixelType,PixelType> > op(value);
    std::transform (this->begin(), this->end(), this->begin(),  op);
    return *this;
}
template<int Dim, typename PixelType>
MatN<Dim, PixelType>  MatN<Dim,PixelType>::operator/(PixelType value)const{
    MatN<Dim, PixelType> h(*this);
    h /=value;
    return h;
}



template<int DIM, typename PixelType>
MatN<DIM,PixelType>  MatN<DIM,PixelType>::deleteRow(unsigned int i)const{
    POP_DbgAssert(i<sizeI());
    MatN<DIM,PixelType> temp(this->sizeI()-1,this->sizeJ());
    for(unsigned int i1=0;i1<temp.sizeI();i1++)
        for(unsigned int j=0;j<temp.sizeJ();j++)
        {
            if(i1<i)
                temp(i1,j)=this->operator ()(i1,j);
            else
                temp(i1,j)=this->operator ()(i1+1,j);
        }
    return temp;
}
template<int DIM, typename PixelType>
MatN<DIM,PixelType>  MatN<DIM,PixelType>::deleteCol(unsigned int j)const{
    POP_DbgAssert(j<sizeJ());
    MatN<DIM,PixelType> temp(this->sizeI(),this->sizeJ()-1);

    for(unsigned int i=0;i<temp.sizeI();i++)
        for(unsigned int j1=0;j1<temp.sizeJ();j1++)
        {
            if(j1<j)
                temp(i,j1)=this->operator ()(i,j1);
            else
                temp(i,j1)=this->operator ()(i,j1+1);
        }
    return temp;
}

template<int DIM, typename PixelType>
Vec<PixelType>  MatN<DIM,PixelType>::getRow(unsigned int i)const{
    Vec<PixelType> v(this->sizeJ());
    std::copy(this->begin()+i*this->_domain(1), this->begin()+(i+1)*this->_domain(1),v.begin());
    return v;
}
template<int DIM, typename PixelType>
Vec<PixelType>  MatN<DIM,PixelType>::getCol(unsigned int j)const{
    Vec<PixelType> v(this->sizeI());
    for(unsigned int i=0;i<this->sizeI();i++){
        v(i)=this->operator ()(i,j);
    }
    return v;
}
template<int DIM, typename PixelType>
void  MatN<DIM,PixelType>::setRow(unsigned int i,const Vec<PixelType> &v){

    POP_DbgAssertMessage(v.size()==this->sizeJ(),"In Matrix::setRow, incompatible size");
    std::copy(v.begin(),v.end(),this->begin()+i*this->_domain(1));
}
template<int DIM, typename PixelType>
void  MatN<DIM,PixelType>::setCol(unsigned int j,const Vec<PixelType>& v){
    POP_DbgAssertMessage(v.size()==this->sizeI(),"In Matrix::setCol, Incompatible size");
    for(unsigned int i=0;i<this->sizeI();i++){
        this->operator ()(i,j)=v(i);
    }
}
template<int DIM, typename PixelType>
void  MatN<DIM,PixelType>::swapRow(unsigned int i_0,unsigned int i_1){
    POP_DbgAssertMessage( (i_0<this->sizeI()&&i_1<this->sizeI()),"In Matrix::swapRow, Over Range in swapRow");
    std::swap_ranges(this->begin()+i_0*this->sizeJ(), this->begin()+(i_0+1)*this->sizeJ(), this->begin()+i_1*this->sizeJ());

}
template<int DIM, typename PixelType>
void  MatN<DIM,PixelType>::swapCol(unsigned int j_0,unsigned int j_1){
    POP_DbgAssertMessage( (j_0<this->sizeJ()&&j_1<this->sizeJ()),"In Matrix::swapCol, Over Range in swapCol");
    for(unsigned int i=0;i<this->sizeI();i++){
        std::swap(this->operator ()(i,j_0) ,this->operator ()(i,j_1));
    }
}
template<int DIM, typename PixelType>
PixelType MatN<DIM,PixelType>::minorDet(unsigned int i,unsigned int j)const{

    return this->deleteRow(i).deleteCol(j).determinant();
}
template<int DIM, typename PixelType>
PixelType MatN<DIM,PixelType>::cofactor(unsigned int i,unsigned int j)const{
    if( (i+j)%2==0)
        return this->minorDet(i,j);
    else
        return -this->minorDet(i,j);
}
template<int DIM, typename PixelType>
MatN<DIM,PixelType>  MatN<DIM,PixelType>::cofactor()const{
    MatN<DIM,PixelType> temp(this->getDomain());
    for(unsigned int i=0;i<this->sizeI();i++)
        for(unsigned int j=0;j<this->sizeJ();j++)
        {
            temp(i,j)=this->cofactor(i,j);
        }
    return temp;
}
template<int DIM, typename PixelType>
MatN<DIM,PixelType>  MatN<DIM,PixelType>::transpose()const
{
    const unsigned int sizei= this->sizeI();
    const unsigned int sizej= this->sizeJ();
    MatN<DIM,PixelType> temp(sizej,sizei);
    for(unsigned int i=0;i<sizei;i++){
        typename  MatN<DIM,PixelType>::const_iterator this_ptr  =  this->begin() + i*sizej;
        typename  MatN<DIM,PixelType>::const_iterator this_end_ptr  =  this_ptr + sizej;
        typename  MatN<DIM,PixelType>::iterator temp_ptr =     temp.begin() + i;
        while(this_ptr!=this_end_ptr){
            * temp_ptr =  * this_ptr;
            temp_ptr   +=  sizei;
            this_ptr++;
        }
    }
    return temp;
}
template<int DIM, typename PixelType>
PixelType MatN<DIM,PixelType>::determinant() const{
    if(this->sizeI()==1)
        return this->operator ()(0,0);
    else
    {
        F det=0;
        for(unsigned int i=0;i<this->sizeI();i++)
        {
            det +=(this->operator ()(i,0)*this->cofactor(i,0));
        }
        return det;
    }

}
template<int DIM, typename PixelType>
PixelType MatN<DIM,PixelType>::trace() const
{
    POP_DbgAssertMessage(this->sizeI()==this->sizeJ(),"In  MatN<DIM,PixelType>::trace, Input  MatN<DIM,PixelType> must be square");

    F t=0;
    for(unsigned int i=0;i<this->sizeI();i++)
    {
        t +=this->operator ()(i,i);
    }
    return t;


}
template<int DIM, typename PixelType>
MatN<DIM,PixelType>  MatN<DIM,PixelType>::identity(int size_mat)const{
    if(size_mat==0)
        size_mat=this->sizeI();
    MatN<DIM,PixelType> I(size_mat,size_mat);
    for(unsigned int i=0;i<I.sizeI();i++){
        I(i,i)=1;
    }
    return I;
}

template<int DIM, typename PixelType>
MatN<DIM,PixelType>  MatN<DIM,PixelType>::inverse()const{
    if(sizeI()==2&&sizeJ()==2){
        MatN<DIM,PixelType> temp(*this);
        const PixelType det= PixelType(1)/ (temp.operator[](0) * temp.operator[](3) - temp.operator[](1) * temp.operator[](2)) ;
                std::swap(temp.operator[](0),temp.operator[](3));
                temp.operator[](1)=-temp.operator[](1)*det;
        temp.operator[](2)=-temp.operator[](2)*det;
        temp.operator[](0)*=det;
        temp.operator[](3)*=det;
        return temp;
    }else if(sizeI()==3&&sizeJ()==3){
        MatN<DIM,PixelType > temp(*this);
        const PixelType det= PixelType(1)/(temp.operator[](0) * (temp.operator[](4)*temp.operator[](8) - temp.operator[](7) * temp.operator[](5))-temp.operator[](1) * (temp.operator[](3)*temp.operator[](8) - temp.operator[](6) * temp.operator[](5)) +temp.operator[](2) * (temp.operator[](3)*temp.operator[](7) - temp.operator[](4) * temp.operator[](6)));
                                                                                                                                                                        const PixelType t0=  temp.operator[](4)*temp.operator[](8)-temp.operator[](7)*temp.operator[](5);
                                                                 const PixelType t1=-(temp.operator[](3)*temp.operator[](8)-temp.operator[](6)*temp.operator[](5));
                const PixelType t2=  temp.operator[](3)*temp.operator[](7)-temp.operator[](6)*temp.operator[](4);
                                           const PixelType t3=-(temp.operator[](1)*temp.operator[](8)-temp.operator[](7)*temp.operator[](2));
                const PixelType t4= temp.operator[](0)*temp.operator[](8)-temp.operator[](6)*temp.operator[](2);
                const PixelType t5=-(temp.operator[](0)*temp.operator[](7)-temp.operator[](6)*temp.operator[](1));
                const PixelType t6= temp.operator[](1)*temp.operator[](5)-temp.operator[](4)*temp.operator[](2);
        const PixelType t7=-(temp.operator[](0)*temp.operator[](5)-temp.operator[](3)*temp.operator[](2));
                const PixelType t8= temp.operator[](0)*temp.operator[](4)-temp.operator[](3)*temp.operator[](1);
        temp.operator[](0)=t0;
        temp.operator[](1)=t1;
        temp.operator[](2)=t2;
        temp.operator[](3)=t3;
        temp.operator[](4)=t4;
        temp.operator[](5)=t5;
        temp.operator[](6)=t6;
        temp.operator[](7)=t7;
        temp.operator[](8)=t8;
        std::swap(temp.operator[](1),temp.operator[](3)); std::swap(temp.operator[](2),temp.operator[](6));std::swap(temp.operator[](5),temp.operator[](7));
                temp.operator[](0)*=det;
        temp.operator[](1)*=det;
        temp.operator[](2)*=det;
        temp.operator[](3)*=det;
        temp.operator[](4)*=det;
        temp.operator[](5)*=det;
        temp.operator[](6)*=det;
        temp.operator[](7)*=det;
        temp.operator[](8)*=det;
        return temp;
    }
    else
    {
        MatN<DIM,PixelType> temp;
        PixelType det = this->determinant();
        temp = this->cofactor();
        temp = temp.transpose();
        temp/=det;
        return temp;
    }
}

template<int Dim1,int Dim2, typename PixelType>
MatN<Dim1+Dim2, PixelType>  productTensoriel(const MatN<Dim1, PixelType>&f,const MatN<Dim2, PixelType>& g)
{
    typename MatN<Dim1+Dim2, PixelType>::E domain;
    for(int i=0;i<Dim1;i++)
    {
        domain(i)=f.getDomain()(i);
    }
    for(int i=0;i<Dim2;i++)
    {
        domain(i+Dim1)=g.getDomain()(i);
    }
    MatN<Dim1+Dim2, PixelType> h(domain);
    typename MatN<Dim1+Dim2, PixelType>::IteratorEDomain it(h.getDomain());

    typename MatN<Dim1, PixelType>::E x1;
    typename MatN<Dim2, PixelType>::E x2;
    while(it.next())
    {
        for(int i=0;i<Dim1;i++)
        {
            x1(i)=it.x()(i);
        }
        for(int i=0;i<Dim2;i++)
        {
            x2(i)=it.x()(Dim1+i);
        }
        h(it.x())=f(x1)*g(x2);
    }
    return h;
}
template<int Dim, typename PixelType>
MatN<Dim, PixelType>  operator*(PixelType value, const MatN<Dim, PixelType>&f)
{
    return f*value;
}
template<int Dim, typename PixelType>
MatN<Dim, PixelType>  operator-(PixelType value, const MatN<Dim, PixelType>&f)
{
    MatN<Dim, PixelType> h(f);
    FunctorF::FunctorArithmeticConstantValueBefore<PixelType,PixelType,PixelType,FunctorF::FunctorSubtractionF2<PixelType,PixelType,PixelType> > op(value);
    std::transform (h.begin(), h.end(), h.begin(),  op);
    return h;
}
template<int Dim, typename PixelType>
MatN<Dim, PixelType>  operator+(PixelType value, const MatN<Dim, PixelType>&f)
{
    MatN<Dim, PixelType> h(f);
    FunctorF::FunctorArithmeticConstantValueBefore<PixelType,PixelType,PixelType,FunctorF::FunctorAdditionF2<PixelType,PixelType,PixelType> > op(value);
    std::transform (h.begin(), h.end(), h.begin(),  op);
    return h;
}
template<int D,typename F1, typename F2>
struct FunctionTypeTraitsSubstituteF<MatN<D,F1>,F2 >
{
    typedef MatN<D,F2> Result;
};
template<int D1,typename F1, int D2>
struct FunctionTypeTraitsSubstituteDIM<MatN<D1,F1>,D2 >
{
    typedef MatN<D2,F1> Result;
};


#define ForEachDomain2D(x,img) \
    pop::Vec2I32 x; \
    for( x(0)=0;x(0)<img.getDomain()(0);x(0)++)\
    for( x(1)=0;x(1)<img.getDomain()(1);x(1)++)



#define ForEachDomain3D(x,img) \
    pop::Vec3I32 x; \
    for( x(2)=0;x(2)<img.getDomain()(2);x(2)++) \
    for( x(0)=0;x(0)<img.getDomain()(0);x(0)++) \
    for( x(1)=0;x(1)<img.getDomain()(1);x(1)++)




template<int D1,int D2,typename F1, typename F2>
void FunctionAssert(const MatN<D1,F1> & f,const MatN<D2,F2> & g ,std::string message)
{
    POP_DbgAssertMessage(D1==D2,"matrixs must have the same Dim\n"+message);
    POP_DbgAssertMessage(f.getDomain()==g.getDomain(),"matrixs must have the same domain\n"+message);
}
template<int DIM,typename PixelType>
struct NumericLimits< MatN<DIM,PixelType> >
{
    static F32 minimumRange() throw()
    { return -NumericLimits<PixelType>::maximumRange();}
    static F32 maximumRange() throw()
    { return NumericLimits<PixelType>::maximumRange();}
};

/*!
* \ingroup MatN
* \brief minimum value for each VecN  \f$h(x)=\min(f(x),g(x))\f$
* \param f first input matrix
* \param g first input matrix
* \return output  matrix
*
*/
template<int Dim, typename PixelType>
pop::MatN<Dim, PixelType>  minimum(const pop::MatN<Dim, PixelType>& f,const pop::MatN<Dim, PixelType>& g)
{
    pop::FunctionAssert(f,g,"In min");
    pop::MatN<Dim, PixelType> h(f);
    pop::FunctorF::FunctorMinF2<PixelType,PixelType> op;

    std::transform (h.begin(), h.end(), g.begin(),h.begin(), op );
    return h;
}
/*!
* \ingroup MatN
* \brief maximum value for each VecN  \f$h(x)=\max(f(x),g(x))\f$
* \param f first input matrix
* \param g first input matrix
* \return output  matrix
*
*/
template<int Dim, typename PixelType>
pop::MatN<Dim, PixelType>  maximum(const pop::MatN<Dim, PixelType>& f,const pop::MatN<Dim, PixelType>& g)
{
    pop::FunctionAssert(f,g,"In max");
    pop::MatN<Dim, PixelType> h(f);
    pop::FunctorF::FunctorMaxF2<PixelType,PixelType> op;
    std::transform (h.begin(), h.end(), g.begin(),h.begin(), op );
    return h;
}
/*!
\ingroup MatN
\brief  absolute value for each VecN  \f$h(x)=abs(f(x))\f$
* \param f first input matrix
* \return output  matrix
*
*/
template<int Dim, typename PixelType>
pop::MatN<Dim, PixelType>  absolute(const pop::MatN<Dim, PixelType>& f)
{
    pop::MatN<Dim, PixelType> h(f.getDomain());
    std::transform (f.begin(), f.end(), h.begin(),(PixelType(*)(PixelType)) abs );
    return h;
}
/*!
* \ingroup MatN
* \brief  square value for each pixel value  \f$h(x)=\sqrt{f(x)}\f$
* \param f first input matrix
* \return output  matrix
*
*/
template<int Dim, typename PixelType>
pop::MatN<Dim, PixelType>  squareRoot(const pop::MatN<Dim, PixelType>& f)
{
    pop::MatN<Dim, PixelType> h(f.getDomain());
    std::transform (f.begin(), f.end(), h.begin(), (PixelType(*)(PixelType)) sqrt );
    return h;
}
/*!
* \ingroup MatN
* \brief  log value in e-base  for each VecN  h(x)=std::log(f(x))
* \param f first input matrix
* \return output  matrix
*
*/
template<int Dim, typename PixelType>
pop::MatN<Dim, PixelType>  log(const pop::MatN<Dim, PixelType>& f)
{
    pop::MatN<Dim, PixelType> h(f.getDomain());
    std::transform (f.begin(), f.end(), h.begin(), (PixelType(*)(PixelType)) std::log );
    return h;
}
/*!
* \ingroup MatN
* \brief  log value in 10-base  for each pixel value  h(x)=std::log10(f(x))
* \param f first input matrix
* \return output  matrix
*
*/
template<int Dim, typename PixelType>
pop::MatN<Dim, PixelType>  log10(const pop::MatN<Dim, PixelType>& f)
{
    pop::MatN<Dim, PixelType> h(f.getDomain());
    std::transform (f.begin(), f.end(), h.begin(), (PixelType(*)(PixelType)) std::log10 );
    return h;
}
/*!
* \ingroup MatN
* \brief  exponentiel value for each pixel value  h(x)=std::exp(f(x))
* \param f first input matrix
* \return output  matrix
*
*/
template<int Dim, typename PixelType>
pop::MatN<Dim, PixelType>  exp(const pop::MatN<Dim, PixelType>& f)
{
    pop::MatN<Dim, PixelType> h(f.getDomain());
    std::transform (f.begin(), f.end(), h.begin(), (PixelType(*)(PixelType)) std::exp );
    return h;
}


/*!
* \ingroup MatN
* \brief  pow value for each pixel value  h(x)=std::pow(f(x),exponant)
* \param f first input matrix
* \param exponant exponant
* \return output  matrix
*
*/
template<int Dim, typename PixelType>
pop::MatN<Dim, PixelType>  pow(const pop::MatN<Dim, PixelType>& f,F32 exponant)
{
    pop::MatN<Dim, PixelType> h(f.getDomain());
    pop::Private::PowF<PixelType> op(exponant);
    std::transform (f.begin(), f.end(), h.begin(), op );
    return h;
}


/*!
* \ingroup MatN
* \brief "Entrywise" norms  normValue(A,p) =\f$ \Vert A \Vert_{p} = \left( \sum_{i=1}^m \sum_{j=1}^n  | a_{ij} |^p \right)^{1/p}.  \f$
* \param A  input matrix
* \param p p-norm
* \return scalar value
*
*
*/
template<int Dim, typename PixelType>
F32  normValue(const pop::MatN<Dim, PixelType>& A,int p=2)
{
    pop::Private::sumNorm<PixelType> op(p);
    if(p!=0)
        return std::pow(std::accumulate(A.begin(),A.end(),0.,op),1./p);
    else
        return std::accumulate(A.begin(),A.end(),0.,op);

}

/*!
* \ingroup MatN
* \brief distance between two vectors \f$\vert u-v \vert^p\f$
* \param A  first matrix
* \param B second matrix
* \param p  p-norm
* \return output scalar value
*
*/
template<int Dim, typename PixelType>
F32 distance(const pop::MatN<Dim, PixelType>& A, const pop::MatN<Dim, PixelType>& B,int p=2)
{
    return normValue(A-B,p);
}

/*!
* \ingroup MatN
* \brief "Entrywise" norms  normPowerValue(A,p) =\f$ \Vert A \Vert_{p}^p = \sum_{i=1}^m \sum_{j=1}^n  | a_{ij} |^p   \f$
* \param f  input matrix
* \param p p-norm
* \return scalar value
*
*
*/
template<int Dim, typename PixelType>
F32  normPowerValue(const pop::MatN<Dim, PixelType>& f,int p=2)
{
    pop::Private::sumNorm<PixelType> op(p);
    return std::accumulate(f.begin(),f.end(),0.,op);
}

namespace Private {
template<int DIM,typename PixelType>
struct ConsoleOutputPixel
{
    void  print(std::ostream& out,  PixelType v){
        out<<v;
    }
};

template<int DIM>
struct ConsoleOutputPixel<DIM,unsigned char>
{
    void  print(std::ostream& out,  unsigned char v){
        out<<(int)v;
    }
};

template<int DIM,typename PixelType>
struct ConsoleInputPixel
{
    void  print(std::istream& in,  PixelType &v){
        in>>v;
    }
};

template<int DIM>
struct ConsoleInputPixel<DIM,unsigned char>
{
    void  print(std::istream& in,  unsigned char& v){
        int value;
        in>>value;
        v= static_cast<unsigned char>(value);
    }
};
}


/*!
* \ingroup MatN
* \param out output stream
* \param in input matrix
* \return output stream
*
*  stream extraction  of the  Vec
*/


template <class PixelType>
std::ostream& operator << (std::ostream& out, const pop::MatN<1,PixelType>& in)
{
    Private::ConsoleOutputPixel<1,PixelType> output;
    for( int i =0;i<in.getDomain()(0);i++){
        output.print(out,(in)(i));
        out<<" ";
    }
    return out;
}

template <class PixelType>
std::ostream& operator << (std::ostream& out, const pop::MatN<2,PixelType>& in)
{
    Private::ConsoleOutputPixel<2,PixelType> output;
    for( int i =0;i<in.getDomain()(0);i++){
        for( int j =0;j<in.getDomain()(1);j++){
            output.print(out,(in)(i,j));
            out<<" ";
        }
        out<<std::endl;
    }
    return out;
}
template <class PixelType>
std::ostream& operator << (std::ostream& out, const pop::MatN<3,PixelType>& in)
{
    Private::ConsoleOutputPixel<3,PixelType> output;
    for( int k =0;k<in.getDomain()(2);k++){
        for( int i =0;i<in.getDomain()(0);i++){
            for( int j =0;j<in.getDomain()(1);j++){
                output.print(out,in(i,j,k));
                out<<" ";
            }
            out<<std::endl;
        }
        out<<std::endl;
    }
    return out;
}
/*!
* \ingroup MatN
* \param in input stream
* \param f ouput matrix
* \return input stream
*
*  stream insertion of the  Vec
*/
template <int Dim,class PixelType>
std::istream& operator >> (std::istream& in,  pop::MatN<Dim,PixelType>& f)
{
    typename pop::MatN<Dim,pop::UI8>::IteratorEOrder it(f.getIteratorEOrder());
    typename pop::MatN<Dim,pop::UI8>::Domain d;
    for(int i=0;i<Dim;i++)
        d(i)=i;
    std::swap(d(0),d(1));
    it.setOrder(d);
    Private::ConsoleInputPixel<Dim,PixelType> input;
    while(it.next())
    {
        input.print(in,f(it.x()));
    }
    return in;
}

/*!
* \ingroup Vec
* \brief  tensorial product of two vector
* \param v1 first vector
* \param v2 second vector
* \return output matrix
*
*
*/
template<typename T1>
pop::MatN<2,T1>  productTensoriel(const pop::Vec<T1>& v1,const pop::Vec<T1>& v2)
{
    POP_DbgAssert( v1.size()==v2.size());

    pop::MatN<2,T1> m(v1.size(),v1.size());
    for(unsigned int i=0;i<v1.size();i++)
        for(unsigned int j=0;j<v1.size();j++)
            m(i,j)=productInner(v1(i),v2(j));
    return m;
}





template<typename Type1,typename Type2,typename FunctorAccumulatorF,typename IteratorEGlobal,typename IteratorELocal>
void forEachGlobalToLocal(const MatN<2,Type1> & f, MatN<2,Type2> &  h, FunctorAccumulatorF facc,IteratorELocal  it_local,typename MatN<2,Type1>::IteratorEDomain ){

    Vec2I32 x;
    for(x(0)=0;x(0)<f.sizeI();x(0)++){
        for(x(1)=0;x(1)<f.sizeJ();x(1)++){
            it_local.init(x);
            h(x)=forEachFunctorAccumulator(f,facc,it_local);
        }
    }

}
template<typename Type1,typename Type2,typename FunctorAccumulatorF,typename IteratorEGlobal,typename IteratorELocal>
void forEachGlobalToLocal(const MatN<3,Type1> & f, MatN<3,Type2> &  h, FunctorAccumulatorF facc,IteratorELocal  it_local,typename MatN<3,Type1>::IteratorEDomain ){
    Vec3I32 x;
    for(x(0)=0;x(0)<f.sizeI();x(0)++){
        for(x(1)=0;x(1)<f.sizeJ();x(1)++){
            for(x(2)=0;x(2)<f.sizeK();x(2)++){
                it_local.init(x);
                h(x)=forEachFunctorAccumulator(f,facc,it_local);
            }
        }
    }
}
template<typename Type1,typename Type2,typename FunctorBinaryFunctionE>
void forEachFunctorBinaryFunctionE(const MatN<2,Type1> & f, MatN<2,Type2> &  h,  FunctorBinaryFunctionE func, typename MatN<2,Type1>::IteratorEDomain it)
{
    int i_max,j_max;
    if(it.getDomain()==f.getDomain()){
        i_max= f.sizeI();
        j_max= f.sizeJ();
    }else{
        i_max= h.sizeI();
        j_max= h.sizeJ();
    }

    Vec2I32 x;
    for(x(0)=0;x(0)<i_max;x(0)++){
        for(x(1)=0;x(1)<j_max;x(1)++){
            h(x)=func( f, x);
        }
    }
}
template<typename Type1,typename Type2,typename FunctorBinaryFunctionE>
void forEachFunctorBinaryFunctionE(const MatN<2,Type1> & f, MatN<2,Type2> &  h,  FunctorBinaryFunctionE func, typename MatN<2,Type1>::IteratorERectangle it)
{

    Vec2I32 x;
    for(x(0)=it.xMin()(0);x(0)<it.xMax()(0);x(0)++){
        for(x(1)=it.xMin()(1);x(1)<it.xMax()(1);x(1)++){
            h(x)=func( f, x);
        }
    }
}


template<typename Type1,typename Type2,typename FunctorBinaryFunctionE>
void forEachFunctorBinaryFunctionE(const MatN<3,Type1> & f, MatN<3,Type2> &  h,  FunctorBinaryFunctionE func, typename MatN<3,Type1>::IteratorEDomain it)
{

    int i_max,j_max,k_max;
    if(it.getDomain()==f.getDomain()){
        i_max= f.sizeI();
        j_max= f.sizeJ();
        k_max= f.sizeK();
    }else{
        i_max= h.sizeI();
        j_max= h.sizeJ();
        k_max= h.sizeK();
    }
    Vec3I32 x;
    for(x(0)=0;x(0)<i_max;x(0)++){
        for(x(1)=0;x(1)<j_max;x(1)++){
            for(x(2)=0;x(2)<k_max;x(2)++){
                h(x)=func( f, x);
            }
        }
    }
}

template<int DIM,typename PixelType,typename FunctorAccumulatorF>
typename FunctorAccumulatorF::ReturnType forEachFunctorAccumulator(const MatN<DIM,PixelType> & f,  FunctorAccumulatorF & func, typename MatN<DIM,PixelType>::IteratorEDomain &){
    for(unsigned int i=0;i<f.size();i++){
        func( f(i));
    }
    return func.getValue();
}
template<int DIM,typename PixelType1,typename PixelType2,typename FunctorUnary_F_F>
void forEachFunctorUnaryF(const MatN<DIM,PixelType1> & f,MatN<DIM,PixelType2> & g,FunctorUnary_F_F & func, typename MatN<DIM,PixelType1>::IteratorEDomain ){
    for(unsigned int i=0;i<f.size();i++){
        g(i)=func(f(i));
    }

}

}
#endif
