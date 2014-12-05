/******************************************************************************\
|*       Population library for C++ X.X.X     *|
|*----------------------------------------------------------------------------*|
The Population License is similar to the MIT license in adding this clause:
for any writing public or private that has resulted from the use of the
software population, the reference of this book "Population library, 2012,
Vincent Tariel" shall be included in it.

So, the terms of the Population License are:

Copyright © 2012, Tariel Vincent

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
#include <iterator>
#include <cmath>
#include <string>
#include <algorithm>
#include <numeric>
#include"data/utility/Exception.h"
#include"PopulationConfig.h"
#include"data/typeF/TypeTraitsF.h"
#include"data/typeF/RGB.h"
#include"data/vec/VecN.h"
#include"data/mat/MatNBoundaryCondition.h"
#include"data/mat/MatNIteratorE.h"
#include"data/functor/FunctorF.h"
#include"algorithm/FunctionProcedureFunctorF.h"
#include"data/utility/BasicUtility.h"

namespace pop
{
template<typename Type, int SIZEI, int SIZEJ>
class  Mat2x;
/*! \ingroup Data
* \defgroup Matrix  Matrix
* \brief n-dimensional matrices as dense array
*/


/*! \ingroup Matrix
* \defgroup MatN Mat{2,3}{UI8,RGBUI8}
* \brief template class for n-dimensional matrices which fixed type
*/

template<int Dim, typename Type>
class POP_EXPORTS MatN : public std::vector<Type>
{
public:

    /*!
    \class pop::MatN
    \ingroup MatN
    \brief template class for matrix (or Image)
    \author Tariel Vincent
    \tparam Dim Space dimension
    \tparam Type Pixel/Voxel type


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

    \section Type Type

    This class is written in generic programming with template parameters. The first one is the dimension, the second one the pixel/voxel type.
    To facilite its utilisation, we use some typedef declarations to define the usual types to allow coding in C-style as these ones:
    - Mat2UI8: a 2d matrix with a pixel type coded in 1 byte for an unsigned integer between 0 and 255,
    - Mat2RGBUI8: a 2d matrix with a pixel type coded with a RGB color with 1 byte per channel,
    - Mat2F64: a 2d matrix with a pixel type coded in float type,
    - Mat2UI32: a 2d matrix with a pixel type coded in unsigned integer in 4 bytes,
    - Mat3UI8: a 3d matrix with a pixel type coded in 1 byte for an unsigned integer between 0 and 255.

    \section Structure Structure

    The cell values are stored in a %vector container, a class template of STL, which works like a dynamic array in a strict linear sequence. The position of the
    cell can be located by an index or a point VecN<Dim,int> as n integers (in 2D Vec2I32=(i,j), in 3D Vec3I32=(i,j,k)) as explained in the below figure:
    \image html vector.png the value f(i,j) corresponds to the element  v[j+i*ColSize] of the %vector container.
    We make the choice to have a single std::vector and not a std::vector of std::vector as the classical ways because the matrix can be nD and also for optimization purpose in the case of iteration.\n
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

    \subsection Domain Domain
    This class provides four ways for the iterative loops on the domain: \f$ \forall x \in \mathcal{D}\f$
    -  \a for: utilisation of the for statement. For instance, this code generate an matrix with a constant value:
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
    - \a  dereferencing: utilisation of the iterator of the STL std::vector. For instance, this code:
    \code
    void constant (unsigned char & v) {
        v= 150;
    }
    Mat2UI8 img(256,256);
    for_each(img.begin(),img.end(),constant);
    img.display();
    \endcode
    The first one is the simplest one but less generic than the second one independant of the space. The last one is more optimized but you lost the VecN position that can be required for the subsequent process as in the erosion algorithm.

    \subsection Neighborhood Neighborhood
    This class provides two ways for the iterative loops on the VecN neighborhood: \f$ \forall x' \in N(x)\f$.
    -  \a for: utilisation of the for statement. For instance, the erosion code is:
    \code
    Mat2UI8 img;
    img.load("../image/Lena.bmp");//load lena matrix
    Mat2UI8 erosion(img.sizeI(),img.sizeJ());//construct an matrix with the same domain
    int radius=4;
    //Domain loop
    for(int i =0;i<img.sizeI();i++){
        for(int j =0;j<img.sizeJ();j++){
            Mat2UI8::F v =NumericLimits<Mat2UI8::F>::maximumRange(); //equivalent to unsigned char v=255
             //Neighborhood loop
            for(int m= i-radius;m<i+radius;m++){
                for(int n= j-radius;n<j+radius;n++){
                    if(img.isValid(m,n)){//test if (i,j) belongs to the domain of definition of the matrix
                        v = minimum(img(m,n),v);
                    }
                }
            }
            erosion(i,j)=v;
       }
    }
    erosion.display();
    \endcode
    -  \a IteratorE: utilisation of the IteratorENeighborhood concept. This class exposes an IteratorENeighborhood model
    with its definition type, \a MatN::IteratorENeighborhood, and its defintion domain, \a MatN::getIteratorENeighborhood(F64 radius=1,F64 norm=1).
    For the IteratorENeighborhood object, the member \a next() advances to the next element in returning a boolean to indicate if the end of the collection is reached,
    the member \a x() returns the current element and the member \a init(const E & x)  initializes the neighborhood on the VecN x. For instance, the erosion code is:
    \code
    Mat2UI8 img;
    img.load("../image/Lena.bmp");
    Mat2UI8 erosion(img.getDomain());//construct an matrix with the same domain
    Mat2UI8::IteratorEDomain itdomain(img.getIteratorEDomain());//Domain IteratorE
    Mat2UI8::IteratorENeighborhood itneigh (img.getIteratorENeighborhood(4,2));//Neighborhood IteratorE with the norm euclidean and the radius 4
    while(itdomain.next()){
        Mat2UI8::F v =NumericLimits<Mat2UI8::F>::maximumRange();
        itneigh.init(itdomain.x());
        while(itneigh.next()){
            v = minimum(v,img(itneigh.x()));
        }
        erosion(itdomain.x())=v;
    }
    erosion.display();
    \endcode
    -  \a ForEachNeighborhood idiom: implicit iteration defined by a preprocessor directive.. Note we do have to set the type of the point
    \code
    Mat2UI8 img;
    img.load("../image/Lena.bmp");
    Mat2UI8 erosion(img.getDomain());//construct an matrix with the same domain
    ForEachDomain2D(x,img)
    {
        Mat2UI8::F v =NumericLimits<Mat2UI8::F>::maximumRange();
        ForEachNeighborhood(y,img,x,4,2){
            v = minimum(v,img(y));
        }
        erosion(x)=v;
    }
    erosion.display();
    \endcode


    The abstraction with IteratorENeighborhood concept provides an efficient and generic ways to iterate through a neighborhood.

    \section Load Load/Save

    The implementation of these methods are Included in the hedader MatNInOut.h. So do not
    forget to Include "#include"data/mat/MatNInOut.h" to avoid a such error message "undefined reference to "'pop::MatN<2, unsigned char>::save(char const*) const'".
    You can load/save various matrix formats, png, pgm, jpg, bmp. , the naturel format is pgm. To load/save a stack of matrices, you
    can use MatN::loadFromDirectory or MatN::saveFromDirectory.
    However, I extend the pgm format to directly save a 3D matrix in a single file.

    \section Exception Exception and execution information

    To handle the error message pexception, a good pratice is to include your code in try-catch statement. Also, many algorithms can give
    information about their execution on the standard output stream. You have the possibility to (des)activate this stream. To manage error and
    execution information, the code can bMatNe like that:
    \code
    CollectorExecutionInformationSingleton::getInstance()->setActivate(true);
    try{
    Mat2RGBUI8 img;
    img.load("../image/Lena.bmp");
    img = PDE::nonLinearAnisotropicDiffusionGaussian(img,10,50);
    img.display();
    }catch(const pexception & e){
    std::cerr<<e.what()<<std::endl;
    }
    \endcode

    \section Display Display

    In good architectur design, the data and its representation are separeted in two classes. However for conveniency, you provide the member display() in this data class for simple display and
    you provide the MatNDisplay class for extended display.

      \sa VecN RGB Complex MatNDisplay
    */

protected:


public:
    VecN<Dim,int> _domain;
    /*!
    \typedef F
    * Pixel/voxel type
    */
    typedef Type F;
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
    img.load("../image/Lena.bmp");
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
    img.load("../image/Lena.bmp");
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
    typedef MatNIteratorEROI<MatN<Dim, Type> >  IteratorEROI;
    /*!
    \typedef IteratorENeighborhood

    This iterator allows the iteration through the neighborhood of an element
    x of E without any order. The corresponding mathematical object is : \f$\forall x' \in N(x)\f$. This
    iterator plays an important role in algorithms of mathematical morphology and region growing.
    Its construction requires a domain of definition as argument given by the member getIteratorENeighborhood(F64 radius,F64 norm).
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
    img.load("../image/Lena.bmp");
    Mat2RGBUI8 img_erosion(img.getDomain());
    Mat2RGBUI8::IteratorEDomain it_total(img.getIteratorEDomain());
    F64 norm =2;
    F64 radius =4.5;
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
    // neighborhood iteration with bounded condition
    typedef MatNIteratorENeighborhood<E,MatNBoundaryConditionBounded> IteratorENeighborhood;
    // neighborhood iteration with mirror condition
    typedef MatNIteratorENeighborhood<E,MatNBoundaryConditionMirror> IteratorENeighborhoodMirror;
    // neighborhood iteration with periodic condition
    typedef MatNIteratorENeighborhood<E,MatNBoundaryConditionPeriodic> IteratorENeighborhoodPeriodic;
    typedef MatNIteratorENeighborhoodAmoebas<MatN> IteratorENeighborhoodAmoebas;

    typedef MatNIteratorEOrder<E> IteratorEOrder;
    typedef MatNBoundaryCondition BoundaryCondition;

    /*!
    \typedef IteratorERectangle

    Iterate in rectangle

     \code
    Mat2UI8 img2;
    img2.load("../image/Lena.bmp");
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

    Stl iterator (vector::iterator)

     \sa  begin() end()
    */
    typedef typename std::vector<Type>::iterator iterator;
    typedef typename std::vector<Type>::value_type					 value_type;
    typedef typename std::vector<Type>::pointer           pointer;
    typedef typename std::vector<Type>::const_pointer     const_pointer;
    typedef typename std::vector<Type>::reference         reference;
    typedef typename std::vector<Type>::const_reference   const_reference;

    typedef typename std::vector<Type>::const_iterator const_iterator;
    typedef typename std::vector<Type>::const_reverse_iterator  const_reverse_iterator;
    typedef typename std::vector<Type>::reverse_iterator		 reverse_iterator;
    typedef typename std::vector<Type>::size_type					 size_type;
    typedef typename std::vector<Type>::difference_type				 difference_type;
    typedef typename std::vector<Type>::allocator_type                        		 allocator_type;



    //-------------------------------------
    //
    //! \name Constructor
    //@{
    //-------------------------------------
    /*!
    \fn MatN()
    * default constructor
    */
    MatN( )
    {
        _domain =0;

    }

    virtual ~MatN(){
    }
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
    explicit MatN(const VecN<Dim,int>& domain,Type v=Type())
        :std::vector<Type>(domain.multCoordinate(),Type(v)),_domain(domain)
    {
    }

//    explicit MatN(unsigned int sizei)
//        :std::vector<Type>(sizei,Type(0))
//    {
//        _domain(0)=sizei;
//    }
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

    explicit MatN(unsigned int sizei,unsigned int sizej)
        :std::vector<Type>(sizei*sizej,Type(0))
    {
        POP_DbgAssertMessage(Dim==2,"In MatN::MatN(int i, int j), your matrix must be 2D");
        _domain(0)=sizei;_domain(1)=sizej;
    }
    /*!
    \param sizei number of  columns
    \param sizej number of  rows
    \param sizek number of  depths
    *
    * construct an matrix of size i,j,k where each pixel/voxel value is set at 0\n
    */
    explicit MatN(unsigned int sizei, unsigned int sizej,unsigned int sizek)
        :std::vector<Type>(sizei*sizej*sizek,Type(0))
    {
        POP_DbgAssertMessage(Dim==3,"In MatN::MatN(int sizei, int sizej,int sizek), your matrix must be 3D");
        _domain(0)=sizei;_domain(1)=sizej;_domain(2)=sizek;
    }
    /*!
    \fn MatN(const VecN<Dim,int>& x,const std::vector<Type>& data)
    \param x domain size of the matrix
    \param data value of each pixel/voxel
    *
    * construct an matrix of size domain(0),domain(1) for 2D matrix and  domain(0),domain(1),domain(2) for 3D matrix where each pixel/voxel is set by the values contained in the std::vector \n
    *   This code: \n
    \code
    Mat2UI8::Domain x;
    x(0)=2;
    x(1)=4;
    std::vector<Mat2UI8::F> v;
    v.push_back(0);v.push_back(1);v.push_back(2);v.push_back(3);
    v.push_back(3);v.push_back(2);v.push_back(1);v.push_back(0);
    Mat2UI8 img(x,v);
    std::cout<<img;
    \endcode
    produce this output\n
    \code
    0 1 2 3
    3 2 1 0
    \endcode
    */
    explicit MatN(const VecN<Dim,int> & x,const std::vector<Type>& data )
        :std::vector<Type>(data),_domain(x)
    {
        POP_DbgAssertMessage((int)data.size()==_domain.multCoordinate(),"In MatN::MatN(const VecN<Dim,int> & x,const std::vector<Type>& data ), the size of input std::vector data must be equal to the number of pixel/voxel");
    }

    /*!
    \fn MatN(const VecN<Dim,int> & x,const Type* v_value )
    \param x domain size of the matrix
    \param v_value affection values for the matrix elements
     *
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
    explicit MatN(const VecN<Dim,int> & x,const Type* v_value )
        :std::vector<Type>(x.multCoordinate(),Type()),_domain(x)
    {
        std::copy(v_value,v_value + _domain.multCoordinate(),this->begin());
    }

    /*!
    \param img object to copy
    *
    * copy construct\n
    *   This code: \n
    \code
    Mat2UI8::Domain x;
    x(0)=2;
    x(1)=4;
    std::vector<Mat2UI8::F> v;
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
    MatN(const MatN<Dim, T1> & img )
        :std::vector<Type>(img.getDomain().multCoordinate()),_domain(img.getDomain())
    {
        std::transform(img.begin(),img.end(),this->begin(),ArithmeticsSaturation<Type,T1>::Range);
    }


#ifndef HAVE_SWIG
    /*!
    \param img object to copy
    *
    * copy construct\n
    *   This code: \n
    \code
    Mat2UI8::Domain x;
    x(0)=2;
    x(1)=4;
    std::vector<Mat2UI8::F> v;
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
    MatN(const MatN & img )
        :std::vector<Type>(img),_domain(img.getDomain())
    {
    }
#endif
    /*!
      * \param m small 2d matrix of size (2,2)
      *
      * type conversion
    */
    MatN(const Mat2x<Type,2,2> m);
    /*!
      * \param m small 2d matrix of size (3,3)
      *
      * type conversion
    */
    MatN(const Mat2x<Type,3,3> m);

    template<int SIZEI,int SIZEJ>
    MatN(const Mat2x<Type,SIZEI,SIZEJ> m);

    /*!
    \param filepath path of the matrix
    *
    *  construct the matrix from an matrix file
    *
    \code
    Mat2UI8 img("lena.bmp");
    img.display();
    \endcode
    */
    MatN(const char * filepath )
    {
        if(filepath!=0)
            load(filepath);
    }
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
    img2.load("../image/Lena.bmp");
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
    MatN(const MatN & img, const VecN<Dim,int>& xmin, const VecN<Dim,int> & xmax  )
    {
        POP_DbgAssertMessage(xmin.allSuperiorEqual(0),"xmin must be superior or equal to 0");
        POP_DbgAssertMessage(xmax.allSuperior(xmin),"xmax must be superior to xmin");
        POP_DbgAssertMessage(xmax.allInferior(img.getDomain()+1),"xmax must be superior or equal to xmin");
        _domain = xmax-xmin;
        std::vector<Type>::resize(_domain.multCoordinate(),0);
        if(  DIM==2 ){
            if(_domain(1)==img.getDomain()(1)){
                if(_domain(0)==img.getDomain()(0))
                    std::copy(img.begin(),img.end(),this->begin());
                else
                    std::copy(img.begin()+ xmin(0)*img._domain(1),img.begin()+xmax(0)*img._domain(1),this->begin());
            }else{

                typename std::vector<Type>::const_iterator itb = img.begin() + xmin(1)+xmin(0)*img._domain(1);
                typename std::vector<Type>::const_iterator ite = img.begin() + xmax(1)+xmin(0)*img._domain(1);
                typename std::vector<Type>::iterator it = this->begin();
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
                    typename std::vector<Type>::const_iterator itb = img.begin() + xmin(0)*img._domain(1) + xmin(2)*intra_slice_add;
                    typename std::vector<Type>::const_iterator ite = img.begin() + xmax(0)*img._domain(1) + xmin(2)*intra_slice_add;
                    typename std::vector<Type>::iterator it        = this->begin();

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
                typename std::vector<Type>::const_iterator itb = img.begin() + xmin(1) +indexmini*img._domain(1) + xmin(2)*intra_slice_add;
                typename std::vector<Type>::const_iterator ite = img.begin() + xmax(1) +indexmini*img._domain(1) + xmin(2)*intra_slice_add;
                typename std::vector<Type>::iterator it        = this->begin();
                unsigned int indexmin = xmin(2);
                unsigned int indexmax = xmax(2);
                for(unsigned int i=indexmin;i<indexmax;i++){
                    typename std::vector<Type>::const_iterator itbb = itb;
                    typename std::vector<Type>::const_iterator itee = ite;
                    typename std::vector<Type>::iterator itt =it;
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
    }
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
    Domain  getDomain()
    const
    {
        return _domain;
    }

    /*!
    \return  number of rows
    *
    * return the number of rows
    */
    unsigned int sizeI()const{
        return this->getDomain()[0];
    }
    /*!
    \return number of rows
    *
    * return the number of rows
    */
    unsigned int rows()const{
        return this->getDomain()[0];
    }
    /*!
    \return number of columns
    *
    * return the number of columns
    */
    unsigned int sizeJ()const{
        return this->getDomain()[1];
    }
    /*!
    \return number of columns
    *
    * return the number of columns
    */
    unsigned int columns()const{
        return this->getDomain()[1];
    }
    /*!
    \return int sizek
    *
    * return the number of depths
    */
    unsigned int sizeK()const{
        POP_DbgAssert(Dim==3);

       if(DIM>=3)
        return this->getDomain()[2];
        else
            return 1;

    }
    /*!
    \return number of depths
    *
    * return the number of depths
    */
    unsigned int depth()const{
        POP_DbgAssert(Dim==3);
        return this->getDomain()[2];
    }
    /*!
    \param x VecN
    \return boolean
    *
    * return true if the VecN belongs to the domain, false otherwise
    */
    bool isValid(const E & x)const{
        if(x.allSuperiorEqual(E(0)) && x.allInferior(this->getDomain()))
            return true;
        else
            return false;
    }
    /*!
    \param i i coordinate of the VecN
    \param j j coordinate of the VecN
    \return boolean
    *
    * return true if the VecN (i,j) belongs to the domain, false otherwise
    */
    bool isValid(int i,int j)const{
        if(i>=0&&j>=0 && i<static_cast<int>(sizeI())&& j<static_cast<int>(sizeJ()))
            return true;
        else
            return false;
    }
    /*!
    \param i i coordinate of the VecN
    \param j j coordinate of the VecN
    \param k k coordinate of the VecN
    \return boolean
    *
    * return true if the VecN (i,j,k) belongs to the domain, false otherwise
    */
    bool isValid(int i,int j,int k)const{
        if(i>=0&&j>=0&&k>=0 && i<static_cast<int>(sizeI())&& j<static_cast<int>(sizeJ())&&k<static_cast<int>(sizeK()))
            return true;
        else
            return false;
    }

    /*!
    \param sizei  row size
    \param sizej coloumn size
    *
    * resize the matrix in loosing the data information
    */
    void resize(unsigned int sizei,unsigned int sizej){
        _domain(0)=sizei;
        _domain(1)=sizej;
        std::vector<Type>::resize(_domain(0)*_domain(1));
    }
    /*!
    \param sizei  row size
    \param sizej  col size
    \param sizek depth size
    *
    * resize the matrix in loosing the data information
    */
    void resize(unsigned int sizei,unsigned int sizej,unsigned int sizek){
        _domain(0)=sizei;
        _domain(1)=sizej;
        _domain(2)=sizek;
        std::vector<Type>::resize(_domain(0)*_domain(1)*_domain(2));
    }
    /*!
    \param d  domain =Vec2(i,j) in 2d  and domain =Vec3(i,j,k) in 3d
    *
    * resize the matrix in loosing the data information
    */
    void resize(const VecN<Dim,int> & d){
        _domain=d;
        std::vector<Type>::resize(_domain.multCoordinate());
    }
    /*!
    \param sizei  row size
    \param sizej coloumn size
    *
    * resize the matrix in keeping the data information
    */
    void resizeInformation(unsigned int sizei,unsigned int sizej){

        Domain d;
        d(0)=sizei;
        d(1)=sizej;
        resizeInformation(d);
    }
    /*!

    \param sizei  row size
    \param sizej  colo size
    \param sizek depth size
    *
    * resize the matrix in keeping the data information
    */
    void resizeInformation(unsigned int sizei,unsigned int sizej,unsigned int sizek){
        Domain d;
        d(0)=sizei;
        d(1)=sizej;
        d(2)=sizek;
        resizeInformation(d);
    }
    /*!
    \param d  domain =Vec2(i,j) in 2d  and domain =Vec3(i,j,k) in 3d
    *
    * resize the matrix in keeping the data information
    */
    void resizeInformation(const VecN<Dim,int>& d){
        MatN temp(*this);
        _domain=d;
        std::vector<Type>::resize(_domain.multCoordinate());

        IteratorEDomain it(this->getIteratorEDomain());
        while(it.next()){
            if(temp.isValid(it.x())){
                this->operator ()(it.x())=temp(it.x());
            }else{
                this->operator ()(it.x())=0;
            }
        }
    }
    /*!
    \return true if matrix is empty
    *
    * return true if the the matrix empty
    */
    bool isEmpty()const{
        if(_domain.multCoordinate()==0)
            return true;
        else
            return false;
    }
    /*!
    *
    * clear the content of the matrix
    */
    void clear(){
        _domain=0;
        std::vector<Type>::clear();
    }

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
    img.load("../image/Lena.bmp");
    Mat2UI8::IteratorEDomain it(img.getIteratorEDomain());
    Distribution d(0,20,"NORMAL");
    FunctorF::FunctorAdditionF2<Mat2UI8::F,F64,Mat2UI8::F> op;
    while(it.next()){
    img(it.x())=op(img(it.x()),d.randomVariable());//access a VecN, add a random variable and set it
    }
    img.display();
    \endcode
    * \sa VecN
    */
    inline F & operator ()(const VecN<Dim,int> & x)
    {
        POP_DbgAssert( x.allSuperiorEqual( E(0))&&x.allInferior(getDomain()));
        return  this->operator[](VecNIndice<Dim>::VecN2Indice(_domain,x));
    }

    /*!
    \param x pixel/voxel position
    \return pixel/voxel value
    *
    * access the reference of the pixel/voxel value at the given position
    * \code
    Mat2UI8 img;
    img.load("../image/Lena.bmp");
    Mat2UI8::IteratorEDomain it(img.getIteratorEDomain());
    Distribution d(0,20,"NORMAL");
    FunctorF::FunctorAdditionF2<Mat2UI8::F,F64,Mat2UI8::F> op;
    while(it.next()){
    img(it.x())=op(img(it.x()),d.randomVariable());//access a VecN, add a random variable and set it
    }
    img.display();
    \endcode
    * \sa VecN
    */
    inline const F & operator ()( const VecN<Dim,int>& x)
    const
    {
        POP_DbgAssert( x.allSuperiorEqual(E(0))&&x.allInferior(getDomain()));
        return  this->operator[](VecNIndice<Dim>::VecN2Indice(_domain,x));
    }

    /*!
    \param i i coordinate (row)
    \param j j coordinate (column)
    \return pixel/voxel value
    *
    * access the reference of the pixel/voxel value at the position (i,j) for a 2D matrix
    */
    inline Type & operator ()(unsigned int i,unsigned int j)
    {
        POP_DbgAssert( i<(sizeI())&&j<(sizeJ()));
        return  this->operator[](j+i*_domain(1));
    }
    /*!
    \param i i coordinate (row)
    \param j j coordinate (column)
    \return pixel/voxel value
    *
    * access the reference of the pixel/voxel value at the position (i,j) for a 2D matrix
    */
    inline const Type & operator ()(unsigned int i,unsigned int j)const
    {
        POP_DbgAssert( i<(sizeI())&&j<(sizeJ()));
        return  this->operator[](j+i*_domain(1));
    }
    /*!
    \param i i coordinate (row)
    \param j j coordinate (column)
    \param k k coordinate (depth)
    \return pixel/voxel value
    *
    * access the reference of the pixel/voxel value at the given position (i,j,k) for a 3D matrix
    */
    inline Type & operator ()(unsigned int i,unsigned int j,unsigned int k)
    {
        POP_DbgAssert(  i<(sizeI())&&j<(sizeJ())&&k<(sizeK()));
        return  this->operator[](j+i*_domain(1)+k*_domain(0)*_domain(1));
    }

    /*!
    \param i i coordinate (row)
    \param j j coordinate (column)
    \param k k coordinate (depth)
    \return pixel/voxel value
    *
    * access the reference of the pixel/voxel value at the given position (i,j,k) for a 3D matrix
    */
    inline const Type & operator ()(unsigned int i,unsigned int j,unsigned int k)const
    {
        POP_DbgAssert(  i<(sizeI())&&j<(sizeJ())&&k<(sizeK()));
        return  this->operator[](j+i*_domain(1)+k*_domain(0)*_domain(1));
    }

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
    MatN operator()(const VecN<Dim,int> & xmin, const VecN<Dim,int> & xmax) const{
        return MatN(*this,xmin,xmax);
    }
    /*!
    \param index vector index
    \return pixel/voxel value
    *
    * access the reference of the pixel/voxel value at the vector index (std::vector contains pixel values)
    */
    inline Type & operator ()(unsigned int index)
    {
        POP_DbgAssert( index<this->size());
        return this->operator[](index);
    }
    /*!
    \param xf vector position in float value
    \return pixel/voxel value
    *
    * access the interpolated pixel/voxel value at the float position
    */
    Type interpolationBilinear(const VecN<DIM,F64> xf)const
    {
        if(DIM==2){
            Vec2F64 x;
            x(0)=xf(0)+EPSILON;
            x(1)=xf(1)+EPSILON;
            typename FunctionTypeTraitsSubstituteF<Type,F64>::Result value=0;
            double sum=0;
            Vec2I32 x1;
            x1(0)=std::floor(x(0));
            x1(1)=std::floor(x(1));
            if(this->isValid(x1(0),x1(1))){
                double norm = (1-(x(0)-x1(0)))*(1-(x(1)-x1(1)));
                value+=this->operator ()(x1(0),x1(1))*norm;
                sum+= norm;
            }
            x1(0)=std::ceil(x(0));
            if(this->isValid(x1(0),x1(1))){
                double norm = (1-(x1(0)-x(0)))* (1-(x(1)-x1(1)));
                value+=this->operator ()(x1(0),x1(1))*norm;
                sum+= norm;
            }
            x1(1)=std::ceil(x(1));
            if(this->isValid(x1(0),x1(1))){
                double norm = (1-(x1(0)-x(0)))*(1-(x1(1)-x(1)));
                value+=this->operator ()(x1(0),x1(1))*norm;
                sum+= norm;
            }
            x1(0)=std::floor(x(0));
            if(this->isValid(x1(0),x1(1))){
                double norm = (1-(x(0)-x1(0)))*(1-(x1(1)-x(1)));
                value+=this->operator ()(x1(0),x1(1))*norm;
                sum+= norm;
            }
            if(sum==0)
                return 0;
            else if(NumericLimits<Type>::is_integer==true)
                return round(value/sum);
            else
                return value/sum;
        }else if(DIM==3){
            Vec3F64 x;
            x(0)=xf(0)+EPSILON;
            x(1)=xf(1)+EPSILON;
            x(2)=xf(2)+EPSILON;
            typename FunctionTypeTraitsSubstituteF<Type,F64>::Result value=0;
            double sum=0;
            Vec3I32 x1;
            x1(0)=std::floor(x(0));
            x1(1)=std::floor(x(1));
            x1(2)=std::floor(x(2));
            if(this->isValid(x1(0),x1(1),x(2))){
                double norm = (1-(x(0)-x1(0)))*(1-(x(1)-x1(1)))*(1-(x(2)-x1(2)));
                value+=this->operator ()(x1(0),x1(1),x(2))*norm;
                sum+= norm;
            }
            x1(0)=std::ceil(x(0));
            if(this->isValid(x1(0),x1(1),x(2))){
                double norm = (1-(x1(0)-x(0)))* (1-(x(1)-x1(1)))*(1-(x(2)-x1(2)));
                value+=this->operator ()(x1(0),x1(1),x(2))*norm;
                sum+= norm;
            }
            x1(1)=std::ceil(x(1));
            if(this->isValid(x1(0),x1(1),x(2))){
                double norm = (1-(x1(0)-x(0)))*(1-(x1(1)-x(1)))*(1-(x(2)-x1(2)));
                value+=this->operator ()(x1(0),x1(1),x(2))*norm;
                sum+= norm;
            }
            x1(0)=std::floor(x(0));
            if(this->isValid(x1(0),x1(1),x(2))){
                double norm = (1-(x(0)-x1(0)))*(1-(x1(1)-x(1)))*(1-(x(2)-x1(2)));
                value+=this->operator ()(x1(0),x1(1),x(2))*norm;
                sum+= norm;
            }
            x1(0)=std::floor(x(0));
            x1(1)=std::floor(x(1));
            x1(2)=std::ceil(x(2));
            if(this->isValid(x1(0),x1(1),x(2))){
                double norm = (1-(x(0)-x1(0)))*(1-(x(1)-x1(1)))*(1-(x1(2)-x(2)));
                value+=this->operator ()(x1(0),x1(1),x(2))*norm;
                sum+= norm;
            }
            x1(0)=std::ceil(x(0));
            if(this->isValid(x1(0),x1(1),x(2))){
                double norm = (1-(x1(0)-x(0)))* (1-(x(1)-x1(1)))*(1-(x1(2)-x(2)));
                value+=this->operator ()(x1(0),x1(1),x(2))*norm;
                sum+= norm;
            }
            x1(1)=std::ceil(x(1));
            if(this->isValid(x1(0),x1(1),x(2))){
                double norm = (1-(x1(0)-x(0)))*(1-(x1(1)-x(1)))*(1-(x1(2)-x(2)));
                value+=this->operator ()(x1(0),x1(1),x(2))*norm;
                sum+= norm;
            }
            x1(0)=std::floor(x(0));
            if(this->isValid(x1(0),x1(1),x(2))){
                double norm = (1-(x(0)-x1(0)))*(1-(x1(1)-x(1)))*(1-(x1(2)-x(2)));
                value+=this->operator ()(x1(0),x1(1),x(2))*norm;
                sum+= norm;
            }
            if(sum==0)
                return 0;
            else if(NumericLimits<Type>::is_integer==true)
                return round(value/sum);
            else
                return value/sum;
        }
        return 0;
    }

    /*!
    * Return a ptr to the first pixel value
    *
    *
    * direct access to the matrix data that can be usefull for optimized purposes
    */
    inline Type *  data()
    {
        return &(*this->begin());
    }
    /*!
    * Return a ptr to the first pixel value
    *
    *
    * direct access to the matrix data that can be usefull for optimized purposes
    */
    virtual inline const Type *  data()
    const
    {
        return &(*this->begin());
    }

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
    * \exception the path dir does not exist or the matrix is not 3D
    *
    *
    * The loadFromDirectory attempts to load all files as 2d slices of the  3D matrix in the  directory pathdir. If the extension is set,
    * we filter all filter all files with the extension. It is the same for basename.\n
    * For instance, this code produces:
    \code
    CollectorExecutionInformationSingleton::getInstance()->setActivate(true);
    Mat3UI8 img;
    img.loadFromDirectory("/home/vincent/Desktop/WorkSegmentation/lavoux/","in","tiff");
    img.display();
    img.save("lavoux3d.pgm");
    \endcode

    */
    void loadFromDirectory(const char * pathdir,const char * basefilename="",const char * extension="")throw(pexception);
    /*!
    \param file input file
    \exception  pexception the input file does not exist
    *
    * The loader attempts to read the matrix using the specified format. Natively, this library support the pgm, png, jpg, bmp formats. However thanks to the CImg library, this library can
    read various matrix formats http://cimg.sourceforge.net/reference/group__cimg__files__io.html if you install Image Magick http://www.imagemagick.org/script/binary-releases.php.
    */
    void load(const char * file)throw(pexception);
    /*!
    * \param file input file
    * \exception  pexception the input file does not exist
    *
    * \sa MatN::load(const char * file)
    */
    void load(const std::string file) throw(pexception){
        this->load(file.c_str());
    }
    /*!
    \param file input file
    \param d  domain of definition of the image
    \exception  pexception  the file should be equal to sizeof(T)*in.getDomain().multCoordinate()
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
    void loadRaw(const char * file,const Domain & d)throw(pexception);
    /*!
    \param pathdir directory path
    \param basefilename filename base by default "toto"
    \param extension by default ".pgm"
    \exception the path dir does not exist or the matrix is not 3D
    *
    * The saveFromdirectory attempts to save Save all slices of the  3D matrix f in the  directory pathdir with the given basefilename and the extenion,\n
    * for instance pathdir="/home/vincent/Project/ENPC/ROCK/Seg/"  basefilename="seg" and extension=".bmp", will save the slices as follows \n
    * "/home/vincent/Project/ENPC/ROCK/Seg/seg0000.bmp", \n
    * "/home/vincent/Project/ENPC/ROCK/Seg/seg0001.bmp",\n
    * "/home/vincent/Project/ENPC/ROCK/Seg/seg0002.bmp"\n
    *  "and so one.
    */
    void saveFromDirectory(const char * pathdir,const char * basefilename="toto",const char * extension=".pgm")const throw(pexception);

    /*!
    \param file input file
    \exception  the input file does not exist or it is not pgm format
    *
    * The saver attempts to write the matrix using the specified format.  Natively, this library support the pgm, png, jpg, bmp format. However thanks to the CImg library, this library can
    save various matrix formats http://cimg.sourceforge.net/reference/group__cimg__files__io.html .
    */
    void save(const char * file)const throw(pexception);

    /*!
    \param file input file
    \exception  pexception the input file does not exist
    *
    * \sa MatN::save(const char * file)
    */
    void save(const std::string file)const throw(pexception){
        save(file.c_str());
    }
    /*!
    \param file input file
    \exception  pexception the input file does not exist
    *
    * save the data in raw format without header
    */
    void saveRaw(const char * file)const throw(pexception);
    /*!
    \param file input file
    \param header header of the file
    \exception  pexception the input file does not exist
    *
    * save the data in ascii format without header
    */
    void saveAscii(const char * file,std::string header="")const throw(pexception);
    /*!
    * \param title windows title
    * \param stoprocess for stoprocess=true, stop the process until the windows is closed, otherwise the process is still running
    * \param automaticresize for automaticresize=true, you scale the matrix before the display, we do nothing otherwise
    *
    * Display the matrix using the CIMG facility.
    * \code
    * Mat2UI8 img;
    * img.load("../image/Lena.bmp");
    * img.display();
    * Mat2F64 gradx(img);
    * gradx = pop::Processing::gradientDeriche(gradx,0,0.5);
    * gradx = pop::Processing::greylevelRange(gradx,0,255);//to display the matrix with a float type, the good thing is to translate the grey-level range between [0-255] before
    * gradx.display();
    * \endcode
    *
    * display the matrix
    */

    void display(const char * title="",bool stoprocess=true, bool automaticresize=true)const throw(pexception);

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
    IteratorEDomain getIteratorEDomain()const
    {
        return IteratorEDomain(getDomain());
    }
    /*!
    \fn typename IteratorEROI getIteratorEROI()const
    \return ROI iterator
    *
    * return the ROI iterator  of the matrix where the iteration is done on
    * pixel/voxel values different to 0.
    *

    */
    IteratorEROI getIteratorEROI()const
    {
        return IteratorEROI(*this);
    }
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
    IteratorENeighborhood getIteratorENeighborhood(F64 radius=1 ,F64 norm=1 )const
    {
        return IteratorENeighborhood(getDomain(),radius , norm);
    }
    /*!
    * \param structural_element structural element
    * \param dilate number of dilation of the structural element
    \return Neighborhood iterator
    *
    * \code
    * Mat2UI8 img;
    * img.load("../image/Lena.bmp");
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
    IteratorENeighborhood getIteratorENeighborhood(const MatN<Dim,Type1> & structural_element,int dilate=1 )const
    {
        Vec<E> _tab;
        typename MatN<Dim,Type1>::IteratorEDomain it(structural_element.getDomain());
        typename MatN<Dim,Type1>::E center = VecN<Dim,F64>(structural_element.getDomain()-1)*0.5;
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
    IteratorEOrder getIteratorEOrder(int coordinatelastloop=0,int direction=1)const
    {
        return IteratorEOrder(getDomain(),coordinatelastloop,direction);
    }
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


    IteratorERectangle getIteratorERectangle(const E & xmin,const E & xmax )const
    {
        return IteratorERectangle(std::make_pair(xmin,xmax));
    }
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


    IteratorENeighborhoodAmoebas getIteratorENeighborhoodAmoebas(F64 distance_max=4,double lambda_param = 0.01 )const
    {
        return IteratorENeighborhoodAmoebas(*this,distance_max,lambda_param );
    }

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
    MatN& operator =(const MatN<Dim, T1> & img ){
        this->resize(img.getDomain());
        std::transform(img.begin(),img.end(),this->begin(),ArithmeticsSaturation<Type,T1>::Range);
        return *this;
    }
    /*!
    * \param img other matrix
    * \return this matrix
    *
    * Basic assignement of this matrix by \a other
    */
    MatN& operator =(const MatN & img ){
        this->resize(img.getDomain());
        std::copy(img.begin(),img.end(),this->begin());
        return *this;
    }
    /*!
    * \param value value
    * \return this matrix
    *
    * Basic assignement of all pixel/voxel values by \a value
    */
    MatN<Dim, Type>&  operator=(Type value)
    {
        std::fill (this->begin(),this->end(),value);
        return *this;
    }
    /*!
    * \param value value
    * \return this matrix
    *
    * Basic assignement of all pixel/voxel values by \a value
    */
    MatN<Dim, Type>&  fill(Type value)
    {
        std::fill (this->begin(),this->end(),value);
        return *this;
    }
    /*!
    * \param mode mode by default 0
    * \return opposite matrix
    *
    * opposite of the matrix  h(x)=max(f::F)-f(x) with max(f::F) is the maximum value of the range defined by the pixel/voxel type for mode =0,\n
    * or h(x)=max(f)-f(x) with max(f) is the maximum value of the field for mode =1
    */
    MatN<Dim, Type>  opposite(int mode=0)const
    {
        MatN<Dim, Type> temp;
        Type maxi;
        if(mode==0)
            maxi=NumericLimits<Type>::maximumRange();
        else{
            FunctorF::FunctorAccumulatorMax<Type > func;
            func = std::for_each (this->begin(), this->end(), func);
            maxi=func.getValue();
        }
        temp=maxi-*this;
        return temp;
    }

    /*!
    \param f input matrix
    \return boolean
    *
    * Equal operator true for all x in E f(x)=(*this)(x), false otherwise
    */
    bool operator==(const MatN<Dim, Type>& f)const
    {
        FunctionAssert(f,*this,"In MatN::operator==");
        return std::equal (f.begin(), f.end(), this->begin());
    }
    /*!
    \fn bool operator!=(const MatN<Dim, Type>& f)const
    \param f input matrix
    \return boolean
    *
    * Difference operator true for at least on x in E f(x)!=(*this)(x), false otherwise
    */
    bool operator!=(const MatN<Dim, Type>& f)const
    {
        FunctionAssert(f,*this,"In MatN::operator==");
        return !std::equal (f.begin(), f.end(), this->begin());
    }
    /*!
    \fn MatN<Dim, Type>&  operator+=(const MatN<Dim, Type>& f)
    \param f input matrix
    \return object reference
    *
    * Addition assignment h(x)+=f(x)
    */
    MatN<Dim, Type>&  operator+=(const MatN<Dim, Type>& f)
    {

        FunctionAssert(f,*this,"In MatN::operator+=");
        FunctorF::FunctorAdditionF2<Type,Type,Type> op;
        std::transform (this->begin(), this->end(), f.begin(),this->begin(),  op);
        return *this;
    }
    /*!
    * \param f input matrix
    * \return object
    *
    *  Addition h(x)= (*this)(x)+f(x)
    */
    MatN<Dim, Type>  operator+(const MatN<Dim, Type>& f)const{
        MatN<Dim, Type> h(*this);
        h +=f;
        return h;
    }
    /*!
    * \param value input value
    * \return object reference
    *
    * Addition assignment h(x)+=value
    */
    MatN<Dim, Type>& operator+=(Type value)
    {
        FunctorF::FunctorArithmeticConstantValueAfter<Type,Type,Type,FunctorF::FunctorAdditionF2<Type,Type,Type> > op(value);
        std::transform (this->begin(), this->end(), this->begin(),  op);
        return *this;
    }
    /*!
    \param value input value
    \return object
    *
    * Addition h(x)= (*this)(x)+value
    */
    MatN<Dim, Type>  operator+(Type value)const{
        MatN<Dim, Type> h(*this);
        h +=value;
        return h;
    }
    /*!
    \fn MatN<Dim, Type>&  operator-=(const MatN<Dim, Type>& f)
    \param f input matrix
    \return object reference
    *
    * Subtraction assignment h(x)-=f(x)
    */
    MatN<Dim, Type>&  operator-=(const MatN<Dim, Type>& f)
    {

        FunctionAssert(f,*this,"In MatN::operator-=");
        FunctorF::FunctorSubtractionF2<Type,Type,Type> op;
        std::transform (this->begin(), this->end(), f.begin(),this->begin(),  op);
        return *this;
    }
    /*!
    \param value input value
    \return object reference
    *
    * Subtraction assignment h(x)-=value
    */
    MatN<Dim, Type>&  operator-=(Type value)
    {
        FunctorF::FunctorArithmeticConstantValueAfter<Type,Type,Type,FunctorF::FunctorSubtractionF2<Type,Type,Type> > op(value);
        std::transform (this->begin(), this->end(), this->begin(),  op);
        return *this;
    }

    /*!
    * \param f input matrix
    * \return output matrix
    *
    *  Subtraction h(x)= (*this)(x)-f(x)
    */
    MatN<Dim, Type>  operator-(const MatN<Dim, Type>& f)const{
        MatN<Dim, Type> h(*this);
        h -=f;
        return h;
    }

    /*!
    * \return output matrix
    *
    *  opposite   h(x)= -this(x)
    */
    MatN<Dim, Type>  operator-()const{
        MatN<Dim, Type> h(this->getDomain(),Type(0));
        h -=*this;
        return h;
    }
    /*!
    * \param value input value
    * \return output matrix
    *
    * Subtraction h(x)= (*this)(x)-value
    */
    MatN<Dim, Type>  operator-(Type value)const{
        MatN<Dim, Type> h(*this);
        h -=value;
        return h;
    }

    /*!
    * \param m  other matrix
    * \return output matrix
    *
    *  matrix multiplication see http://en.wikipedia.org/wiki/Matrix_multiplication
    *
    *  \code
    Mat2F64 m1(2,3);
    m1(0,0)=1; m1(0,1)=2; m1(0,2)=0;
    m1(1,0)=4; m1(1,1)=3; m1(1,2)=-1;

    Mat2F64 m2(3,2);
    m2(0,0)=5; m2(0,1)=1;
    m2(1,0)=2; m2(1,1)=3;
    m2(2,0)=3; m2(2,1)=4;
    Mat2F64 m3 = m1*m2;
    std::cout<<m3<<std::endl;
    *  \endcode
    *
    */
    MatN  operator*(const MatN &m)const
    {
        POP_DbgAssertMessage(DIM==2&&this->sizeJ()==m.sizeI() ,"In Matrix::operator*, Not compatible size for the operator * of the class Matrix (A_{n,k}*B_{k,p})");
        MatN mtrans = m.transpose();
        MatN mout(this->sizeI(),m.sizeJ());
        Type sum = 0;
        for(unsigned int i=0;i<this->sizeI();i++){
            for(unsigned int j=0;j<m.sizeJ();j++){
                sum = 0;
                typename MatN::const_iterator this_it  = this->begin() +  i*this->sizeJ();
                typename MatN::const_iterator mtrans_it= mtrans.begin() + j*mtrans.sizeJ();
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
    /*!
    * \param m  other matrix
    * \return output matrix
    *
    *  matrix multiplication see http://en.wikipedia.org/wiki/Matrix_multiplication
    */
    MatN & operator*=(const MatN &m)
    {
        *this = this->operator *(m);
        return *this;
    }
    /*!
    \param v  vector
    \return output vector
    *
    *  matrix vector  multiplication
    */
    Vec<Type>  operator*(const Vec<Type> & v)const{
        POP_DbgAssertMessage(DIM==2&&this->sizeJ()==v.size() ,"In Matrix::operator*, Not compatible size for the operator *=(Vec) of the class Matrix (A_{n,k}*v_{k})");
        Vec<Type> temp(this->sizeI());
        for(unsigned int i=0;i<this->sizeI();i++){
            Type sum = 0;
            typename MatN::const_iterator this_it  = this->begin() +  i*this->sizeJ();
            typename Vec<Type>::const_iterator mtrans_it= v.begin();
            for(;mtrans_it!=v.end();this_it++,mtrans_it++){
                sum+=(* this_it) * (* mtrans_it);
            }
            temp(i)=sum;
        }
        return temp;
    }

    /*!
    \param f  matrix
    \return output matrix
    *
    *  multTermByTerm h(x)= (*this)(x)*f(x) (to avoid the the confusion with the matrix multiplication, we use this signature)
    */
    MatN  multTermByTerm(const MatN& f)const{
        FunctionAssert(f,*this,"In MatN::operator*=");
        FunctorF::FunctorMultiplicationF2<Type,Type,Type> op;
        MatN out(*this);
        std::transform (out.begin(), out.end(), f.begin(),out.begin(),  op);
        return out;
    }
    /*!
    \param value input value
    \return object reference
    *
    * Multiplication assignment h(x)*=value
    */
    MatN<Dim, Type>&  operator*=(Type  value)
    {
        FunctorF::FunctorArithmeticConstantValueAfter<Type,Type,Type,FunctorF::FunctorMultiplicationF2<Type,Type,Type> > op(value);
        std::transform (this->begin(), this->end(), this->begin(),  op);
        return *this;
    }

    /*!
    \param value input value
    \return object
    *
    * Multiplication h(x)= (*this)(x)*value
    */
    MatN<Dim, Type>  operator*(Type value)const{
        MatN<Dim, Type> h(*this);
        h *=value;
        return h;
    }
    /*!
    \param f  matrix
    \return output matrix
    *
    *  division term by term h(x)= (*this)(x)/f(x) (to avoid the the confusion with the matrix division, we use this signature)
    */
    MatN<Dim, Type>  divTermByTerm(const MatN& f){
        FunctionAssert(f,*this,"In MatN::divTermByTerm");
        FunctorF::FunctorDivisionF2<Type,Type,Type> op;
        std::transform (this->begin(), this->end(), f.begin(),this->begin(),  op);
        return *this;
    }

    /*!
    \param value input value
    \return object reference
    *
    * Division assignment h(x)/=value
    */
    MatN<Dim, Type>&  operator/=(Type value)
    {
        FunctorF::FunctorArithmeticConstantValueAfter<Type,Type,Type,FunctorF::FunctorDivisionF2<Type,Type,Type> > op(value);
        std::transform (this->begin(), this->end(), this->begin(),  op);
        return *this;
    }
    /*!
    \param value input value
    \return object
    *
    * Division h(x)= (*this)(x)/value
    */

    MatN<Dim, Type>  operator/(Type value)const{
        MatN<Dim, Type> h(*this);
        h /=value;
        return h;
    }
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
    * \return the row in a std::vector
    *
    * the output std::vector contained the row at the given index i
    * \sa Vec
    */
    Vec<F> getRow(unsigned int i)const;
    /*!
    * \param j  column entry
    * \return the column in a std::vector
    *
    * the output std::vector contained the column at the given index j
    * \sa Vec
    */
    Vec<F> getCol(unsigned int j)const;

    /*!
    * \param i  row entry
    * \param v  std::vector
    *
    * set the row at the given row entry with the given std::vector of size equal to number of column
    * \sa Vec
    */
    void setRow(unsigned int i,const Vec<F>& v);

    /*!
    * \param j  column entry
    * \param v  std::vector
    *
    * set the column at the given column entry with the given std::vector of size equal to number of row
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
    Mat2F64 m(3,3);
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
    Mat2F64 m(3,3);
    m(0,0)=1;m(0,1)=1;m(0,2)=2;
    m(1,0)=2;m(1,1)=1;m(1,2)=2;
    m(2,0)=1;m(2,1)=3;m(2,2)=3;

    Mat2F64 minverse;
    minverse = m.inverse();
    std::cout<<minverse<<std::endl;
    std::cout<<m*minverse<<std::endl;
    \endcode
    For large matrix, you should use LinearAlgebra::inverseGaussianElimination()
    */
    MatN inverse()const;
    //@}

#ifdef HAVE_SWIG
    MatN(const MatN<Dim,UI8> &img)
        :std::vector<Type>(img.getDomain().multCoordinate()),_domain(img.getDomain())
    {
        std::transform(img.begin(),img.end(),this->begin(),ArithmeticsSaturation<Type,UI8>::Range);
    }
    MatN(const MatN<Dim,UI16> &img)
        :std::vector<Type>(img.getDomain().multCoordinate()),_domain(img.getDomain())
    {
        std::transform(img.begin(),img.end(),this->begin(),ArithmeticsSaturation<Type,UI16>::Range);
    }
    MatN(const MatN<Dim,UI32> &img)
        :std::vector<Type>(img.getDomain().multCoordinate()),_domain(img.getDomain())
    {
        std::transform(img.begin(),img.end(),this->begin(),ArithmeticsSaturation<Type,UI32>::Range);
    }
    MatN(const MatN<Dim,F64> &img)
        :std::vector<Type>(img.getDomain().multCoordinate()),_domain(img.getDomain())
    {
        std::transform(img.begin(),img.end(),this->begin(),ArithmeticsSaturation<Type,F64>::Range);
    }
    MatN(const MatN<Dim,RGBUI8> &img)
        :std::vector<Type>(img.getDomain().multCoordinate()),_domain(img.getDomain())
    {
        std::transform(img.begin(),img.end(),this->begin(),ArithmeticsSaturation<Type,RGBUI8>::Range);
    }
    MatN(const MatN<Dim,RGBF64> &img)
        :std::vector<Type>(img.getDomain().multCoordinate()),_domain(img.getDomain())
    {
        std::transform(img.begin(),img.end(),this->begin(),ArithmeticsSaturation<Type,RGBF64>::Range);
    }
    MatN(const MatN<Dim,ComplexF64> &img)
        :std::vector<Type>(img.getDomain().multCoordinate()),_domain(img.getDomain())
    {
        std::transform(img.begin(),img.end(),this->begin(),ArithmeticsSaturation<Type,ComplexF64>::Range);
    }
    Type getValue(int i, int j)const{
        return  this->operator[](j+i*_domain(1));
    }
    Type getValue(int i, int j, int k )const{
        return  this->operator[](j+i*_domain(1)+k*_domain(0)*_domain(1));
    }
    Type getValue(const E & x )const{
        return  this->operator[](VecNIndice<Dim>::VecN2Indice(_domain,x));
    }
    void setValue(int i, int j , Type value){
        this->operator[](j+i*_domain(1)) =value;
    }
    void setValue(int i, int j , int k, Type value){
        this->operator[](j+i*_domain(1)+k*_domain(0)*_domain(1)) =value;
    }
    void setValue(const E & x, Type value){
        this->operator[](VecNIndice<Dim>::VecN2Indice(_domain,x)) =value;
    }

#endif

};

typedef MatN<2,UI8> Mat2UI8;
typedef MatN<2,UI16> Mat2UI16;
typedef MatN<2,UI32> Mat2UI32;
typedef MatN<2,F32> Mat2F32;
typedef MatN<2,F64> Mat2F64;

typedef MatN<2,RGBUI8> Mat2RGBUI8;
typedef MatN<2,RGBF64> Mat2RGBF64;
typedef MatN<2,ComplexF64> Mat2ComplexF64;
typedef MatN<2,Vec2F64 >  Mat2Vec2F64;


typedef MatN<3,UI8> Mat3UI8;
typedef MatN<3,UI16> Mat3UI16;
typedef MatN<3,UI32> Mat3UI32;
typedef MatN<3,F32> Mat3F32;
typedef MatN<3,F64> Mat3F64;

typedef MatN<3,RGBUI8> Mat3RGBUI8;
typedef MatN<3,RGBF64> Mat3RGBF64;
typedef MatN<3,ComplexF64> Mat3ComplexF64;
typedef MatN<3,VecN<3,F64> >  Mat3Vec3F64;




template<int DIM, typename Type>
MatN<DIM,Type> MatN<DIM,Type>::deleteRow(unsigned int i)const{
    POP_DbgAssert(i<sizeI());
    MatN<DIM,Type> temp(*this);
    temp._domain(0)--;
    temp.erase( temp.begin()+i*temp._domain(1), temp.begin()+(i+1)*temp._domain(1)  );
    return temp;
}
template<int DIM, typename Type>
MatN<DIM,Type> MatN<DIM,Type>::deleteCol(unsigned int j)const{
    POP_DbgAssert(j<sizeJ());
    MatN<DIM,Type> temp(this->sizeI(),this->sizeJ()-1);

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

template<int DIM, typename Type>
Vec<Type> MatN<DIM,Type>::getRow(unsigned int i)const{
    Vec<Type> v(this->sizeJ());
    std::copy(this->begin()+i*this->_domain(1), this->begin()+(i+1)*this->_domain(1),v.begin());
    return v;
}
template<int DIM, typename Type>
Vec<Type> MatN<DIM,Type>::getCol(unsigned int j)const{
    Vec<Type> v(this->sizeI());
    for(unsigned int i=0;i<this->sizeI();i++){
        v(i)=this->operator ()(i,j);
    }
    return v;
}
template<int DIM, typename Type>
void MatN<DIM,Type>::setRow(unsigned int i,const Vec<Type> &v){

    POP_DbgAssertMessage(v.size()==this->sizeJ(),"In Matrix::setRow, incompatible size");
    std::copy(v.begin(),v.end(),this->begin()+i*this->_domain(1));
}
template<int DIM, typename Type>
void MatN<DIM,Type>::setCol(unsigned int j,const Vec<Type>& v){
    POP_DbgAssertMessage(v.size()==this->sizeI(),"In Matrix::setCol, Incompatible size");
    for(unsigned int i=0;i<this->sizeI();i++){
        this->operator ()(i,j)=v(i);
    }
}
template<int DIM, typename Type>
void MatN<DIM,Type>::swapRow(unsigned int i_0,unsigned int i_1){
    POP_DbgAssertMessage( (i_0<this->sizeI()&&i_1<this->sizeI()),"In Matrix::swapRow, Over Range in swapRow");
    std::swap_ranges(this->begin()+i_0*this->sizeJ(), this->begin()+(i_0+1)*this->sizeJ(), this->begin()+i_1*this->sizeJ());

}
template<int DIM, typename Type>
void MatN<DIM,Type>::swapCol(unsigned int j_0,unsigned int j_1){
    POP_DbgAssertMessage( (j_0<this->sizeJ()&&j_1<this->sizeJ()),"In Matrix::swapCol, Over Range in swapCol");
    for(unsigned int i=0;i<this->sizeI();i++){
        std::swap(this->operator ()(i,j_0) ,this->operator ()(i,j_1));
    }
}
template<int DIM, typename Type>
Type MatN<DIM,Type>::minorDet(unsigned int i,unsigned int j)const{

    return this->deleteRow(i).deleteCol(j).determinant();
}
template<int DIM, typename Type>
Type MatN<DIM,Type>::cofactor(unsigned int i,unsigned int j)const{
    if( (i+j)%2==0)
        return this->minorDet(i,j);
    else
        return -this->minorDet(i,j);
}
template<int DIM, typename Type>
MatN<DIM,Type> MatN<DIM,Type>::cofactor()const{
    MatN<DIM,Type> temp(this->getDomain());
    for(unsigned int i=0;i<this->sizeI();i++)
        for(unsigned int j=0;j<this->sizeJ();j++)
        {
            temp(i,j)=this->cofactor(i,j);
        }
    return temp;
}
template<int DIM, typename Type>
MatN<DIM,Type> MatN<DIM,Type>::transpose()const
{
    const unsigned int sizei= this->sizeI();
    const unsigned int sizej= this->sizeJ();
    MatN<DIM,Type> temp(sizej,sizei);
    for(unsigned int i=0;i<sizei;i++){
        typename MatN<DIM,Type>::const_iterator this_ptr  =  this->begin() + i*sizej;
        typename MatN<DIM,Type>::iterator temp_ptr =     temp.begin() + i;
        for(unsigned int j=0;j<sizej;j++){
            * temp_ptr =  * this_ptr;
            temp_ptr   +=  sizei;
            this_ptr++;
        }
    }
    return temp;
}
template<int DIM, typename Type>
Type MatN<DIM,Type>::determinant() const{
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
template<int DIM, typename Type>
Type MatN<DIM,Type>::trace() const
{
    POP_DbgAssertMessage(this->sizeI()==this->sizeJ(),"In MatN<DIM,Type>::trace, Input MatN<DIM,Type> must be square");

    F t=0;
    for(unsigned int i=0;i<this->sizeI();i++)
    {
        t +=this->operator ()(i,i);
    }
    return t;


}
template<int DIM, typename Type>
MatN<DIM,Type> MatN<DIM,Type>::inverse()const{
    if(sizeI()==2&&sizeJ()==2){
        MatN<DIM,Type> temp(*this);
        const Type det= Type(1)/ (temp.operator[](0) * temp.operator[](3) - temp.operator[](1) * temp.operator[](2)) ;
                std::swap(temp.operator[](0),temp.operator[](3));
                temp.operator[](1)=-temp.operator[](1)*det;
        temp.operator[](2)=-temp.operator[](2)*det;
        temp.operator[](0)*=det;
        temp.operator[](3)*=det;
        return temp;
    }else if(sizeI()==3&&sizeJ()==3){
        MatN<DIM,Type> temp(*this);
        const Type det= Type(1)/(temp.operator[](0) * (temp.operator[](4)*temp.operator[](8) - temp.operator[](7) * temp.operator[](5))-temp.operator[](1) * (temp.operator[](3)*temp.operator[](8) - temp.operator[](6) * temp.operator[](5)) +temp.operator[](2) * (temp.operator[](3)*temp.operator[](7) - temp.operator[](4) * temp.operator[](6)));
                                                                                                                                                              const Type t0=  temp.operator[](4)*temp.operator[](8)-temp.operator[](7)*temp.operator[](5);
                                                       const Type t1=-(temp.operator[](3)*temp.operator[](8)-temp.operator[](6)*temp.operator[](5));
                const Type t2=  temp.operator[](3)*temp.operator[](7)-temp.operator[](6)*temp.operator[](4);
                                 const Type t3=-(temp.operator[](1)*temp.operator[](8)-temp.operator[](7)*temp.operator[](2));
                const Type t4= temp.operator[](0)*temp.operator[](8)-temp.operator[](6)*temp.operator[](2);
                const Type t5=-(temp.operator[](0)*temp.operator[](7)-temp.operator[](6)*temp.operator[](1));
                const Type t6= temp.operator[](1)*temp.operator[](5)-temp.operator[](4)*temp.operator[](2);
        const Type t7=-(temp.operator[](0)*temp.operator[](5)-temp.operator[](3)*temp.operator[](2));
                const Type t8= temp.operator[](0)*temp.operator[](4)-temp.operator[](3)*temp.operator[](1);
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
        MatN<DIM,Type> temp;
        Type det = this->determinant();
        temp = this->cofactor();
        temp = temp.transpose();
        temp/=det;
        return temp;
    }
}

template<int Dim1,int Dim2, typename Type>
MatN<Dim1+Dim2, Type>  productTensoriel(const MatN<Dim1, Type>&f,const MatN<Dim2, Type>& g)
{
    typename MatN<Dim1+Dim2, Type>::E domain;
    for(int i=0;i<Dim1;i++)
    {
        domain(i)=f.getDomain()(i);
    }
    for(int i=0;i<Dim2;i++)
    {
        domain(i+Dim1)=g.getDomain()(i);
    }
    MatN<Dim1+Dim2, Type> h(domain);
    typename MatN<Dim1+Dim2, Type>::IteratorEDomain it(h.getDomain());

    typename MatN<Dim1, Type>::E x1;
    typename MatN<Dim2, Type>::E x2;
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
template<int Dim, typename Type>
MatN<Dim, Type>  operator*(Type value, const MatN<Dim, Type>&f)
{
    return f*value;
}
template<int Dim, typename Type>
MatN<Dim, Type>  operator-(Type value, const MatN<Dim, Type>&f)
{
    MatN<Dim, Type> h(f);
    FunctorF::FunctorArithmeticConstantValueBefore<Type,Type,Type,FunctorF::FunctorSubtractionF2<Type,Type,Type> > op(value);
    std::transform (h.begin(), h.end(), h.begin(),  op);
    return h;
}
template<int Dim, typename Type>
MatN<Dim, Type>  operator+(Type value, const MatN<Dim, Type>&f)
{
    MatN<Dim, Type> h(f);
    FunctorF::FunctorArithmeticConstantValueBefore<Type,Type,Type,FunctorF::FunctorAdditionF2<Type,Type,Type> > op(value);
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
template<int DIM,typename Type>
struct NumericLimits<MatN<DIM,Type> >
{
    static F64 min() throw()
    { return -NumericLimits<Type>::maximumRange();}
    static F64 max() throw()
    { return NumericLimits<Type>::maximumRange();}
};

/*!
* \ingroup MatN
* \brief minimum value for each VecN  \f$h(x)=\min(f(x),g(x))\f$
* \param f first input matrix
* \param g first input matrix
* \return output  matrix
*
*/
template<int Dim, typename Type>
pop::MatN<Dim, Type>  minimum(const pop::MatN<Dim, Type>& f,const pop::MatN<Dim, Type>& g)throw(pop::pexception)
{
    pop::FunctionAssert(f,g,"In min");
    pop::MatN<Dim, Type> h(f);
    pop::FunctorF::FunctorMinF2<Type,Type> op;

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
template<int Dim, typename Type>
pop::MatN<Dim, Type>  maximum(const pop::MatN<Dim, Type>& f,const pop::MatN<Dim, Type>& g)throw(pop::pexception)
{
    pop::FunctionAssert(f,g,"In max");
    pop::MatN<Dim, Type> h(f);
    pop::FunctorF::FunctorMaxF2<Type,Type> op;
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
template<int Dim, typename Type>
pop::MatN<Dim, Type>  absolute(const pop::MatN<Dim, Type>& f)
{
    pop::MatN<Dim, Type> h(f.getDomain());
    std::transform (f.begin(), f.end(), h.begin(),(Type(*)(Type)) abs );
    return h;
}
/*!
* \ingroup MatN
* \brief  square value for each pixel value  \f$h(x)=\sqrt{f(x)}\f$
* \param f first input matrix
* \return output  matrix
*
*/
template<int Dim, typename Type>
pop::MatN<Dim, Type>  squareRoot(const pop::MatN<Dim, Type>& f)
{
    pop::MatN<Dim, Type> h(f.getDomain());
    std::transform (f.begin(), f.end(), h.begin(), (Type(*)(Type)) sqrt );
    return h;
}
/*!
* \ingroup MatN
* \brief  log value in e-base  for each VecN  h(x)=std::log(f(x))
* \param f first input matrix
* \return output  matrix
*
*/
template<int Dim, typename Type>
pop::MatN<Dim, Type>  log(const pop::MatN<Dim, Type>& f)
{
    pop::MatN<Dim, Type> h(f.getDomain());
    std::transform (f.begin(), f.end(), h.begin(), (Type(*)(Type)) std::log );
    return h;
}
/*!
* \ingroup MatN
* \brief  log value in 10-base  for each pixel value  h(x)=std::log10(f(x))
* \param f first input matrix
* \return output  matrix
*
*/
template<int Dim, typename Type>
pop::MatN<Dim, Type>  log10(const pop::MatN<Dim, Type>& f)
{
    pop::MatN<Dim, Type> h(f.getDomain());
    std::transform (f.begin(), f.end(), h.begin(), (Type(*)(Type)) std::log10 );
    return h;
}
/*!
* \ingroup MatN
* \brief  exponentiel value for each pixel value  h(x)=std::exp(f(x))
* \param f first input matrix
* \return output  matrix
*
*/
template<int Dim, typename Type>
pop::MatN<Dim, Type>  exp(const pop::MatN<Dim, Type>& f)
{
    pop::MatN<Dim, Type> h(f.getDomain());
    std::transform (f.begin(), f.end(), h.begin(), (Type(*)(Type)) std::exp );
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
template<int Dim, typename Type>
pop::MatN<Dim, Type>  pow(const pop::MatN<Dim, Type>& f,double exponant)
{
    pop::MatN<Dim, Type> h(f.getDomain());
    pop::Private::PowF<Type> op(exponant);
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
template<int Dim, typename Type>
double  normValue(const pop::MatN<Dim, Type>& A,int p=2)
{
    pop::Private::sumNorm<Type> op(p);
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
template<int Dim, typename Type>
double distance(const pop::MatN<Dim, Type>& A, const pop::MatN<Dim, Type>& B,int p=2)
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
template<int Dim, typename Type>
double  normPowerValue(const pop::MatN<Dim, Type>& f,int p=2)
{
    pop::Private::sumNorm<Type> op(p);
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


template <class Type>
std::ostream& operator << (std::ostream& out, const pop::MatN<1,Type>& in)
{
    Private::ConsoleOutputPixel<1,Type> output;
    for( int i =0;i<in.getDomain()(0);i++){
        output.print(out,(in)(i));
        out<<" ";
    }
    return out;
}

template <class Type>
std::ostream& operator << (std::ostream& out, const pop::MatN<2,Type>& in)
{
    Private::ConsoleOutputPixel<2,Type> output;
    for( int i =0;i<in.getDomain()(0);i++){
        for( int j =0;j<in.getDomain()(1);j++){
            output.print(out,(in)(i,j));
            out<<" ";
        }
        out<<std::endl;
    }
    return out;
}
template <class Type>
std::ostream& operator << (std::ostream& out, const pop::MatN<3,Type>& in)
{
    Private::ConsoleOutputPixel<3,Type> output;
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
template <int Dim,class Type>
std::istream& operator >> (std::istream& in,  pop::MatN<Dim,Type>& f)
{
    typename pop::MatN<Dim,pop::UI8>::IteratorEOrder it(f.getIteratorEOrder());
    typename pop::MatN<Dim,pop::UI8>::Domain d;
    for(int i=0;i<Dim;i++)
        d(i)=i;
    std::swap(d(0),d(1));
    it.setOrder(d);
    Private::ConsoleInputPixel<Dim,Type> input;
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


}
#endif
