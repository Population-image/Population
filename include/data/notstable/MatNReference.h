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
#ifndef MatNReference_HPP
#define MatNReference_HPP
#include "data/mat/MatN.h"

namespace pop
{
template<typename PixelType, int SIZEI, int SIZEJ>
class  Mat2x;


template<int Dim, typename PixelType>
class POP_EXPORTS MatNReference
{
protected:
   VecN<Dim,int> _domain;
   PixelType* _data;
public:
    typedef PixelType F;
    enum {DIM=Dim};
    typedef VecN<Dim,I32> E;
    typedef VecN<Dim,int> Domain;
    typedef MatNIteratorEDomain<E>  IteratorEDomain;
    typedef MatNIteratorEROI<MatNReference<Dim, PixelType> >  IteratorEROI;
    typedef MatNIteratorENeighborhood<E,MatNBoundaryConditionBounded> IteratorENeighborhood;// neighborhood iteration with bounded condition
    typedef MatNIteratorENeighborhood<E,MatNBoundaryConditionMirror> IteratorENeighborhoodMirror;// neighborhood iteration with mirror condition
    typedef MatNIteratorENeighborhood<E,MatNBoundaryConditionPeriodic> IteratorENeighborhoodPeriodic;// neighborhood iteration with periodic condition
    typedef MatNIteratorENeighborhoodAmoebas<MatN<DIM,PixelType> > IteratorENeighborhoodAmoebas;
    typedef MatNIteratorEOrder<E> IteratorEOrder;
    typedef MatNBoundaryCondition BoundaryCondition;
    typedef MatNIteratorERectangle<E> IteratorERectangle;
    typedef typename Vec<PixelType>::iterator iterator;
    typedef typename Vec<PixelType>::value_type					 value_type;
    typedef typename Vec<PixelType>::pointer           pointer;
    typedef typename Vec<PixelType>::const_pointer     const_pointer;
    typedef typename Vec<PixelType>::reference         reference;
    typedef typename Vec<PixelType>::const_reference   const_reference;
    typedef typename Vec<PixelType>::const_iterator const_iterator;
    typedef typename Vec<PixelType>::const_reverse_iterator  const_reverse_iterator;
    typedef typename Vec<PixelType>::reverse_iterator		 reverse_iterator;
    typedef typename Vec<PixelType>::size_type					 size_type;
    typedef typename Vec<PixelType>::difference_type				 difference_type;
    typedef typename Vec<PixelType>::allocator_type                        		 allocator_type;

	MatNReference()
	:_domain(0),_data(NULL){}

    MatNReference(const VecN<Dim,int>& domain,PixelType *v);
    MatN<Dim,PixelType> toMatN()const;

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
    unsigned int size()const;
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
    \param index vector index
    \return pixel/voxel value
    *
    * access the reference of the pixel/voxel value at the vector index (Vec contains pixel values)
    */
    inline PixelType & operator ()(unsigned int index);
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
    //@}

    //-------------------------------------
    //
    //! \name Arithmetics
    //@{
    //-------------------------------------


    /*!
    * \param value value
    * \return this matrix
    *
    * Basic assignement of all pixel/voxel values by \a value
    */
    MatNReference<Dim, PixelType>&  operator=(PixelType value);
    /*!
    * \param value value
    * \return this matrix
    *
    * Basic assignement of all pixel/voxel values by \a value
    */
    MatNReference<Dim, PixelType>&  fill(PixelType value);
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
    bool operator==(const MatNReference<Dim, PixelType>& f)const;
    /*!
    \param f input matrix
    \return boolean
    *
    * Difference operator true for at least on x in E f(x)!=(*this)(x), false otherwise
    */
    bool operator!=(const MatNReference<Dim, PixelType>& f)const;
    /*!
    \param f input matrix
    \return object reference
    *
    * Addition assignment h(x)+=f(x)
    */
    MatNReference<Dim, PixelType>&  operator+=(const MatNReference<Dim, PixelType>& f);
    /*!
    * \param f input matrix
    * \return object
    *
    *  Addition h(x)= (*this)(x)+f(x)
    */
    MatN<Dim, PixelType>  operator+(const MatNReference<Dim, PixelType>& f)const;
    /*!
    * \param value input value
    * \return object reference
    *
    * Addition assignment h(x)+=value
    */
    MatNReference<Dim, PixelType>& operator+=(PixelType value);
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
    MatNReference<Dim, PixelType>&  operator-=(const MatNReference<Dim, PixelType>& f);
    /*!
    \param value input value
    \return object reference
    *
    * Subtraction assignment h(x)-=value
    */
    MatNReference<Dim, PixelType>&  operator-=(PixelType value);
    /*!
    * \param f input matrix
    * \return output matrix
    *
    *  Subtraction h(x)= (*this)(x)-f(x)
    */
    MatN<Dim, PixelType>  operator-(const MatNReference<Dim, PixelType>& f)const;
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
    MatNReference  operator*(const MatNReference &m)const;
    /*!
    * \param m  other matrix
    * \return output matrix
    *
    *  matrix multiplication see http://en.wikipedia.org/wiki/Matrix_multiplication
    */
    MatNReference & operator*=(const MatNReference &m);
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
    MatNReference  multTermByTerm(const MatNReference& f)const;
    /*!
    \param value input value
    \return object reference
    *
    * Multiplication assignment h(x)*=value
    */
    MatNReference<Dim, PixelType>&  operator*=(PixelType  value);
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
    MatN<Dim, PixelType>  divTermByTerm(const MatNReference& f);
    /*!
    \param value input value
    \return object reference
    *
    * Division assignment h(x)/=value
    */
    MatNReference<Dim, PixelType>&  operator/=(PixelType value);
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
    MatNReference deleteRow(unsigned int i)const;
    /*!
    * \param j  column entry
    *
    * delete the column of index j
    */
    MatNReference deleteCol(unsigned int j)const;
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
    MatNReference cofactor()const;
    /*!
    *
    * the ith row, jth column element of transpose matrix is the jth row, ith column element of matrix:
    */
    MatNReference transpose()const;
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
    MatNReference inverse()const;

    /*! \brief  \f$I_n = \begin{bmatrix}1 & 0 & \cdots & 0 \\0 & 1 & \cdots & 0 \\\vdots & \vdots & \ddots & \vdots \\0 & 0 & \cdots & 1 \end{bmatrix}\f$
     * \param size_mat size of the output matrix
     * \return  Identity matrix
     *
     *  Generate the identity matrix or unit matrix of square matrix with the given size for size!=0 or this matrix size with ones on the main diagonal and zeros elsewhere
    */
    MatNReference identity(int size_mat=0)const;

    //@}

};





template<int Dim, typename PixelType>
MatNReference<Dim,PixelType>::MatNReference(const VecN<Dim,int>& domain,PixelType* v)
    :_domain(domain),_data(v)
{
}
template<int Dim, typename PixelType>
MatN<Dim,PixelType> MatNReference<Dim,PixelType>::toMatN()const
{
    return MatN<Dim,PixelType>(this->getDomain(),this->_data);
}
template<int Dim, typename PixelType>
typename MatNReference<Dim,PixelType>::Domain  MatNReference<Dim,PixelType>::getDomain()
const
{
    return _domain;
}
template<int Dim, typename PixelType>
unsigned int MatNReference<Dim,PixelType>::size()const{
    return this->getDomain().multCoordinate();
}

template<int Dim, typename PixelType>
unsigned int MatNReference<Dim,PixelType>::sizeI()const{
    return this->getDomain()[0];
}
template<int Dim, typename PixelType>
unsigned int MatNReference<Dim,PixelType>::rows()const{
    return this->getDomain()[0];
}
template<int Dim, typename PixelType>
unsigned int MatNReference<Dim,PixelType>::sizeJ()const{
    return this->getDomain()[1];
}
template<int Dim, typename PixelType>
unsigned int MatNReference<Dim,PixelType>::columns()const{
    return this->getDomain()[1];
}
template<int Dim, typename PixelType>
unsigned int MatNReference<Dim,PixelType>::sizeK()const{
    POP_DbgAssert(Dim==3);
    return this->getDomain()[2];
}
template<int Dim, typename PixelType>
unsigned int MatNReference<Dim,PixelType>::depth()const{
    POP_DbgAssert(Dim==3);
    return this->getDomain()[2];
}
template<int Dim, typename PixelType>
bool MatNReference<Dim,PixelType>::isValid(const E & x)const{
    if(x.allSuperiorEqual(E(0)) && x.allInferior(this->getDomain()))
        return true;
    else
        return false;
}
template<int Dim, typename PixelType>
bool MatNReference<Dim,PixelType>::isValid(int i,int j)const{
    if(i>=0&&j>=0 && i<static_cast<int>(sizeI())&& j<static_cast<int>(sizeJ()))
        return true;
    else
        return false;
}
template<int Dim, typename PixelType>
bool MatNReference<Dim,PixelType>::isValid(int i,int j,int k)const{
    if(i>=0&&j>=0&&k>=0 && i<static_cast<int>(sizeI())&& j<static_cast<int>(sizeJ())&&k<static_cast<int>(sizeK()))
        return true;
    else
        return false;
}

template<int Dim, typename PixelType>
bool MatNReference<Dim,PixelType>::isEmpty()const{
    if(_domain.multCoordinate()==0)
        return true;
    else
        return false;
}
template<int Dim, typename PixelType>
void MatNReference<Dim,PixelType>::clear(){
    _domain=0;
    Vec<PixelType>::clear();
}
//template<int Dim, typename PixelType>
//PixelType & MatNReference<Dim,PixelType>::operator ()(int i)
//{
//    return  this->_data[i];
//}
//template<int Dim, typename PixelType>
//const PixelType & MatNReference<Dim,PixelType>::operator ()(int i)const
//{
//    return  this->_data[i];
//}
template<int Dim, typename PixelType>
PixelType & MatNReference<Dim,PixelType>::operator ()(const VecN<Dim,int> & x)
{
    POP_DbgAssert( x.allSuperiorEqual( E(0))&&x.allInferior(getDomain()));
    return  this->_data[VecNIndice<Dim>::VecN2Indice(_domain,x)];
}

template<int Dim, typename PixelType>
const PixelType & MatNReference<Dim,PixelType>::operator ()( const VecN<Dim,int>& x)
const
{
    POP_DbgAssert( x.allSuperiorEqual(E(0))&&x.allInferior(getDomain()));
    return  this->_data[VecNIndice<Dim>::VecN2Indice(_domain,x)];
}
template<int Dim, typename PixelType>
PixelType & MatNReference<Dim,PixelType>::operator ()(unsigned int i,unsigned int j)
{
    POP_DbgAssert( i<(sizeI())&&j<(sizeJ()));
    return  this->_data[j+i*_domain(1)];
}
template<int Dim, typename PixelType>
const PixelType & MatNReference<Dim,PixelType>::operator ()(unsigned int i,unsigned int j)const
{
    POP_DbgAssert( i<(sizeI())&&j<(sizeJ()));
    return  this->_data[j+i*_domain(1)];
}
template<int Dim, typename PixelType>
PixelType & MatNReference<Dim,PixelType>::operator ()(unsigned int i,unsigned int j,unsigned int k)
{
    POP_DbgAssert(  i<(sizeI())&&j<(sizeJ())&&k<(sizeK()));
    return  this->_data[j+i*_domain(1)+k*_domain(0)*_domain(1)];
}

template<int Dim, typename PixelType>
const PixelType & MatNReference<Dim,PixelType>::operator ()(unsigned int i,unsigned int j,unsigned int k)const
{
    POP_DbgAssert(  i<(sizeI())&&j<(sizeJ())&&k<(sizeK()));
    return  this->_data[j+i*_domain(1)+k*_domain(0)*_domain(1)];
}


template<int Dim, typename PixelType>
PixelType & MatNReference<Dim,PixelType>::operator ()(unsigned int index)
{
    POP_DbgAssert( index<this->size());
    return this->_data[index];
}
template<int Dim, typename PixelType>
const PixelType & MatNReference<Dim,PixelType>::operator ()(unsigned int index)const
{
    POP_DbgAssert( index<this->size());
    return this->_data[index];
}
template<int Dim, typename PixelType>
PixelType MatNReference<Dim,PixelType>::interpolationBilinear(const VecN<DIM,F32> xf)const
{
    VecN<PowerGP<2,Dim>::value,std::pair<F32,VecN<Dim,I32> > > v_out =MatNInterpolationBiliniear::getWeightPosition(this->getDomain(),xf);
    typedef typename FunctionTypeTraitsSubstituteF<PixelType,F32>::Result  PixelTypeFloat;
    PixelTypeFloat value=0;
    for( int i=0;i<PowerGP<2,Dim>::value;i++){
        value+=m(v_out(i).second)*v_out(i).first;
    }
    return ArithmeticsSaturation<PixelType,PixelTypeFloat>::Range(value);

}
template<int Dim, typename PixelType>
PixelType *  MatNReference<Dim,PixelType>::data()
{
    return this->_data;
}
template<int Dim, typename PixelType>
const PixelType *  MatNReference<Dim,PixelType>::data()
const
{
    return this->_data;
}

template<int Dim, typename PixelType>
typename MatNReference<Dim,PixelType>::IteratorEDomain MatNReference<Dim,PixelType>::getIteratorEDomain()const
{
    return IteratorEDomain(getDomain());
}
template<int Dim, typename PixelType>
typename MatNReference<Dim,PixelType>::IteratorEROI MatNReference<Dim,PixelType>::getIteratorEROI()const
{
    return IteratorEROI(*this);
}
template<int Dim, typename PixelType>
typename MatNReference<Dim,PixelType>::IteratorENeighborhood MatNReference<Dim,PixelType>::getIteratorENeighborhood(F32 radius ,int norm )const
{
    return IteratorENeighborhood(getDomain(),radius , norm);
}
template<int Dim, typename PixelType>
template<typename Type1>
typename MatNReference<Dim,PixelType>::IteratorENeighborhood MatNReference<Dim,PixelType>::getIteratorENeighborhood(const MatN<Dim,Type1> & structural_element,int dilate )const
{
    Vec<E> _tab;
    typename MatNReference<Dim,Type1>::IteratorEDomain it(structural_element.getDomain());
    typename MatNReference<Dim,Type1>::E center = VecN<Dim,F32>(structural_element.getDomain()-1)*0.5;
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
typename MatNReference<Dim,PixelType>::IteratorEOrder MatNReference<Dim,PixelType>::getIteratorEOrder(int coordinatelastloop,int direction)const
{
    return IteratorEOrder(getDomain(),coordinatelastloop,direction);
}
template<int Dim, typename PixelType>
typename MatNReference<Dim,PixelType>::IteratorERectangle MatNReference<Dim,PixelType>::getIteratorERectangle(const E & xmin,const E & xmax )const
{
    return IteratorERectangle(std::make_pair(xmin,xmax));
}
template<int Dim, typename PixelType>
typename MatNReference<Dim,PixelType>::IteratorENeighborhoodAmoebas MatNReference<Dim,PixelType>::getIteratorENeighborhoodAmoebas(F32 distance_max,F32 lambda_param )const
{
    return IteratorENeighborhoodAmoebas(*this,distance_max,lambda_param );
}


template<int Dim, typename PixelType>
MatNReference<Dim, PixelType>&  MatNReference<Dim,PixelType>::operator=(PixelType value)
{
   std::fill (this->_data,this->_data+_domain.multCoordinate(),value);
    return *this;
}
template<int Dim, typename PixelType>
MatNReference<Dim, PixelType>&  MatNReference<Dim,PixelType>::fill(PixelType value)
{
    std::fill (this->_data,this->_data+_domain.multCoordinate(),value);
    return *this;
}
template<int Dim, typename PixelType>
MatN<Dim, PixelType>  MatNReference<Dim,PixelType>::opposite(int mode)const
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
bool MatNReference<Dim,PixelType>::operator==(const MatNReference<Dim, PixelType>& f)const
{
    FunctionAssert(f,*this,"In MatNReference::operator==");
    return std::equal (f.begin(), f.end(), this->begin());
}
template<int Dim, typename PixelType>
bool MatNReference<Dim,PixelType>::operator!=(const MatNReference<Dim, PixelType>& f)const
{
    FunctionAssert(f,*this,"In MatNReference::operator==");
    return !std::equal (f.begin(), f.end(), this->begin());
}
template<int Dim, typename PixelType>
MatNReference<Dim, PixelType>&  MatNReference<Dim,PixelType>::operator+=(const MatNReference<Dim, PixelType>& f)
{

    FunctionAssert(f,*this,"In MatNReference::operator+=");
    FunctorF::FunctorAdditionF2<PixelType,PixelType,PixelType> op;
    std::transform (this->begin(), this->end(), f.begin(),this->begin(),  op);
    return *this;
}
template<int Dim, typename PixelType>
MatN<Dim, PixelType>  MatNReference<Dim,PixelType>::operator+(const MatNReference<Dim, PixelType>& f)const{
    MatN<Dim, PixelType> h=this->toMatN();
    h +=f;
    return h;
}
template<int Dim, typename PixelType>
MatNReference<Dim, PixelType>& MatNReference<Dim,PixelType>::operator+=(PixelType value)
{
    FunctorF::FunctorArithmeticConstantValueAfter<PixelType,PixelType,PixelType,FunctorF::FunctorAdditionF2<PixelType,PixelType,PixelType> > op(value);
    std::transform (this->begin(), this->end(), this->begin(),  op);
    return *this;
}
template<int Dim, typename PixelType>
MatN<Dim, PixelType>  MatNReference<Dim,PixelType>::operator+(PixelType value)const{
    MatN<Dim, PixelType> h=this->toMatN();
    h +=value;
    return h;
}
template<int Dim, typename PixelType>
MatNReference<Dim, PixelType>&  MatNReference<Dim,PixelType>::operator-=(const MatNReference<Dim, PixelType>& f)
{
    FunctionAssert(f,*this,"In MatNReference::operator-=");
    FunctorF::FunctorSubtractionF2<PixelType,PixelType,PixelType> op;
    std::transform (this->begin(), this->end(), f.begin(),this->begin(),  op);
    return *this;
}
template<int Dim, typename PixelType>
MatNReference<Dim, PixelType>&  MatNReference<Dim,PixelType>::operator-=(PixelType value)
{
    FunctorF::FunctorArithmeticConstantValueAfter<PixelType,PixelType,PixelType,FunctorF::FunctorSubtractionF2<PixelType,PixelType,PixelType> > op(value);
    std::transform (this->begin(), this->end(), this->begin(),  op);
    return *this;
}

template<int Dim, typename PixelType>
MatN<Dim, PixelType>  MatNReference<Dim,PixelType>::operator-(const MatNReference<Dim, PixelType>& f)const{
    MatN<Dim, PixelType> h=this->toMatN();
    h -=f;
    return h;
}
template<int Dim, typename PixelType>
MatN<Dim, PixelType>  MatNReference<Dim,PixelType>::operator-()const{
    MatN<Dim, PixelType> h(this->getDomain(),PixelType(0));
    h -=*this;
    return h;
}
template<int Dim, typename PixelType>
MatN<Dim, PixelType>  MatNReference<Dim,PixelType>::operator-(PixelType value)const{
    MatN<Dim, PixelType> h=this->toMatN();
    h -=value;
    return h;
}

template<int Dim, typename PixelType>
MatNReference<Dim,PixelType>  MatNReference<Dim,PixelType>::operator*(const MatNReference<Dim,PixelType> &m)const
{
    POP_DbgAssertMessage(DIM==2&&this->sizeJ()==m.sizeI() ,"In Matrix::operator*, Not compatible size for the operator * of the class Matrix (A_{n,k}*B_{k,p})");
    MatNReference<Dim,PixelType> mtrans = m.transpose();
    MatNReference<Dim,PixelType> mout(this->sizeI(),m.sizeJ());
    PixelType sum = 0;
    int i,j;
    typename MatNReference::const_iterator this_it,mtrans_it;
#if defined(HAVE_OPENMP)
#pragma omp parallel shared(mtrans,mout) private(i,j,sum,this_it,mtrans_it)
#endif
    {
#if defined(HAVE_OPENMP)
#pragma omp for schedule (static)
#endif
        for(i=0;i<static_cast<int>(this->sizeI());i++){
            for(j=0;j<static_cast<int>(m.sizeJ());j++){
                sum = 0;
                this_it  = this->begin() +  i*this->sizeJ();
                mtrans_it= mtrans.begin() + j*mtrans.sizeJ();
                for(unsigned int k=0;k<this->sizeJ();k++){
                    sum+=(* this_it) * (* mtrans_it);
                    this_it++;
                    mtrans_it++;
                }
                mout(i,j)=sum;
            }

        }
    }
    return mout;


}
template<int Dim, typename PixelType>
MatNReference<Dim,PixelType> & MatNReference<Dim,PixelType>::operator*=(const MatNReference<Dim,PixelType> &m)
{
    *this = this->operator *(m);
    return *this;
}
template<int Dim, typename PixelType>
Vec<PixelType>  MatNReference<Dim,PixelType>::operator*(const Vec<PixelType> & v)const{
    POP_DbgAssertMessage(DIM==2&&this->sizeJ()==v.size() ,"In Matrix::operator*, Not compatible size for the operator *=(Vec) of the class Matrix (A_{n,k}*v_{k})");
    Vec<PixelType> temp(this->sizeI());
    for(unsigned int i=0;i<this->sizeI();i++){
        PixelType sum = 0;
        typename MatNReference::const_iterator this_it  = this->begin() +  i*this->sizeJ();
        typename Vec<PixelType>::const_iterator mtrans_it= v.begin();
        for(;mtrans_it!=v.end();this_it++,mtrans_it++){
            sum+=(* this_it) * (* mtrans_it);
        }
        temp(i)=sum;
    }
    return temp;
}

template<int Dim, typename PixelType>
MatNReference<Dim,PixelType>  MatNReference<Dim,PixelType>::multTermByTerm(const MatNReference& f)const{
    FunctionAssert(f,*this,"In MatNReference::operator*=");
    FunctorF::FunctorMultiplicationF2<PixelType,PixelType,PixelType> op;
    MatNReference<Dim,PixelType> out(*this);
    std::transform (out.begin(), out.end(), f.begin(),out.begin(),  op);
    return out;
}
template<int Dim, typename PixelType>
MatNReference<Dim, PixelType>&  MatNReference<Dim,PixelType>::operator*=(PixelType  value)
{
    FunctorF::FunctorArithmeticConstantValueAfter<PixelType,PixelType,PixelType,FunctorF::FunctorMultiplicationF2<PixelType,PixelType,PixelType> > op(value);
    std::transform (this->begin(), this->end(), this->begin(),  op);
    return *this;
}
template<int Dim, typename PixelType>
MatN<Dim, PixelType>  MatNReference<Dim,PixelType>::operator*(PixelType value)const{
    MatN<Dim, PixelType> h=this->toMatN();
    h *=value;
    return h;
}
template<int Dim, typename PixelType>
MatN<Dim, PixelType>  MatNReference<Dim,PixelType>::divTermByTerm(const MatNReference& f){
    FunctionAssert(f,*this,"In MatNReference::divTermByTerm");
    FunctorF::FunctorDivisionF2<PixelType,PixelType,PixelType> op;
    std::transform (this->begin(), this->end(), f.begin(),this->begin(),  op);
    return *this;
}
template<int Dim, typename PixelType>
MatNReference<Dim, PixelType>&  MatNReference<Dim,PixelType>::operator/=(PixelType value)
{
    FunctorF::FunctorArithmeticConstantValueAfter<PixelType,PixelType,PixelType,FunctorF::FunctorDivisionF2<PixelType,PixelType,PixelType> > op(value);
    std::transform (this->begin(), this->end(), this->begin(),  op);
    return *this;
}
template<int Dim, typename PixelType>
MatN<Dim, PixelType>  MatNReference<Dim,PixelType>::operator/(PixelType value)const{
    MatN<Dim, PixelType> h=this->toMatN();
    h /=value;
    return h;
}



template<int DIM, typename PixelType>
MatNReference<DIM,PixelType>  MatNReference<DIM,PixelType>::deleteRow(unsigned int i)const{
    POP_DbgAssert(i<sizeI());
    MatNReference<DIM,PixelType> temp(*this);
    temp._domain(0)--;
    temp.erase( temp.begin()+i*temp._domain(1), temp.begin()+(i+1)*temp._domain(1)  );
    return temp;
}
template<int DIM, typename PixelType>
MatNReference<DIM,PixelType>  MatNReference<DIM,PixelType>::deleteCol(unsigned int j)const{
    POP_DbgAssert(j<sizeJ());
    MatNReference<DIM,PixelType> temp(this->sizeI(),this->sizeJ()-1);

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
Vec<PixelType>  MatNReference<DIM,PixelType>::getRow(unsigned int i)const{
    Vec<PixelType> v(this->sizeJ());
    std::copy(this->begin()+i*this->_domain(1), this->begin()+(i+1)*this->_domain(1),v.begin());
    return v;
}
template<int DIM, typename PixelType>
Vec<PixelType>  MatNReference<DIM,PixelType>::getCol(unsigned int j)const{
    Vec<PixelType> v(this->sizeI());
    for(unsigned int i=0;i<this->sizeI();i++){
        v(i)=this->operator ()(i,j);
    }
    return v;
}
template<int DIM, typename PixelType>
void  MatNReference<DIM,PixelType>::setRow(unsigned int i,const Vec<PixelType> &v){

    POP_DbgAssertMessage(v.size()==this->sizeJ(),"In Matrix::setRow, incompatible size");
    std::copy(v.begin(),v.end(),this->begin()+i*this->_domain(1));
}
template<int DIM, typename PixelType>
void  MatNReference<DIM,PixelType>::setCol(unsigned int j,const Vec<PixelType>& v){
    POP_DbgAssertMessage(v.size()==this->sizeI(),"In Matrix::setCol, Incompatible size");
    for(unsigned int i=0;i<this->sizeI();i++){
        this->operator ()(i,j)=v(i);
    }
}
template<int DIM, typename PixelType>
void  MatNReference<DIM,PixelType>::swapRow(unsigned int i_0,unsigned int i_1){
    POP_DbgAssertMessage( (i_0<this->sizeI()&&i_1<this->sizeI()),"In Matrix::swapRow, Over Range in swapRow");
    std::swap_ranges(this->begin()+i_0*this->sizeJ(), this->begin()+(i_0+1)*this->sizeJ(), this->begin()+i_1*this->sizeJ());

}
template<int DIM, typename PixelType>
void  MatNReference<DIM,PixelType>::swapCol(unsigned int j_0,unsigned int j_1){
    POP_DbgAssertMessage( (j_0<this->sizeJ()&&j_1<this->sizeJ()),"In Matrix::swapCol, Over Range in swapCol");
    for(unsigned int i=0;i<this->sizeI();i++){
        std::swap(this->operator ()(i,j_0) ,this->operator ()(i,j_1));
    }
}
template<int DIM, typename PixelType>
PixelType MatNReference<DIM,PixelType>::minorDet(unsigned int i,unsigned int j)const{

    return this->deleteRow(i).deleteCol(j).determinant();
}
template<int DIM, typename PixelType>
PixelType MatNReference<DIM,PixelType>::cofactor(unsigned int i,unsigned int j)const{
    if( (i+j)%2==0)
        return this->minorDet(i,j);
    else
        return -this->minorDet(i,j);
}
template<int DIM, typename PixelType>
MatNReference<DIM,PixelType>  MatNReference<DIM,PixelType>::cofactor()const{
    MatNReference<DIM,PixelType> temp(this->getDomain());
    for(unsigned int i=0;i<this->sizeI();i++)
        for(unsigned int j=0;j<this->sizeJ();j++)
        {
            temp(i,j)=this->cofactor(i,j);
        }
    return temp;
}
template<int DIM, typename PixelType>
MatNReference<DIM,PixelType>  MatNReference<DIM,PixelType>::transpose()const
{
    const unsigned int sizei= this->sizeI();
    const unsigned int sizej= this->sizeJ();
    MatNReference<DIM,PixelType> temp(sizej,sizei);
    for(unsigned int i=0;i<sizei;i++){
        typename  MatNReference<DIM,PixelType>::const_iterator this_ptr  =  this->begin() + i*sizej;
        typename  MatNReference<DIM,PixelType>::const_iterator this_end_ptr  =  this_ptr + sizej;
        typename  MatNReference<DIM,PixelType>::iterator temp_ptr =     temp.begin() + i;
        while(this_ptr!=this_end_ptr){
            * temp_ptr =  * this_ptr;
            temp_ptr   +=  sizei;
            this_ptr++;
        }
    }
    return temp;
}
template<int DIM, typename PixelType>
PixelType MatNReference<DIM,PixelType>::determinant() const{
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
PixelType MatNReference<DIM,PixelType>::trace() const
{
    POP_DbgAssertMessage(this->sizeI()==this->sizeJ(),"In  MatNReference<DIM,PixelType>::trace, Input  MatNReference<DIM,PixelType> must be square");

    F t=0;
    for(unsigned int i=0;i<this->sizeI();i++)
    {
        t +=this->operator ()(i,i);
    }
    return t;


}
template<int DIM, typename PixelType>
MatNReference<DIM,PixelType>  MatNReference<DIM,PixelType>::identity(int size_mat)const{
    if(size_mat==0)
        size_mat=this->sizeI();
    MatNReference<DIM,PixelType> I(size_mat,size_mat);
    for(unsigned int i=0;i<I.sizeI();i++){
        I(i,i)=1;
    }
    return I;
}

template<int DIM, typename PixelType>
MatNReference<DIM,PixelType>  MatNReference<DIM,PixelType>::inverse()const{
    if(sizeI()==2&&sizeJ()==2){
        MatNReference<DIM,PixelType> temp(*this);
        const PixelType det= PixelType(1)/ (temp.operator[](0) * temp.operator[](3) - temp.operator[](1) * temp.operator[](2)) ;
                std::swap(temp.operator[](0),temp.operator[](3));
                temp.operator[](1)=-temp.operator[](1)*det;
        temp.operator[](2)=-temp.operator[](2)*det;
        temp.operator[](0)*=det;
        temp.operator[](3)*=det;
        return temp;
    }else if(sizeI()==3&&sizeJ()==3){
        MatNReference<DIM,PixelType > temp(*this);
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
        MatNReference<DIM,PixelType> temp;
        PixelType det = this->determinant();
        temp = this->cofactor();
        temp = temp.transpose();
        temp/=det;
        return temp;
    }
}

template<int Dim1,int Dim2, typename PixelType>
MatNReference<Dim1+Dim2, PixelType>  productTensoriel(const MatNReference<Dim1, PixelType>&f,const MatNReference<Dim2, PixelType>& g)
{
    typename MatNReference<Dim1+Dim2, PixelType>::E domain;
    for(int i=0;i<Dim1;i++)
    {
        domain(i)=f.getDomain()(i);
    }
    for(int i=0;i<Dim2;i++)
    {
        domain(i+Dim1)=g.getDomain()(i);
    }
    MatNReference<Dim1+Dim2, PixelType> h(domain);
    typename MatNReference<Dim1+Dim2, PixelType>::IteratorEDomain it(h.getDomain());

    typename MatNReference<Dim1, PixelType>::E x1;
    typename MatNReference<Dim2, PixelType>::E x2;
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
MatNReference<Dim, PixelType>  operator*(PixelType value, const MatNReference<Dim, PixelType>&f)
{
    return f*value;
}
template<int Dim, typename PixelType>
MatNReference<Dim, PixelType>  operator-(PixelType value, const MatNReference<Dim, PixelType>&f)
{
    MatN<Dim, PixelType> h(f);
    FunctorF::FunctorArithmeticConstantValueBefore<PixelType,PixelType,PixelType,FunctorF::FunctorSubtractionF2<PixelType,PixelType,PixelType> > op(value);
    std::transform (h.begin(), h.end(), h.begin(),  op);
    return h;
}
template<int Dim, typename PixelType>
MatNReference<Dim, PixelType>  operator+(PixelType value, const MatNReference<Dim, PixelType>&f)
{
    MatN<Dim, PixelType> h(f);
    FunctorF::FunctorArithmeticConstantValueBefore<PixelType,PixelType,PixelType,FunctorF::FunctorAdditionF2<PixelType,PixelType,PixelType> > op(value);
    std::transform (h.begin(), h.end(), h.begin(),  op);
    return h;
}
template<int D,typename F1, typename F2>
struct FunctionTypeTraitsSubstituteF<MatNReference<D,F1>,F2 >
{
    typedef MatNReference<D,F2> Result;
};
template<int D1,typename F1, int D2>
struct FunctionTypeTraitsSubstituteDIM<MatNReference<D1,F1>,D2 >
{
    typedef MatNReference<D2,F1> Result;
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
void FunctionAssert(const MatNReference<D1,F1> & f,const MatNReference<D2,F2> & g ,std::string message)
{
    POP_DbgAssertMessage(D1==D2,"matrixs must have the same Dim\n"+message);
    POP_DbgAssertMessage(f.getDomain()==g.getDomain(),"matrixs must have the same domain\n"+message);
}
template<int DIM,typename PixelType>
struct NumericLimits< MatNReference<DIM,PixelType> >
{
    static F32 (min)() throw()
    { return -NumericLimits<PixelType>::maximumRange();}
    static F32 (max)() throw()
    { return NumericLimits<PixelType>::maximumRange();}
};

/*!
* \ingroup MatNReference
* \brief minimum value for each VecN  \f$h(x)=\min(f(x),g(x))\f$
* \param f first input matrix
* \param g first input matrix
* \return output  matrix
*
*/
template<int Dim, typename PixelType>
pop::MatN<Dim, PixelType>  minimum(const pop::MatNReference<Dim, PixelType>& f,const pop::MatNReference<Dim, PixelType>& g)
{
    pop::FunctionAssert(f,g,"In min");
    pop::MatNReference<Dim, PixelType> h(f);
    pop::FunctorF::FunctorMinF2<PixelType,PixelType> op;

    std::transform (h.begin(), h.end(), g.begin(),h.begin(), op );
    return h;
}
template<int Dim, typename PixelType>
pop::MatNReference<Dim, PixelType>  maximum(const pop::MatNReference<Dim, PixelType>& f,const pop::MatNReference<Dim, PixelType>& g)
{
    pop::FunctionAssert(f,g,"In max");
    pop::MatNReference<Dim, PixelType> h(f);
    pop::FunctorF::FunctorMaxF2<PixelType,PixelType> op;
    std::transform (h.begin(), h.end(), g.begin(),h.begin(), op );
    return h;
}
template<int Dim, typename PixelType>
pop::MatN<Dim, PixelType>  absolute(const pop::MatNReference<Dim, PixelType>& f)
{
    pop::MatN<Dim, PixelType> h =f.toMatN();
    std::transform (f.begin(), f.end(), h.begin(),(PixelType(*)(PixelType)) abs );
    return h;
}
template<int Dim, typename PixelType>
pop::MatN<Dim, PixelType>  squareRoot(const pop::MatNReference<Dim, PixelType>& f)
{
    pop::MatN<Dim, PixelType> h =f.toMatN();
    std::transform (f.begin(), f.end(), h.begin(), (PixelType(*)(PixelType)) sqrt );
    return h;
}

template<int Dim, typename PixelType>
pop::MatN<Dim, PixelType>  log(const pop::MatNReference<Dim, PixelType>& f)
{
    pop::MatN<Dim, PixelType> h =f.toMatN();
    std::transform (f.begin(), f.end(), h.begin(), (PixelType(*)(PixelType)) std::log );
    return h;
}

template<int Dim, typename PixelType>
pop::MatN<Dim, PixelType>  log10(const pop::MatNReference<Dim, PixelType>& f)
{
    pop::MatN<Dim, PixelType> h =f.toMatN();
    std::transform (f.begin(), f.end(), h.begin(), (PixelType(*)(PixelType)) std::log10 );
    return h;
}
template<int Dim, typename PixelType>
pop::MatN<Dim, PixelType>  exp(const pop::MatNReference<Dim, PixelType>& f)
{
    pop::MatN<Dim, PixelType> h =f.toMatN();
    std::transform (f.begin(), f.end(), h.begin(), (PixelType(*)(PixelType)) std::exp );
    return h;
}
template<int Dim, typename PixelType>
pop::MatN<Dim, PixelType>  pow(const pop::MatNReference<Dim, PixelType>& f,F32 exponant)
{
    pop::MatN<Dim, PixelType> h =f.toMatN();
    pop::Private::PowF<PixelType> op(exponant);
    std::transform (f.begin(), f.end(), h.begin(), op );
    return h;
}


template<int Dim, typename PixelType>
F32  normValue(const pop::MatNReference<Dim, PixelType>& A,int p=2)
{
    pop::Private::sumNorm<PixelType> op(p);
    if(p!=0)
        return std::pow(std::accumulate(A.begin(),A.end(),0.,op),1./p);
    else
        return std::accumulate(A.begin(),A.end(),0.,op);

}
template<int Dim, typename PixelType>
F32 distance(const pop::MatNReference<Dim, PixelType>& A, const pop::MatNReference<Dim, PixelType>& B,int p=2)
{
    return normValue(A-B,p);
}


template<int Dim, typename PixelType>
F32  normPowerValue(const pop::MatNReference<Dim, PixelType>& f,int p=2)
{
    pop::Private::sumNorm<PixelType> op(p);
    return std::accumulate(f.begin(),f.end(),0.,op);
}


template <class PixelType>
std::ostream& operator << (std::ostream& out, const pop::MatNReference<2,PixelType>& in)
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













template<typename Type1,typename Type2,typename FunctorAccumulatorF,typename IteratorEGlobal,typename IteratorELocal>
void forEachGlobalToLocal(const MatNReference<2,Type1> & f, MatNReference<2,Type2> &  h, FunctorAccumulatorF facc,IteratorELocal  it_local,typename MatNReference<2,Type1>::IteratorEDomain ){
    int i,j;
#if defined(HAVE_OPENMP)
#pragma omp parallel shared(f,h) private(i,j) firstprivate(facc,it_local)
#endif
    {
#if defined(HAVE_OPENMP)
#pragma omp for schedule (static)
#endif
        for(i=0;i<f.sizeI();i++){
            for(j=0;j<f.sizeJ();j++){
                it_local.init(Vec2I32(i,j));
                h(i,j)=forEachFunctorAccumulator(f,facc,it_local);
            }
        }
    }
}
template<typename Type1,typename Type2,typename FunctorAccumulatorF,typename IteratorEGlobal,typename IteratorELocal>
void forEachGlobalToLocal(const MatNReference<3,Type1> & f, MatNReference<3,Type2> &  h, FunctorAccumulatorF facc,IteratorELocal  it_local,typename MatNReference<3,Type1>::IteratorEDomain ){
    int i,j,k;
#if defined(HAVE_OPENMP)
#pragma omp parallel shared(f,h) private(i,j,k) firstprivate(facc,it_local)
#endif
    {
#if defined(HAVE_OPENMP)
#pragma omp for schedule (static)
#endif
        for(i=0;i<f.sizeI();i++){
            for(j=0;j<f.sizeJ();j++){
                for(k=0;k<f.sizeK();k++){
                    it_local.init(Vec3I32(i,j,k));
                    h(i,j,k)=forEachFunctorAccumulator(f,facc,it_local);
                }
            }
        }
    }
}
template<typename Type1,typename Type2,typename FunctorBinaryFunctionE>
void forEachFunctorBinaryFunctionE(const MatNReference<2,Type1> & f, MatNReference<2,Type2> &  h,  FunctorBinaryFunctionE func, typename MatNReference<2,Type1>::IteratorEDomain)
{
    int i,j;
#if defined(HAVE_OPENMP)
#pragma omp parallel shared(f,h) private(i,j) firstprivate(func)
#endif
    {
#if defined(HAVE_OPENMP)
#pragma omp for schedule (static)
#endif
        for(i=0;i<static_cast<int>(f.sizeI());i++){
            for(j=0;j<static_cast<int>(f.sizeJ());j++){
                h(Vec2I32(i,j))=func( f, Vec2I32(i,j));
            }
        }
    }
}
template<typename Type1,typename Type2,typename FunctorBinaryFunctionE>
void forEachFunctorBinaryFunctionE(const MatNReference<3,Type1> & f, MatNReference<3,Type2> &  h,  FunctorBinaryFunctionE func, typename MatNReference<3,Type1>::IteratorEDomain)
{
    int i,j,k;
#if defined(HAVE_OPENMP)
#pragma omp parallel shared(f,h) private(i,j,k) firstprivate(func)
#endif
    {
#if defined(HAVE_OPENMP)
#pragma omp for schedule (static)
#endif
        for(i=0;i<static_cast<int>(f.sizeI());i++){
            for(j=0;j<static_cast<int>(f.sizeJ());j++){
                for(k=0;k<static_cast<int>(f.sizeK());k++){
                    h(Vec3I32(i,j,k))=func( f, Vec3I32(i,j,k));
                }
            }
        }
    }
}


}
#endif
