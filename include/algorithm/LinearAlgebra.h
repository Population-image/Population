/******************************************************************************\
|*                   Population library for C++ X.X.X                         *|
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
#ifndef LINEARALGEBRA_H
#define LINEARALGEBRA_H


#include"data/distribution/Distribution.h"
#include"data/mat/MatN.h"
#include"data/mat/Mat2x.h"

namespace pop
{

/*! \defgroup LinearAlgebra LinearAlgebra
    \ingroup Algorithm
    \brief Matrix In -> Matrix Out (gaussian inversion, eigen value/vector, orthogonalization...)
*
* I develop this module only for image processing purpose. So it is not the state of art of algorithms for linear algebra computing.
* However, this module provides some basic linear algebra operations as matrix inversion by gaussian elimination, find eigen vector (see the code below).
*
* \code
* Mat2F64 m(3,3);
* m(0,0)=1.5;m(0,1)=0;m(0,2)=1;
* m(1,0)=-0.5;m(1,1)=0.5;m(1,2)=-0.5;
* m(2,0)=-0.5;m(2,1)=0;m(2,2)=0;
* VecF64 v_eigen_value = LinearAlgebra::eigenValue(m);
* std::cout<<v_eigen_value<<std::endl;
* Mat2F64 m_eigen_vector= LinearAlgebra::eigenVectorGaussianElimination(m,v_eigen_value);
* std::cout<<m_eigen_vector<<std::endl;
* std::cout<<m*m_eigen_vector.getCol(0)<<" is equal to "<<v_eigen_value(0)*m_eigen_vector.getCol(0)<<std::endl;
* std::cout<<m*m_eigen_vector.getCol(1)<<" is equal to "<<v_eigen_value(1)*m_eigen_vector.getCol(1)<<std::endl;
* std::cout<<m*m_eigen_vector.getCol(2)<<" is equal to "<<v_eigen_value(2)*m_eigen_vector.getCol(2)<<std::endl;
* \endcode

*/

struct POP_EXPORTS LinearAlgebra
{


    /*!
     * \class pop::LinearAlgebra
     * \ingroup LinearAlgebra
     * \brief class to act on Mat2F64
     * \author Tariel Vincent
     *
    */


    /*! \brief Determine if matrix is diagonal
     *  \param m input matrix
     *  \param error tolerance value
     *
     */
    template<typename Matrix>
    static bool isDiagonal(const Matrix &m,double error=0.01);



    /*! \brief \f$I_n = \begin{bmatrix}X_{0,0} & X_{0,1} & \cdots & X_{0,n-1} \\X_{1,0} & 1 & \cdots & X_{1,n-1} \\\vdots & \vdots & \ddots & \vdots \\X_{n-1,0} & 0 & \cdots & X_{n-1,n-1} \end{bmatrix}\f$ with \f$X_{n-1,n-1}\f$ independent and identically distributed random variable following the probability distribution
     * \param size_i row size
     * \param size_j column size
     * \param proba probability distribution by default random real number
     * \return  Identity Mat2F64
     *
     * Generate the random matrix such that each element is a random variable thrown following the probability distribution \a proba
     * \code
     * Distribution d(0,10,"NORMAL");//centered normal distribution with sigma=10
     * Mat2F64 R=LinearAlgebra::random(5,5,d);
     * std::cout<<R<<std::endl;
     * \endcode
     * \verbatim
      -14.8008 -9.45266 18.2426 -12.7995 -10.4097
      18.1754 4.37007 -4.74885 1.95643 1.56716
      -4.85429 -5.86872 -20.0965 -3.04209 -4.52438
      0.574132 13.5789 -13.2111 12.6382 -3.02713
      -15.2847 23.5578 -7.78712 -4.01587 -0.0216708
      \endverbatim
     * \sa Distribution
    */
    static Mat2F64 random(int size_i,int size_j, Distribution proba=Distribution(-MTRand_int32::maxValue(),MTRand_int32::maxValue(),"UNIFORMREAL"));
    /*!
     * \brief inverse the matrix by gaussian elimination algorithm
     * \param m input invertible Mat2F64
     * \return  inverse Mat2F64
     *
     * Inverse the matrix m by gaussian elimination algorithm
     * \code
     * Mat2F64 m =LinearAlgebra::random(10,10);
     * Mat2F64 minverse= LinearAlgebra::inverseGaussianElimination(m);
     * //We get the identity matrix if you multiply M^-1 by M
     * std::cout<<m*minverse<<std::endl;
     * \endcode
     */
    static Mat2F64 inverseGaussianElimination(const Mat2F64 &m);
    /*!
     * \brief Mat2F64 r orthonormalising a set of vectors
     * \param m input vectors (each column is a vector)
     * \return  output vector (each column is a orthonormalised vector)
     *
     * the Gram–Schmidt process is a method for orthonormalising a set of std::vectors defined by the columns of the matrix \a m.
     * The set of orthonormalising std::vectors are the columns of the output matrix.

    \code
    Mat2F64 m(3,3);
    m(0,0)=1;m(0,1)=1;m(0,2)=1;
    m(1,0)=1;m(1,1)=0;m(1,2)=1;
    m(2,0)=1;m(2,1)=1;m(2,2)=3;
    std::cout<<m<<std::endl;
    Mat2F64 mortho= LinearAlgebra::orthogonalGramSchmidt(m);
    std::cout<<mortho<<std::endl;

    VecF64 v0 = mortho.getCol(0);
    VecF64 v1 = mortho.getCol(1);
    VecF64 v2 = mortho.getCol(2);
    //Check the orthogonality
    std::cout<<productInner(v0,v1)<<std::endl;
    std::cout<<productInner(v0,v2)<<std::endl;
    //Check the normalization
    std::cout<<normValue(v0,2)<<std::endl;
    std::cout<<normValue(v1,2)<<std::endl;
    std::cout<<normValue(v2,2)<<std::endl;
    \endcode
    */

    static Mat2F64 orthogonalGramSchmidt(const Mat2F64 &m);


    /*! \brief M=Q*R with Q is an orthogonal matrix and R is an upper triangular matrix
     * \param m input Mat2F64
     * \param  Q Q Mat2F64
     * \param  R R Mat2F64
     *
     * a QR decomposition (also called a QR factorization) of a matrix is a decomposition of a matrix A
     * into a product A=QR of an orthogonal matrix Q and an upper triangular matrix R
     * (see <a href=http://en.wikipedia.org/wiki/QR_decomposition>wikipedia </a>)
     * \code
     * Mat2F64 m(3,3);
     * m(0,0)=1;m(0,1)=1;m(0,2)=1;
     * m(1,0)=1;m(1,1)=0;m(1,2)=1;
     * m(2,0)=1;m(2,1)=1;m(2,2)=3;
     * std::cout<<m<<std::endl;
     * Mat2F64 Q;
     * Mat2F64 R;
     * LinearAlgebra::QRDecomposition(m,Q,R);
     * std::cout<<Q<<std::endl;
     * std::cout<<R<<std::endl;
     * std::cout<<Q*R<<std::endl;
     * //CHECK that Q is an orthogonal matrix (Q^tQ = I)
     * Mat2F64 Qt=Q.transpose();
     * std::cout<<Q*Qt<<std::endl;
     * \endcode
    */
    static void QRDecomposition(const Mat2F64 &m, Mat2F64 &Q, Mat2F64 &R);


    /*! \brief eigen values with QR algorithm
     * \param m input Mat2F64
     * \param error maximum error
     * \return  eigen values
     *
     * the QR algorithm allows to evaluate the eigenvalues of the matrix \a m.
     * (see <a href=http://en.wikipedia.org/wiki/QR_algorithm>wikipedia </a>)
     * \code
     * Mat2F64 m(3,3);
     * m(0,0)=1.5;m(0,1)=0;m(0,2)=1;
     * m(1,0)=-0.5;m(1,1)=0.5;m(1,2)=-0.5;
     * m(2,0)=-0.5;m(2,1)=0;m(2,2)=0;

     * VecF64 v_eigen_value = LinearAlgebra::eigenValue(m);
     * std::cout<<v_eigen_value<<std::endl;

     * Mat2F64 m_eigen_vector= LinearAlgebra::eigenVectorGaussianElimination(m,v_eigen_value);
     * std::cout<<m_eigen_vector<<std::endl;
     * std::cout<<m*m_eigen_vector.getCol(0)<<" is equal to "<<v_eigen_value(0)*m_eigen_vector.getCol(0)<<std::endl;
     * std::cout<<m*m_eigen_vector.getCol(1)<<" is equal to "<<v_eigen_value(1)*m_eigen_vector.getCol(1)<<std::endl;
     * std::cout<<m*m_eigen_vector.getCol(2)<<" is equal to "<<v_eigen_value(2)*m_eigen_vector.getCol(2)<<std::endl;
     * \endcode
    */
    enum EigenValueMethod{
        QR,
        Symmetric
    };

    static VecF64 eigenValue(const Mat2F64 &m,EigenValueMethod method = QR,F64 error=0.01);
    static VecF64 eigenValue(const Mat2x<F64,3,3> &m,EigenValueMethod method = QR,F64 error=0.01);
    static VecF64 eigenValue(const Mat2x<F64,2,2> &m);


    /*! \brief  gaussian elimination algorithm
     * \param m input Mat2F64
     * \return  gaussian elimination Mat2F64
     *
     *  Solving a Linear System by Gaussian Elimination, (see <a href=http://en.wikipedia.org/wiki/Gaussian_elimination>wikipedia </a>)
     * \code
     * Mat2F64 M(3,3);
     * M(0,0)=2;M(0,1)=1;M(0,2)=-1;
     * M(1,0)=-3;M(1,1)=-1;M(1,2)=2;
     * M(2,0)=-2;M(2,1)=1;M(2,2)=2;
     * VecF64 a(3);
     * a(0)=8;
     * a(1)=-11;
     * a(2)=-3;
     * //Find x such Mx=a
     * Mat2F64 Ma(3,4);
     * for(int i =0;i<M.sizeI();i++)
     *     for(int j =0;j<M.sizeJ();j++)
     *        Ma(i,j)=M(i,j);
     * for(int i =0;i<M.sizeI();i++)
     *     Ma(i,3)=a(i);

     * Ma=LinearAlgebra::solvingLinearSystemGaussianElimination(Ma);
     * VecF64 x(3);
     * for(int i =0;i<M.sizeI();i++)
     *     x(i)=Ma(i,3);
     * std::cout<<"The result is:"<<x<<std::endl;
     * \endcode
    */
    static Mat2F64 solvingLinearSystemGaussianElimination(const Mat2F64 &m);


    /*!
     * \brief find x such that A*x=b
     * \param A input Mat2F64
     * \param b input Vec
     * \return  x
     *
     *  Solving this linear equation A*x =b a by Gaussian Elimination, (see <a href=http://en.wikipedia.org/wiki/Gaussian_elimination>wikipedia </a>)
     * \code
     * Mat2F64 M(3,3);
     * M(0,0)=2;M(0,1)=1;M(0,2)=-1;
     * M(1,0)=-3;M(1,1)=-1;M(1,2)=2;
     * M(2,0)=-2;M(2,1)=1;M(2,2)=2;
     * VecF64 a(3);
     * a(0)=8;
     * a(1)=-11;
     * a(2)=-3;

     * VecF64 X=LinearAlgebra::solvingLinearSystemGaussianElimination(M,a);
     * std::cout<<"The result is:"<<X<<std::endl;
     * \endcode
    */
    static VecF64 solvingLinearSystemGaussianElimination(const Mat2F64 &A,const VecF64 & b);
    static VecF64 solvingLinearSystemGaussianElimination(const pop::Mat2x22F64 &A,const VecF64 & b);
    /*!
     * \brief eigen vectors
     * \param m input Mat2F64
     * \param v_eigen_value eigen values
     * \return  Eigen std::vector Mat2F64
     *
     * return the set of eigen vectors contained in the columns of the output matrix based on the gaussiand elimination algorithm\n
     * \code
     * Mat2F64 m(3,3);
     * m(0,0)=1.5;m(0,1)=0;m(0,2)=1;
     * m(1,0)=-0.5;m(1,1)=0.5;m(1,2)=-0.5;
     * m(2,0)=-0.5;m(2,1)=0;m(2,2)=0;

     * VecF64 v_eigen_value = LinearAlgebra::eigenValue(m);
     * std::cout<<v_eigen_value<<std::endl;

     * Mat2F64 m_eigen_vector= LinearAlgebra::eigenVectorGaussianElimination(m,v_eigen_value);
     * std::cout<<m_eigen_vector<<std::endl;
     * std::cout<<m*m_eigen_vector.getCol(0)<<" is equal to "<<v_eigen_value(0)*m_eigen_vector.getCol(0)<<std::endl;
     * std::cout<<m*m_eigen_vector.getCol(1)<<" is equal to "<<v_eigen_value(1)*m_eigen_vector.getCol(1)<<std::endl;
     * std::cout<<m*m_eigen_vector.getCol(2)<<" is equal to "<<v_eigen_value(2)*m_eigen_vector.getCol(2)<<std::endl;
     *  * \endcode
     \sa eigenValue
    */
    static Mat2F64 eigenVectorGaussianElimination(const Mat2F64 &m,VecF64 v_eigen_value);


    /*!
     * \brief M=L*U L is lower triangular matrix and U an upper triangular matrix
     * \param M input Mat2F64
     * \param L lower triangular Mat2F64
     * \param U upper triangular Mat2F64
     *
     * LU decomposition (also called LU factorization) factorizes a Mat2F64 as the product of a lower triangular Mat2F64 and an upper triangular Mat2F64 http://en.wikipedia.org/wiki/LU_decomposition
     * \code
     * Mat2F64 m(3,3);
     * m(0,0)=1.5;m(0,1)=0;m(0,2)=1;
     * m(1,0)=-0.5;m(1,1)=0.5;m(1,2)=-0.5;
     * m(2,0)=-0.5;m(2,1)=0;m(2,2)=0;
     * Mat2F64 L,U;
     * std::cout<<m<<std::endl;
     * LinearAlgebra::LUDecomposition(m,L,U);
     * std::cout<<L<<std::endl;
     * std::cout<<U<<std::endl;
     * std::cout<<L*U<<std::endl;
     * \endcode
    */
    static void LUDecomposition(const Mat2F64 &M,Mat2F64 & L,  Mat2F64 & U);


    /*! \fn Mat2F64 AATransposeEqualMDecomposition(const Mat2F64 &M);
     * \param M input Mat2F64
     * \return upper triangular Mat2F64
     * \brief AA^t=M for a symmetric, positive-definite Mat2F64
     *
     * AA^t=M factorization based on LU decomposition for a symmetric, positive-definite Matrix
     * \code
     * Mat2F64 m(3,3);
     * m(0,0)=2;m(0,1)=-1;m(0,2)=0;
     * m(1,0)=-1;m(1,1)=2;m(1,2)=-1;
     * m(2,0)=0;m(2,1)=-1;m(2,2)=2;
     * std::cout<<m<<std::endl;
     * Mat2F64 A = LinearAlgebra::AATransposeEqualMDecomposition(m);

     * Mat2F64 At;
     * At =At.transpose();
     * std::cout<<A*At<<std::endl;
     * \endcode
    */
    static Mat2F64 AATransposeEqualMDecomposition(const Mat2F64 &M);

    /*!
     * \brief find beta such that \f$\hat{\boldsymbol{\beta}} = \underset{\boldsymbol{\beta}}{\operatorname{arg\,min}}\, \bigl\|\mathbf y - \mathbf X \boldsymbol \beta \bigr\|^2\f$
     * \param X input matrix
     * \param Y output vector
     * \return beta
     *
     * See  <a href=http://en.wikipedia.org/wiki/Linear_least_squares_%28mathematics%29>wikipedia </a>
     * \code
     * Mat2F64 X(4,2);
     * X(0,0)=1;X(0,1)=1;
     * X(1,0)=1;X(1,1)=2;
     * X(2,0)=1;X(2,1)=3;
     * X(3,0)=1;X(3,1)=4;
     * VecF64 Y(4);
     * Y(0)=6;
     * Y(1)=5;
     * Y(2)=7;
     * Y(3)=10;
     * std::cout<<LinearAlgebra::linearLeastSquares(X,Y)<<std::endl;
     * \endcode
     * \sa LinearLeastSquareRANSACModel
    */
    static Vec2F64  linearLeastSquares(const Mat2F64 &X,const VecF64 &Y);
private:
    static void _solvingLinearSystemGaussianEliminationNonInvertible(Mat2F64 &M);
};

template<typename Matrix>
bool LinearAlgebra::isDiagonal(const Matrix &m,double error){
    double error_diff=0;
    for(unsigned int i=0;i<m.sizeI();i++){
        for(unsigned int j=0;j<m.sizeJ();j++){
            if(i!=j){
                error_diff = std::max(error_diff,std::abs(m(i,j)));
            }
        }
    }
    if(error_diff>error)
        return false;
    else
        return true;
}

}
#endif // LINEARALGEBRA_H
