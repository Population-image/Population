
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
#ifndef FUNCTORMATN_H
#define FUNCTORMATN_H

#include"PopulationConfig.h"
#include"data/vec/VecN.h"
#include"data/mat/MatN.h"
#include"algorithm/Convertor.h"
namespace pop
{
/// @cond DEV

/*! \ingroup Functor
* \defgroup FunctorMatN FunctorMatN
* \brief return a pixel value from a position element and a matrix
*
*/


struct POP_EXPORTS FunctorMatN
{
    /*!
    * \class pop::FunctorMatN
    * \ingroup FunctorMatN
    * \brief  treturn a pixel value from a position element and a matrix
    *
    */

    //-------------------------------------
    //
    //! \name Convolution algorithms
    //@{
    //-------------------------------------

    template<int DIM,typename PixelType1,typename PixelType2,typename IteratorE,typename BoundaryCondition>
    static MatN<DIM,PixelType1> convolutionSeperable(const MatN<DIM,PixelType1> & f, const Vec<PixelType2> & kernel,int direction,IteratorE itglobal,BoundaryCondition)
    {

        MatN<DIM,PixelType1> h(f.getDomain());
        typedef typename FunctionTypeTraitsSubstituteF<PixelType1,F32>::Result Type_F32;
        typedef typename FunctionTypeTraitsSubstituteF<PixelType2,F32>::Result Type2_F32;
        int radius = (kernel.size()-1)/2;
        while(itglobal.next()){
            typename MatN<DIM,PixelType1>::E x = itglobal.x()  ;
            Type_F32 value(0);
            for(unsigned int k=0;k<kernel.size();k++){
                x(direction)= itglobal.x()(direction)+(radius-k);
                if(BoundaryCondition::isValid(f.getDomain(),x,direction)){
                    BoundaryCondition::apply(f.getDomain(),x,direction);
                    value+=Type_F32(f(x))*Type2_F32(kernel(k));
                }
            }

            h(itglobal.x())=ArithmeticsSaturation<PixelType1,Type_F32>::Range (value);
        }
        return h;
    }
    template<typename PixelType1,typename PixelType2,typename BoundaryCondition>
    static MatN<2,PixelType1> convolutionSeperable(const MatN<2,PixelType1> & f, const Vec<PixelType2> & kernel,int direction,MatNIteratorEDomain<Vec2I32> ,BoundaryCondition)
    {

        MatN<2,PixelType1> h(f.getDomain());
        typedef typename FunctionTypeTraitsSubstituteF<PixelType1,F32>::Result Type_F32;
        typedef typename FunctionTypeTraitsSubstituteF<PixelType2,F32>::Result Type2_F32;
        int radius;
        int i,j,k;
        int dir;
        Type_F32 value;
        Vec2I32 x;
#if defined(HAVE_OPENMP)
#pragma omp parallel shared(h) private(i,j,k,dir,value,x,radius)
#endif
        {
            radius = static_cast<int>((kernel.size()-1)/2);
#if defined(HAVE_OPENMP)
#pragma omp for schedule (static)
#endif
            for(i=0;i<static_cast<int>(f.sizeI());i++){
                for(j=0;j<static_cast<int>(f.sizeJ());j++){
                    x(0)=i;x(1)=j;
                    dir = x(direction);
                    value=0;
                    for(k=0;k<static_cast<int>(kernel.size());k++){
                        x(direction)= dir+(radius-k);
                        if(BoundaryCondition::isValid(f.getDomain(),x,direction)){
                            BoundaryCondition::apply(f.getDomain(),x,direction);
                            value+=Type_F32(f(x))*Type2_F32(kernel(k));
                        }
                    }
                    h(i,j)=ArithmeticsSaturation<PixelType1,Type_F32>::Range (value);
                }
            }
        }
        return h;
    }
    template<typename PixelType1,typename PixelType2,typename BoundaryCondition>
    static MatN<3,PixelType1> convolutionSeperable(const MatN<3,PixelType1> & f, const Vec<PixelType2> & kernel,int direction,MatNIteratorEDomain<Vec3I32> ,BoundaryCondition)
    {

        MatN<3,PixelType1> h(f.getDomain());
        typedef typename FunctionTypeTraitsSubstituteF<PixelType1,F32>::Result Type_F32;
        typedef typename FunctionTypeTraitsSubstituteF<PixelType2,F32>::Result Type2_F32;
        int radius;
        int i,j,z,k,dir;
        Type_F32 value;
        Vec3I32 x;
#if defined(HAVE_OPENMP)
#pragma omp parallel shared(f,h) private(i,j,z,k,dir,value,x,radius)
#endif
        {
            radius = (kernel.size()-1)/2;
#if defined(HAVE_OPENMP)
#pragma omp for schedule (static)
#endif
            for(i=0;i<static_cast<int>(f.sizeI());i++){
                for(j=0;j<static_cast<int>(f.sizeJ());j++){
                    for(z=0;z<static_cast<int>(f.sizeK());z++){
                        x(0)=i;x(1)=j;x(2)=z;
                        dir = x(direction);
                        value=0;
                        for(k=0;k<static_cast<int>(kernel.size());k++){
                            x(direction)= dir+(radius-k);
                            if(BoundaryCondition::isValid(f.getDomain(),x,direction)){
                                BoundaryCondition::apply(f.getDomain(),x,direction);
                                value+=Type_F32(f(x))*Type2_F32(kernel(k));
                            }
                        }
                        h(i,j,z)=ArithmeticsSaturation<PixelType1,Type_F32>::Range (value);
                    }
                }
            }
        }
        return h;
    }
    template<int DIM,typename PixelType1,typename PixelType2,typename IteratorE,typename BoundaryCondition>
    static MatN<DIM,PixelType1> convolution(const MatN<DIM,PixelType1> & f, const MatN<DIM,PixelType2> & kernel,IteratorE itglobal,BoundaryCondition)
    {
        MatN<DIM,PixelType1> h(f.getDomain());
        typedef typename FunctionTypeTraitsSubstituteF<PixelType1,F32>::Result Type_F32;
        typedef typename FunctionTypeTraitsSubstituteF<PixelType2,F32>::Result Type2_F32;
        typename MatN<DIM,PixelType2>::IteratorEDomain itlocal(kernel.getIteratorEDomain());
        typename MatN<DIM,PixelType2>::E center = (kernel.getDomain()-1)/2;

        while(itglobal.next()){
            itlocal.init();
            Type_F32 value(0);
            while(itlocal.next()){
                typename MatN<DIM,PixelType2>::E x = itglobal.x()-itlocal.x()+center;
                if(BoundaryCondition::isValid(f.getDomain(),x)){
                    BoundaryCondition::apply(f.getDomain(),x);
                    value+=Type_F32(f(x))*Type2_F32(kernel(itlocal.x()));
                }
            }
            h(itglobal.x())=ArithmeticsSaturation<PixelType1,Type_F32>::Range (value);
        }
        return h;
    }

    //@}
    //-------------------------------------
    //
    //! \name Convolution applications : Gaussian
    //@{
    //-------------------------------------

    static Vec<F32> createGaussianKernelOneDimension(F32 sigma,int radius_kernel){
        Vec<F32> gaussian_kernel(2*radius_kernel+1);
        //initialisation one-dimension
        F32 sum=0;
        for(int i=0;i<static_cast<int>(gaussian_kernel.size());i++){
            F32  value =std::exp(-0.5f*(radius_kernel-i)*(radius_kernel-i)/(sigma*sigma));
            gaussian_kernel[i]=value;
            sum+=value;
        }
        sum=1/sum;
        //normalisation
        for(unsigned int i=0;i<gaussian_kernel.size();i++){
            gaussian_kernel[i]*=sum;
        }
        return gaussian_kernel;
    }


    template< int DIM>
    static MatN<DIM,F32> createGaussianKernelMultiDimension(F32 sigma,int radius_kernel){
        VecN<DIM,int> domain(2*radius_kernel+1);
        MatN<DIM,F32> gaussian_kernel(domain);
        typename MatN<DIM,F32>::IteratorEDomain it = gaussian_kernel.getIteratorEDomain();
        F32 sum=0;
        while(it.next()){
            F32 dist = (it.x()-VecN<DIM,int>(radius_kernel)).normPower(2);
            F32  value = std::exp(-0.5f*dist/(sigma*sigma));
            gaussian_kernel(it.x())=value;
            sum+=pop::absolute(value);
        }
        sum=1/sum;
        //normalisation
        for(unsigned int i=0;i<gaussian_kernel.size();i++){
            gaussian_kernel[i]*=sum;
        }
        return gaussian_kernel;
    }
    static Mat2F32 createGaussianKernelTwoDimension(F32 sigma,int radius_kernel){
        return createGaussianKernelMultiDimension<2>(sigma,radius_kernel);
    }

    static Vec<F32> createGaussianDerivateKernelOneDimension(F32 sigma,int radius_kernel){
        Vec<F32> gaussian_kernel(2*radius_kernel+1);
        //initialisation one-dimension
        F32 sum=0;
        for(int i=0;i<static_cast<int>(gaussian_kernel.size());i++){
            F32  value =(radius_kernel-i)*std::exp(-0.5f*(radius_kernel-i)*(radius_kernel-i)/(sigma*sigma));
            gaussian_kernel[i]=value;
            if(value>=0)
                sum+=value;

        }
        //normalisation
        for( int i=0;i<static_cast<int>(gaussian_kernel.size());i++){
            gaussian_kernel[i]*=1/(sqrt(pop::PI*2)*sum*sigma);
        }
        return gaussian_kernel;
    }

    template< int DIM>
    static MatN<DIM,F32> createGaussianDerivateKernelMultiDimension(int direction,F32 sigma,int radius_kernel){
        VecN<DIM,int> domain(2*radius_kernel+1);
        MatN<DIM,F32> gaussian_kernel(domain);
        typename MatN<DIM,F32>::IteratorEDomain it = gaussian_kernel.getIteratorEDomain();
        F32 sum=0;
        while(it.next()){
            F32 dist = (it.x()-VecN<DIM,int>(radius_kernel)).normPower(2);
            F32  value = (it.x()-VecN<DIM,int>(radius_kernel))(direction)*std::exp(-0.5f*dist/(sigma*sigma));
            gaussian_kernel(it.x())=value;
            sum+=value;
        }
        sum=1/sum;
        //normalisation
        for(unsigned int i=0;i<gaussian_kernel.size();i++){
            gaussian_kernel[i]*=sum;
        }
        return gaussian_kernel;
    }
    static Mat2F32 createGaussianDerivateKernelTwoDimension(int direction,F32 sigma,int radius_kernel){
        return createGaussianDerivateKernelMultiDimension<2>(direction,sigma,radius_kernel);
    }

    template<int DIM,typename PixelType>
    static MatN<DIM,PixelType> convolutionGaussian(const MatN<DIM,PixelType>& in, F32 sigma,int radius_kernel){
        VecF32 kernel = createGaussianKernelOneDimension(sigma,radius_kernel);
        MatN<DIM,PixelType> out(in);
        typename MatN<DIM,PixelType>::IteratorEDomain it=out.getIteratorEDomain();
        for(unsigned int i=0;i<DIM;i++){
            it.init();
            out = FunctorMatN::convolutionSeperable(out,kernel,i,it,MatNBoundaryConditionMirror());
        }
        return out;
    }
    template<int DIM,typename PixelType,typename IteratorE>
    static MatN<DIM,PixelType> convolutionGaussian(const MatN<DIM,PixelType>& in,IteratorE &it, F32 sigma,int radius_kernel){
        VecF32 kernel = createGaussianKernelOneDimension(sigma,radius_kernel);
        MatN<DIM,PixelType> out(in);
        for(unsigned int i=0;i<DIM;i++){
            it.init();
            out = FunctorMatN::convolutionSeperable(out,kernel,i,it,MatNBoundaryConditionMirror());
        }
        return out;
    }


    template<int DIM,typename PixelType>
    static MatN<DIM,PixelType> convolutionGaussianDerivate(const MatN<DIM,PixelType>& in,int direction, F32 sigma,int radius_kernel){
        VecF32 kernel = createGaussianKernelOneDimension(sigma,radius_kernel);
        VecF32 kernel_derivate = createGaussianDerivateKernelOneDimension(sigma,radius_kernel);
        MatN<DIM,PixelType> out(in);
        typename MatN<DIM,PixelType>::IteratorEDomain it=out.getIteratorEDomain();
        for(unsigned int i=0;i<DIM;i++){
            it.init();
            if(i==(unsigned int)direction)
                out = FunctorMatN::convolutionSeperable(out,kernel_derivate,i,it,MatNBoundaryConditionMirror());
            else
                out = FunctorMatN::convolutionSeperable(out,kernel,i,it,MatNBoundaryConditionMirror());
        }
        return out;
    }
    template<int DIM,typename PixelType,typename IteratorE>
    static MatN<DIM,PixelType> convolutionGaussianDerivate(const MatN<DIM,PixelType>& in,IteratorE& it,int direction, F32 sigma,int radius_kernel){
        VecF32 kernel = createGaussianKernelOneDimension(sigma,radius_kernel);
        VecF32 kernel_derivate = createGaussianDerivateKernelOneDimension(sigma,radius_kernel);
        MatN<DIM,PixelType> out(in);
        for(unsigned int i=0;i<DIM;i++){
            it.init();
            if(i==(unsigned int)direction)
                out = FunctorMatN::convolutionSeperable(out,kernel_derivate,i,it,MatNBoundaryConditionMirror());
            else
                out = FunctorMatN::convolutionSeperable(out,kernel,i,it,MatNBoundaryConditionMirror());
        }
        return out;
    }
    //@}
    //-------------------------------------
    //
    //! \name Recursive algorithms
    //@{
    //-------------------------------------
    /*! \fn static Function1    recursive(const Function1 & f,FunctorRecursive & func,int direction=0, int way=1)
     *
      * \brief recursive filter
      * \param f input function
      * \param func recursive functor as FunctorF::FunctorRecursiveOrder1 or FunctorF::FunctorRecursiveOrder2
      * \param direction recursive direction 0=x-direction, 1=y-direction
      * \param way  way 1=by increasing the position, -1=by deceasing the position
      * \return h output function
      *
      * A recursive algorithm uses one or more of its outputs as an input at each step.
      * For a 2d grid matrix, we select the way and the direction to scan the grid: left to right along each row or right to left along each row
      * or up to down along each column or down to up along each column. Then, we apply a recursive formula as the \f$\alpha\f$-recursive filter:
      * \f$g(n)=\alpha f(n)+ (1-\alpha)g(n-1)\f$.\n The code of this filter is
      * \sa FunctorMatN::smoothDeriche
        */
    template< typename Function1, typename FunctorRecursive>
    static Function1  recursive(const Function1 & f,FunctorRecursive & func,int direction=0, int way=1)
    {
        typename Function1::IteratorEOrder itorder (f.getIteratorEOrder());
        itorder.setLastLoop(direction);
        itorder.setDirection(way);
        return recursive(f,itorder,func);
    }
    /*! \fn static Function1 recursiveAllDirections(const Function1 & f, FunctorCausal & funccausal, FunctorAntiCausal & funcanticausal )
      * \brief recursive filter \f$f_i = causal(f_{i-1})+ anticausal(f_{i-1})\f$ with i through the directions
      * \param f input function
      * \param funccausal causal functor as FunctorF::FunctorRecursiveOrder1 or FunctorF::FunctorRecursiveOrder2
      * \param funcanticausal anticausal functor as FunctorF::FunctorRecursiveOrder1 or FunctorF::FunctorRecursiveOrder2
      * \return h output function
      *
      * We apply successively the recursive though all directions such that \f$f_i = causal(f_{i-1})+ anticausal(f_{i-1})\f$. F
      * \sa FunctorF::FunctorRecursiveOrder1 FunctorF::FunctorRecursiveOrder2
      */


    template<typename Function1,typename FunctorCausal, typename FunctorAntiCausal>
    static Function1 recursiveAllDirections(const Function1 & f, FunctorCausal & funccausal, FunctorAntiCausal & funcanticausal )
    {
        typename Function1::IteratorEOrder itorder (f.getIteratorEOrder());
        typedef typename  FunctionTypeTraitsSubstituteF<typename Function1::F,F32>::Result TypeF32;
        typename FunctionTypeTraitsSubstituteF<Function1,TypeF32>::Result fprevious(f);
        typename FunctionTypeTraitsSubstituteF<Function1,TypeF32>::Result fcausal(f.getDomain());
        typename FunctionTypeTraitsSubstituteF<Function1,TypeF32>::Result fanticausal(f.getDomain());
        fprevious = f;
        for(I32 i=0;i <Function1::DIM;i++)
        {
            itorder.setLastLoop(i);
            typename Function1::E dir;
            dir = 1;
            itorder.setDirection(dir);
            itorder.init();
            fcausal = recursive(fprevious,itorder,funccausal);
            dir = -1;
            itorder.setDirection(dir);
            itorder.init();
            fanticausal= recursive(fprevious,itorder,funcanticausal);

            fprevious = fcausal + fanticausal;
        }
        Function1 h(fprevious);
        return h;
    }
    /*! \fn Function1 recursiveAllDirections(const Function1 & f, I32 direction,FunctorCausal & funccausal, FunctorAntiCausal & funcanticausal ,FunctorCausalDirection & funccausaldirection, FunctorAntiCausalDirection & funcanticausaldirection)
      * \brief recursive filter \f$f_i = causal(f_{i-1})+ anticausal(f_{i-1})\f$ with i through the directions exception direction, \f$f_i = causaldirection(f_{i-1})+ anticausaldirection(f_{i-1})\f$ otherwise
      * \param f input function
      * \param direction apply the direction functors in this direction
      * \param funccausal causal functor in all directions exception direction
      * \param funcanticausal anticausal functor as FunctorF::FunctorRecursiveOrder1 or FunctorF::FunctorRecursiveOrder2
      * \param funccausaldirection causal functor in the input direction
      * \param funcanticausaldirection anticausal functor in the input direction
      * \return h output function
      *
      * We apply successively the recursive though all directions such that \f$f_i = causal(f_{i-1})+ anticausal(f_{i-1})\f$ for \f$i\neq\f$direction,  \f$f_i = causaldirection(f_{i-1})+ anticausaldirection(f_{i-1})\f$ otherwise.
      * \sa FunctorMatN::smoothDeriche
      */
    template<typename Function1,typename FunctorCausal, typename FunctorAntiCausal,typename FunctorCausalDirection, typename FunctorAntiCausalDirection >
    static Function1 recursiveAllDirections(const Function1 & f, I32 direction,FunctorCausal & funccausal, FunctorAntiCausal & funcanticausal ,FunctorCausalDirection & funccausaldirection, FunctorAntiCausalDirection & funcanticausaldirection)
    {
        typename Function1::IteratorEOrder itorder (f.getIteratorEOrder());
        typedef typename  FunctionTypeTraitsSubstituteF<typename Function1::F,F32>::Result TypeF32;
        typename FunctionTypeTraitsSubstituteF<Function1,TypeF32>::Result fprevious(f);
        typename FunctionTypeTraitsSubstituteF<Function1,TypeF32>::Result fcausal(f.getDomain());
        typename FunctionTypeTraitsSubstituteF<Function1,TypeF32>::Result fanticausal(f.getDomain());
        for(I32 i=0;i <Function1::DIM;i++)
        {
            itorder.setLastLoop(i);
            typename Function1::E dir;
            dir = 1;
            itorder.setDirection(dir);
            itorder.init();
            if(i==direction)
                fcausal = recursive(fprevious,itorder,funccausaldirection);
            else
                fcausal = recursive(fprevious,itorder,funccausal);
            dir = -1;
            itorder.setDirection(dir);
            itorder.init();
            if(i==direction)
                fanticausal = recursive(fprevious,itorder,funcanticausaldirection);
            else
                fanticausal = recursive(fprevious,itorder,funcanticausal);
            fprevious = (fcausal  + fanticausal);
        }
        Function1 h(fprevious);
        return h;
    }



    /*!
      * \brief recursive filter
      * \param f input function
      * \param it order iterator
      * \param func recursive functor func
      * \return h output function
      *
      */
    template< typename Function1, typename IteratorEOrder, typename FunctorRecursive>
    static Function1    recursive(const Function1 & f,IteratorEOrder & it, FunctorRecursive & func)
    {
        Function1 h(f.getDomain());
        while(it.next()){
            h(it.x())=func(f,h,it.x(),it.getBordeLenghtLastLoop(),it.getIndexLastLoop(),it.getWayLastLoop());
        }
        return h;
    }
    //@}
    //-------------------------------------
    //
    //! \name Recursive Functors
    //@{
    //-------------------------------------
    class POP_EXPORTS FunctorRecursiveOrder1
    {
        /*!
     * \brief recursive functor
     * \author Tariel Vincent
     *
     * I am not happy with this implementation because the iterator and this function are strongly coupled.. But since its utilisation is limited to the recursive algorithm, I do not want to spend time to find a better implementation.
     *
     *
    */
    private:

        F32 _a0;
        F32 _a1;
        F32 _b1;
        F32 _a0border0;
    public:
        FunctorRecursiveOrder1(F32 a0,F32 a1,
                               F32 b1,
                               F32 a0border0)
            :_a0(a0),_a1(a1),_b1(b1),_a0border0(a0border0)
        {}
        template<typename Function1,typename Function2>
        typename    Function2::F   operator()(const Function1 & f,const Function2 & g, typename Function1::E & x, I32 lenghtborder, typename Identity<typename Function1::E>::Result::E
                                              indice, typename Identity<typename Function1::E>::Result::E direction)
        {
            if(lenghtborder==0)return _a0border0 *f(x);
            else
            {
                typename Function2::F value = _a0 *f(x);
                x(indice)-=direction;
                value+=_a1 *f(x)+_b1 *g(x);
                x(indice)+=direction;
                return value;
            }
        }
    };
    class POP_EXPORTS FunctorRecursiveOrder2
    {
    private:
        /*!
     * \brief recursive functor
     * \author Tariel Vincent
     *
     * I am not happy with this implementation because the iterator and this function are strongly coupled.. But since its utilisation is limited to the recursive algorithm, I do not want to spend time to find a better implementation.
     *
     *
    */
        F32 _a0;
        F32 _a1;
        F32 _a2;
        F32 _b1;
        F32 _b2;

        F32 _a0border0;

        F32 _a0border1;
        F32 _a1border1;
        F32 _b1border1;
    public:
        FunctorRecursiveOrder2(F32 a0,F32 a1,F32 a2,
                               F32 b1,F32 b2,
                               F32 a0border0,
                               F32 a0border1,F32 a1border1,F32 b1border1)
            :_a0(a0),_a1(a1),_a2(a2),
              _b1(b1),_b2(b2),
              _a0border0(a0border0),
              _a0border1(a0border1),_a1border1(a1border1),_b1border1(b1border1)
        {}

        template<
                typename Function1,
                typename Function2
                >
        typename    Function2::F   operator()(const Function1 & f,const Function2 & g, typename Function1::E & x, int lenghtborder, typename Identity<typename Function1::E>::Result::E
                                              indice, typename Identity<typename Function1::E>::Result::E direction)
        {
            if(lenghtborder==0)return f(x)*_a0border0 ;
            else if(lenghtborder==1)
            {
                typename Function2::F value =  f(x)*_a0border1;
                x(indice)-=direction;
                value+= f(x)*_a1border1+ g(x)*_b1border1;
                x(indice)+=direction;
                return  value;
            }
            else
            {
                typename Function2::F value =  f(x)*_a0;
                x(indice)-=direction;
                value+=f(x)*_a1 +g(x)*_b1;
                x(indice)-=direction;
                value+=f(x)*_a2 +g(x)*_b2 ;
                x(indice)+=2*direction;
                return value;
            }
        }

    };
    //@}
    //-------------------------------------
    //
    //! \name Recursive applications : Deriche
    //@{
    //-------------------------------------
    /*!
     * \brief Deriche's smooth filter
     * \param f input matrix
     * \param alpha inverse scale parameter
     * \return h output function
     *
     * Smooth the input matrix with the inverse scale parameter (alpha=2=low, alpha=0.5=high)
     * \code
     * Mat2RGBUI8 img;
     * img.load((std::string(POP_PROJECT_SOURCE_DIR)+"/image/Lena.bmp").c_str());
     * img = Processing::smoothDeriche(img,0.5);
     * img.display();
     * \endcode
     */
    template<typename Function1>
    static Function1 smoothDeriche(const Function1 & f, F32 alpha=1)
    {
        F32 e_a = std::exp(- alpha);
        F32 e_2a = std::exp(- 2.f * alpha);
        F32 k = (1.f - e_a) * (1.f - e_a) / (1.f + (2 * alpha * e_a) - e_2a);

        F32 a0_c= k;
        F32 a1_c=  k * e_a * (alpha - 1.f);
        F32 a2_c=  0;
        F32 a0_ac= 0;
        F32 a1_ac=  k * e_a * (alpha + 1.f);
        F32 a2_ac=  - k * e_2a;

        F32 b1= 2 * e_a;
        F32 b2 = - e_2a;


        F32 a0_c_border0 = ((a0_c + a1_c) / (1.f - b1 - b2));
        F32 a0_c_border1 = a0_c ;
        F32 a1_c_border1 = a1_c ;

        F32 a0_ac_border0 = ((a1_ac + a2_ac) / (1.f - b1 - b2));
        F32 a0_ac_border1 = 0 ;
        F32 a1_ac_border1 = a1_ac + a2_ac ;

        F32 b1_border1 = b1 + b2 ;

        FunctorMatN::FunctorRecursiveOrder2 funccausal
                (a0_c,a1_c,a2_c,
                 b1,b2,
                 a0_c_border0,
                 a0_c_border1,a1_c_border1,b1_border1) ;

        FunctorMatN::FunctorRecursiveOrder2 funcanticausal
                (a0_ac,a1_ac,a2_ac,
                 b1,b2,
                 a0_ac_border0,
                 a0_ac_border1,a1_ac_border1,b1_border1) ;

        return FunctorMatN::recursiveAllDirections(f,funccausal,funcanticausal);
    }

    /*! \fn Function1 gradientDeriche(const Function1 & f, F32 alpha,I32 direction)
     * \brief Deriche's smooth filter
     * \param f input matrix used float type as pixel/voxel type
     * \param alpha inverse scale parameter
     * \param direction derivate in the following direction
     * \return h output function
     *
     * Derivate the input matrix in the following direction with the inverse scale parameter (alpha=2=low, alpha=0.5=high)
     * \code
        Mat2RGBUI8 img;
        img.load((std::string(POP_PROJECT_SOURCE_DIR)+"/image/Lena.bmp").c_str());
        Mat2RGBF32 gradx(img);
        gradx = Processing::gradientDeriche(gradx,0,1);//Calculate the gradient in the direction 0
        img = Processing::greylevelRange(gradx,0,255);//to display the matrix with a float type, the
        img.display();
     * \endcode
     * \image html LenaGradDeriche.jpg
     */
    template<typename Function1>
    static typename FunctionTypeTraitsSubstituteF<Function1,typename FunctionTypeTraitsSubstituteF<typename Function1::F,F32 >::Result >::Result  gradientDeriche(const Function1 & f, I32 direction, F32 alpha=1)
    {
        if(NumericLimits<typename Function1::F>::is_integer==true){
            typedef typename FunctionTypeTraitsSubstituteF<Function1,typename FunctionTypeTraitsSubstituteF<typename Function1::F,F32 >::Result >::Result FunctionFloat;
            FunctionFloat ffloat(f);
            return gradientDeriche(ffloat, direction,alpha);
        }
        F32 e_a = std::exp(- alpha);
        F32 e_2a = std::exp(- 2.f * alpha);
        F32 k = (1.f - e_a) * (1.f - e_a) / (1.f + (2 * alpha * e_a) - e_2a);

        F32 a0_c= k;
        F32 a1_c=  k * e_a * (alpha - 1.f);
        F32 a2_c=  0;
        F32 a0_ac= 0;
        F32 a1_ac=  k * e_a * (alpha + 1.f);
        F32 a2_ac=  - k * e_2a;

        F32 b1= 2 * e_a;
        F32 b2 = - e_2a;


        F32 a0_c_border0 = ((a0_c + a1_c) / (1.f - b1 - b2));
        F32 a0_c_border1 = a0_c ;
        F32 a1_c_border1 = a1_c ;

        F32 a0_ac_border0 = ((a1_ac + a2_ac) / (1.f - b1 - b2));
        F32 a0_ac_border1 = 0 ;
        F32 a1_ac_border1 = a1_ac + a2_ac ;


        F32 b1_border1 = b1 + b2 ;


        F32 kp = - (1.f - e_a) * (1.f - e_a) / e_a;
        F32 a0_c_d= 0;
        F32 a1_c_d=  kp * e_a;
        F32 a2_c_d=  0;
        F32 a0_ac_d= 0;
        F32 a1_ac_d= - kp * e_a;
        F32 a2_ac_d= 0;

        F32 a0_c_border0_d = ((a0_c_d + a1_c_d) / (1.f - b1 - b2));
        F32 a0_c_border1_d = a0_c_d ;
        F32 a1_c_border1_d = a1_c_d ;

        F32 a0_ac_border0_d = ((a1_ac_d + a2_ac_d) / (1.f - b1 - b2));
        F32 a0_ac_border1_d = 0 ;
        F32 a1_ac_border1_d = a1_ac_d + a2_ac_d ;

        FunctorMatN::FunctorRecursiveOrder2 funccausalsmooth
                (a0_c,a1_c,a2_c,
                 b1,b2,
                 a0_c_border0,
                 a0_c_border1,a1_c_border1,b1_border1) ;

        FunctorMatN::FunctorRecursiveOrder2 funcanticausalsmooth
                (a0_ac,a1_ac,a2_ac,
                 b1,b2,
                 a0_ac_border0,
                 a0_ac_border1,a1_ac_border1,b1_border1) ;

        FunctorMatN::FunctorRecursiveOrder2 funccausalgrad
                (a0_c_d,a1_c_d,a2_c_d,
                 b1,b2,
                 a0_c_border0_d,
                 a0_c_border1_d,a1_c_border1_d,b1_border1) ;

        FunctorMatN::FunctorRecursiveOrder2 funcanticausalgrad
                (a0_ac_d,a1_ac_d,a2_ac_d,
                 b1,b2,
                 a0_ac_border0_d,
                 a0_ac_border1_d,a1_ac_border1_d,b1_border1) ;

        return recursiveAllDirections(f,direction,funccausalsmooth,funcanticausalsmooth,funccausalgrad,funcanticausalgrad);
    }
    /*!
     *  \brief Vector field of Deriche's gradient
     * \param f input function
     * \param alpha inverse scale parameter
     * \return h output function
     *
     *  Deriche's gradient
     * \code
    Mat2UI8 img;
    img.load("/usr/share/doc/opencv-doc/examples/c/lena.jpg");
    Mat2Vec2F32 gradx = Processing::gradientVecDeriche(img);
    Visualization::vectorField2DToArrows(gradx,RGBUI8(0,0,255),RGBUI8(255,0,0),8).display();
     *  \endcode
    */
    template<class Function1>
    static typename FunctionTypeTraitsSubstituteF<Function1,VecN<Function1::DIM,F32> >::Result gradientVecDeriche(const Function1  & f,F32 alpha=1)
    {
        typedef typename FunctionTypeTraitsSubstituteF<Function1,F32 >::Result  FunctionFloat;
        VecN<Function1::DIM,FunctionFloat> v_der;
        for(int i =0;i<Function1::DIM;i++){
            v_der[i]= gradientDeriche(f,i,alpha);

        }
        typename FunctionTypeTraitsSubstituteF<Function1,VecN<Function1::DIM,F32> >::Result f_grad(f.getDomain());
        Convertor::fromVecN(v_der,f_grad);
        return f_grad;
    }

    //@}




};





}
#endif // FUNCTORMATN_H
