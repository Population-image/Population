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
#ifndef FUNCTORF_HPP
#define FUNCTORF_HPP
#include<algorithm>
#include"PopulationConfig.h"
#include"data/typeF/TypeTraitsF.h"
#include"data/utility/BasicUtility.h"
#include<vector>
namespace pop
{
/// @cond DEV
/*! \ingroup Data
* \defgroup Functor Functor
* \brief  transforms one category into another category for functionnal programming allowing code factorization and optimization
*/

/*! \ingroup Functor
* \defgroup FunctorF FunctorF
* \brief  transforms a pixel value into another pixel value
*
*/
struct POP_EXPORTS FunctorF
{
    /*!
    * \class pop::FunctorF
    * \ingroup FunctorF
    * \brief  transforms a pixel value into another pixel value
    *
    *
    * This class contains many functors divided in diffrent categories
    *
    *  \section ElementaryArithmetic  Elementary arithmetic
    *
    * Elementary arithmetic refers to the simpler properties when using the traditional operations of
    * addition, subtraction, multiplication, division, max and min between numbers. In mathematic,
    * elementary arithmetic is usually defined on infinite sets as natural numbers, integers, rational
    * numbers, real numbers, complex number. Because computer memory/CPU instruction are
    * represented with a finite number of byte, the arithmetic set must be finite in computer. Let
    * us consider the finite set F = (00, 01, 10, 11) in the binary numeral system, and three elements
    * of this set, a = 11, b = 01 and c = a+b, result of the addition of these two previous numbers.
    * We expect the value 100 for the element c. But, this value produces an arithmetic overflow
    * because it is greater than the definition range. We can impose two behaviors to handle this
    * arithmetic overflow : periodic condition property given the result 11 or saturation condition
    * property given 11. For this propose, we define binary functor in generic programming.
    *
    * \section Accumulatorfunctor  Accumulator
    *
    * The accumulator functor returns a value depending on the list of accumulated values.
    * These functors are usefull coupled with a local/global iteration (see chapter of my book) for code factorization. For this propose, we define unary functor in generic programming.
    *
    * \section FacilityFunctor  Generic programming facility
    *
    * Some generic programming functor to operate generic programming facilities as extract a given channel of a vector value (pop::RGB, pop::VecN, pop::Complex )
    *
    */

    template<typename Result >
    struct FunctorConst{const Result & _v;FunctorConst(const Result & v ):_v(v){}Result operator()(){return _v;}};

    //-------------------------------------
    //
    //! \name Elementary arithmetic
    //@{
    //-------------------------------------
    template<
            typename T1,
            typename T2=T1,
            typename R =T1,
            template<typename,typename> class PolicyClassOverFlow=ArithmeticsSaturation
            >
    class POP_EXPORTS FunctorAdditionF2 : public PolicyClassOverFlow<R,typename ArithmeticsTrait<T1,T2>::Result >
    {
    /*!
    * \class pop::FunctorF::FunctorAdditionF2
    * \brief This template class is a binary functor to represent the addition
    * \author Tariel Vincent
    * \tparam T1 first input type
    * \tparam T2 second input type
    * \tparam R  returned type
    * \tparam PolicyClassOverFlow Overflow policy class by default saturation

    * This class allows the addition of two input variables of type T1 and T2 to return the result of
    * type R with the given policy for the overflow bahavior
    * \code
    *  unsigned char a = 100;
    *  unsigned char b = 200;
    *  c = FunctorAdditionF2<unsigned char>(a,b); //c is equal to 255 since saturation is the default bahavior
    * \endcode


    */
    public:

        typedef R ResultType;
        typedef typename ArithmeticsTrait<T1,T2>::Result CastParm;
    public:
     /*!
     *  \brief addition of the two input numbers
     *  \param p1 fist number.
     *  \param p2 second number.
     *  \return result of the addition.
     */
        inline  R operator()(T1 p1,T2 p2)
        {
            return PolicyClassOverFlow<R,typename ArithmeticsTrait<T1,T2>::Result >::Range( static_cast<CastParm>(p1)+static_cast<CastParm>(p2));
        }
        inline  static R op(T1 p1,T2 p2)
        {
            return PolicyClassOverFlow<R,typename ArithmeticsTrait<T1,T2>::Result >::Range( static_cast<CastParm>(p1)+static_cast<CastParm>(p2));
        }
        static R neutralElement()
        {
            return R(0);
        }
    };
    template<
            typename T1,
            typename T2,
            typename R =T1,
            template<typename,typename> class PolicyClassOverFlow=ArithmeticsSaturation
            >
    class POP_EXPORTS FunctorSubtractionF2 : public PolicyClassOverFlow<R,typename ArithmeticsTrait<T1,T2>::Result >
    {
   /*!
    * \class pop::FunctorF::FunctorSubtractionF2
    * \brief This template class is a binary functor to represent the subtraction
    * \author Tariel Vincent
    * \sa pop::FunctorF::FunctorAdditionF2
    */
    public:
        typedef R ResultType;
        typedef typename ArithmeticsTrait<T1,T2>::Result CastParm;
    public:
        inline  R operator()(T1 p1,T2 p2)
        {
            return PolicyClassOverFlow<R,typename ArithmeticsTrait<T1,T2>::Result >::Range( static_cast<CastParm>(p1)-static_cast<CastParm>(p2));
        }
        inline static R op(T1 p1,T2 p2)
        {
            return PolicyClassOverFlow<R,typename ArithmeticsTrait<T1,T2>::Result >::Range( static_cast<CastParm>(p1)-static_cast<CastParm>(p2));
        }
        static R neutralElement()
        {
            return R(0);
        }
    };
    template<
            typename T1,
            typename T2,
            typename R =T1,
            template<typename,typename> class PolicyClassOverFlow=ArithmeticsSaturation
            >
    class POP_EXPORTS FunctorMultiplicationF2 : public PolicyClassOverFlow<R,typename ArithmeticsTrait<T1,T2>::Result >
    {
      /*!
    * \class pop::FunctorF::FunctorMultiplicationF2

    * \brief This template class is a binary functor to represent the multiplication
    * \author Tariel Vincent
    * \sa pop::FunctorF::FunctorAdditionF2
    */
    public:
        typedef R ResultType;
        typedef typename ArithmeticsTrait<T1,T2>::Result CastParm;


    public:
        inline  R operator()(T1 p1,T2 p2)
        {
            return PolicyClassOverFlow<R,typename ArithmeticsTrait<T1,T2>::Result >::Range( static_cast<CastParm>(p1)*static_cast<CastParm>(p2));
        }
        inline static R op(T1 p1,T2 p2)
        {
            return PolicyClassOverFlow<R,typename ArithmeticsTrait<T1,T2>::Result >::Range( static_cast<CastParm>(p1)*static_cast<CastParm>(p2));
        }
        static R neutralElement()
        {
            return R(1);
        }
    };
    template<
            typename T1,
            typename T2,
            typename R =T1,
            template<typename,typename> class PolicyClassOverFlow=ArithmeticsSaturation
            >
    class POP_EXPORTS FunctorDivisionF2 : public PolicyClassOverFlow<R,typename ArithmeticsTrait<T1,T2>::Result >
    {
        /*!
        \class pop::FunctorF::FunctorDivisionF2

        \brief This template class is a binary functor to represent the division
        \author Tariel Vincent
        \sa pop::FunctorF::FunctorAdditionF2
    */
    public:
        typedef R ResultType;
        typedef typename ArithmeticsTrait<T1,T2>::Result CastParm;
    public:
        inline  R operator()(T1 p1,T2 p2)
        {
            return PolicyClassOverFlow<R,typename ArithmeticsTrait<T1,T2>::Result >::Range( static_cast<CastParm>(p1)/static_cast<CastParm>(p2));
        }
        inline static R op(T1 p1,T2 p2)
        {
            return PolicyClassOverFlow<R,typename ArithmeticsTrait<T1,T2>::Result >::Range( static_cast<CastParm>(p1)/static_cast<CastParm>(p2));
        }
        static R neutralElement()
        {
            return R(1);
        }
    };
    template<
            typename T1,
            typename T2,
            typename R =T1,
            template<typename,typename> class PolicyClassOverFlow=ArithmeticsSaturation
            >
    class POP_EXPORTS FunctorModuloF2 : public PolicyClassOverFlow<R,typename ArithmeticsTrait<T1,T2>::Result >
    {
        /*!
        \class pop::FunctorF::FunctorModuloF2

        \brief This template class is a binary functor to represent the min
        \author Tariel Vincent
        \sa pop::FunctorF::FunctorAdditionF2
    */
    public:

        typedef R ResultType;
        typedef typename ArithmeticsTrait<T1,T2>::Result CastParm;
    public:
        inline  R operator()(T1 p1,T2 p2)
        {
            return PolicyClassOverFlow<R,typename ArithmeticsTrait<T1,T2>::Result >::Range( static_cast<CastParm>(p1)%static_cast<CastParm>(p2));
        }
        inline static R op(T1 p1,T2 p2)
        {
            return PolicyClassOverFlow<R,typename ArithmeticsTrait<T1,T2>::Result >::Range( static_cast<CastParm>(p1)%static_cast<CastParm>(p2));
        }
        static R neutralElement()
        {
            return R(1);
        }
    };
    template<
            typename T1,
            typename T2,
            typename R =T1,
            template<typename,typename> class PolicyClassOverFlow=ArithmeticsSaturation
            >
    class POP_EXPORTS FunctorMaxF2 : public PolicyClassOverFlow<R,typename ArithmeticsTrait<T1,T2>::Result >
    {
      /*!
    * \class pop::FunctorF::FunctorMaxF2

    * \brief This template class is a binary functor to represent the max
    * \author Tariel Vincent
    * \sa pop::FunctorF::FunctorAdditionF2
    */

    public:

        typedef R ResultType;
        typedef typename ArithmeticsTrait<T1,T2>::Result CastParm;
    public:
        inline  R operator()(T1 p1,T2 p2)
        {
            return maximum(p1,p2);
        }
        inline static R op(T1 p1,T2 p2)
        {
            return maximum(p1,p2);
        }
        static R neutralElement()
        {
            return NumericLimits<R>::minimumRange();
        }
    };
    template<
            typename T1,
            typename T2,
            typename R =T1,
            template<typename,typename> class PolicyClassOverFlow=ArithmeticsSaturation
            >
    class POP_EXPORTS FunctorMinF2 : public PolicyClassOverFlow<R,typename ArithmeticsTrait<T1,T2>::Result >
    {
      /*!
    * \class pop::FunctorF::FunctorMinF2

    * \brief This template class is a binary functor to represent the min
    * \author Tariel Vincent
    * \sa pop::FunctorF::FunctorAdditionF2
    */
    public:
        typedef R ResultType;
        typedef typename ArithmeticsTrait<T1,T2>::Result CastParm;
    public:
        inline  R operator()(T1 p1,T2 p2)
        {
            return minimum(p1,p2);
        }
        inline static R op(T1 p1,T2 p2)
        {
            return minimum(p1,p2);
        }
        static R neutralElement()
        {
            return NumericLimits<R>::maximumRange();
        }
    };
    template<typename Parm,typename Value,typename Result, typename BinaryFunctor >
    class POP_EXPORTS FunctorArithmeticConstantValueAfter
    {
        /*!
         * \class pop::FunctorF::FunctorArithmeticConstantValueAfter
         * \brief operate an arithmetic operation with a constant value in a unary functor
         * \author Tariel Vincent
         *
         * This functor allows to iterate a arithmetic operation on each pixel value with a constant scalar value in stl algorithm. See the implepmentation of this method
         *
         * \sa MatN::operator+=(G value)
        */
    private:
        Value _value;
    public:
        FunctorArithmeticConstantValueAfter(Value value)
            :_value(value)
        {}

        Result operator()(Parm p)
        const
        {
            return BinaryFunctor::op(p,_value);
        }
    };
    template<typename Value,typename Parm,typename Result, typename BinaryFunctor >
    class POP_EXPORTS FunctorArithmeticConstantValueBefore
    {
        /*!
         * \class pop::FunctorF::FunctorArithmeticConstantValueBefore
         * \brief operate an arithmetic operation with a constant value in a unary functor
         * \author Tariel Vincent
         *
         *
        */
    private:
        Value _value;
    public:
        FunctorArithmeticConstantValueBefore(Value value)
            :_value(value)
        {}

        Result operator()(Parm p)
        const
        {
            return BinaryFunctor::op(_value,p);
        }
    };

    //@}
    //-------------------------------------
    //
    //! \name Accumulator
    //@{
    //-------------------------------------




    template<typename Type>
    class POP_EXPORTS FunctorAccumulatorMax
    {
        /*!
         * \class pop::FunctorF::FunctorAccumulatorMax
         * \brief max value of the list of acculuated values
         * \author Tariel Vincent
         * \tparam Type pixel type
         *
         * \sa pop::Processing::dilation
        */
    private:
        Type _value;
    public:
        typedef Type ReturnType;
        FunctorAccumulatorMax()
            :_value(NumericLimits<Type>::minimumRange()){}
        FunctorAccumulatorMax(const FunctorAccumulatorMax & m)
            :_value(m.getValue()){}

        void operator()(Type p1)
        {
            _value =  maximum(_value,p1);
        }
        void init()
        {
            _value = NumericLimits<Type>::minimumRange();
        }
        Type getValue()const
        {
            return _value;
        }
    };
    template<typename Type>
    class POP_EXPORTS FunctorAccumulatorMin
    {
        /*!
         * \class pop::FunctorF::FunctorAccumulatorMin
         * \brief min value of the list of acculuated values
         * \author Tariel Vincent
         * \tparam Type pixel type
         *
         * \sa pop::Processing::erosion
        */
    private:
        Type _value;
    public:
        typedef Type ReturnType;
        FunctorAccumulatorMin()
            :_value(NumericLimits<Type>::maximumRange()){}

        void operator()(Type p1)
        {
            _value =  minimum(_value,p1);
        }
        void init()
        {
            _value= NumericLimits<Type>::maximumRange();
        }
        Type getValue()
        {
            return _value;
        }
    };

    template<typename Type>
    class POP_EXPORTS FunctorAccumulatorMedian
    {
        /*!
         * \class pop::FunctorF::FunctorAccumulatorMedian
         * \brief median value of the list of acculuated values
         * \author Tariel Vincent
         * \tparam Type pixel type
         *
         * \sa pop::Processing::median
        */
    private:
        std::vector<Type> _collector;
        I32 _occurence;
    public:
        typedef Type ReturnType;
        FunctorAccumulatorMedian()
            :_occurence(0)
        {}
        void operator()(Type p)
        {
            if((I32)_collector.size()<=_occurence)
                _collector.push_back(p);
            else
                _collector[_occurence]=p;
            _occurence++;
        }
        void init()
        {
            _occurence=0;
        }
        Type getValue()
        {
            std::sort(_collector.begin(),_collector.begin()+_occurence);
            return _collector[_occurence/2];
        }
    };
    template<typename Type>
    class POP_EXPORTS FunctorAccumulatorMean
    {
        /*!
         * \class pop::FunctorF::FunctorAccumulatorMean
         * \brief mean value of the list of acculuated values
         * \author Tariel Vincent
         * \tparam Type pixel type
         *
         * \sa pop::Processing::mean
        */
    private:
        typedef typename FunctionTypeTraitsSubstituteF<Type,F64>::Result SuperType;
        SuperType _sum;
        I32 _occurence;
    public:
        typedef SuperType ReturnType;
        FunctorAccumulatorMean()
            :_sum(0),_occurence(0)
        {}

        void operator()(Type p)
        {
            _sum+=SuperType(p);
            _occurence++;
        }
        void init()
        {
            _sum = SuperType(0);
            _occurence=0;
        }
        SuperType getValue()const
        {
            return _sum/_occurence;
        }
    };
    template<typename Type>
    class POP_EXPORTS FunctorAccumulatorVariance
    {
        /*!
         * \class pop::FunctorF::FunctorAccumulatorMedian
         * \brief variance of the list of acculuated values
         * \author Tariel Vincent
         * \tparam Type pixel type
         *
         * \sa pop::Analysis::standardDeviationValue
        */
    private:
        typedef typename FunctionTypeTraitsSubstituteF<Type,F64>::Result SuperType;
        SuperType _sum;
        I32 _occurence;
        F64 _mean;
    public:
        typedef SuperType ReturnType;
        FunctorAccumulatorVariance(F64 mean)
            :_sum(0),_occurence(0),_mean(mean)
        {}
        void operator()(Type p)
        {
            _sum+=(p-_mean)*(p-_mean);
            _occurence++;
        }
        void init()
        {
            _sum = SuperType(0);
            _occurence=0;
        }
        SuperType getValue()const
        {
            return _sum/_occurence;
        }
    };
    //@}
    //-------------------------------------
    //
    //! \name Generic programming functor facilities
    //@{
    //-------------------------------------
    template<typename Scalar,typename Vec>
    struct FunctorFromVectorToScalarCoordinate
    {
        /*!
         * \class pop::FunctorF::FunctorFromVectorToScalarCoordinate
         * \brief coordinate value of a vector value as pop::RGB, pop::Vec, pop::VecN
         * \author Tariel Vincent
         * \tparam Scalar coordinate type
         * \tparam Vec vector type
         *
        */
        const int _coordinate;
        FunctorFromVectorToScalarCoordinate(int coordinate)
            :_coordinate(coordinate){

        }
        Scalar operator ()(const Vec r){
            return r(_coordinate);
        }
    };
    template<typename Scalar,typename Vec>
    struct FunctorFromMultiCoordinatesToVector
    {
        /*!
         * \class pop::FunctorF::FunctorFromMultiCoordinatesToVector
         * \brief  vector value from multi coordinate values
         * \author Tariel Vincent
         * \tparam Scalar coordinate type
         * \tparam Vec vector type
         *
        */

        Vec operator ()(Scalar r1){
            return Vec(r1);
        }
        Vec operator ()(Scalar r1,Scalar r2){
            return Vec(r1,r2);
        }
        Vec operator ()(Scalar r1,Scalar r2,Scalar r3){
            return Vec(r1,r2,r3);
        }
        Vec operator ()(Scalar r1,Scalar r2,Scalar r3,Scalar r4){
            return Vec(r1,r2,r3,r4);
        }
    };
    //@}


    template<typename R,typename TypeBound,typename TypeIn>
    class POP_EXPORTS FunctorThreshold
    {
        /*!
         * \class pop::FunctorF::FunctorThreshold
         * \brief  Threshold functor
         * \author Tariel Vincent
         * \tparam R return type
         * \tparam TypeBound bound type
         * \tparam TypeIn type of the unary functor argument
         *
        */
        TypeBound _valuemin;
        TypeBound _valuemax;
        R _return_value;
    public:
        FunctorThreshold(TypeBound valuemin, TypeBound valuemax,R return_value=NumericLimits<R>::maximumRange())
            :_valuemin(valuemin),_valuemax(valuemax),_return_value(return_value)
        {}
        R operator()(TypeIn p1){
            if( p1>=_valuemin && p1<=_valuemax){
                return _return_value;
            }
            else return 0;
        }
    };



};


template<typename Value>
class POP_EXPORTS FunctorStatistic
{
private:
    std::vector<Value> _v_value;
    F64 _variance;
    F64 _mean;
public:
    typedef F64 ReturnType;
    FunctorStatistic()
    {
        init();
    }
    void init()
    {
        _variance=NumericLimits<double>::maximumRange();_mean=NumericLimits<double>::maximumRange();_v_value.clear();
    }
    template<typename Function>
    void operator()(const Function & f, const typename Function::E & x)
    {
        _v_value.push_back(f(x));
    }
    F64 mean()
    {
        if(_mean==NumericLimits<double>::maximumRange())
        {
            for(I32 i =0;i<(I32)_v_value.size();i++)
            {
                _mean+=_v_value[i]/(1.*_v_value.size());
            }
        }
        return _mean;
    }
    F64 variance()
    {
        if(_variance==NumericLimits<double>::maximumRange())
        {
            if(_mean==NumericLimits<double>::maximumRange())this->mean();
            for(I32 i =0;i<(I32)_v_value.size();i++)
            {
                _variance+=(_v_value[i]-_mean)*(_v_value[i]-_mean)/(1.*_v_value.size());
            }
        }
        return _variance;
    }
};
namespace Private{
    template< typename T>struct sumNorm{sumNorm(int norm):_norm(norm){}int _norm;double operator()(double p1,T p2){if(_norm!=0)return p1+normPowerValue(p2,_norm);else return  maximum( p1,static_cast<double>(absolute(p2)));}};
    template<typename T1>struct PowF{double _exponent;PowF(double exponent):_exponent(exponent){}T1 operator()(T1 p1){return pop::ArithmeticsSaturation<T1,T1 >::Range(std::pow(p1,_exponent));}};


}
/// @endcond
}

#endif // FUNCTORF_HPP
