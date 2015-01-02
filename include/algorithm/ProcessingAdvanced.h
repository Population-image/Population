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
#ifndef PROCESSINGADVANCED_H
#define PROCESSINGADVANCED_H


#include"data/typeF/TypeTraitsF.h"
#include"data/population/PopulationData.h"
#include"data/distribution/DistributionFromDataStructure.h"
#include"data/utility/BasicUtility.h"
#include"data/functor/FunctorF.h"
#include"data/population/PopulationFunctor.h"
#include"data/utility/CollectorExecutionInformation.h"
#include"data/functor/FunctorPDE.h"
#include"algorithm/Convertor.h"
#include"algorithm/FunctionProcedureFunctorF.h"
#include"algorithm/AnalysisAdvanced.h"
#include"algorithm/Statistics.h"
namespace pop
{
namespace Private{
template<typename Result >
struct FunctorConst{const Result & _v;FunctorConst(const Result & v ):_v(v){}Result operator()(){return _v;}};
template<typename Result, typename Type,bool isVectoriel, template<typename,typename> class PolicyClassOverFlow=ArithmeticsSaturation >
class FunctorDistribution: public PolicyClassOverFlow<Result,F64 >
{
private:
    Distribution & _f;
public:
    FunctorDistribution(Distribution & f )
        :_f(f)
    {}
    Result operator()(Type  x)
    {
        return PolicyClassOverFlow<Result,F64 >::Range(_f(x));
    }
};
template<typename Result, typename Type>
class FunctorDistribution<Result,  Type,true,ArithmeticsSaturation>: public ArithmeticsSaturation<Result,F64 >
{
private:
    Distribution & _f;
public:
    FunctorDistribution(Distribution & f )
        :_f(f)
    {}
    Result operator()(Type  x)
    {
        Result r;
        for(int i=0;i<Result::DIM;i++)
            r(i)=ArithmeticsSaturation<typename Result::F,F64 >::Range(_f(x(i)));
        return r;
    }
};
}
struct ProcessingAdvanced
{
    template<typename Function,int DIM>
    static Function  integral(const Function & f,Loki::Int2Type<DIM>);
    template<typename Function>
    static Function  integral(const Function & f,Loki::Int2Type<2>)
    {
        Function s (f.getDomain());
        Function out(f.getDomain());
        for(int i=0;i<f.getDomain()(0);i++){

            for(int j=0;j<f.getDomain()(1);j++){
                if(j==0)
                {
                    s(i,j)=f(i,j);
                }
                else
                {
                    s(i,j)=f(i,j)+s(i,j-1);
                }
                if(i==0)
                {
                    out(i,j)=s(i,j);
                }
                else
                {
                    out(i,j)=s(i,j)+out(i-1,j);
                }
            }
        }
        return out;
    }
    template<typename Function>
    static Function  integral(const Function & f,Loki::Int2Type<3>)
    {
        Function s (f.getDomain());
        Function integral2d (f.getDomain());
        Function out(f.getDomain());
        for(int i=0;i<f.getDomain()(0);i++){

            for(int j=0;j<f.getDomain()(1);j++){

                for(int k=0;k<f.getDomain()(2);k++){
                    if(k==0)
                    {
                        s(i,j,k)=f(i,j,k);
                    }
                    else
                    {
                        s(i,j,k)=f(i,j,k)+s(i,j,k-1);
                    }
                    if(j==0)
                    {
                        integral2d(i,j,k)=s(i,j,k);
                    }
                    else
                    {
                        integral2d(i,j,k)=s(i,j,k)+integral2d(i,j-1,k);
                    }
                    if(i==0)
                    {
                        out(i,j,k)=integral2d(i,j,k);
                    }
                    else
                    {
                        out(i,j,k)=integral2d(i,j,k)+out(i-1,j,k);
                    }
                }
            }
        }
        return out;
    }


    template<typename Function>
    static Function  nonMaximumSuppression(const Function & img,double sigma, typename FunctionTypeTraitsSubstituteF<Function,F64>::Result &gradnorm)
    {

        typedef typename FunctionTypeTraitsSubstituteF<Function,VecN<Function::DIM,F64> >::Result FunctionVecFloat;
        typedef typename FunctionTypeTraitsSubstituteF<Function,F64>::Result FunctionFloat;
        typedef typename FunctionTypeTraitsSubstituteF<Function,UI8 >::Result FunctionBinary;
        gradnorm= ProcessingAdvanced::gradNormGaussian(FunctionFloat(img),sigma,std::min(9.,sigma*3),img.getIteratorEDomain());
        FunctionVecFloat grad= ProcessingAdvanced::gradientVecGaussian(FunctionFloat(img),sigma);

        std::vector<double> vtan;
        vtan.push_back(std::tan(-3*PI/8));
        vtan.push_back(std::tan(-PI/8));
        vtan.push_back(std::tan( PI/8));
        vtan.push_back(std::tan(3*PI/8));
        FunctionBinary edge(img.getDomain());

        typename Function::IteratorERectangle it = img.getIteratorERectangle(1,img.getDomain()-2);
        while(it.next())
        {
            Mat2UI8::E x = it.x();
            double slop=-grad(x)(1)/grad(x)(0);
            int direction = (int)(std::lower_bound (vtan.begin(), vtan.end(), slop)-vtan.begin());
            if(direction==2){
                if(gradnorm(x(0),x(1))>=gradnorm(x(0)-1,x(1))&&gradnorm(x(0),x(1))>=gradnorm(x(0)+1,x(1))){
                    edge(x)=255;
                }
            }else if(direction==3){
                if(gradnorm(x(0),x(1))>=gradnorm(x(0)-1,x(1)+1)&&gradnorm(x(0),x(1))>=gradnorm(x(0)+1,x(1)-1)){
                    edge(x)=255;
                }
            }
            else if(direction==0||direction==4){
                if(gradnorm(x(0),x(1))>=gradnorm(x(0),x(1)-1)&&gradnorm(x(0),x(1))>=gradnorm(x(0),x(1)+1)){
                    edge(x)=255;
                }
            }
            else{
                if(gradnorm(x(0),x(1))>=gradnorm(x(0)-1,x(1)-1)&&gradnorm(x(0),x(1))>=gradnorm(x(0)+1,x(1)+1)){
                    edge(x)=255;
                }
            }
        };
        return edge;
    }

    template< typename Function, typename IteratorEDomain, typename IteratorENeighborhood >
    static typename FunctionTypeTraitsSubstituteF<Function,UI32 >::Result minimaLocalMap(const Function & f,IteratorEDomain& itd,IteratorENeighborhood&itn){
        typename FunctionTypeTraitsSubstituteF<Function,UI32 >::Result map(f.getDomain());
        UI32 index=1;
        while(itd.next()){
            typename Function::F value = f(itd.x());
            bool minima=true;
            itn.init(itd.x());
            while(itn.next()&&minima==true){
                if(f(itn.x())<value){
                    minima = false;
                }
            }
            if(minima==true){
                map(itd.x())=index;
                index++;
            }
        }
        return map;
    }

    template< typename Function, typename IteratorEDomain, typename IteratorENeighborhood >
    static Vec<typename Function::E> minimaLocal(const Function & f,IteratorEDomain& itd,IteratorENeighborhood&itn){
        Vec<typename Function::E> v_minima;
        while(itd.next()){
            typename Function::F value = f(itd.x());
            bool minima=true;
            itn.init(itd.x());
            while(itn.next()&&minima==true){
                if(f(itn.x())<value){
                    minima = false;
                }
            }
            if(minima==true)
                v_minima.push_back(itd.x());
        }
        return v_minima;
    }
    template< typename Function, typename IteratorEDomain, typename IteratorENeighborhood >
    static std::vector<typename Function::E> maximaLocal(const Function & f,IteratorEDomain& itd,IteratorENeighborhood&itn){
        Vec<typename Function::E> v_maxima;
        while(itd.next()){
            typename Function::F value = f(itd.x());
            bool maxima=true;
            itn.init(itd.x());
            while(itn.next()&&maxima==true){
                if(f(itn.x())>value){
                    maxima = false;
                }
            }
            if(maxima==true)
                v_maxima.push_back(itd.x());
        }
        return v_maxima;
    }
    template< typename Function, typename IteratorEDomain, typename IteratorENeighborhood >
    static typename FunctionTypeTraitsSubstituteF<Function,UI32 >::Result maximaLocalMap(const Function & f,IteratorEDomain& itd,IteratorENeighborhood&itn){
        typename FunctionTypeTraitsSubstituteF<Function,UI32 >::Result map(f.getDomain());
        UI32 index=1;
        while(itd.next()){
            typename Function::F value = f(itd.x());
            bool maxima=true;
            itn.init(itd.x());
            while(itn.next()&&maxima==true){
                if(f(itn.x())>value){
                    maxima = false;
                }
            }
            if(maxima==true){
                map(itd.x())=index;
                index++;
            }
        }
        return map;
    }
    template< typename Function, typename IteratorEDomain, typename IteratorENeighborhood >
    static Vec<typename Function::E> extremaLocal(const Function & f,IteratorEDomain& itd,IteratorENeighborhood&itn){
        Vec<typename Function::E> v_maxima;
        while(itd.next()){
            typename Function::F value = f(itd.x());
            bool maxima=true;
            bool minima=true;
            itn.init(itd.x());
            while(itn.next()&&(maxima==true||minima==true)){
                if(f(itn.x())>value){
                    maxima = false;
                }
                if(f(itn.x())<value){
                    minima = false;
                }
            }
            if(maxima==true||minima==true)
                v_maxima.push_back(itd.x());
        }
        return v_maxima;
    }
    template< typename Function, typename IteratorEDomain, typename IteratorENeighborhood >
    static typename FunctionTypeTraitsSubstituteF<Function,UI32 >::Result extremaLocalMap(const Function & f,IteratorEDomain& itd,IteratorENeighborhood&itn){
        typename FunctionTypeTraitsSubstituteF<Function,UI32 >::Result map(f.getDomain());
        UI32 index=1;
        while(itd.next()){
            typename Function::F value = f(itd.x());
            bool maxima=true;
            bool minima=true;
            itn.init(itd.x());
            while(itn.next()&&(maxima==true||minima==true)){
                if(f(itn.x())>value){
                    maxima = false;
                }
                if(f(itn.x())<value){
                    minima = false;
                }
            }
            if(maxima==true||minima==true){

                map(itd.x())=index;
                index++;
            }
        }
        return map;
    }

    template<typename Function,typename Iterator>
    static Function  randomField(const typename Function::Domain & domain, Distribution &d,Iterator &it){
        Function out (domain);
        while(it.next()){
            out(it.x())=d.randomVariable();
        }
        return out;
    }


    template<typename Function,typename Iterator>
    static Function  fill(const Function & in,const typename Function::F & y,Iterator &it){
        Function out (in.getDomain());
        Private::FunctorConst<typename Function::F> func(y);
        FunctionProcedureGenerator(func,it,out);
        return out;
    }



    template<typename Function,typename Iterator>
    static Function  fofx(const Function& in,Distribution &d,Iterator &it){
        Function out (in.getDomain());

        Private::FunctorDistribution<typename Function::F,typename Function::F,isVectoriel<typename Function::F>::value > func(d);
        FunctionProcedureFunctorUnaryF(in,func,it,out);
        return out;
    }
    template<typename Function,typename Iterator>
    static typename FunctionTypeTraitsSubstituteF<Function,UI8>::Result  threshold(const Function& in,typename Function::F ymin,typename Function::F ymax,Iterator &it){

        typename FunctionTypeTraitsSubstituteF<Function,UI8>::Result out(in.getDomain());
        FunctorF::FunctorThreshold<unsigned char,typename Function::F,typename  Function::F> func(ymin,ymax);
        FunctionProcedureFunctorUnaryF(in,func,it,out);
        return out;
    }
    template<typename Function,typename Iterator>
    static Function greylevelScaleContrast(const Function & f,F64 scale, Iterator & it)throw(pexception){

        Function out(f.getDomain());
        typedef typename FunctionTypeTraitsSubstituteF<typename Function::F,F64>::Result FloatF;
        FunctorF::FunctorAccumulatorMean<typename Function::F> func;
        it.init();
        FloatF mean = FunctionProcedureFunctorAccumulatorF(f,func,it);
        it.init();
        while(it.next()){
            FloatF value(f(it.x()));
            value = (value-mean)*scale+mean;
            out(it.x())= ArithmeticsSaturation<typename Function::F,FloatF>::Range(value);
        }
        return out;
    }
    template<typename Function,typename Iterator,int isVec>
    static Function greylevelRange(const Function & f,Iterator & it,typename Function::F min, typename Function::F max,Loki::Int2Type<isVec>)throw(pexception);

    template<typename Function,typename Iterator>
    static Function greylevelRange(const Function & f,Iterator & it,typename Function::F min, typename Function::F max,Loki::Int2Type<true>)throw(pexception)
    {
        typedef typename TypeTraitsTypeScalar<typename Function::F>::Result  FScalar;
        typedef typename FunctionTypeTraitsSubstituteF<Function,FScalar>::Result  FunctionScalar;
        VecN< Function::F::DIM,FunctionScalar > v ;
        pop::Convertor::toVecN(f,v);
        for(int i=0;i<Function::F::DIM;i++){
            it.init();
            v[i] = ProcessingAdvanced::greylevelRange(v[i],it,min(i),max(i),Loki::Int2Type< isVectoriel<FScalar>::value>());
        }
        Function dilat;
        Convertor::fromVecN(v,dilat);
        return dilat;
    }

    template<typename Function,typename Iterator>
    static Function greylevelRange(const Function & f,Iterator & it,typename Function::F min, typename Function::F max,Loki::Int2Type<false>)throw(pexception)
    {
        Function h(f.getDomain());
        FunctorF::FunctorAccumulatorMin<typename Function::F > funcmini;
        typename Function::F mini = FunctionProcedureFunctorAccumulatorF(f,funcmini,it);
        it.init();
        FunctorF::FunctorAccumulatorMax<typename Function::F > funcmaxi;
        typename Function::F maxi = FunctionProcedureFunctorAccumulatorF(f,funcmaxi,it);
        typename FunctionTypeTraitsSubstituteF<typename Function::F,F64>::Result ratio;
        if(maxi!=mini)
            ratio= 1.0*(max-min)/(maxi-mini);
        else
            ratio =1;

        std::string exp =BasicUtility::Any2String(ratio)+"*(x-"+BasicUtility::Any2String(double(mini))+")+"+BasicUtility::Any2String(double(min));
        DistributionExpression dist(exp);
        Private::FunctorDistribution<typename Function::F,typename Function::F,isVectoriel<typename Function::F>::value > func(dist);
        it.init();
        FunctionProcedureFunctorUnaryF(f,func,it,h);
        return h;
    }
    template<typename Function,typename Iterator>
    static Function greylevelRange(const Function & f,Iterator & it,typename Function::F min=NumericLimits<typename Function::F>::minimumRange(), typename Function::F max=NumericLimits<typename Function::F>::maximumRange())throw(pexception)
    {
        Function h(f.getDomain());
        FunctorF::FunctorAccumulatorMin<typename Function::F > funcmini;
        typename Function::F mini = FunctionProcedureFunctorAccumulatorF(f,funcmini,it);
        it.init();
        FunctorF::FunctorAccumulatorMax<typename Function::F > funcmaxi;
        typename Function::F maxi = FunctionProcedureFunctorAccumulatorF(f,funcmaxi,it);
        typename FunctionTypeTraitsSubstituteF<typename Function::F,F64>::Result ratio= 1.0*(max-min)/(maxi-mini);

        std::string exp =BasicUtility::Any2String(ratio)+"*(x-"+BasicUtility::Any2String(mini)+")+"+BasicUtility::Any2String(min);
        DistributionExpression dist(exp);
        Private::FunctorDistribution<typename Function::F,typename Function::F,isVectoriel<typename Function::F>::value > func(dist);
        it.init();
        FunctionProcedureFunctorUnaryF(f,func,it,h);
        return h;
    }

    template<typename Function,typename Iterator>
    static Function greylevelRemoveEmptyValue(const Function & f,  Iterator & it)throw(pexception)
    {
        Function h(f.getDomain());
        std::vector<typename Function::F> valueinf;
        while(it.next())
        {
            typename Function::F i= f(it.x());
            if( std::find(valueinf.begin(),valueinf.end(),i)==valueinf.end() )valueinf.push_back(i);
        }
        std::sort (valueinf.begin(), valueinf.end());

        typename std::vector<typename Function::F>::iterator  itvalue;
        it.init();
        while(it.next())
        {
            typename Function::F i= f(it.x());
            itvalue=std::find(valueinf.begin(),valueinf.end(),i);
            i=static_cast<I32>(itvalue-valueinf.begin());
            h(it.x())=i;
        }
        return h;
    }
    template<typename Function,typename Type>
    static Function greylevelTranslateMeanValueCast(const Function& f, typename Function::F mean,Loki::Type2Type<Type> )throw(pexception)
    {
        return  greylevelTranslateMeanValueScalar(f,mean);
    }
    template<typename Function>
    static Function greylevelTranslateMeanValueCast(const Function& f, typename Function::F mean,Loki::Type2Type<RGBUI8> )throw(pexception)
    {
        typename FunctionTypeTraitsSubstituteF<Function,typename TypeTraitsTypeScalar<RGBUI8>::Result>::Result r,g,b;
        Convertor::toRGB(f,r,g,b);
        typename TypeTraitsTypeScalar<RGBUI8>::Result rmean,gmean,bmean;
        rmean=mean.r();
        gmean=mean.g();
        bmean=mean.b();
        r = greylevelTranslateMeanValueScalar(r,rmean);
        g = greylevelTranslateMeanValueScalar(g,gmean);
        b = greylevelTranslateMeanValueScalar(b,bmean);
        Function h(f.getDomain());
        Convertor::fromRGB(r,g,b,h);
        return h;
    }
    template<typename Function>
    static Function greylevelTranslateMeanValueCast(const Function& f, typename Function::F mean,Loki::Type2Type<RGBF64> )throw(pexception)
    {
        typename FunctionTypeTraitsSubstituteF<Function,typename TypeTraitsTypeScalar<RGBF64>::Result>::Result r,g,b;
        Convertor::toRGB(f,r,g,b);
        typename TypeTraitsTypeScalar<RGBF64>::Result rmean,gmean,bmean;
        rmean=mean.r();
        gmean=mean.g();
        bmean=mean.b();
        r = greylevelTranslateMeanValueScalar(r,rmean);
        g = greylevelTranslateMeanValueScalar(g,gmean);
        b = greylevelTranslateMeanValueScalar(b,bmean);
        Function h(f.getDomain());
        Convertor::fromRGB(r,g,b,h);
        return h;
    }
    template<typename Function>
    static Function greylevelTranslateMeanValueScalar(const Function& f, typename Function::F mean )throw(pexception)
    {
        typename Function::IteratorEDomain it(f.getIteratorEDomain());
        Mat2F64 m = AnalysisAdvanced::histogram(f,it);
        it.init();
        F64 pow_min=0;
        F64 pow_max=1000;
        F64 pow_current=1;
        F64 error_max=0.1;
        F64 error_current=1;
        int number=0;
        bool test=false;
        while(test==false){
            number++;
            DistributionExpression exp;
            std::string str = "(x/256)^("+BasicUtility::Any2String(pow_current)+")*256";
            exp.fromRegularExpression(str);
            F64 meantemp=0;
            for(unsigned int i=0;i<m.sizeI();i++){
                meantemp  +=  exp.operator ()(m(i,0))*m(i,1);
            }
            error_current = absolute(meantemp-mean);
            if(error_current<error_max)
                test =true;
            else{
                if(meantemp<mean)
                    pow_max = pow_current;
                else
                    pow_min  = pow_current;
                pow_current =(pow_max-pow_min)/2+pow_min;
                if(number>2000){
                    test =true;
                }
            }
        }
        it.init();
        //small covolution
        typedef typename FunctionTypeTraitsSubstituteF<Function,F64>::Result FunctionF64;
        FunctionF64 fd(f.getDomain());
        DistributionUniformReal d(-0.5,0.5);
        while(it.next()){
            fd(it.x())=f(it.x())+d.randomVariable();
        }

        Function outcast(f.getDomain());
        DistributionExpression exp;
        std::string str = "(x/256)^("+BasicUtility::Any2String(pow_current)+")*256";
        exp.fromRegularExpression(str);
        Private::FunctorDistribution<typename Function::F,F64,false> func(exp);
        it.init();
        FunctionProcedureFunctorUnaryF(fd,func,it,outcast);
        return outcast;
    }
    template<typename Function,typename FunctionMask, typename Iterator>
    static Function mask(const Function & f,const FunctionMask & mask,  Iterator & it)throw(pexception)
    {
        Function h(f.getDomain());
        it.init();
        while(it.next())
        {
            if(mask(it.x())!=0)
                h(it.x())=f(it.x());
            else
                h(it.x())=0;
        }
        return h;
    }

    /*! \fn Function erosion(const Function & f,IteratorGlobal & itglobal, IteratorLocal & itlocal)throw(pexception)
     *  \brief Erosion of the input matrix
     * \param f input function
     * \param itglobal Global IteratorE
     * \param itlocal Local  IteratorE
     * \return h output function
     *
     *  Erosion of the input matrix:\n
     * \f$\forall x \in E':\quad h(x) =\min_{\forall x'\in N(x) }f(x) \f$ where the iteration trough  \f$\forall x \in E'\f$ is done by the global IteratorE
     * and the iteration trough  \f$\forall x'\in N(x)\f$  is done by the local  IteratorE
    */
    template< typename Function,typename IteratorGlobal, typename IteratorLocal >
    static Function erosion(const Function & f,IteratorGlobal & itglobal, IteratorLocal & itlocal)throw(pexception)
    {
        CollectorExecutionInformationSingleton::getInstance()->startExecution("Erosion",COLLECTOR_EXECUTION_NOINFO);
        CollectorExecutionInformationSingleton::getInstance()->info("Slow algorithm for large structural element");
        Function h(f.getDomain());
        typedef FunctorF::FunctorAccumulatorMin<typename Function::F > FunctorAccumulator;
        FunctorAccumulator funcAccumulator;
        FunctionProcedureLocal(f,itglobal,itlocal,funcAccumulator, h);
        CollectorExecutionInformationSingleton::getInstance()->endExecution("Erosion");
        return h;
    }
    /*! \fn Function dilation(const Function & f,IteratorGlobal & itglobal, IteratorLocal & itlocal )throw(pexception)
     *  \brief Dilation of the input matrix
     * \param f input function
     * \param itglobal Global IteratorE
     * \param itlocal Local  IteratorE
     * \return h output function
     *
     *  Dilation of the input matrix:\n
     * \f$\forall x \in E':\quad h(x) =\max_{\forall x'\in N(x) }f(x) \f$ where the iteration trough \f$\forall x \in E'\f$ is done by the global IteratorE
     * and the iteration trough  \f$\forall x'\in N(x)\f$  is done by the local  IteratorE
    */
    template< typename Function, typename IteratorGlobal,typename IteratorLocal>
    static Function dilation(const Function & f,IteratorGlobal & itglobal, IteratorLocal & itlocal )throw(pexception)
    {
        Function h(f.getDomain());
        CollectorExecutionInformationSingleton::getInstance()->startExecution("Dilation",COLLECTOR_EXECUTION_NOINFO);
        CollectorExecutionInformationSingleton::getInstance()->info("Slow algorithm for large structural element");
        typedef FunctorF::FunctorAccumulatorMax<typename Function::F > FunctorAccumulator;
        FunctorAccumulator funcAccumulator;
        FunctionProcedureLocal(f,itglobal,itlocal,funcAccumulator, h);
        CollectorExecutionInformationSingleton::getInstance()->endExecution("Dilation");
        return h;
    }
    /*! \fn Function closing(const Function & f,IteratorGlobal & itglobal, IteratorLocal & itlocal )throw(pexception)
     *  \brief Closing of the input matrix
     * \param f input function
     * \param itglobal Global IteratorE
     * \param itlocal Local  IteratorE
     * \param h output function
     *
     *  Closing of the input matrix:\n
     * \f$\forall x \in E':\quad h(x) =Erosion(Dilation(f,E',N),E',N) \f$ where the iteration trough \f$E'\f$ is done by the global IteratorE
     * and the iteration trough \f$N(x)\f$  is done by the local  IteratorE
    */
    template<typename Function,typename IteratorGlobal,typename IteratorLocal>
    static Function closing(const Function & f,IteratorGlobal & itglobal, IteratorLocal & itlocal )throw(pexception)
    {
        Function h(f.getDomain());
        CollectorExecutionInformationSingleton::getInstance()->startExecution("Closing",COLLECTOR_EXECUTION_NOINFO);
        CollectorExecutionInformationSingleton::getInstance()->info("Slow algorithm for large structural element");
        Function temp =ProcessingAdvanced::dilation(f,itglobal,itlocal);
        itglobal.init();
        h = ProcessingAdvanced::erosion(temp,itglobal,itlocal);
        CollectorExecutionInformationSingleton::getInstance()->endExecution("Closing");
        return h;
    }

    /*! \fn Function opening(const Function & f,IteratorGlobal & itglobal, IteratorLocal & itlocal)throw(pexception)
     *  \brief Opening of the input matrix
     * \param f input function
     * \param itglobal Global IteratorE
     * \param itlocal Local  IteratorE
     * \return h output function
     *
     *  Opening of the input matrix:\n
     * \f$\forall x \in E':\quad h(x) =Dilation(Erosion(f,E',N),E',N) \f$ where the iteration trough \f$E'\f$ is done by the global IteratorE
     * and the iteration trough \f$N(x)\f$  is done by the local  IteratorE
    */
    template<typename Function, typename IteratorGlobal,typename IteratorLocal >
    static Function opening(const Function & f,IteratorGlobal & itglobal, IteratorLocal & itlocal)throw(pexception)
    {
        Function h(f.getDomain());
        CollectorExecutionInformationSingleton::getInstance()->startExecution("Opening",COLLECTOR_EXECUTION_NOINFO);
        CollectorExecutionInformationSingleton::getInstance()->info("Slow algorithm for large structural element");
        Function temp =ProcessingAdvanced::erosion(f,itglobal,itlocal);
        itglobal.init();
        h = ProcessingAdvanced::dilation(temp,itglobal,itlocal);
        CollectorExecutionInformationSingleton::getInstance()->endExecution("Opening");
        return h;
    }

    /*! \fn Function median(const Function & f,IteratorGlobal & itglobal, IteratorLocal & itlocal)throw(pexception)
     *  \brief Median filter of the input matrix
     * \param f input function
     * \param itglobal Global IteratorE
     * \param itlocal Local  IteratorE
     * \return h output function
     *
     *  Median filter of the input matrix:\n
     * \f$\forall x \in E':\quad h(x) =\mbox{median}_{\forall x'\in N(x) }f(x) \f$ where the operator median returns the median value of the list of input values
     * , the iteration trough  \f$\forall x \in E'\f$ is done by the global IteratorE
     * and the iteration trough  \f$\forall x'\in N(x)\f$  is done by the local  IteratorE
    */
    template<typename Function,typename IteratorGlobal,typename IteratorLocal>
    static Function median(const Function & f,IteratorGlobal & itglobal, IteratorLocal & itlocal)throw(pexception)
    {
        Function h(f.getDomain());
        CollectorExecutionInformationSingleton::getInstance()->startExecution("Median",COLLECTOR_EXECUTION_NOINFO);
        CollectorExecutionInformationSingleton::getInstance()->info("Slow algorithm for large structural element");
        typedef FunctorF::FunctorAccumulatorMedian<typename Function::F> FunctorAccumulator;
        FunctorAccumulator funcAccumulator;
        FunctionProcedureLocal(f,itglobal,itlocal,funcAccumulator, h);
        CollectorExecutionInformationSingleton::getInstance()->endExecution("Median");
        return h;
    }

    /*! \fn Function mean(const Function & f,IteratorGlobal & itglobal, IteratorLocal & itlocal)throw(pexception)
     *  \brief Median filter of the input matrix
     * \param f input function
     * \param itglobal Global IteratorE
     * \param itlocal Local  IteratorE
     * \return h output function
     *
     *  Median filter of the input matrix:\n
     * \f$\forall x \in E':\quad h(x) =\mbox{mean}_{\forall x'\in N(x) }f(x) \f$ where the operator mean returns the mean value of the list of input values
     * , the iteration trough  \f$\forall x \in E'\f$ is done by the global IteratorE
     * and the iteration trough  \f$\forall x'\in N(x)\f$  is done by the local  IteratorE
    */
    template<typename Function,typename IteratorGlobal,typename IteratorLocal>
    static Function mean(const Function & f,IteratorGlobal & itglobal, IteratorLocal & itlocal)throw(pexception)
    {
        Function h(f.getDomain());
        CollectorExecutionInformationSingleton::getInstance()->startExecution("Mean",COLLECTOR_EXECUTION_NOINFO);
        CollectorExecutionInformationSingleton::getInstance()->info("Slow algorithm for large structural element");
        typedef FunctorF::FunctorAccumulatorMean<typename Function::F> FunctorAccumulator;
        FunctorAccumulator funcAccumulator;
        FunctionProcedureLocal(f,itglobal,itlocal,funcAccumulator, h);
        CollectorExecutionInformationSingleton::getInstance()->endExecution("Mean");
        return h;
    }


    /*! \fn Function alternateSequentialCOStructuralElement(const Function & f,IteratorGlobal & itglobal,IteratorLocal & itlocal, int maxradius)throw(pexception)
     *  \brief Sequential Alternate filter of the input matrix
     * \param f input function
     * \param itglobal Global IteratorE
     * \param itlocal initial structural element
     * \param maxradius max radius
     * \return h output function
     *
     *  Successive application of Closing and opening by increasing the scale factor of structure element until max radius
    */
    template<typename Function,typename IteratorGlobal,typename IteratorLocal >
    static Function alternateSequentialCO(const Function & f,IteratorGlobal & itglobal,IteratorLocal & itlocal, int maxradius)throw(pexception)
    {
        Function h(f.getDomain());
        CollectorExecutionInformationSingleton::getInstance()->startExecution("AlternateSequentialCOStructuralElement",COLLECTOR_EXECUTION_NOINFO);
        CollectorExecutionInformationSingleton::getInstance()->info("Slow algorithm for large structural element");

        IteratorLocal  itlocalsuccessive(itlocal);
        Function temp(f);
        h=f;
        for(int radius=1;radius<=maxradius;radius++){
            CollectorExecutionInformationSingleton::getInstance()->info("Radius : "+BasicUtility::Any2String(radius));
            itglobal.init();
            temp = ProcessingAdvanced::closing(h,itglobal,itlocalsuccessive);
            itglobal.init();
            h = ProcessingAdvanced::opening(temp,itglobal,itlocalsuccessive);
            itlocalsuccessive.dilate(itlocal);
        }
        CollectorExecutionInformationSingleton::getInstance()->endExecution("AlternateSequentialCOStructuralElement");
        return h;

    }
    /*! \fn Function alternateSequentialOCStructuralElement(const Function & f,IteratorGlobal & itglobal,IteratorLocal & itlocal, int maxradius)throw(pexception)
     *  \brief Sequential Alternate filter of the input matrix
     * \param f input function
     * \param itglobal Global IteratorE
      * \param itlocal initial structural element
     * \param maxradius max radius
     * \return h output function
     *
     *  Successive application of opening and closing by increasing the scale factor of structure element until max radius
    */
    template< typename Function,typename IteratorGlobal,typename IteratorLocal>
    static Function alternateSequentialOC(const Function & f,IteratorGlobal & itglobal,IteratorLocal & itlocal, int maxradius)throw(pexception)
    {
        CollectorExecutionInformationSingleton::getInstance()->startExecution("AlternateSequentialOCStructuralElement",COLLECTOR_EXECUTION_NOINFO);
        CollectorExecutionInformationSingleton::getInstance()->info("Slow algorithm for large structural element");
        Function h(f.getDomain());

        IteratorLocal  itlocalsuccessive(itlocal);
        Function temp(f);
        h=f;
        for(int radius=1;radius<=maxradius;radius++){
            itglobal.init();
            temp = ProcessingAdvanced::opening(h,itglobal,itlocalsuccessive);
            itglobal.init();
            h = ProcessingAdvanced::closing(temp,itglobal,itlocalsuccessive);
            itlocalsuccessive.dilate(itlocal);
        }
        CollectorExecutionInformationSingleton::getInstance()->endExecution("AlternateSequentialOCStructuralElement");
        return h;

    }

    /*! \fn static Function hitOrMiss(const Function & f,IteratorGlobal & itglobal,IteratorLocal & itC, IteratorLocal & itD)throw(pexception)
     *  \brief Hit or miss filter
     * \param f input function
     * \param itglobal Global IteratorE
     * \param itC  local iterator for the iteration through the set C
     * \param itD local iterator for the iteration through the set D
     * \return h output function
     *
     *  \f$ H = (X\ominus C)\cap (X^c\ominus D) \f$ with \f$\ominus\f$ the erosion, \f$X=\{x:f(x)\neq 0 \} \f$ and \f$H=\{x:h(x)\neq 0 \}\f$.\n
     * For instance a direct implementation of the thinning algorithm http://en.wikipedia.org/wiki/Hit-or-miss_transform is
     * \code
    Mat2UI8 img;
    img.load("../image/outil.bmp");
    Processing processing;
    Mat2UI8 C_1(3,3),D_1(3,3);
    Mat2UI8 C_2(3,3),D_2(3,3);
    C_1(1,1)=255;C_1(0,0)=255;C_1(1,0)=255;C_1(2,0)=255;
    D_1(0,2)=255;D_1(1,2)=255;D_1(2,2)=255;
    C_2(0,1)=255;C_2(1,1)=255;C_2(0,0)=255;C_2(1,0)=255;
    D_2(1,2)=255;D_2(2,2)=255;D_2(2,1)=255;
    LinearAlgebra algebra;

    Mat2UI8 temp(img);
    Mat2UI8 temp2;
    int nbr_equal=0;
    while(nbr_equal<8){
        temp2 = processing.hitOrMiss(temp,C_1,D_1);
        temp = temp -temp2;
        C_1.rotate(algebra.generate2DRotation(PI/2),Vec2I32(1,1) );
        C_1 =processing.threshold(C_1,125);//due to the interpolation with the rotation, the value can fluctuate to remove these fluctuation, we apply a threshold
        D_1.rotate(algebra.generate2DRotation(PI/2),Vec2I32(1,1) );
        D_1 =processing.threshold(D_1,125);

        temp2 = processing.hitOrMiss(temp,C_2,D_2);
        temp = temp -temp2;
        C_2.rotate(algebra.generate2DRotation(PI/2),Vec2I32(1,1) );
        C_2 =processing.threshold(C_2,125);

        D_2.rotate(algebra.generate2DRotation(PI/2),Vec2I32(1,1) );
        D_2 =processing.threshold(D_2,125);
        if(temp==img){
            nbr_equal++;
        }else{
            nbr_equal=0;
        }
        img =temp;
    }
    img.display();
    \endcode
    */

    template< typename Function, typename IteratorGlobal,typename IteratorLocal >
    static Function hitOrMiss(const Function & f,IteratorGlobal & itglobal,IteratorLocal & itC, IteratorLocal & itD)throw(pexception)
    {
        Function h(f.getDomain());
        CollectorExecutionInformationSingleton::getInstance()->startExecution("HitOrMiss",COLLECTOR_EXECUTION_NOINFO);
        CollectorExecutionInformationSingleton::getInstance()->info("Slow algorithm for large structural element");

        Function erosion1(f.getDomain());
        itglobal.init();
        //        std::cout<<f<<std::endl;
        erosion1 =ProcessingAdvanced::erosion(f,itglobal,itC);
        //        std::cout<<erosion1<<std::endl;
        //        getchar();
        Function temp(f.getDomain(),NumericLimits<typename Function::F>::maximumRange());
        temp =   temp-f;
        //        std::cout<<"TEMP"<<std::endl;
        //        std::cout<<temp<<std::endl;
        Function erosion2(f.getDomain());
        itglobal.init();
        erosion2 = ProcessingAdvanced::erosion(temp,itglobal,itD);
        //        std::cout<<erosion2<<std::endl;
        //        getchar();
        h=minimum(erosion1,erosion2);
        //        std::cout<<h<<std::endl;
        //        getchar();
        CollectorExecutionInformationSingleton::getInstance()->endExecution("HitOrMiss");
        return h;
    }

    template<int DIM, typename Type,typename IteratorE>
    static MatN<DIM,Type> gradSobel(const MatN<DIM,Type> & f, I32 direction, IteratorE it)
    {

        pop::Vec<F64> der (3),smooth(3);
        der(0)=1; der(1)=0; der(2)=-1;
        smooth(0)=1;smooth(1)=2;smooth(2)=1;

        MatN<DIM,Type> fout(f);

        for(I32 i=0;i <DIM;i++)
        {
            it.init();

            if(i==direction)
               fout = FunctorMatN::convolutionSeperable(fout,der,direction,it,MatNBoundaryConditionMirror());
            else
                fout = FunctorMatN::convolutionSeperable(fout,smooth,direction,it,MatNBoundaryConditionMirror());
        }
        return fout;
    }
    template<int DIM, typename Type,typename Iterator>
    static MatN<DIM,Type> gradNormSobel(const MatN<DIM,Type> & f, Iterator it)
    {
        typedef typename FunctionTypeTraitsSubstituteF<Type,F64>::Result TypeF64;
        MatN<DIM,TypeF64> ffloat(f);
        MatN<DIM,TypeF64> fdir(f.getDomain());
        MatN<DIM,TypeF64> fsum(f.getDomain());

        for(I32 i=0;i <DIM;i++)
        {
            it.init();
            fdir = ProcessingAdvanced::gradSobel(ffloat,i,it);
            fsum+= fdir.multTermByTerm(fdir);
        }
        MatN<DIM,TypeF64> grad(f.getDomain());
        it.init();
        while(it.next())
        {
            TypeF64 value = pop::squareRoot(fsum(it.x()));
            grad(it.x())=ArithmeticsSaturation<Type,TypeF64>::Range(value);
        }
        return grad;
    }
    template<int DIM,typename PixelType,typename IteratorE>
    static MatN<DIM,PixelType> gradGaussian(const MatN<DIM,PixelType> & f, I32 direction,F64 sigma, int size_kernel, IteratorE it)
    {
        return FunctorMatN::convolutionGaussianDerivate(f,it,direction,sigma,size_kernel);
    }
    template<int DIM,typename PixelType,typename IteratorE>
    static MatN<DIM,PixelType> smoothGaussian(const MatN<DIM,PixelType> & f, F64 sigma, int size_kernel,IteratorE it){
        return FunctorMatN::convolutionGaussian(f,it,sigma,size_kernel);
    }
    template<class Function1,typename Iterator>
    static  Function1 gradNormGaussian(const Function1 & f, F64 sigma,int size_kernel,Iterator it)
    {
        typedef typename FunctionTypeTraitsSubstituteF<typename Function1::F,F64>::Result Type_F64;
        typename FunctionTypeTraitsSubstituteF<Function1,Type_F64>::Result ffloat(f);
        typename FunctionTypeTraitsSubstituteF<Function1,Type_F64>::Result fdir(f.getDomain());
        typename FunctionTypeTraitsSubstituteF<Function1,Type_F64>::Result fsum(f.getDomain());

        for(I32 i=0;i <Function1::DIM;i++)
        {
            it.init();
            fdir= FunctorMatN::convolutionGaussianDerivate(ffloat,i,sigma,size_kernel);
            fsum+= fdir.multTermByTerm(fdir);
        }
        Function1 g(f.getDomain());
        it.init();
        while(it.next())
        {
            g(it.x())=ArithmeticsSaturation<typename Function1::F,Type_F64>::Range(squareRoot(fsum(it.x())));
        }
        return g;
    }
    template<class Function1>
    static typename FunctionTypeTraitsSubstituteF<Function1,VecN<Function1::DIM,F64> >::Result gradientVecGaussian(const Function1  & f,double sigma=1)
    {
        typedef typename FunctionTypeTraitsSubstituteF<Function1,F64 >::Result  FunctionFloat;
        VecN<Function1::DIM,FunctionFloat> v_der;
        for(int i =0;i<Function1::DIM;i++){
            v_der[i]= ProcessingAdvanced::gradGaussian(f,i,sigma,3*sigma,f.getIteratorEDomain());
        }
        typename FunctionTypeTraitsSubstituteF<Function1,VecN<Function1::DIM,F64> >::Result f_grad(f.getDomain());
        Convertor::fromVecN(v_der,f_grad);
        return f_grad;
    }

    /*! \fn static Function labelMerge(const Function & label1, const Function &label2, Iterator & it)throw(pexception)
     *  \brief merge of the labelled matrix
     * \param label1 labelled matrix1
     * \param label1 labelled matrix2
     * \param it order iterator
     * \return h output labelled function
     *
     * Operate the merge of the two input labelled matrixs that can contain multi-labels
     *  \code
     * Mat2UI8 img;
     * img.load("../image/iex.png");
     * //filtering
     * img = pop::Processing::median(img,4);
     * //seed localisation
     * Mat2UI8 seed1 = pop::Processing::threshold(img,0,100);//seed in the grains
     * Mat2UI8 seed2 = pop::Processing::threshold(img,160);//seed in the background
     * Mat2UI8 seeds = pop::Processing::labelMerge(seed1,seed2);//merge of the seeds
     * //for a good segmentation, each seed should be include in its associated object and touch component of its associated object
     * //Test of the condition with a visal checking
     * Mat2RGBUI8 RGB = pop::Visualization::labelForeground(seed1,img);
     * RGB.display();
     *  \endcode
    */
    template<typename Function,typename Iterator>
    static Function labelMerge(const Function & label1, const Function & label2, Iterator & it)throw(pexception)
    {
        FunctionAssert(label1,label2,"pop::Processing::labelMerge");
        Function h(label1.getDomain());
        std::vector<typename Function::F> vf;
        std::vector<typename Function::F> vg;
        while(it.next())
        {
            if(label1(it.x())!=0)
            {
                typename Function::F valuef= label1(it.x());
                if( std::find(vf.begin(),vf.end(),valuef)==vf.end() )vf.push_back(valuef);
            }
            if(label2(it.x())!=0)
            {
                typename Function::F valueg= label2(it.x());
                if( std::find(vg.begin(),vg.end(),valueg)==vg.end() )vg.push_back(valueg);
            }
        }
        std::sort (vf.begin(), vf.end());
        std::sort (vg.begin(), vg.end());
        if(vf.size()+vg.size()>=NumericLimits<typename Function::F>::maximumRange())
        {
            throw(pexception("In pop::Processing::labelMerge, we have more labels than the grey-level range. Convert your labels images with the type Mat2UI32 before to call this algorithm (Mat2UI32 label=your_image_label"));
        }
        typename std::vector<typename Function::F>::iterator  itvalue;
        it.init();
        while(it.next())
        {
            if(label1(it.x())!=0)
            {
                typename Function::F value= label1(it.x());
                itvalue=std::find(vf.begin(),vf.end(),value);
                value=static_cast<I32>(itvalue-vf.begin())+1;
                h(it.x())=value;
            } else if(label2(it.x())!=0)
            {
                typename Function::F value= label2(it.x());
                itvalue=std::find(vg.begin(),vg.end(),value);
                value=static_cast<I32>(itvalue-vg.begin())+1+static_cast<I32>(vf.size());
                h(it.x())=value;
            }
            else h(it.x())=0;
        }
        return h;
    }
    /*! \fn static Function2 labelFromSingleSeed(const Function1 & label,const Function2& seed, Iterator & it)throw(pexception)
     *  \brief extract the label including the binary seed
     * \param label multi-labelled matrix
     * \param seed binary seed
     * \param it order iterator
     * \return h output binary matrix function
     *
     * From the multi-labelled matrix, we extract the label including the seed
     *  \code
     * Mat2UI8 img;
     * img.load("../image/iex.png");
     * //filtering
     * img = pop::Processing::median(img,4);
     * //seed localisation
     * Mat2UI8 seed1 = pop::Processing::threshold(img,0,100);//seed in the grains
     * Mat2UI8 seed2 = pop::Processing::threshold(img,160);//seed in the background
     * Mat2UI8 seeds = pop::Processing::labelMerge(seed1,seed2);//merge of the seeds
     * //for a good segmentation, each seed should be include in its associated object (here the grains and the background) and touch each component of its associated object
     * //Test of the condition with a visal checking
     * Mat2RGBUI8 RGB = pop::Visualization::labelForeground(seed1,img);
     * RGB.display();

     *  //topographic surface is the magnitude gradient of the input matrix
     * Mat2UI8 topo = pop::Processing::gradientMagnitudeDeriche(img,0.5);
     * //watershed as region growing on the topographic surface with seeds
     * Mat2UI8 regions = pop::Processing::watershed(seeds,topo);


     * //test the agreement between visual segmentation and numerical one
     * RGB = pop::Visualization::labelForeground(regions,img);
     * RGB.display();

     * //Extract the grain label
     * Mat2UI8 grain = pop::Processing::labelFromSingleSeed(regions,seeds);
     * grain.display();
     *  \endcode
    */

    template<typename Function1,typename Iterator,typename Function2>
    static typename FunctionTypeTraitsSubstituteF<Function1,UI8>::Result  labelFromSingleSeed(const Function1 & label,const Function2& seed, Iterator & it)throw(pexception)
    {
        FunctionAssert(label,seed,"pop::Processing::labelFromSingleSeed");
        typename FunctionTypeTraitsSubstituteF<Function1,UI8>::Result h(seed.getDomain());
        it.init();
        typename Function2::F value(0);
        while(it.next())
        {
            if(seed(it.x())!=0) value = label(it.x());
        }
        it.init();
        while(it.next())
        {
            if(label(it.x())==value) h(it.x()) = NumericLimits<typename Function2::F>::maximumRange();
            else h(it.x())=0;
        }
        return h;
    }

    /*! \fn FunctionBinary holeFilling( const FunctionBinary& bin,typename FunctionBinary::IteratorENeighborhood itneigh)
     * \param bin input binary matrix
     * \param itneigh domain of the neighborhood iterator
     * \return hole output matrix
     *
     *  hole filling of the input binary matrix
    */

    template<typename FunctionBinary>
    static FunctionBinary holeFilling( const FunctionBinary& bin,typename FunctionBinary::IteratorENeighborhood itneigh)
    {
        CollectorExecutionInformationSingleton::getInstance()->startExecution("HoleFilling",COLLECTOR_EXECUTION_NOINFO);
        CollectorExecutionInformationSingleton::getInstance()->info("Fast algorithm");
        FunctorZero f;
        Population<FunctionBinary,FunctorZero> pop(bin.getDomain(),f,itneigh);
        typename FunctionBinary::IteratorEDomain it(bin.getIteratorEDomain());
        it.init();
        while(it.next()){
            for(int i = 0; i<FunctionBinary::DIM;i++){
                if(it.x()(i)==0||it.x()(i)==bin.getDomain()(i)-1){
                    if(bin(it.x())==0)
                        pop.growth(0,it.x());
                }
            }
            if(bin(it.x())!=0){
                pop.setRegion(1,it.x());
            }
        }
        while(pop.next()){
            pop.growth(pop.x().first,pop.x().second);
        }
        FunctionBinary hole(bin.getDomain());
        it.init();
        while(it.next()){
            if(pop.getRegion()(it.x())==0)
                hole(it.x())=0;
            else
                hole(it.x())=255;
        }
        CollectorExecutionInformationSingleton::getInstance()->endExecution("HoleFilling");
        return hole;
    }

    /*! \fn FunctionLabel regionGrowingAdamsBischofMeanOverStandardDeviation(const FunctionLabel & seed,const FunctionTopo & topo, typename FunctionTopo::IteratorENeighborhood  itneigh )
     * \param seed input seeds
     * \param topo topographic surface
     * \param itneigh neighborhood IteratorE domain
     * \return  regions
     *
     *  Classical region growing algorithm  of Adams and Bischof such that the ordering attribute function is:\n
     *  \f$ \delta(x,i) = \frac{|f(x)- \mu_i|}{\sigma_i}\f$   where
     *  with f the topograhic surface, X_i the region, \f$\mu_i\f$ the mean value inside the seed and \f$\sigma_i\f$ is the standard deviation \f$\sqrt[]{\frac{\sum_{y\in X_i}(f(y)-\mu_i)^2}{\sum_{x\in X_i}1}} \f$
     *
     *     * \code
     * Mat2UI8 img;
     * img.load("../image/iex.png");
     * //filtering
     * img = pop::Processing::median(img,3);
     * //seed localisation
     * Mat2UI8 seed1 = pop::Processing::threshold(img,0,100);//seed in the grains
     * Mat2UI8 seed2 = pop::Processing::threshold(img,160);//seed in the background
     * Mat2UI8 seeds = pop::Processing::labelMerge(seed1,seed2);//merge of the seeds
     * //for a good segmentation, each seed should be include in its associated object (here the grains and the background) and touch each component of its associated object
     * //Test of the condition with a visal checking
     * Mat2RGBUI8 RGB = pop::Visualization::labelForeground(seed1,img);
     * RGB.display();
     * //region growing on the topographic surface with seeds
     * Mat2UI8 regions = pop::Processing::regionGrowingAdamsBischofMeanOverStandardDeviation(seeds,img);
     * //test the agreement between visual segmentation and numerical one
     * RGB = pop::Visualization::labelForeground(regions,img);
     * RGB.display();
     * \endcode
     *
    */
    template<typename FunctionTopo,typename FunctionLabel >
    static FunctionLabel regionGrowingAdamsBischofMeanOverStandardDeviation(const FunctionLabel & seed,const FunctionTopo & topo, typename FunctionTopo::IteratorENeighborhood  itneigh )
    {
        CollectorExecutionInformationSingleton::getInstance()->startExecution("AdamsBischof",COLLECTOR_EXECUTION_NOINFO);
        CollectorExecutionInformationSingleton::getInstance()->info("Fast algorithm");
        FunctorMeanStandardDeviation<FunctionTopo> functopo(topo);
        Population<FunctionLabel,FunctorMeanStandardDeviation<FunctionTopo>, RestrictedSetWithoutALL,SQFIFONextSmallestLevel,Growth> pop(seed.getDomain(),functopo,itneigh);
        typename FunctionLabel::IteratorEDomain it(seed.getDomain());
        while(it.next()){
            if(seed(it.x())!=0){
                functopo.addPoint(seed(it.x()),it.x());
                pop.setRegion(seed(it.x()),it.x());
            }
        }
        it.init();
        while(it.next()){
            if(seed(it.x())!=0){
                pop.growth(seed(it.x()),it.x());
            }
        }
        while(pop.next()){
            functopo.addPoint(pop.x().first,pop.x().second);
            pop.growth(pop.x().first,pop.x().second);

        }
        CollectorExecutionInformationSingleton::getInstance()->endExecution("AdamsBischof");
        return pop.getRegion();
    }

/*! \fn FunctionLabel regionGrowingAdamsBischofMean(const FunctionLabel & seed, const FunctionTopo & topo, typename FunctionTopo::IteratorENeighborhood  itneigh )

     * \param seed input seeds
     * \param topo topographic surface
     * \param itneigh neighborhood IteratorE domain
      * \return  regions
     *
     *  Classical region growing algorithm  of Adams and Bischof such that the ordering attribute function is:\n
     *  \f$ \delta(x,i) = |f(x)- \mu_i|\f$\n
     *  with f the topograhic surface, X_i the region and \f$\mu_i\f$ the mean value inside the seed
     *
     * \code
     * Mat2UI8 img;
     * img.load("../image/iex.png");
     * //filtering
     * img = pop::Processing::median(img,3);
     * //seed localisation
     * Mat2UI8 seed1 = pop::Processing::threshold(img,0,100);//seed in the grains
     * Mat2UI8 seed2 = pop::Processing::threshold(img,160);//seed in the background
     * Mat2UI8 seeds = pop::Processing::labelMerge(seed1,seed2);//merge of the seeds
     * //for a good segmentation, each seed should be include in its associated object (here the grains and the background) and touch each component of its associated object
     * //Test of the condition with a visal checking
     * Mat2RGBUI8 RGB = pop::Visualization::labelForeground(seed1,img);
     * RGB.display();
     * //region growing on the topographic surface with seeds
     * Mat2UI8 regions = pop::Processing::regionGrowingAdamsBischofMean(seeds,img);
     * //test the agreement between visual segmentation and numerical one
     * RGB = pop::Visualization::labelForeground(regions,img);
     * RGB.display();
     * \endcode
    */
template< typename FunctionTopo,typename FunctionLabel>
static FunctionLabel regionGrowingAdamsBischofMean(const FunctionLabel & seed, const FunctionTopo & topo, typename FunctionTopo::IteratorENeighborhood  itneigh )
{
    CollectorExecutionInformationSingleton::getInstance()->startExecution("AdamsBischofMean",COLLECTOR_EXECUTION_NOINFO);
    CollectorExecutionInformationSingleton::getInstance()->info("Fast algorithm");
    FunctorMean<FunctionTopo> functopo(topo);
    Population<FunctionLabel,FunctorMean<FunctionTopo>, RestrictedSetWithoutALL,SQFIFONextSmallestLevel,Growth> pop(seed.getDomain(),functopo,itneigh);
    typename FunctionLabel::IteratorEDomain it(seed.getDomain());
    while(it.next()){
        if(seed(it.x())!=0){
            functopo.addPoint(seed(it.x()),it.x());
            pop.setRegion(seed(it.x()),it.x());
        }
    }
    it.init();
    while(it.next()){
        if(seed(it.x())!=0){
            pop.growth(seed(it.x()),it.x());
        }
    }
    while(pop.next()){
        functopo.addPoint(pop.x().first,pop.x().second);
        pop.growth(pop.x().first,pop.x().second);
    }
    CollectorExecutionInformationSingleton::getInstance()->endExecution("AdamsBischofMean");
    return pop.getRegion();
}
template<typename FunctionTopo,typename FunctionLabel >
static FunctionLabel regionGrowingMergingLevel(const FunctionLabel & seed,const FunctionTopo & topo,int diff, typename FunctionTopo::IteratorENeighborhood  it_local )
{
    FunctorMeanMerge<FunctionTopo> functopo(topo);
    Population<FunctionLabel,FunctorMeanMerge<FunctionTopo>, RestrictedSetWithMySelf,SQFIFONextSmallestLevel,Growth> pop(seed.getDomain(),functopo,it_local);
    typename FunctionLabel::IteratorEDomain it(seed.getDomain());
    while(it.next()){
        if(seed(it.x())!=0){
            functopo.addPoint(seed(it.x()),it.x());
            pop.setRegion(seed(it.x()),it.x());
        }
    }
    int maxi=0;
    it.init();
    while(it.next()){
        if(seed(it.x())!=0){
            pop.growth(seed(it.x()),it.x());
            maxi =std::max(maxi,(int)seed(it.x()));
        }
    }
    std::vector<MasterSlave> v_individu(maxi+1);
    for(unsigned int i=0;i<v_individu.size();i++){
        v_individu[i]._my_label=i;
    }

    while(pop.next()){
        int label1=v_individu[pop.x().first].getLabelMaster();
        if(pop.getRegion()(pop.x().second)==RestrictedSetWithMySelf<typename FunctionLabel::F>::NoRegion){
            functopo.addPoint(label1,pop.x().second);
            pop.growth(label1,pop.x().second);
        }else{
            int label2=v_individu[pop.getRegion()(pop.x().second)].getLabelMaster();
            if(label1!=label2){
                if(functopo.diff(label1,label2 )<diff){
                    if(v_individu[label1]._my_slaves.size()<v_individu[label2]._my_slaves.size())
                        std::swap(label1,label2);
                    v_individu[label1].addSlave(v_individu[label2]);
                    functopo.merge(label1,label2);
                }
            }
            pop.pop();

        }
    }
    it.init();
    while(it.next()){
        if(pop.getRegion()(it.x())< v_individu.size()){
            int label1=v_individu[pop.getRegion()(it.x())].getLabelMaster();
            pop.setRegion(label1,it.x());
        }
    }
    CollectorExecutionInformationSingleton::getInstance()->endExecution("regionGrowing");
    return pop.getRegion();
}


/*!

     * \param cluster input binary matrix
     * \param itneigh neighborhood IteratorE
     * \param itneigh domain IteratorE
     * \return  label output label matrix
     *
     *  Each cluster of the input binary matrix has a specific label in the output label matrix
     *  Mat2UI8 img;
     *  img.load("../image/outil.bmp");
     *  img.display();
     *  Mat2UI32 label = ProcessingAdvanced::clusterToLabel(img,img.getIteratorENeighborhood(),img.getIteratorEDomain());
     *  Mat2RGBUI8 RGB = pop::Visualization::label2RGB(label);
     *  RGB.display();
    */
template<typename FunctionBinary,typename IteratorE>
static typename FunctionTypeTraitsSubstituteF<FunctionBinary,UI32>::Result clusterToLabel(const FunctionBinary & cluster, typename FunctionBinary::IteratorENeighborhood  itneigh,IteratorE it)
{
    CollectorExecutionInformationSingleton::getInstance()->startExecution("clusterToLabel",COLLECTOR_EXECUTION_NOINFO);
    CollectorExecutionInformationSingleton::getInstance()->info("Fast algorithm");
    typedef typename FunctionTypeTraitsSubstituteF<FunctionBinary,UI32>::Result FunctionLabel;
    FunctorZero f;
    Population<FunctionLabel,FunctorZero> pop(cluster.getDomain(),f,itneigh);
    it.init();
    while(it.next()){
        if(cluster(it.x())==0)
            pop.setRegion(0,it.x());
    }
    typename FunctionLabel::F i=0;
    pop.setLevel(0);

    it.init();
    while(it.next()){
        if(pop.getRegion()(it.x())==pop.getLabelNoRegion()){
            i++;

            pop.growth(i,it.x());
            while(pop.next()){
                pop.growth(i,pop.x().second);
            }
        }
    }
    CollectorExecutionInformationSingleton::getInstance()->endExecution("clusterToLabel");
    return pop.getRegion();
}
/*! \fn FunctionBinary clusterMax(const FunctionBinary & bin, typename FunctionBinary::IteratorENeighborhood  itneigh)
     * \param bin input binary matrix
     * \param itneigh neighborhood IteratorE domain
      *\return  max cluster
     *
     *  The ouput matrix is the max cluster of the input binary matrix
    */
template<typename FunctionBinary>
static FunctionBinary clusterMax(const FunctionBinary & bin, typename FunctionBinary::IteratorENeighborhood  itneigh)
{
    CollectorExecutionInformationSingleton::getInstance()->startExecution("ClusterMax",COLLECTOR_EXECUTION_NOINFO);
    CollectorExecutionInformationSingleton::getInstance()->info("Fast algorithm");
    typename FunctionTypeTraitsSubstituteF<FunctionBinary,UI32 >::Result label(bin.getDomain());
    label = ProcessingAdvanced::clusterToLabel(bin,itneigh,bin.getIteratorEDomain());
    typename FunctionBinary::IteratorEDomain it(bin.getIteratorEDomain());
    std::vector<UI32> occurence;
    while(it.next())
    {

        if(label(it.x())>=(UI32)occurence.size())occurence.resize(label(it.x())+1,0);
        if(label(it.x())!=0)
            occurence[label(it.x())]++;
    }

    UI32 maxoccurence=0;
    UI32 maxlabel=0;
    for(I32 i=1;i<(I32)occurence.size();i++)
    {
        if(maxoccurence<occurence[i])
        {
            maxlabel=i;
            maxoccurence=occurence[i];
        }
    }
    FunctionBinary clustermax(bin.getDomain());
    it.init();
    while(it.next())
    {
        if(label(it.x())==maxlabel)
            clustermax(it.x())=NumericLimits<UI8>::maximumRange();
        else
            clustermax(it.x())=0;
    }
    CollectorExecutionInformationSingleton::getInstance()->endExecution("ClusterMax");
    return clustermax;
}

/*! \fn void FunctionProcedureMinimaRegional(const FunctionTopo & topo, typename FunctionTopo::IteratorENeighborhood  itneigh , FunctionLabel & minima)
     * \param topo input topographic surface
     * \param itneigh neighborhood IteratorE domain
      *\param  minima labelled minima
     *
     *  The labelled minima is the minima of the input binary matrix such that each minumum has a specific label
    */

template< typename FunctionTopo>
static typename FunctionTypeTraitsSubstituteF<FunctionTopo,UI32>::Result minimaRegional(const FunctionTopo & topo, typename FunctionTopo::IteratorENeighborhood  itneigh)
{
    CollectorExecutionInformationSingleton::getInstance()->startExecution("MinimaRegional",COLLECTOR_EXECUTION_NOINFO);
    CollectorExecutionInformationSingleton::getInstance()->info("Fast algorithm");

    FunctorZero functortopo;
    typedef typename FunctionTypeTraitsSubstituteF<FunctionTopo,UI32>::Result FunctionLabel;
    Population<FunctionLabel,FunctorZero, RestrictedSetWithMySelf> pop(topo.getDomain(),functortopo,itneigh);
    typename FunctionLabel::F labelminima=0;
    std::vector<bool> globalminima(1,true);//the label 0 is for the blank region
    pop.setLevel(0);
    typename FunctionTopo::IteratorEDomain it(topo.getIteratorEDomain());
    while(it.next()){
        if(pop.getRegion()(it.x())==pop.getLabelNoRegion()){
            labelminima++;
            pop.growth(labelminima,it.x());
            globalminima.push_back(true);
            typename FunctionTopo::F value = topo(it.x());
            while(pop.next()){
                typename FunctionTopo::F temp = topo(pop.x().second);
                if(temp==value)
                    pop.growth(pop.x().first,pop.x().second);
                else
                {
                    pop.pop();
                    if(temp<value)
                        *(globalminima.rbegin())=false;
                }
            }
        }
    }
    it.init();
    while(it.next()){
        if(globalminima[pop.getRegion()(it.x())]==false  )
            pop.setRegion(0,it.x());//set blank region
    }

    CollectorExecutionInformationSingleton::getInstance()->endExecution("MinimaRegional");
    it.init();
    return ProcessingAdvanced::greylevelRemoveEmptyValue(pop.getRegion(),  it);
}

/*! \fn FunctionLabel watershed(const FunctionLabel & seed, const FunctionTopo & topo, typename FunctionTopo::IteratorENeighborhood itneigh )
      * \param seed input seed
     * \param topo input topographic surface
     * \param itneigh neighborhood IteratorE domain
      *\return  basins of the watershed transformation
     *
     *  Watershed transformation on the topographic surface initialiased by the seeds withoutboundary
    */

template< typename FunctionLabel,typename FunctionTopo>
static FunctionLabel watershed(const FunctionLabel & seed, const FunctionTopo & topo, typename FunctionTopo::IteratorENeighborhood itneigh )
{
    CollectorExecutionInformationSingleton::getInstance()->startExecution("Watershed",COLLECTOR_EXECUTION_NOINFO);
    CollectorExecutionInformationSingleton::getInstance()->info("Fast algorithm");
    FunctorTopography<FunctionTopo  > functortopo(topo);
    Population<FunctionLabel,FunctorTopography<FunctionTopo>, RestrictedSetWithoutALL,SQFIFO,Growth> pop(topo.getDomain(),functortopo,itneigh);
    typename FunctionTopo::IteratorEDomain it(topo.getDomain());
    while(it.next()){
        if(seed(it.x())!=0){
            pop.setRegion(seed(it.x()),it.x());
        }
    }
    it.init();
    while(it.next()){
        if(seed(it.x())!=0){
            pop.growth(seed(it.x()),it.x());
        }
    }
    for(I32 i=0;i<functortopo.nbrLevel();i++)
    {
        pop.setLevel(i);
        functortopo.setLevel(i);
        while(pop.next())
        {
            pop.growth(pop.x().first,pop.x().second);
        }
    }
    CollectorExecutionInformationSingleton::getInstance()->endExecution("Watershed");
    return pop.getRegion();
}
/*! \fn FunctionLabel watershed(const FunctionLabel & seed, const FunctionTopo & topo, const FunctionMask & mask , typename FunctionTopo::IteratorENeighborhood itneigh )

      * \param seed input seed matrix
     * \param topo input topographic surface
     * \param mask mask restricted the region growing
     * \param itneigh neighborhood IteratorE domain
      *\return  basins of the watershed transformation
     *
     *  Watershed transformation on the topographic surface initialiased by the seeds restricted by the mask
    */


template<
        typename FunctionLabel,
        typename FunctionTopo,
        typename FunctionMask
        >
static FunctionLabel watershed(const FunctionLabel & seed, const FunctionTopo & topo, const FunctionMask & mask , typename FunctionTopo::IteratorENeighborhood itneigh )
{
    CollectorExecutionInformationSingleton::getInstance()->startExecution("Watershed",COLLECTOR_EXECUTION_NOINFO);
    CollectorExecutionInformationSingleton::getInstance()->info("Fast algorithm");
    FunctorTopography<FunctionTopo  > functortopo(topo);
    Population<FunctionLabel,FunctorTopography<FunctionTopo>, RestrictedSetWithoutALL,SQFIFO,Growth> pop(topo.getDomain(),functortopo,itneigh);
    typename FunctionTopo::IteratorEDomain it(topo.getDomain());
    while(it.next()){
        if(mask(it.x())==0){
            pop.setRegion(0,it.x());
        }
        else if(seed(it.x())!=0){
            pop.setRegion(seed(it.x()),it.x());
        }
    }
    it.init();
    while(it.next()){
        if(seed(it.x())!=0&&mask(it.x())!=0){
            pop.growth(seed(it.x()),it.x());
        }
    }
    for(I32 i=0;i<functortopo.nbrLevel();i++)
    {
        pop.setLevel(i);
        functortopo.setLevel(i);
        while(pop.next())
        {
            pop.growth(pop.x().first,pop.x().second);
        }
    }
    it.init();
    while(it.next()){
        if(pop.getRegion()(it.x())==pop.getLabelNoRegion()){
            pop.setRegion(0,it.x());
        }
    }
    CollectorExecutionInformationSingleton::getInstance()->endExecution("Watershed");
    return pop.getRegion();
}
/*! \fn FunctionLabel watershedBoundary(const FunctionLabel & seed, const FunctionTopo & topo, typename FunctionTopo::IteratorENeighborhood itneigh )
      * \param seed input seed
     * \param topo input topographic surface
     * \param itneigh neighborhood IteratorE domain
      *\return  basins of the watershed transformation
     *
     *  Watershed transformation on the topographic surface initialiased by the seeds with a boundary region to separate the basins
    */

template<
        typename FunctionTopo,
        typename FunctionLabel
        >
static FunctionLabel watershedBoundary(const FunctionLabel & seed, const FunctionTopo & topo, typename FunctionTopo::IteratorENeighborhood itneigh )
{
    CollectorExecutionInformationSingleton::getInstance()->startExecution("WatershedBoundary",COLLECTOR_EXECUTION_NOINFO);
    CollectorExecutionInformationSingleton::getInstance()->info("Fast algorithm");
    FunctorTopography<FunctionTopo  > functortopo(topo);
    Population<FunctionLabel,FunctorTopography<FunctionTopo>, RestrictedSetWithoutSuperiorLabel> pop(topo.getDomain(),functortopo,itneigh);

    typename FunctionTopo::IteratorEDomain it(topo.getDomain());
    while(it.next()){
        if(seed(it.x())!=0){
            pop.setRegion(seed(it.x()),it.x());
        }
    }
    it.init();
    while(it.next()){
        if(seed(it.x())!=0){
            pop.growth(seed(it.x()),it.x());
        }
    }

    for(I32 i=0;i<functortopo.nbrLevel();i++)
    {
        pop.setLevel(i);
        functortopo.setLevel(i);
        while(pop.next())
        {
            if(pop.getRegion()(pop.x().second)==pop.getLabelNoRegion())
            {
                pop.growth(pop.x().first,pop.x().second);
            }
            else{
                pop.setRegion(0,pop.x().second);
            }
        }
    }
    CollectorExecutionInformationSingleton::getInstance()->endExecution("WatershedBoundary");
    return pop.getRegion();
}

/*! \fn FunctionLabel watershedBoundary(const FunctionLabel & seed,const FunctionTopo & topo,const FunctionMask & mask, typename FunctionTopo::IteratorENeighborhood itneigh )
      * \param seed input seed matrix
     * \param topo input topographic surface
     * \param mask mask restricted the region growing
     * \param itneigh neighborhood IteratorE domain
      *\return  basins of the watershed transformation
     *
     *  Watershed transformation on the topographic surface initialiased by the seeds restricted by the mask with a boundary region to separate the basins
    */

template<
        typename FunctionTopo,
        typename FunctionLabel,
        typename FunctionMask
        >
static FunctionLabel watershedBoundary(const FunctionLabel & seed,const FunctionTopo & topo,const FunctionMask & mask, typename FunctionTopo::IteratorENeighborhood itneigh )
{
    CollectorExecutionInformationSingleton::getInstance()->startExecution("WatershedBoundary",COLLECTOR_EXECUTION_NOINFO);
    CollectorExecutionInformationSingleton::getInstance()->info("Fast algorithm");
    FunctorTopography<FunctionTopo  > functortopo(topo);
    Population<FunctionLabel,FunctorTopography<FunctionTopo>, RestrictedSetWithoutSuperiorLabel> pop(topo.getDomain(),functortopo,itneigh);
    typename FunctionTopo::IteratorEDomain it(topo.getIteratorEDomain());
    while(it.next()){
        if(mask(it.x())==0){
            pop.setRegion(0,it.x());
        }
        else if(seed(it.x())!=0){
            pop.setRegion(seed(it.x())+1,it.x());
        }
    }
    it.init();
    while(it.next()){
        if(seed(it.x())!=0&&mask(it.x())!=0){
            pop.growth(seed(it.x())+1,it.x());
        }
    }

    for(I32 i=0;i<functortopo.nbrLevel();i++)
    {
        pop.setLevel(i);
        functortopo.setLevel(i);
        while(pop.next())
        {
            if(pop.getRegion()(pop.x().second)==pop.getLabelNoRegion())
                pop.growth(pop.x().first,pop.x().second);
            else
                pop.setRegion(1,pop.x().second);
        }
    }
    CollectorExecutionInformationSingleton::getInstance()->endExecution("WatershedBoundary");
    return pop.getRegion();
}
/*! \fn Function1 geodesicReconstruction(const Function1 & f,const Function2 & g, typename Function1::IteratorENeighborhood  itneigh)
      * \param f input matrix
     * \param g input matrix
     * \param itneigh neighborhood IteratorE domain
      *\return  the geodesic reconstruction
     *
     *  The geodesic reconstruction is the infinitely iterated geodesic erosion \f$E_g^\infty(f)\f$ such as \f$E_g^{t+1}(f)=\sup (E_g^{t}(f)\ominus N,g)\f$ with \f$E_g^{0}(f)=f\f$
     * \sa dynamic
    */

template<typename Function1,typename Function2>
static Function1 geodesicReconstruction(const Function1 & f,const Function2 & g, typename Function1::IteratorENeighborhood  itneigh)
{
    CollectorExecutionInformationSingleton::getInstance()->startExecution("GeodesicReconstruction",COLLECTOR_EXECUTION_NOINFO);
    CollectorExecutionInformationSingleton::getInstance()->info("Fast algorithm");
    FunctorZero zero;
    Population<Function1,FunctorZero,RestrictedSetWithoutSuperiorLabel,SQFIFO> pop(f.getDomain(),zero,itneigh) ;
    typename Function1::IteratorEDomain it(f.getIteratorEDomain());

    while(it.next()){

        pop.setRegion(f(it.x()),it.x());
    }
    it.init();
    while(it.next()){
        pop.growth(f(it.x()),it.x());

    }
    pop.setLevel(0);
    while(pop.next())
    {
        if( g(pop.x().second)<pop.x().first)
            pop.growth(pop.x().first,pop.x().second);
        else
            pop.pop();
    }
    CollectorExecutionInformationSingleton::getInstance()->endExecution("GeodesicReconstruction");
    return pop.getRegion();
}
template<typename Function1,typename Function2>
static Function1 dynamicNoRegionGrowing(const Function1 & f,const Function2 & g, typename Function1::IteratorENeighborhood  itneigh)
{
    Function1 fi;
    Function1 fiplusun(f);
    do{
        fi = fiplusun;
        typename Function1::IteratorEDomain it(fi.getIteratorEDomain());
        fiplusun =maximum(erosion(fi,it,itneigh),g);
    }while(!(fi==fiplusun));
    return fiplusun;
}

/*! \fn static Function1 dynamic(const Function1 & f, typename Function1::F value, typename Function1::IteratorENeighborhood  itneigh)
      * \param f input matrix
     * \param value dynamic value
      * \param itneigh neighborhood IteratorE domain
      *\return  the geodesic reconstruction
     *
     *  Geodesic reconstruction with f = f+value and g=f
     * \sa FunctionProcedureGeodesicReconstruction
    */

template<typename Function1>
static Function1 dynamic(const Function1 & f, typename Function1::F value, typename Function1::IteratorENeighborhood  itneigh)
{
    Function1 h(f);
    h+=value;
    return geodesicReconstruction(h, f,itneigh);
}
/*! \fn std::pair<FunctionRegion,typename FunctionTypeTraitsSubstituteF<FunctionRegion,plabel>::Result > voronoiTesselation(const FunctionRegion & seed, typename FunctionRegion::IteratorENeighborhood  itneigh)

      * \param seed input seed
      * \param itneigh neighborhood IteratorE domain
      * \return the first element of the pair contain the voronoi tesselation and the second the mapping
      *
      *  Voronoi tesselation based on the seeds \f$ region_i(x) = \{y :  d(y ,s_i) \leq d(y , s_j), j\neq i\}\f$ (work only for 1-norm and \f$\infty-norm\f$)
    */
template<typename FunctionRegion>
static std::pair<FunctionRegion,typename FunctionTypeTraitsSubstituteF<FunctionRegion,UI16>::Result > voronoiTesselation(const FunctionRegion & seed, typename FunctionRegion::IteratorENeighborhood  itneigh)
{
    CollectorExecutionInformationSingleton::getInstance()->startExecution("VoronoiTesselation",COLLECTOR_EXECUTION_NOINFO);
    CollectorExecutionInformationSingleton::getInstance()->info("Fast algorithm");
    FunctorSwitch f;
    Population<FunctionRegion,FunctorSwitch> pop(seed.getDomain(),f,itneigh);
    typename FunctionTypeTraitsSubstituteF<FunctionRegion,UI16>::Result dist(seed.getDomain());
    typename FunctionRegion::IteratorEDomain it(seed.getIteratorEDomain());

    while(it.next()){
        if(seed(it.x())!=0){
            dist(it.x())=0;
            pop.setRegion(seed(it.x()),it.x());
        }
    }
    it.init();
    while(it.next()){
        if(seed(it.x())!=0){
            pop.growth(seed(it.x()),it.x());
        }
    }
    UI16 distancevalue=0;
    bool atleastonegrowth=false;
    do
    {
        pop.setLevel(f.getFlipFlop());
        f.switchFlipFlop();
        distancevalue++;
        atleastonegrowth=false;
        while(pop.next()){
            atleastonegrowth=true;
            pop.growth(pop.x().first,pop.x().second);
            dist(pop.x().second)=distancevalue;
        }
    }while(atleastonegrowth==true);

    CollectorExecutionInformationSingleton::getInstance()->endExecution("VoronoiTesselation");
    return std::make_pair(pop.getRegion(),dist);
}
template<typename FunctionRegion,typename FunctionMask>
static FunctionRegion voronoiTesselationWithoutDistanceFunction(const FunctionRegion & seed, const FunctionMask & mask, typename FunctionRegion::IteratorENeighborhood  itneigh)
{
    CollectorExecutionInformationSingleton::getInstance()->startExecution("VoronoiTesselation",COLLECTOR_EXECUTION_NOINFO);
    CollectorExecutionInformationSingleton::getInstance()->info("Fast algorithm");
    FunctorZero f;
    Population<FunctionRegion,FunctorZero> pop(seed.getDomain(),f,itneigh);
    typename FunctionRegion::IteratorEDomain it(seed.getIteratorEDomain());
    while(it.next()){
        if(mask(it.x())==0){
            pop.setRegion(0,it.x());
        }
        else   if(seed(it.x())!=0){
            pop.setRegion(seed(it.x()),it.x());

        }
    }
    it.init();
    while(it.next()){
        if(mask(it.x())!=0 && seed(it.x())!=0){
            pop.growth(seed(it.x()),it.x());
        }
    }
    while(pop.next()){
        pop.growth(pop.x().first,pop.x().second);
    }
    it.init();
    while(it.next()){
        if(pop.getRegion()(it.x())==pop.getLabelNoRegion())
            pop.getRegion()(it.x())=0;
    }
    CollectorExecutionInformationSingleton::getInstance()->endExecution("VoronoiTesselation");
    return pop.getRegion();
}
template<typename FunctionRegion>
static FunctionRegion voronoiTesselationWithoutDistanceFunction(const FunctionRegion & seed,  typename FunctionRegion::IteratorENeighborhood  itneigh)
{
    CollectorExecutionInformationSingleton::getInstance()->startExecution("VoronoiTesselation",COLLECTOR_EXECUTION_NOINFO);
    CollectorExecutionInformationSingleton::getInstance()->info("Fast algorithm");
    FunctorZero f;
    Population<FunctionRegion,FunctorZero> pop(seed.getDomain(),f,itneigh);
    typename FunctionRegion::IteratorEDomain it(seed.getIteratorEDomain());

    while(it.next()){
        if(seed(it.x())!=0)
            pop.setRegion(seed(it.x()),it.x());
    }
    it.init();
    while(it.next()){

        if(seed(it.x())!=0)
            pop.growth(seed(it.x()),it.x());
    }
    while(pop.next()){
        pop.growth(pop.x().first,pop.x().second);
    }


    CollectorExecutionInformationSingleton::getInstance()->endExecution("VoronoiTesselation");
    return pop.getRegion();
}
/*! \fn     std::pair<FunctionRegion,typename FunctionTypeTraitsSubstituteF<FunctionRegion,plabel>::Result > voronoiTesselation(const FunctionRegion & seed, const FunctionMask & mask, typename FunctionRegion::IteratorENeighborhood  itneigh)
      * \param seed input seed
      * \param mask mask restricted the region growing
      * \param itneigh neighborhood IteratorE domain
      * \return the first element of the pair contain the voronoi tesselation and the second the mapping
      *
      *  Voronoi tesselation based on the seeds\f$ region_i(x) = \{y :  d(y ,s_i) \leq d(y , s_j), j\neq i\}\f$ such that the distunce function
      * is restricted by the mask (work only for 1-norm and \f$\infty-norm\f$)
    */
template<typename FunctionRegion,typename FunctionMask>
static std::pair<FunctionRegion,typename FunctionTypeTraitsSubstituteF<FunctionRegion,UI16>::Result > voronoiTesselation(const FunctionRegion & seed, const FunctionMask & mask, typename FunctionRegion::IteratorENeighborhood  itneigh)
{
    CollectorExecutionInformationSingleton::getInstance()->startExecution("VoronoiTesselation",COLLECTOR_EXECUTION_NOINFO);
    CollectorExecutionInformationSingleton::getInstance()->info("Fast algorithm");
    FunctorSwitch f;
    Population<FunctionRegion,FunctorSwitch> pop(seed.getDomain(),f,itneigh);
    typename FunctionTypeTraitsSubstituteF<FunctionRegion,UI16>::Result dist(seed.getDomain());
    typename FunctionRegion::IteratorEDomain it(seed.getIteratorEDomain());

    while(it.next()){
        if(mask(it.x())==0){
            pop.setRegion(0,it.x());
        }
        else
        {
            if(seed(it.x())!=0){
                dist(it.x())=0;
                pop.setRegion(seed(it.x()),it.x());
            }
        }
    }

    it.init();
    while(it.next()){
        if(mask(it.x())!=0 && seed(it.x())!=0){
            pop.growth(seed(it.x()),it.x());
        }
    }
    int distancevalue=0;
    bool atleastonegrowth=false;
    do
    {
        pop.setLevel(f.getFlipFlop());
        f.switchFlipFlop();
        distancevalue++;
        atleastonegrowth=false;
        while(pop.next()){
            atleastonegrowth=true;
            pop.growth(pop.x().first,pop.x().second);
            dist(pop.x().second)=distancevalue;
        }
    }while(atleastonegrowth==true);
    CollectorExecutionInformationSingleton::getInstance()->endExecution("VoronoiTesselation");
    return std::make_pair(pop.getRegion(),dist);
}
/*! \fn std::pair<FunctionRegion,typename FunctionTypeTraitsSubstituteF<FunctionRegion,F64>::Result >  voronoiTesselationEuclidean(const FunctionRegion & seed)
      * \param seed input seed
      * \param region ouput region
      * \param dist distunce function
      * \return the first element of the pair contain the voronoi tesselation and the second the mapping
      *
      *  Quasi-voronoi tesselation based on the seeds \f$ region_i(x) = \{y :  d(y ,s_i) \leq d(y , s_j), j\neq i\}\f$ calculated with the euclidean norm
    */

template<typename FunctionRegion>
static std::pair<FunctionRegion,typename FunctionTypeTraitsSubstituteF<FunctionRegion,F64>::Result >  voronoiTesselationEuclidean(const FunctionRegion & seed)
{
    CollectorExecutionInformationSingleton::getInstance()->startExecution("VoronoiTesselationEuclidean",COLLECTOR_EXECUTION_INFO);
    typename FunctionRegion::IteratorENeighborhood itn(seed.getIteratorENeighborhood(1,0));
    FunctorZero f;
    typedef typename FunctionTypeTraitsSubstituteF<FunctionRegion ,UI32 >::Result FunctionLabel;
    Population<FunctionLabel,FunctorZero,RestrictedSetWithMySelf> pop(seed.getDomain(),f,seed.getIteratorENeighborhood(1,0));


    FunctionRegion region(seed.getDomain());
    typedef typename FunctionTypeTraitsSubstituteF<FunctionRegion,F64>::Result FunctionDistance;
    FunctionDistance dist(seed.getDomain());
    std::vector<typename FunctionRegion::E > vrand;
    typename FunctionRegion::IteratorEDomain it(seed.getIteratorEDomain());

    FunctionRegion  seedinner(seed.getDomain());
    seedinner = ProcessingAdvanced::erosion(seed,it,itn);
    seedinner = seed-seedinner;

    it.init();
    while(it.next()){
        if(seed(it.x())==0){
            dist(it.x())=NumericLimits<typename FunctionDistance::F>::maximumRange();
        }else{
            dist(it.x())=0;
        }
        if(seedinner(it.x())!=0){
            vrand.push_back(it.x());
        }
        region(it.x())=seed(it.x());
    }
    /* initialize random seed: */
    srand ( time(NULL) );
    pop.setLevel(0);
    int index =0;
    int display_step=1;
    while(vrand.empty()==false)
    {
        if(index>=display_step-1){
            display_step*=2;
            CollectorExecutionInformationSingleton::getInstance()->info("Random seed number="+BasicUtility::Any2String(index));
        }
        I32 i = rand()%((I32)vrand.size());
        typename FunctionRegion::E x = vrand[i];
        vrand[i]= *(vrand.rbegin());
        vrand.pop_back();
        int label = seed(x);
        pop.growth(index,x);
        region(it.x())=label;
        index++;
        while(pop.next())
        {
            typename FunctionRegion::E diff = pop.x().second-x;
            F64 disttemp = diff.normPower();
            if(disttemp<= dist(pop.x().second))
            {
                pop.growth(pop.x().first,pop.x().second);
                region(pop.x().second)=label;
                dist(pop.x().second)=disttemp;
            }
            else
                pop.pop();
        }
    }
    it.init();
    while(it.next()){
        if(dist(it.x())==NumericLimits<typename FunctionDistance::F>::maximumRange())
            dist(it.x()) =0;
        else
            dist(it.x()) =std::sqrt(static_cast<double>(dist(it.x())));
    }
    CollectorExecutionInformationSingleton::getInstance()->endExecution("VoronoiTesselationEuclidean");
    return std::make_pair(region,dist);
}

/*! \fn static Function  erosionRegionGrowing(const Function & f,F64 radius, int norm=1)
          * \param bin input binary matrix
          * \param radius radius
          * \param norm norm
          * \return erosion ouput matrix
          *
          *  erosion(x) =  \min_{x'\in B(x,r,n)}f(x') \f$, where \f$B(x,norm)=\{x':|x'-x|_n\leq r\}\f$ the ball centered in 0 of radius r and the norm n
        */
template<typename Function>
static Function  erosionRegionGrowing(const Function & f,F64 radius, int norm=1)
{
    return erosionRegionGrowing(f,radius,  norm, Loki::Int2Type<isVectoriel<typename Function::F>::value > ());
}
template<typename Function>
static Function  erosionRegionGrowing(const Function & f,F64 radius, int norm,Loki::Int2Type<true>)
{
    typedef typename Identity<typename Function::F>::Result::F TypeScalar;
    VecN<Function::F::DIM,typename FunctionTypeTraitsSubstituteF<Function,TypeScalar>::Result > V;
    Convertor::toVecN(f,V);
    for(int i=0;i<Function::DIM;i++){
        V(i) = erosionRegionGrowing(V(i),radius,norm);
    }
    Function exit;
    Convertor::fromVecN(V,exit);
    return exit;
}

template<typename Function>
static Function  erosionRegionGrowing(const Function & f,F64 radius, int norm,Loki::Int2Type<false>)
{
    if(norm<=1){
        CollectorExecutionInformationSingleton::getInstance()->startExecution("erosionRegionGrowing",COLLECTOR_EXECUTION_NOINFO);
        CollectorExecutionInformationSingleton::getInstance()->info("Fast algorithm");
        FunctorSwitch func;
        Population<Function,FunctorSwitch,RestrictedSetWithoutSuperiorLabel> pop(f.getDomain(),func,f.getIteratorENeighborhood(1,norm));

        typename Function::IteratorEDomain it(f.getIteratorEDomain());

        while(it.next()){
            pop.setRegion(f(it.x()),it.x());
        }
        it.init();
        while(it.next()){
            pop.growth(f(it.x()),it.x());
        }
        int distancevalue=0;
        bool atleastonegrowth=true;
        while(atleastonegrowth==true&&distancevalue<radius)
        {
            pop.setLevel(func.getFlipFlop());
            func.switchFlipFlop();
            distancevalue++;
            atleastonegrowth=false;
            while(pop.next()){
                atleastonegrowth=true;
                pop.growth(pop.x().first,pop.x().second);
            }
        }
        CollectorExecutionInformationSingleton::getInstance()->endExecution("erosionRegionGrowing");
        return pop.getRegion();
    }else{
        CollectorExecutionInformationSingleton::getInstance()->startExecution("erosionRegionGrowing",COLLECTOR_EXECUTION_NOINFO);
        CollectorExecutionInformationSingleton::getInstance()->info("Fast algorithm");
        FunctorSwitch func;
        PopulationInformation<Function,FunctorSwitch,typename Function::E,RestrictedSetWithoutSuperiorLabel> pop(f.getDomain(),func,f.getIteratorENeighborhood(1,0));

        typename Function::IteratorEDomain it(f.getIteratorEDomain());

        while(it.next()){
            pop.setRegion(f(it.x()),it.x());
        }
        it.init();
        while(it.next()){
            pop.growth(f(it.x()),it.x(),it.x());
        }
        double radiuspower2= radius*radius;
        int distancevalue=0;
        bool atleastonegrowth=true;
        while(atleastonegrowth==true&&distancevalue<radius)
        {
            pop.setLevel(func.getFlipFlop());
            func.switchFlipFlop();
            distancevalue++;
            atleastonegrowth=false;
            while(pop.next()){
                typename Function::E diff = pop.x().first.second-pop.x().second;
                F64 disttemp = diff.normPower();
                if(disttemp<= radiuspower2)
                {
                    atleastonegrowth=true;
                    pop.growth(pop.x().first.first,pop.x().first.second,pop.x().second);
                }
                else{
                    pop.pop();
                }
            }

        }
        CollectorExecutionInformationSingleton::getInstance()->endExecution("erosionRegionGrowing");
        return pop.getRegion();
    }
}
/*! \fn static Function  dilationRegionGrowing(const Function & f,F64 radius, int norm=1)
          * \param bin input binary matrix
          * \param radius radius
          * \param norm norm
          * \return dilation ouput matrix
          *
          *  dilation(x) =  \max_{x'\in B(x,r,n)}f(x') \f$, where \f$B(x,norm)=\{x':|x'-x|_n\leq r\}\f$ the ball centered in 0 of radius r and the norm n
        */
template<typename Function>
static Function  dilationRegionGrowing(const Function & f,F64 radius, int norm=1)
{
    return dilationRegionGrowing(f,radius,  norm, Loki::Int2Type<isVectoriel<typename Function::F>::value > ());
}
template<typename Function>
static Function  dilationRegionGrowing(const Function & f,F64 radius, int norm,Loki::Int2Type<true>)
{
    typedef typename Identity<typename Function::F>::Result::F TypeScalar;
    VecN<Function::F::DIM,typename FunctionTypeTraitsSubstituteF<Function,TypeScalar>::Result > V;
    Convertor::toVecN(f,V);
    for(int i=0;i<Function::DIM;i++){
        V(i) = dilationRegionGrowing(V(i),radius,norm);
    }
    Function exit;
    Convertor::fromVecN(V,exit);
    return exit;
}

template<typename Function>
static Function  dilationRegionGrowing(const Function & f,F64 radius, int norm,Loki::Int2Type<false>)
{
    if(norm<=1){
        CollectorExecutionInformationSingleton::getInstance()->startExecution("dilationRegionGrowing",COLLECTOR_EXECUTION_NOINFO);
        CollectorExecutionInformationSingleton::getInstance()->info("Fast algorithm");
        FunctorSwitch func;
        Population<Function,FunctorSwitch,RestrictedSetWithoutInferiorLabel> pop(f.getDomain(),func,f.getIteratorENeighborhood(1,norm));

        typename Function::IteratorEDomain it(f.getIteratorEDomain());

        while(it.next()){
            pop.setRegion(f(it.x()),it.x());
        }
        it.init();
        while(it.next()){
            pop.growth(f(it.x()),it.x());
        }
        int distancevalue=0;
        bool atleastonegrowth=true;
        while(atleastonegrowth==true&&distancevalue<radius)
        {
            pop.setLevel(func.getFlipFlop());
            func.switchFlipFlop();
            distancevalue++;
            atleastonegrowth=false;
            while(pop.next()){
                atleastonegrowth=true;
                pop.growth(pop.x().first,pop.x().second);
            }
        }
        CollectorExecutionInformationSingleton::getInstance()->endExecution("dilationRegionGrowing");
        return pop.getRegion();
    }else{
        CollectorExecutionInformationSingleton::getInstance()->startExecution("dilationRegionGrowing",COLLECTOR_EXECUTION_NOINFO);
        CollectorExecutionInformationSingleton::getInstance()->info("Fast algorithm");
        FunctorSwitch func;
        PopulationInformation<Function,FunctorSwitch,typename Function::E,RestrictedSetWithoutInferiorLabel> pop(f.getDomain(),func,f.getIteratorENeighborhood(1,0));

        typename Function::IteratorEDomain it(f.getIteratorEDomain());

        while(it.next()){
            pop.setRegion(f(it.x()),it.x());
        }
        it.init();
        while(it.next()){
            pop.growth(f(it.x()),it.x(),it.x());
        }
        double radiuspower2= radius*radius;
        int distancevalue=0;
        bool atleastonegrowth=true;
        while(atleastonegrowth==true&&distancevalue<radius)
        {
            pop.setLevel(func.getFlipFlop());
            func.switchFlipFlop();
            distancevalue++;
            atleastonegrowth=false;
            while(pop.next()){
                typename Function::E diff = pop.x().first.second-pop.x().second;
                F64 disttemp = diff.normPower();
                if(disttemp<= radiuspower2)
                {
                    atleastonegrowth=true;
                    pop.growth(pop.x().first.first,pop.x().first.second,pop.x().second);
                }
                else{
                    pop.pop();
                }
            }
        }
        CollectorExecutionInformationSingleton::getInstance()->endExecution("dilationRegionGrowing");
        return pop.getRegion();

    }
}
template<typename Function>
static Function  erosionRegionGrowing(const Function & f,const typename Function::IteratorENeighborhood & itneigh,double radius)
{

    CollectorExecutionInformationSingleton::getInstance()->startExecution("erosionRegionGrowing",COLLECTOR_EXECUTION_NOINFO);
    CollectorExecutionInformationSingleton::getInstance()->info("Fast algorithm");
    FunctorSwitch func;
    Population<Function,FunctorSwitch,RestrictedSetWithoutSuperiorLabel> pop(f.getDomain(),func,itneigh);

    typename Function::IteratorEDomain it(f.getIteratorEDomain());

    while(it.next()){
        pop.setRegion(f(it.x()),it.x());
    }
    it.init();
    while(it.next()){
        pop.growth(f(it.x()),it.x());
    }
    int distancevalue=0;
    bool atleastonegrowth=true;
    while(atleastonegrowth==true&&distancevalue<radius)
    {
        pop.setLevel(func.getFlipFlop());
        func.switchFlipFlop();
        distancevalue++;
        atleastonegrowth=false;
        while(pop.next()){
            atleastonegrowth=true;
            pop.growth(pop.x().first,pop.x().second);
        }
    }
    CollectorExecutionInformationSingleton::getInstance()->endExecution("erosionRegionGrowing");
    return pop.getRegion();
}
template<typename Function>
static Function  dilationRegionGrowing(const Function & f,const typename Function::IteratorENeighborhood & itneigh,double radius)
{
    FunctorSwitch func;
    Population<Function,FunctorSwitch,RestrictedSetWithoutInferiorLabel> pop(f.getDomain(),func,itneigh);

    typename Function::IteratorEDomain it(f.getIteratorEDomain());

    while(it.next()){
        pop.setRegion(f(it.x()),it.x());
    }
    it.init();
    while(it.next()){
        pop.growth(f(it.x()),it.x());
    }
    int distancevalue=0;
    bool atleastonegrowth=true;
    while(atleastonegrowth==true&&distancevalue<radius)
    {
        pop.setLevel(func.getFlipFlop());
        func.switchFlipFlop();
        distancevalue++;
        atleastonegrowth=false;
        while(pop.next()){
            atleastonegrowth=true;
            pop.growth(pop.x().first,pop.x().second);
        }
    }
    CollectorExecutionInformationSingleton::getInstance()->endExecution("dilationRegionGrowing");
    return pop.getRegion();
}
template<typename Function>
static Function  closingRegionGrowing(const Function & f,F64 radius, int norm=1)
{
    Function temp(f.getDomain());
    temp = dilationRegionGrowing(f,radius,norm);
    return erosionRegionGrowing(temp,radius,norm);
}
template<typename Function>
static Function  openingRegionGrowing(const Function & f,F64 radius, int norm=1)
{
    Function temp(f.getDomain());
    temp = erosionRegionGrowing(f,radius,norm);
    return dilationRegionGrowing(temp,radius,norm);
}
};
}
#endif // PROCESSINGADVANCED_H
