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
#ifndef PDEADVANCED_H
#define PDEADVANCED_H
#include<algorithm>
#include"data/utility/CollectorExecutionInformation.h"
#include"data/functor/FunctorPDE.h"
#include"data/mat/MatN.h"
#include"data/mat/Mat2x.h"

namespace pop
{
struct PDEAdvanced
{

    /*! \brief Solve the phase field equation by iterating over the range of time \f$\frac{\partial_i\phi(x,t)}{\partial t}= F(\phi_i(x,t),1_{\max\phi)}(x,t) ,x)\f$
        \brief The optimisation comes from the evolution on a single phase field and a second field  to localize the different fields
        \param field is the input function phi at time t=0 and return the function phi  at time t=deltat*nbrstep
        \param multiphase localize the phase in the space
        \param nbrstep is the number of time iterations
        \param deltat is the time step
        \param it is iterator iterating through the space definition
        \param f is the binary functor taking for the first parameter a function  or for the second a VecN
        *
        *
    */
    template<typename FunctionScalar,typename FunctionLabel,typename Iterator,typename FunctorLaplacien>
    static void allenCahnInMutliPhaseField( FunctionScalar & field,FunctionLabel & labelfield,int nbrstep, Iterator & it,FunctorLaplacien& laplacien,double width=2 )
    {
        FunctionScalar fieldtimetdelatt(field);
        typename FunctionLabel::IteratorENeighborhood itn(labelfield.getIteratorENeighborhood());

        FunctorPDE::FreeEnergy free;
        double witdhpowerminus1=1/(width*width);
        DistributionUniformInt d(0,256);

        for(int i=0;i<nbrstep;i++)
        {
            it.init();
            while(it.next()){
                fieldtimetdelatt(it.x()) = field(it.x())+ 0.5/(2.*FunctionScalar::DIM)*(free(field(it.x()))*witdhpowerminus1+laplacien(field,it.x()));
                if(fieldtimetdelatt(it.x())<0){
                    fieldtimetdelatt(it.x())=-fieldtimetdelatt(it.x());
                    typename FunctionLabel::F label =  labelfield(it.x());
                    std::vector<typename FunctionLabel::F> v_label;
                    itn.init(it.x());
                    while(itn.next()){
                        typename FunctionLabel::F labelneight =  labelfield(itn.x());
                        if(label!=labelneight &&labelneight!=NumericLimits<typename FunctionLabel::F>::maximumRange()){
                            v_label.push_back(labelneight);
                        }
                    }
                    if(v_label.size()>0){
                        int index = d.randomVariable();
                        index = index%v_label.size();
                        labelfield(it.x())= v_label[index];
                    }

                }
            }
            field = fieldtimetdelatt;
        }
    }
    template<typename FunctionScalar,typename Iterator,typename FunctorLaplacien>
    static void allenCahnInSinglePhaseField( FunctionScalar & field,int nbrstep, Iterator & it,FunctorLaplacien& laplacien,double width=2 )
    {
        FunctionScalar fieldtimetdelatt(field);
        FunctorPDE::FreeEnergy free;
        double witdhpowerminus1=1/(width*width);

        for(int i=0;i<nbrstep;i++)
        {
            it.init();
            while(it.next()){
                fieldtimetdelatt(it.x()) = field(it.x())+ 0.5/(2.*FunctionScalar::DIM)*(free(field(it.x()))*witdhpowerminus1+laplacien(field,it.x()));
            }
            field = fieldtimetdelatt;
        }
    }

    //typical value errorpressuremax=0.001, relaxationSOR=0.9,   relaxationpressure=1


    /*! \brief Solve the phase field equation by iterating over the range of time \f$\frac{\partial_i\phi(x,t)}{\partial t}= F(\phi_i(x,t),1_{\max\phi)}(x,t) ,x)\f$
        \brief The optimisation comes from the evolution on a single phase field and a second field  to localize the different fields
        \param field is the input function phi at time t=0 and return the function phi  at time t=deltat*nbrstep
        \param multiphase localize the phase in the space
        \param nbrstep is the number of time iterations
        \param deltat is the time step
        \param it is iterator iterating through the space definition
        \param f is the binary functor taking for the first parameter a function  or for the second a VecN
        *
        *
    */

    template<int DIM>
    static void permeability(const  MatN<DIM,UI8> & pore,int direction, F64 errorpressuremax,F64 relaxationSOR, F64 relaxationpressure, MatN<DIM,VecN<DIM,F64> > & velocity, VecN<DIM,F64> & permeability )
    {
        F64 BOUNDARYVALUE = -10;
        velocity = createVelocityFieldMAKGrid(pore,direction,BOUNDARYVALUE);

        MatN<DIM,F64>  pressure(pore.getDomain());
        typename MatN<DIM,UI8>::IteratorEDomain ittotal (pressure.getIteratorEDomain());
        while(ittotal.next()){
            if(pore(ittotal.x())!=0)
                pressure(ittotal.x())=pressure.getDomain()(direction)-ittotal.x()(direction)-1;
            else
                pressure(ittotal.x())=-1;
        }

        typename MatN<DIM,UI8>::IteratorEDomain it (velocity.getIteratorEDomain());
        typename MatN<DIM,UI8>::IteratorEDomain itp (pressure.getIteratorEDomain());
        typedef  FunctorGaussSeidelStockes<MatN<DIM,F64>,MatN<DIM,VecN<DIM,F64> >  > FunctorGausss;
        FunctorGausss func(pressure,velocity,direction,velocity.getDomain()(direction)-1,BOUNDARYVALUE);

        FunctorGaussSeidelSOR<FunctorGausss > ff(relaxationSOR,func);
        FunctorRelaxationPressure<MatN<DIM,F64>,MatN<DIM,VecN<DIM,F64> > > div(pressure,velocity,BOUNDARYVALUE);
        FunctorEvolution<FunctorRelaxationPressure<MatN<DIM,F64>,MatN<DIM,VecN<DIM,F64> > > > func_evol(div);
        func_evol.relaxationpressure = relaxationpressure;
//        double errorpressurecurrent;
        int tour =0;
        do{
            forEachFunctorBinaryFunctionE(velocity, velocity,ff);
            func_evol.error=0;
            forEachFunctorBinaryFunctionE(pressure, pressure,func_evol);
            if(tour%10==0){
                std::string str = "At iteration "+BasicUtility::Any2String(tour)+ ", the ratio error_pressure_current/error_pressure_convergence is equal to "+BasicUtility::Any2String(func_evol.error/errorpressuremax);
                std::cout<<str<<std::endl;
            }
            tour++;
        }while(func_evol.error>errorpressuremax);


        it.init();
        int occurence =0;
        while(it.next())
        {
            occurence++;
            for(int i=0;i<DIM;i++)
            {
                if(velocity(it.x())(i)!=BOUNDARYVALUE){
                    permeability(i) += (velocity(it.x())(i));
                }
                else
                {
                    (velocity(it.x())(i))= 0;
                }
            }
        }
        for(int i=0;i<DIM;i++){
            permeability(i)/=occurence;
        }
    }

private:
    template<typename Functor>
    struct FunctorEvolution
    {
        Functor & div;
        FunctorEvolution(Functor & ddiv)
            :div(ddiv){

        }

        static double error;
        double relaxationpressure;
        template<int DIM,typename PixelType>
        PixelType operator()(const MatN<DIM,PixelType> & pressure, const VecN<DIM,int> & x){
            PixelType temp = pressure(x);
            PixelType next= pressure(x)-relaxationpressure*div(x);
            error = maximum(error, absolute(next-temp));
            return next;
        }
    };

    template<typename Function1,typename Function2>
    class FunctorGaussSeidelStockes
    {
    private:
        Function1 & _pressure;
        Function2 & _velocity;
        int _direction;
        F64 _pressureboundary;
        F64 _velocityboundary;
    public:
        FunctorGaussSeidelStockes(Function1 & pressure, Function2 & velocity,int direction,F64 pressureboundary, F64 velocityboundary)
            :_pressure(pressure),_velocity(velocity),_direction(direction),_pressureboundary(pressureboundary),_velocityboundary(velocityboundary)
        {}
        VecN<Function1::DIM, typename Function1::F> operator()( typename Function1::E x){

            VecN<Function1::DIM, typename Function1::F> coefficient(0);
            //cout<<"x: "<<x<<std::endl;
            for(int i=0;i<Function1::DIM;i++)
            {
                //cout<<"Direction: "<<i<<std::endl;
                if(_velocity(x)(i)!=_velocityboundary)
                {
                    for(int k=0;k<Function1::DIM;k++)
                    {
                        //cout<<"Laplacien direction: "<<k<<std::endl;
                        x(k)++;
                        //cout<<"x++:"<<x<<std::endl;
                        if(_velocity.isValid(x)==true)
                        {
                            if(_velocity(x)(i)!=_velocityboundary)
                            {
                                coefficient(i)+=_velocity(x)(i);
                            }
                        }
                        else
                        {
                            if(i==_direction && k ==i)
                            {
                                x(k)--;
                                coefficient(i)+=_velocity(x)(i);
                                x(k)++;
                            }
                        }
                        //cout<<"coefficient++: "<<coefficient<<std::endl;
                        //cout<<"x--:"<<x<<std::endl;
                        x(k)-=2;
                        if(_velocity.isValid(x)==true)
                        {
                            if(_velocity(x)(i)!=_velocityboundary)
                            {
                                coefficient(i)+=_velocity(x)(i);
                            }

                        }
                        else
                        {
                            if(i==_direction && k ==i)
                            {
                                x(k)++;
                                coefficient(i)+=_velocity(x)(i);
                                x(k)--;
                            }
                        }
                        //cout<<"coefficient--: "<<coefficient<<std::endl;
                        x(k)++;
                        //cout<<"coefficient==: "<<coefficient<<std::endl;
                    }
                    //coefficient(i)*=VISCOUS;
                    if(_pressure.isValid(x)==true)
                    {
                        coefficient(i)-=_pressure(x);
                    }
                    x(i)--;
                    if(_pressure.isValid(x)==true)
                    {
                        coefficient(i)+=_pressure(x);
                    }
                    else
                    {
                        if(x(i)==-1&&i==_direction)
                            coefficient(i)+= _pressureboundary;
                    }
                    x(i)++;

                    coefficient(i)/=(2*Function1::DIM);
                }
                else
                    coefficient(i) =   _velocityboundary;
            }

            return coefficient;
        }
    };

    template<typename Function1,typename Function2>
    class FunctorRelaxationPressure
    {
    private:
        Function1 & _pressure;
        Function2 & _velocity;
        F64 _velocityboundary;
    public:
        FunctorRelaxationPressure(Function1 & pressure, Function2 & velocity, F64 velocityboundary)
            :_pressure(pressure),_velocity(velocity),_velocityboundary(velocityboundary)
        {}
        typename Function1::F operator()( typename Function1::E  x){
            typename Function1::F divergence(0);
            //cout<<"x:"<<x<<std::endl;
            for(int i=0;i<Function1::DIM;i++)
            {
                //cout<<"direction:"<<i<<std::endl;
                if(_velocity(x)(i)!=_velocityboundary)
                {
                    divergence-=_velocity(x)(i);
                }
                //cout<<"value==:"<<divergence<<std::endl;
                x(i)++;
                if(_velocity(x)(i)!=_velocityboundary)
                {
                    divergence+=_velocity(x)(i);
                }
                //cout<<"value--:"<<divergence<<std::endl;
                x(i)--;
            }
            return divergence;
        }
    };

    template<typename FunctorGaussSeidelLocal>
    class FunctorGaussSeidelSOR
    {
    private:
        const F64 _ratio;
        FunctorGaussSeidelLocal & _func;
    public:
        FunctorGaussSeidelSOR(F64 ratio,FunctorGaussSeidelLocal & func)
            :_ratio(ratio),_func(func)
        {}
        template<typename Function>
        typename Function::F operator()(const Function & f,const  typename Function::E & x)
        {
            return (1-_ratio)*f(x)+_ratio*_func(x);
        }
    };


    template<typename Function>
    static typename FunctionTypeTraitsSubstituteF<Function, VecN<Function::DIM,F64> >::Result createVelocityFieldMAKGrid(Function porespace, int direction, F64 velocityboundary)
    {

        typename Function::E domain;
        domain= porespace.getDomain()+1;
        typename FunctionTypeTraitsSubstituteF<Function, VecN<Function::DIM,F64> >::Result velocity(domain);
        typename Function::IteratorEDomain it (velocity.getIteratorEDomain());
        while(it.next()){
            velocity(it.x())=velocityboundary;
            VecN<Function::DIM,int> x=it.x();
            for(int i = 0;i<Function::DIM;i++){
                x(i)--;
                if(porespace.isValid(it.x())==true&&porespace.isValid(x)==true)
                {
                    if(porespace(it.x())!=0&&porespace(x)!=0 ){
                        velocity(it.x())(i)=0;
                    }
                }else if(porespace.isValid(it.x())==true){
                    if(porespace(it.x())!=0&&i==direction){
                        velocity(it.x())(i)=0;
                    }
                }else if(porespace.isValid(x)==true){
                    if(porespace(x)!=0&&i==direction){
                        velocity(it.x())(i)=0;
                    }
                }
                x(i)++;

            }
        }
        return velocity;

    }
};
template<typename Functor>
double PDEAdvanced::FunctorEvolution<Functor>::error=0;
}
#endif // PDEADVANCED_H
