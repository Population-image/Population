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

#ifndef TOOLPDEREGIONGROWING_H
#define TOOLPDEREGIONGROWING_H


#include"data/population/PopulationData.h"
#include"data/population/PopulationFunctor.h"
namespace pop
{

template<typename E>
class POP_EXPORTS IteratorVecOneRegion
{
private:
    std::vector<E> _v;
    I32 _index;
public:
    void push_x(const E & x_value)
    {
        _v.push_back(x_value);
    }
    void pop_x()
    {
        _v[_index]=*_v.rbegin();
        _v.pop_back();
    }
    bool next()
    {
        _index--;
        if(_index>=0)return true;
        else return false;
    }
    E  & x()
    {
        //assert(_index<(I32)_v.size());
        return _v[_index];
    }
    void init()
    {
        _index = (I32)_v.size();
    }
};
template<typename PopulationSingleRegionZIGP,template<typename>class IteratorOneRegion=IteratorVecOneRegion >
class POP_EXPORTS PopulationOneRegionZIIteratotGP:public  PopulationSingleRegionZIGP
{
protected:
     IteratorOneRegion<typename PopulationSingleRegionZIGP::Function::E> _itregion;
public:
     PopulationOneRegionZIIteratotGP(const typename PopulationSingleRegionZIGP::Function::Domain & domain, typename PopulationSingleRegionZIGP::Functor & f,typename PopulationSingleRegionZIGP::Function::IteratorENeighborhood  itn)
        :PopulationSingleRegionZIGP(domain,  f,itn)
    {}

    void initRegion()
    {
        _itregion.init();
    }
    bool nextRegion()
    {
        while(_itregion.next())
        {
            if( this->_region(_itregion.x())==PopulationSingleRegionZIGP::RestrictedSet::SingleRegion)
                return true;
            else
                _itregion.pop_x();
        }
        return false;
    }
    typename PopulationSingleRegionZIGP::Function::E  & xRegion()
    {
        return _itregion.x();
    }
    virtual void growth( const  typename PopulationSingleRegionZIGP::Function::E & x  )
    {
        PopulationSingleRegionZIGP::growth(0,x);
        _itregion.push_x(x);
    }
};

template<typename Population,typename FunctionField>
class IteratorRegionZIPhaseField
{
private:
    Population & _pop;
    FunctionField & _field;
    typename FunctionField::F _threshold;
    bool regionzi;
    int _size;
public:
    IteratorRegionZIPhaseField(Population & pop,FunctionField  &field,typename FunctionField::F threshold)
        :_pop(pop),_field(field),_threshold(threshold){}


    void initField(FunctionField & field)
    {
        _field = field;
    }
    int size()
    {
       return _size;
    }
    void init()
    {

        _pop.initRegion();
        _pop.init();
        _size =0;
    }
    bool next()
    {
        _size++;
        if(_pop.nextRegion()==true)
        {
            regionzi =true;
            return true;
        }
        else
        {
            regionzi =false;
            return _pop.next();
        }
    }

    typename FunctionField::E & x()
    {
        if(regionzi==true){
            if(absolute(_field(_pop.xRegion()))>_threshold)
                _pop.degrowth(_pop.xRegion());
            return _pop.xRegion();
        }
        else{
            if(absolute(_field(_pop.x().second))<_threshold)
                _pop.growth(_pop.x().second);
            return _pop.x().second;
        }
    }
    typename Population::Function &getZI()
    {
        return _pop.getZI();
    }
    const typename Population::Function &getRegion()
    {
        return _pop.getRegion   ();
    }
};

template<typename FunctionPhaseField>
class POP_EXPORTS RegionGrowingMultiPhaseField
{
public:
    typedef typename  FunctionTypeTraitsSubstituteF<FunctionPhaseField,UI16>::Result FunctionOneLabel;

    typedef PopulationSingleRegionZIGP<FunctionOneLabel,FunctorZero>  Population;
    typedef PopulationOneRegionZIIteratotGP<Population,IteratorVecOneRegion>  PopulationIteratorRegion;
    typedef IteratorRegionZIPhaseField<PopulationIteratorRegion,FunctionPhaseField > IteratorE;
private:
    FunctorZero _funczero;

    typename FunctionPhaseField::IteratorENeighborhood _itn;
    PopulationIteratorRegion _pop;

    IteratorE _it;

public:
    template<typename FunctionMultiphase>
    RegionGrowingMultiPhaseField(FunctionPhaseField & field,FunctionMultiphase & multiphase, F64 treshold)
        :_itn(field.getIteratorENeighborhood()),_pop(field.getDomain(),_funczero,field.getIteratorENeighborhood()),_it(_pop,field,treshold)
    {
       initField(_pop,multiphase,field);
    }
    template<typename FunctionMultiphase,typename FunctionBulk>
    RegionGrowingMultiPhaseField(FunctionPhaseField & field,FunctionMultiphase & multiphase,F64 treshold,FunctionBulk & bulk)
        :_itn(field.getIteratorENeighborhood()),_pop(field.getDomain(),_funczero,field.getIteratorENeighborhood()),_it(_pop,field,treshold)
    {
        initField(_pop,multiphase,field,bulk);
    }

    template<typename Population,typename FunctionMultiField>
    void  initField(Population & pop,FunctionMultiField &multifield,FunctionPhaseField & phasefield )
    {
        typename FunctionMultiField::IteratorEDomain  it(multifield.getIteratorEDomain());
        typename FunctionMultiField::IteratorENeighborhood itn(multifield.getIteratorENeighborhood(1,0));

        while(it.next()){
            typename FunctionMultiField::E x = it.x();
            typename FunctionMultiField::F value = multifield(x);
            itn.init(x);
            bool grow=false;
            while(itn.next()&&grow==false){
                if(multifield(itn.x())!=value){
                    grow=true;
                }
            }
            if(grow==true){
                pop.growth(x);
                phasefield(x)=0.5;
            }
            else{
                phasefield(x)=1;
            }

        }
    }
    template<typename Population,typename FunctionMultiField,typename FunctionBulk>
    void  initField(Population & pop,FunctionMultiField &multifield,FunctionPhaseField & phasefield ,FunctionBulk & bulk )
    {
        typename FunctionMultiField::IteratorEDomain  it(multifield.getIteratorEDomain());
        typename FunctionMultiField::IteratorENeighborhood itn(multifield.getIteratorENeighborhood(1,0));

        while(it.next()){
            typename FunctionMultiField::E x = it.x();
            if(bulk(x)==0)
            {
                pop.setDeadRegion(x);
                phasefield(x)=NumericLimits<typename FunctionPhaseField::F>::maximumRange();
            }
            else
            {
                typename FunctionMultiField::F value = multifield(x);
                itn.init(x);
                bool grow=false;
                while(itn.next()&&grow==false){
                    if(multifield(itn.x())!=value&& bulk(itn.x())!=0){
                        grow=true;
                    }
                }
                if(grow==true){
                    pop.growth(x);
                    phasefield(x)=0.5;
                }
                else{
                    phasefield(x)=1;
                }
            }

        }
    }
    IteratorE &getIterator()
    {
        return _it;
    }
};
}

#endif // TOOLPDEREGIONGROWING_H
