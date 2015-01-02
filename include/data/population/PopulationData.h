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

#ifndef POPULATIONDATA_HPP
#define POPULATIONDATA_HPP
#include<iostream>
#include<limits>
#include"data/population/PopulationQueues.h"
#include"data/population/PopulationRestrictedSet.h"
#include"data/population/PopulationGrows.h"
namespace pop
{

template<typename FunctionSeed,typename FunctorOrderingAttribute,template<typename> class RestrictedSet=RestrictedSetWithoutALL,template<typename> class  SQ=SQFIFO,template<typename,typename>class RuleGrowth=Growth>
class POP_EXPORTS Population
{
protected:
    FunctionSeed _region;
    FunctorOrderingAttribute & _f;
    typedef std::pair<typename FunctionSeed::F, typename FunctionSeed::E> Element;
    SQ<Element > _SQ;
    RuleGrowth< typename FunctionSeed::IteratorENeighborhood,RestrictedSet<typename FunctionSeed::F> > _rulegrowth;
    RestrictedSet<typename FunctionSeed::F> _restrictedset;
public:
    typedef FunctionSeed Function;
    typedef FunctorOrderingAttribute Functor;
    typedef typename FunctionSeed::IteratorENeighborhood IteratorENeighborhood;
    Population(const typename FunctionSeed::Domain & domain, FunctorOrderingAttribute & f,const typename FunctionSeed::IteratorENeighborhood & itneigh)
        :_region(domain,RestrictedSet<typename FunctionSeed::F>::NoRegion),
          _f(f),
          _SQ(f.nbrLevel()),
          _rulegrowth(itneigh,_restrictedset)
    {

    }
    virtual ~Population(){
		//std::cout<<"delete Population"<<std::endl;
	}

    virtual void growth( const typename FunctionSeed::F & labelregion,const  typename FunctionSeed::E & x  ){
        if(labelregion==RestrictedSet<typename FunctionSeed::F>::NoRegion)
        {
            _region(x)=labelregion-1;
            _rulegrowth.growth(labelregion-1,x, _region,_f,_SQ);
        }else{
            _region(x)=labelregion;
            _rulegrowth.growth(labelregion,x, _region,_f,_SQ);
        }
    }

    void setLevel(I32 level){
        _SQ.setLevel(level);
    }
    Element & x(){
        return _SQ.x();
    }
    virtual bool next(){
        while(_SQ.next()){
            if(_restrictedset.noBelong( _SQ.x().first,_region( _SQ.x().second)))
                return true;
            else
                _SQ.pop();
        }
        return false;
    }
    virtual void pop(){
        _SQ.pop();
    }
    FunctionSeed & getRegion(){
        return _region;
    }
    void setRegion( const typename FunctionSeed::F & labelregion,const  typename Function::E & x  ){
        if(labelregion==RestrictedSet<typename FunctionSeed::F>::NoRegion)
            _region(x)=labelregion-1;
        else
            _region(x)=labelregion;
    }

    typename FunctionSeed::F  getLabelNoRegion(){
        return RestrictedSet<typename FunctionSeed::F>::NoRegion;
    }
    FunctorOrderingAttribute & getFunctor()
    {
        return _f;
    }
    RestrictedSet<typename FunctionSeed::F> &  getRestrictedSet()
    {
        return _restrictedset;
    }
    RuleGrowth< typename FunctionSeed::IteratorENeighborhood,RestrictedSet<typename FunctionSeed::F> >  & getRuleGrowth()
    {
        return _rulegrowth;
    }

    SQ<Element > &getSQ()
    {
        return _SQ;
    }

};


template<typename FunctionSeed,typename FunctorOrderingAttribute,template<typename> class RestrictedSet=RestrictedSetWithoutALL,template<typename> class  SQ=SQFIFO,template<typename,typename>class RuleGrowth=Growth>
class POP_EXPORTS PopulationTimeToken : public Population<FunctionSeed,FunctorOrderingAttribute,RestrictedSet,SQ,RuleGrowth>
{
private:
    bool _istimetoken;
public:
    typedef FunctionSeed Function;
    typedef FunctorOrderingAttribute Functor;
    typedef typename FunctionSeed::IteratorENeighborhood IteratorENeighborhood;
    PopulationTimeToken(const typename FunctionSeed::Domain & domain, FunctorOrderingAttribute & f,const typename FunctionSeed::IteratorENeighborhood & itneigh)
        :Population<FunctionSeed,FunctorOrderingAttribute,RestrictedSet,SQ,RuleGrowth>(domain,f,itneigh)
    {
    }

    void addTimeToken(int level)
    {

        this->_SQ.push(std::make_pair(RestrictedSet<typename FunctionSeed::F>::NoRegion,0),level);
    }
    virtual bool next(){
        while(this->_SQ.next()){
            if(this->_SQ.x().first==RestrictedSet<typename FunctionSeed::F>::NoRegion)
            {
                _istimetoken = true;
                this->_SQ.pop();
                return true;
            }
            else
            {
                _istimetoken = false;
                if(this->_restrictedset.noBelong( this->_SQ.x().first,this->_region( this->_SQ.x().second)))
                    return true;
                else
                    this->_SQ.pop();
            }
        }
        return false;
    }
    bool isTimeToken()
    {
        return _istimetoken;
    }
};




template<typename FunctionSeed,typename FunctorOrderingAttribute,template<typename> class  SQ=SQVectorAdvitamAertenam,template<typename,typename>class RuleGrowth=GrowthZISingleRegion >
class POP_EXPORTS PopulationSingleRegionZIGP
{
public:
    typedef RestrictedSetSingleRegion<UI8> RestrictedSet;
protected:

    FunctionSeed _region;
    FunctorOrderingAttribute & _f;
    typedef std::pair<typename FunctionSeed::F, typename FunctionSeed::E> Element;
    SQ<Element > _SQ;
    RuleGrowth< typename FunctionSeed::IteratorENeighborhood,RestrictedSet> _rulegrowth;
    RestrictedSet _restrictedset;
    FunctionSeed _zi;
    typename FunctionSeed::IteratorENeighborhood  _itn;


public:
    typedef FunctionSeed Function;
    typedef FunctorOrderingAttribute Functor;
    typedef typename FunctionSeed::IteratorENeighborhood IteratorENeighborhood;
    PopulationSingleRegionZIGP(const typename FunctionSeed::Domain & domain, FunctorOrderingAttribute & f,typename FunctionSeed::IteratorENeighborhood  itneigh)
        :_region(domain,RestrictedSet::NoRegion),
          _f(f),
          _SQ(f.nbrLevel()),
          _rulegrowth(itneigh,_restrictedset),
          _zi(domain,RestrictedSet::NoRegion),
          _itn(itneigh)
    {

    }
    virtual void growth( const typename FunctionSeed::F &, const  typename FunctionSeed::E & x  )
    {
        this->_region(x)=RestrictedSet::SingleRegion;
        this->_zi(x)=RestrictedSet::NoRegion;
        this->_rulegrowth.growth(x, this->_region,this->_f,this->_SQ,this->_zi);
    }
    virtual void setDeadRegion( const  typename FunctionSeed::E & x  )
    {
        _region(x)=RestrictedSet::DeadRegion;
        this->_zi(x)=RestrictedSet::DeadRegion;
    }
    virtual void degrowth( const  typename FunctionSeed::E & x  )
    {
        this->_region(x)=RestrictedSet::NoRegion;
        this->_rulegrowth.degrowthZI(x, this->_region,this->_f,this->_SQ,this->_zi);
    }
    bool next()
    {
        while(this->_SQ.next()==true)
        {
            if(this->_restrictedset.noBelong(this->_region(this->_SQ.x().second))&& this->_zi(this->_SQ.x().second)==RestrictedSet::SingleRegion)
                return true;
            else
                this->_SQ.pop();
        }
        return false;
    }
    void setLevel(I32 level){
        _SQ.setLevel(level);
    }
    Element & x(){
        return _SQ.x();
    }
    void pop()
    {
        this->_SQ.pop();
    }
    void init()
    {
        this->_SQ.init();
    }

    FunctionSeed & getZI()
    {
        return this->_zi;
    }
    const FunctionSeed & getRegion(){
        return _region;
    }

};

template<typename FunctionSeed,typename FunctorOrderingAttribute,template<typename> class RestrictedSet=RestrictedSetWithoutALL,template<typename> class  SQ=SQFIFO,template<typename,typename>class RuleGrowth=GrowthZIMultiRegion>
class POP_EXPORTS PopulationMultiRegionZIIteratorRegionGP:public  Population<FunctionSeed,FunctorOrderingAttribute,RestrictedSet,SQ,RuleGrowth>
{
protected:


    FunctionSeed _zi;
    typedef typename FunctionSeed::E Element;
    typedef  Population<FunctionSeed,FunctorOrderingAttribute,RestrictedSet,SQ,RuleGrowth> PopulationMother;
    typename FunctionSeed::IteratorENeighborhood  _itn;
public:
    typedef FunctionSeed Function;
    typedef FunctorOrderingAttribute Functor;
    typedef typename FunctionSeed::IteratorENeighborhood IteratorENeighborhood;

    PopulationMultiRegionZIIteratorRegionGP(const typename FunctionSeed::Domain & domain, FunctorOrderingAttribute & f,const typename FunctionSeed::IteratorENeighborhood & itneigh)
        :PopulationMother(domain,  f,itneigh),_zi(domain,RestrictedSet<typename FunctionSeed::F>::NoRegion),_itn(itneigh)
    {}
    void setInitRegion(const FunctionSeed & labelinit)
    {
        typename FunctionSeed::IteratorEDomain it(labelinit.getIteratorEDomain());
        while(it.next())
        {
            this->_region(it.x())=labelinit(it.x());
        }
        it.init();
        while(it.next())
        {
            this->_rulegrowth.growthZIWithoutDegrowth(this->_region(it.x()), it.x(), this->_region,this->_f,this->_SQ,this->_zi);
        }
    }
    virtual void growth( const typename Function::F & labelregion,const  typename Function::E & x   )
    {

        typename Function::F  oldlabelregion = this->_region(x);
        this->_region(x)=labelregion;
        this->_rulegrowth.growthZI(oldlabelregion,labelregion, x, this->_region,this->_f,this->_SQ,this->_zi);
    }

    bool next()
    {
        while(this->_SQ.next()==true)
        {
            typename Function::F label = this->_SQ.x().first;
            typename Function::E pos = this->_SQ.x().second;
            if(this-> _restrictedset.noBelong(label,this->_region(pos))  &&
                    this->_zi(pos)==label
                    )
                return true;
            else
                this->_SQ.pop();
        }
        return false;
    }
    void init()
    {
        this->_SQ.init();
    }
    FunctionSeed & getZI()
    {
        return this->_zi;
    }
};


template<typename FunctionSeed,typename FunctorOrderingAttribute,typename Information,template<typename> class RestrictedSet=RestrictedSetWithoutALL,template<typename> class  SQ=SQFIFO>
class POP_EXPORTS PopulationInformation
{
protected:
    FunctionSeed _region;
    FunctorOrderingAttribute & _f;
    typedef std::pair<std::pair<typename FunctionSeed::F, typename FunctionSeed::E>,Information> Element;
    SQ<Element > _SQ;
    GrowthInformation< typename FunctionSeed::IteratorENeighborhood,RestrictedSet<typename FunctionSeed::F> > _rulegrowth;
    RestrictedSet<typename FunctionSeed::F> _restrictedset;
public:
    typedef FunctionSeed Function;
    typedef FunctorOrderingAttribute Functor;
    typedef typename FunctionSeed::IteratorENeighborhood IteratorENeighborhood;
    PopulationInformation(const typename FunctionSeed::Domain & domain, FunctorOrderingAttribute & f,const typename FunctionSeed::IteratorENeighborhood & itneigh)
        :_region(domain,RestrictedSet<typename FunctionSeed::F>::NoRegion),
          _f(f),
          _SQ(f.nbrLevel()),
          _rulegrowth(itneigh,_restrictedset)
    {

    }
    virtual void growth( const typename FunctionSeed::F & labelregion,const  typename FunctionSeed::E & x,Information &info  ){
        if(labelregion==RestrictedSet<typename FunctionSeed::F>::NoRegion)
        {
            _region(x)=labelregion-1;
            _rulegrowth.growth(labelregion-1,x, _region,_f,_SQ,info);
        }else{
            _region(x)=labelregion;
            _rulegrowth.growth(labelregion,x, _region,_f,_SQ,info);
        }
    }

    void setLevel(I32 level){
        _SQ.setLevel(level);
    }
    Element & x(){
        return _SQ.x();
    }
    virtual bool next(){
        while(_SQ.next()){
            if(_restrictedset.noBelong( _SQ.x().first.first,_region( _SQ.x().first.second)))
                return true;
            else
                _SQ.pop();
        }
        return false;
    }
    virtual void pop(){
        _SQ.pop();
    }
    FunctionSeed & getRegion(){
        return _region;
    }
    void setRegion( const typename FunctionSeed::F & labelregion,const  typename Function::E & x  ){
        if(labelregion==RestrictedSet<typename FunctionSeed::F>::NoRegion)
            _region(x)=labelregion-1;
        else
            _region(x)=labelregion;
    }

    typename FunctionSeed::F  getLabelNoRegion(){
        return RestrictedSet<typename FunctionSeed::F>::NoRegion;
    }
    FunctorOrderingAttribute & getFunctor()
    {
        return _f;
    }
    RestrictedSet<typename FunctionSeed::F> &  getRestrictedSet()
    {
        return _restrictedset;
    }
    GrowthInformation< typename FunctionSeed::IteratorENeighborhood,RestrictedSet<typename FunctionSeed::F> >  & getRuleGrowth()
    {
        return _rulegrowth;
    }

    SQ<Element > &getSQ()
    {
        return _SQ;
    }

};


//template<typename FunctionSeed,typename FunctorOrderingAttribute,template<typename> class RestrictedSet=RestrictedSetWithoutALL,template<typename> class  SQ=SQFIFO,template<typename,typename>class RuleGrowth=Growth>
//class PopulationMultiRegionTokenTimeGP:public  Population<FunctionSeed,FunctorOrderingAttribute,RestrictedSet,SQ,RuleGrowth>
//{

//protected:
//    typedef std::pair<typename FunctionSeed::F, typename FunctionSeed::E> Element;
//public:
//    typedef FunctionSeed Function;
//    typedef FunctorOrderingAttribute Functor;
//    typedef typename FunctionSeed::IteratorENeighborhood IteratorENeighborhood;
//    PopulationMultiRegionTokenTimeGP(const typename FunctionSeed::Domain & domain, FunctorOrderingAttribute & f,const typename FunctionSeed::IteratorENeighborhood & itneigh)
//        :Population<FunctionSeed,FunctorOrderingAttribute,RestrictedSet,SQ,RuleGrowth>(domain,  f,itneigh)
//    {
//    }
//    void putTimeToken(I32 level)
//    {
//        this->_SQ.push(std::make_pair( this->_blank,typename FunctionSeed::E() ),level);
//    }
//    bool isTimeToken(const Element & x)
//    {
//        if(x.first==this->_blank)return true;
//        else return false;
//    }
//    bool next()
//    {
//        while(this->_SQ.next())
//        {
//            if(this->_SQ.x().first==this->_blank)return true;
//            if(this->_restrictedset.noBelong(this->_SQ.x().first,this->_region(this->_SQ.x().second)))
//                return true;
//             else
//               this->_SQ.pop();
//        }
//        return false;
//    }
//};






//template<typename FunctionSeed,typename FunctorOrderingAttribute,template<typename> class  SQ=SQFIFO,template<typename,typename>class RuleGrowth=Growth>
//class PopulationOneRegionGP
//{
//protected:

//    typedef RestrictedSetSingleRegion<UI8> RestrictedSet;
//    FunctionSeed _region;
//    FunctorOrderingAttribute & _f;
//    typedef typename FunctionSeed::E Element;
//    SQ<Element > _SQ;
//    RuleGrowth< typename FunctionSeed::IteratorENeighborhood,RestrictedSet> _rulegrowth;
//    RestrictedSet _restrictedset;
//public:
//    typedef FunctionSeed Function;
//    typedef FunctorOrderingAttribute Functor;
//    typedef typename FunctionSeed::IteratorENeighborhood IteratorENeighborhood;
//    PopulationOneRegionGP(const typename FunctionSeed::Domain & domain, FunctorOrderingAttribute & f,typename FunctionSeed::IteratorENeighborhood  itneigh)
//        :_region(domain,RestrictedSet::NoRegion),
//        _f(f),
//        _SQ(f.nbrLevel()),
//        _rulegrowth(itneigh,_restrictedset)
//    {
//    }
//    virtual void growth( const  typename FunctionSeed::E & x  )
//    {
//        _region(x)=RestrictedSet::SingleRegion;
//        _rulegrowth.growth(x, _region,_f,_SQ);
//    }
//    virtual bool next(){
//        while(_SQ.next()==true)
//        {
//            if(_restrictedset.noBelong(_region(_SQ.x())))
//                return true;
//             else
//               _SQ.pop();
//        }
//        return false;
//    }
//    virtual void pop(){
//        _SQ.pop();
//    }
//    virtual void setRegion( const  typename FunctionSeed::E & x  )
//    {
//        _region(x)=RestrictedSet::SingleRegion;
//    }
//    void setLevel(I32 level)
//    {
//        _SQ.setLevel(level);

//    }
//    Element & x()
//    {
//        return _SQ.x();
//    }
//    FunctionSeed & getRegion()
//    {
//        return _region;
//    }
//    typename FunctionSeed::F  getRegion(const typename FunctionSeed::E & x )
//    {
//        return _region(x);
//    }
//    typename FunctionSeed::F  getLabelNoRegion()
//    {
//        return RestrictedSet::NoRegion;
//    }
//};

//template<typename E>
//class PolicyBlankTokenMinusOne
//{
//public:
//    E getBlankToken()
//    {
//        return E(-1);
//    }
//    bool isBlankToken(const E & token)
//    {
//        if(token(0)==-1)return true;
//        else return false;
//    }
//};

//template<typename FunctionSeed,typename FunctorOrderingAttribute,template<typename> class  SQ=SQFIFO,template<typename,typename>class RuleGrowth=Growth,template<typename>class PoliycBlankToken=PolicyBlankTokenMinusOne >
//class PopulationOneRegionTokenTimeGP:public  PopulationOneRegionGP<FunctionSeed,FunctorOrderingAttribute,SQ,RuleGrowth>, public PoliycBlankToken<typename FunctionSeed::E>
//{

//protected:
//    typedef typename FunctionSeed::E Element;
//public:
//    typedef FunctionSeed Function;
//    typedef FunctorOrderingAttribute Functor;
//    typedef typename FunctionSeed::IteratorENeighborhood IteratorENeighborhood;
//    PopulationOneRegionTokenTimeGP(const typename FunctionSeed::Domain & domain, FunctorOrderingAttribute & f,typename FunctionSeed::IteratorENeighborhood & itneigh)
//        :PopulationOneRegionGP<FunctionSeed,FunctorOrderingAttribute,SQ,RuleGrowth>(domain,  f,itneigh)
//    {
//    }
//    void putTimeToken(I32 level)
//    {
//        this->_SQ.push(PoliycBlankToken<typename FunctionSeed::E>::getBlankToken(),level);
//    }
//    bool isTimeToken(const Element & x)
//    {
//        if(isBlankToken(x))return true;
//        else return false;
//    }
//    bool next()
//    {
//        while(this->_SQ.next())
//        {
//            if(isBlankToken(this->_SQ.x()))
//            {
//                this->_SQ.pop();
//                return true;
//            }
//            if(this->_restrictedset.noBelong(_region(this->_SQ.x())))
//                return true;
//             else
//               this->_SQ.pop();
//        }
//        return false;
//    }
//};
}
#endif // POPULATIONDATA_HPP
