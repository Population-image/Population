/******************************************************************************\
|*                   Population library for C++ X.X.X                         *|
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

#ifndef GROWS_H
#define GROWS_H


namespace pop
{

// we can have diffferent strategies in order for adding VecN in the list of queues

//We add all the VecN at the neighborhood of the growing VecN
template<typename IteratorENeighborhood,typename RestrictedSet>
class POP_EXPORTS Growth
{
protected:
    IteratorENeighborhood   _neigh;
    RestrictedSet & _restrictedset;
public:
    Growth(const IteratorENeighborhood  & itneigh, RestrictedSet & restrictedset)
        :_neigh(itneigh),_restrictedset(restrictedset)
    {
        ;
        _neigh.removeCenter();
    }

    template<typename Function,typename FunctorOrderingAttribute,typename SystemContainer>
    void growth(typename Function::F labelregion, const  typename Function::E & x, const Function & regions, FunctorOrderingAttribute & f,SystemContainer & sq)
    {
        _neigh.init(x);
        while(_neigh.next())
        {
            if(_restrictedset.noBelong(labelregion, regions(_neigh.x())))
            {
                sq.push(std::make_pair(labelregion,_neigh.x()),f(labelregion,_neigh.x()));
            }
        }
    }
    template<typename Function,typename FunctorOrderingAttribute,typename SystemContainer>
    void growth(const  typename Function::E & x, const Function & regions,FunctorOrderingAttribute & f,SystemContainer & lqueues)
    {

        _neigh.init(x);
        while(_neigh.next())
        {
            if(_restrictedset.noBelong(regions(_neigh.x())))
            {

                lqueues.push(_neigh.x(),f(_neigh.x()));

            }
        }
    }
};
template<typename IteratorENeighborhood,typename RestrictedSet>
class POP_EXPORTS GrowthZISingleRegion
{
private:
    IteratorENeighborhood   _neigh;
    IteratorENeighborhood   _neigh_minus1;
        RestrictedSet & _restrictedset;
public:
    GrowthZISingleRegion(const IteratorENeighborhood  & itneigh, RestrictedSet & restrictedset)
        :_neigh(itneigh),_neigh_minus1(itneigh),_restrictedset(restrictedset)
    {
        _neigh.removeCenter();
        _neigh_minus1.removeCenter();
    }

    template<typename Function,typename Functor,typename SystemContainer,typename FunctionZI>
    void growth(const  typename Function::E & x, const Function & regions,Functor & f,SystemContainer & lqueues,FunctionZI & zi)
    {

        _neigh.init(x);
        while(_neigh.next())
        {
            if(_restrictedset.noBelong(regions(_neigh.x()))&&zi(_neigh.x())==RestrictedSet::NoRegion)
            {
                zi(_neigh.x())=RestrictedSet::SingleRegion;
                lqueues.push(std::make_pair(RestrictedSet::SingleRegion,_neigh.x()),f(_neigh.x()));
            }
        }
    }
    template<typename Function,typename Functor,typename SystemContainer,typename FunctionZI>
    void degrowthZI(const  typename Function::E & x, const Function & regions,Functor & f,SystemContainer & lqueues,FunctionZI & zi)
    {

        _neigh.init(x);
        while(_neigh.next())
        {
            if(regions(_neigh.x())==1&&zi(x)==RestrictedSet::NoRegion)
            {
                zi(x)=RestrictedSet::SingleRegion;
                lqueues.push(std::make_pair(RestrictedSet::SingleRegion,x),f(x));
                break;
            }
        }
        _neigh_minus1.init(x);
        while(_neigh_minus1.next())
        {

            if(zi(_neigh_minus1.x())==RestrictedSet::SingleRegion)
            {
                bool hitregion = false;
                _neigh.init(_neigh_minus1.x());
                while(_neigh.next()&&hitregion==false)
                {
                    if(regions(_neigh.x())==RestrictedSet::SingleRegion)

                        hitregion = true;

                }
                if(hitregion==false)
                    zi(_neigh_minus1.x())=RestrictedSet::NoRegion;
            }

        }

    }
};
template<typename IteratorENeighborhood,typename RestrictedSet>
class POP_EXPORTS GrowthZIMultiRegion
{

private:
    IteratorENeighborhood   _neigh;
    IteratorENeighborhood   _neigh_minus1;
        RestrictedSet & _restrictedset;
public:
    GrowthZIMultiRegion(const IteratorENeighborhood  & itneigh, RestrictedSet & restrictedset)
        :_neigh(itneigh),_neigh_minus1(itneigh),_restrictedset(restrictedset)
    {
        _neigh.removeCenter();
        _neigh_minus1.removeCenter();
    }
    template<typename Function,typename Functor,typename SystemContainer>
    void growth(typename Function::F , const  typename Function::E & , const Function & , Functor & ,SystemContainer & )
    {

    }

    template<typename Function,typename Functor,typename SystemContainer>
    void growth(const  typename Function::E & , const Function & ,Functor & ,SystemContainer &)
    {

    }
    template<typename Function,typename Functor,typename SystemContainer,typename FunctionZI>
    void growthZI(typename Function::F oldlabelregion, typename Function::F newlabelregion,const  typename Function::E & x, const Function & regions,Functor & f,SystemContainer & lqueues,FunctionZI & zi)
    {
        this->_neigh.init(x);
        while(this->_neigh.next()){
            if(this->_restrictedset.noBelong(newlabelregion,   regions(this->_neigh.x()))&&zi(this->_neigh.x())!=newlabelregion){
                zi(this->_neigh.x())=newlabelregion;
                lqueues.push(std::make_pair(newlabelregion,this->_neigh.x()),f(newlabelregion,this->_neigh.x()));

            }
        }
        zi(x)=RestrictedSet::NoRegion;
        this->_neigh.init(x);
        while(this->_neigh.next()){
            if(this->_restrictedset.noBelong(regions(this->_neigh.x()),   regions(x)) ){
                zi(x)=regions(this->_neigh.x());
                lqueues.push(std::make_pair(regions(this->_neigh.x()),x),f(regions(this->_neigh.x()),x));
                break;
            }
        }

        this->_neigh_minus1.init(x);
        while(this->_neigh_minus1.next()){
            if(zi(this->_neigh_minus1.x())==oldlabelregion){
                bool hitregion = false;
                this->_neigh.init(this->_neigh_minus1.x());
                while(this->_neigh.next()&&hitregion==false){
                    if(regions(this->_neigh.x())==oldlabelregion)
                        hitregion = true;
                }
                if(hitregion==false){
                    zi(this->_neigh_minus1.x())=RestrictedSet::NoRegion;
                    this->_neigh.init(this->_neigh_minus1.x());
                    while(this->_neigh.next()){
                        if(this->_restrictedset.noBelong(   regions(this->_neigh.x())  , regions(this->_neigh_minus1.x())  )&&zi(this->_neigh_minus1.x())==RestrictedSet::NoRegion){
                            zi(this->_neigh_minus1.x())=regions(this->_neigh.x());
                            lqueues.push(std::make_pair(regions(this->_neigh.x()),this->_neigh_minus1.x()),f(regions(this->_neigh.x()),this->_neigh_minus1.x()));
                            break;
                        }
                    }
                }
            }

        }

    }
    template<typename Function,typename Functor,typename SystemContainer,typename FunctionZI>
    void growthZIWithoutDegrowth(typename Function::F newlabelregion,const  typename Function::E & x, const Function & regions,Functor & f,SystemContainer & lqueues,FunctionZI & zi)
    {
        this->_neigh.init(x);
        while(this->_neigh.next()){
            if(this->_restrictedset.noBelong(newlabelregion,   regions(this->_neigh.x()))&&zi(this->_neigh.x())==RestrictedSet::NoRegion){
                zi(this->_neigh.x())=newlabelregion;
                lqueues.push(std::make_pair(newlabelregion,this->_neigh.x()),f(newlabelregion,this->_neigh.x()));
            }
        }
    }
};


template<typename IteratorENeighborhood,typename RestrictedSet>
class POP_EXPORTS GrowthInformation
{
protected:
    IteratorENeighborhood   _neigh;
    RestrictedSet & _restrictedset;
public:
    GrowthInformation(const IteratorENeighborhood  & itneigh, RestrictedSet & restrictedset)
        :_neigh(itneigh),_restrictedset(restrictedset)
    {

        _neigh.removeCenter();
    }


    template<typename Function,typename Functor,typename SystemContainer,typename Information>
    void growth(typename Function::F labelregion, const  typename Function::E & x, const Function & regions, Functor & f,SystemContainer & sq,Information & info)
    {
        _neigh.init(x);
        while(_neigh.next())
        {

            if(_restrictedset.noBelong(labelregion, regions(_neigh.x())))
            {
                std::pair<typename Function::F, typename Function::E> p = std::make_pair(labelregion,_neigh.x());
                std::pair<std::pair<typename Function::F, typename Function::E>,Information> pp=std::make_pair(p,info);
                sq.push(pp,f(labelregion,_neigh.x()));
            }
        }
    }
};


}
#endif // GROWS_H
