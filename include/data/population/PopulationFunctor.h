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

#ifndef FUNCTORPOP_HPP
#define FUNCTORPOP_HPP
#include<limits>
#include <stdio.h>
#include <stdlib.h>
#include<cmath>
#include"data/typeF/TypeTraitsF.h"
#include"data/functor/FunctorF.h"
#include"algorithm/ForEachFunctor.h"
namespace pop
{
class POP_EXPORTS FunctorZero
{
public:
    I32 nbrLevel()
    {
        return 1;
    }
    template<typename  Label ,typename VecN>
    I32 operator()(Label , const VecN&)
            const
    {
        return 0;
    }
    template<typename VecN>
    I32 operator()( const VecN&)
            const
    {
        return 0;
    }
};
template<typename Function>
class POP_EXPORTS FunctorTopography
{
private:
    const Function &_f;
    typename Function::F _level;
public:
    FunctorTopography( const Function & f)
        :_f(f),_level(0)
    {}
    I32 nbrLevel()
    {
        if(NumericLimits<typename Function::F>::maximumRange()<=NumericLimits<UI16>::maximumRange())
            return NumericLimits<typename Function::F>::maximumRange()+1;
        else{
            typename Function::IteratorEDomain it(_f.getIteratorEDomain());
            FunctorF::FunctorAccumulatorMax<typename Function::F > func;
            typename Function::F value =  forEachFunctorAccumulator(_f,func,it);
            return value+1;
        }
    }
    void setLevel(typename Function::F level)
    {
        _level=level;
    }
    template<typename  Label ,typename VecN>
    I32 operator()(Label , const VecN&x)
            const
    {
        return maximum(_f(x),_level);
    }
};

class POP_EXPORTS FunctorLabel
{
private:
    I32 _level;
public:
    FunctorLabel(I32 nbrlabel)
        :_level(nbrlabel)
    {}
    I32 nbrLevel()
    {
        return _level;
    }
    template<typename  Label ,typename VecN>
    I32 operator()(Label label, const VecN&)
            const
    {
        return label;
    }
};
class POP_EXPORTS FunctorSwitch
{
private:
    int _flipflop;
public:
    FunctorSwitch()
        :_flipflop(0)
    {
    }
    I32 nbrLevel()
    {
        return 2;
    }
    template<typename  Label ,typename VecN>
    I32 operator()(Label , const VecN&)
            const
    {
        return _flipflop;
    }
    template<typename VecN>
    I32 operator()( const VecN&)
            const
    {
        return _flipflop;
    }
    void switchFlipFlop()
    {
        _flipflop = (_flipflop+1)%2;
    }
    int getFlipFlop()
    {
        return _flipflop;
    }

};



template<typename FunctionTopo>
class POP_EXPORTS FunctorMean
{
private:

    const FunctionTopo &_topo;
    std::vector<  F32 > _v_mean;
    std::vector<  int    > _v_number;
public:
    FunctorMean(const FunctionTopo & topo)
        :_topo(topo)
    {}
    I32 nbrLevel()
    {
        return NumericLimits<typename FunctionTopo::F>::maximumRange()+1;
    }

    void addPoint(int label, const typename FunctionTopo::E&x){
        if(_v_mean.size()<label){
            _v_mean.resize(label+1,0);
            _v_number.resize(label+1,0);
        }
        if(_v_number[label] ==0)
             _v_mean[label] = _topo(x);
        else
            _v_mean[label] =  (_v_mean[label]*_v_number[label] + _topo(x))/(_v_number[label] +1);
        _v_number[label]++;
    }


    I32 operator()(int label, const typename FunctionTopo::E&x)
    const
    {
        return maximum(0.0,absolute(1.0*_topo(x)-_v_mean[label]) );
    }
};



template<typename FunctionTopo>
class POP_EXPORTS FunctorMeanStandardDeviation
{
private:

    const FunctionTopo &_topo;
    std::vector<  F32 > _v_X;
    std::vector<  F32 > _v_X_power_2;
    std::vector<  F32 > _v_standard_deviation;
    std::vector<  int    > _v_number;
public:
    FunctorMeanStandardDeviation(const FunctionTopo & topo)
        :_topo(topo)
    {}
    I32 nbrLevel()
    {
        return NumericLimits<typename FunctionTopo::F>::maximumRange()+1;
    }

    void addPoint(int label, const typename FunctionTopo::E&x){
        if(_v_X.size()<label){
            _v_X.resize(label+1,0);
            _v_X_power_2.resize(label+1,0);
            _v_number.resize(label+1,0);
            _v_standard_deviation.resize(label+1,0);
        }
        if(_v_number[label] ==0){
             _v_X[label] = _topo(x);
             _v_X_power_2[label]  = _topo(x)*_topo(x);
             _v_standard_deviation[label] = 1;
        }
        else{
            _v_X[label] =  (_v_X[label]*_v_number[label] + _topo(x))/(_v_number[label] +1);
            _v_X_power_2[label] =  (_v_X_power_2[label]*_v_number[label] + _topo(x)*_topo(x))/(_v_number[label] +1);
            _v_standard_deviation[label]  = std::sqrt(_v_X_power_2[label] -_v_X[label]*_v_X[label]);
        }
        _v_number[label]++;
    }


    I32 operator()(int label, const typename FunctionTopo::E&x)
    const
    {
        return maximum(F32(0.0),absolute( (_topo(x)-_v_X[label]))/_v_standard_deviation[label] );
    }
};
template<typename FunctionTopo>
class POP_EXPORTS FunctorMeanMerge
{
private:
public:
    const FunctionTopo &_topo;
    std::vector<  F32 > _v_mean;
    std::vector<  int    > _v_number;


    FunctorMeanMerge(const FunctionTopo & topo)
        :_topo(topo)
    {}
    I32 nbrLevel()
    {
        return NumericLimits<typename FunctionTopo::F>::maximumRange()+1;
    }

    void addPoint(int label, const typename FunctionTopo::E&x){
        if(static_cast<int>(_v_mean.size())<=label){
            _v_mean.resize(label+1,0);
            _v_number.resize(label+1,0);
        }
        if(_v_number[label] ==0)
            _v_mean[label] = _topo(x);
        else
            _v_mean[label] =  (_v_mean[label]*_v_number[label] + _topo(x))/(_v_number[label] +1);
        _v_number[label]++;
    }
    F32 diff(int label_1,int label_2){
        return absolute(_v_mean[label_1]-_v_mean[label_2]);
    }

    void merge(int label_mix,int label_disapear){
        _v_mean[label_mix]  = (_v_mean[label_mix]*_v_number[label_mix] + _v_mean[label_disapear]*_v_number[label_disapear])/(_v_number[label_mix] +_v_number[label_disapear]);
        _v_number[label_mix] = (_v_number[label_mix] +_v_number[label_disapear]);
    }

    I32 operator()(int label, const typename FunctionTopo::E&x)
    const
    {
        return maximum(F32(0.0),absolute(_topo(x)-_v_mean[label]) );
    }
};
struct MasterSlave
{
    int _my_label;
    bool _is_master;
    MasterSlave * _my_master;
    std::vector<MasterSlave * > _my_slaves;
    MasterSlave()
        :_is_master(true),_my_master(NULL)
    {}
    ~    MasterSlave()
    {}
    void addSlave(MasterSlave & individu){
        if(_my_label!=individu._my_label){
            MasterSlave * master;
            if(individu._is_master==true){
                master = &individu;
            }else{
                master = individu._my_master;
            }
            for(unsigned int i=0;i<master->_my_slaves.size();i++){
                MasterSlave * slave = master->_my_slaves[i];
                slave->_my_master = this;
            }
            int size = this->_my_slaves.size() ;
            this->_my_slaves.resize( this->_my_slaves.size() + master->_my_slaves.size() ); // preallocate memory
           std::copy(master->_my_slaves.begin(), master->_my_slaves.end(),this->_my_slaves.begin()+size);
            master->_my_slaves.clear();
            master->_is_master = false;
            master->_my_master = this;
            this->_my_slaves.push_back(master);
        }
    }
    int getLabelMaster(){
        if(_is_master==true){
            return _my_label;
        }else{
            return _my_master->_my_label;
        }
    }
};



}

#endif // FUNCTORPOP_HPP
