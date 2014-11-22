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

#ifndef QUEUES_HPP
#define QUEUES_HPP
#include<queue>
#include<vector>
#include"data/distribution/DistributionAnalytic.h"
namespace pop
{
template<typename Element>
class POP_EXPORTS SQFIFO
{
private:
    std::vector<std::queue<Element> > _v;
    I32 _level;
    Element _elt;
public:
    SQFIFO(I32 numberqueus)
        :_v(numberqueus),_level(0){}
    void push(const Element & element, I32 level){
        _v[level].push(element);
    }
    void pushCurrentLevel(const Element & element){
        _v[_level].push(element);
    }

    void setLevel(I32 level){
        _level = level;
    }
    bool next(){
        return !_v[_level].empty();
    }
    Element & x(){
        return _v[_level].front();
    }
    void init(){}
    void pop(){_v[_level].pop();}
};

template<typename Element>
class POP_EXPORTS SQFIFONextSmallestLevel
{
private:
    std::vector<std::queue<Element> > _v;
    I32 _level;
public:
    SQFIFONextSmallestLevel(I32 numberqueue)
        :_v(numberqueue),_level(numberqueue+1){}
    void push(const Element & element, I32 level){
        if(level>=0 && level<(I32)_v.size()){
            _v[level].push(element);
            _level = minimum(_level,level);
        }
    }
    void pushCurrentLevel(const Element & element){
        _v[_level].push(element);
    }
    bool next(){
        if(_level>=(I32)_v.size())return false;
        if(_v[_level].empty()){
            _level++;
            return next();
        }
        else{
            return true;
        }
    }
    Element & x(){
        return _v[_level].front();
    }
    void init()
    {}
    void pop(){_v[_level].pop();}
};
template<typename Element>
class POP_EXPORTS SQVectorAdvitamAertenam
{
private:
    std::vector<std::vector<Element> > _v;
    I32 _level;
    I32 _index;
    Element _elt;
public:
    SQVectorAdvitamAertenam(I32 numberqueus)
        :_v(numberqueus),_level(0),_index(0){}
    void push(const Element & element, I32 level){
        _v[level].push_back(element);
    }
    void setLevel(I32 level){
        _level = level;
        _index= (I32)_v[_level].size();
    }
    void pushCurrentLevel(const Element & element){
        _v[_level].push(element);
    }
    void init()
    {
        _index= (I32)_v[_level].size();
    }
    bool next(){

        _index--;
        if(_index<0)
        {
            return false;
        }
        else
            return true;
    }
    Element & x(){
        return _v[_level][_index];
    }

    void pop()
    {
        _v[_level][_index]=*(_v[_level].rbegin());
        _v[_level].pop_back();
    }
};
template<typename Element>
class POP_EXPORTS SQRandomAccess
{
private:
    std::vector<std::vector<Element> > _v;
    I32 _level;
    Element _elt;
    I32 _index;
    DistributionUniformReal gen;
public:
    SQRandomAccess(I32 numberqueus)
        :_v(numberqueus),_level(0),_index(0),gen(0,1)

    {}
    void push(const Element & element, I32 level){
        _v[level].push_back(element);
    }
    void pushCurrentLevel(const Element & element){
        _v[_level].push(element);
    }
    void setLevel(I32 level){
        _level = level;
    }
    void init()
    {
        _index= (I32)_v[_level].size();
    }
    bool next(){

        _index=floor(gen.randomVariable()*_v[_level].size());
        if(_index==_v[_level].size())_index=_v[_level].size()-1;
        if((I32)_v[_level].size()==0)
        {
            return false;
        }
        else
            return true;
    }
    Element & x(){
        return _v[_level][_index];
    }

    void pop( )
    {
        _v[_level][_index]=*(_v[_level].rbegin());
        _v[_level].pop_back();
    }
};
}

#endif // QUEUES_HPP