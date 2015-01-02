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

#ifndef REGION_HPP
#define REGION_HPP
#include<limits>


namespace pop
{

template<typename Label>
class POP_EXPORTS RestrictedSetWithoutALL
{ 
public:
    static Label NoRegion;
    static bool noBelong(Label ,Label regionpos )
    {
        if(regionpos==NoRegion)
            return true;
        else
            return false;
    }
    static bool noBelong(Label regionpos )
    {
        if(regionpos==NoRegion)
            return true;
        else
            return false;
    }    
};
template<typename Label>
Label RestrictedSetWithoutALL<Label>::NoRegion = NumericLimits<Label>::maximumRange();


template<typename Label>
class POP_EXPORTS RestrictedSetWithMySelf
{
public:
    static Label NoRegion;

    static bool noBelong(Label regiongrow,Label regionpos )
    {
        if(regionpos!=regiongrow)
            return true;
        else
            return false;
    }
};
template<typename Label>
Label RestrictedSetWithMySelf<Label>::NoRegion = NumericLimits<Label>::maximumRange();
template<typename Label>
class POP_EXPORTS RestrictedSetWithMySelfAndOneOther
{
private:
    Label _other;
public:
    static Label NoRegion;

    void setOtherRegion(Label other)
    {
        _other = other;
    }
    bool noBelong(Label regiongrow,Label regionpos )
    {
        if(regionpos!=regiongrow && regionpos!=_other )
            return true;
        else
            return false;
    }
};
template<typename Label>
Label RestrictedSetWithMySelfAndOneOther<Label>::NoRegion = NumericLimits<Label>::maximumRange();

template<typename Label>
class POP_EXPORTS RestrictedSetSingleRegion
{
public:
    static Label NoRegion;
    static Label SingleRegion;
    static Label DeadRegion;
    static bool noBelong(Label ,Label regionpos )
    {
        if(regionpos==NoRegion)
            return true;
        else
            return false;
    }
    static bool noBelong(Label regionpos )
    {
        if(regionpos==NoRegion)
            return true;
        else
            return false;
    }
};
template<typename Label>
Label RestrictedSetSingleRegion<Label>::NoRegion = NumericLimits<Label>::maximumRange();
template<typename Label>
Label RestrictedSetSingleRegion<Label>::SingleRegion = 0;
template<typename Label>
Label RestrictedSetSingleRegion<Label>::DeadRegion = 1;


template<typename Label>
class POP_EXPORTS RestrictedSetWithoutSuperiorLabel
{
public:
    static Label NoRegion;
    static bool noBelong(Label label,Label regionpos )
    {
        if(label<regionpos)
            return true;
        else
            return false;
    }
};
template<typename Label>
Label RestrictedSetWithoutSuperiorLabel<Label>::NoRegion = NumericLimits<Label>::maximumRange();

template<typename Label>
class POP_EXPORTS RestrictedSetWithoutInferiorLabel
{
public:
    static Label NoRegion;
    static bool noBelong(Label label,Label regionpos )
    {
        if(label>regionpos)
            return true;
        else
            return false;
    }
};
template<typename Label>
Label RestrictedSetWithoutInferiorLabel<Label>::NoRegion = NumericLimits<Label>::maximumRange();
}
//template<typename Label,int Dead = NumericLimits<Label>::maximumRange()>
//class RestrictedSetWithoutALL2
//{
//public:
//    static Label NoRegion;
//    static bool noBelong(Label ,Label regionpos )
//    {
//        if(regionpos==NoRegion)
//            return true;
//        else
//            return false;
//    }
//    static bool noBelong(Label regionpos )
//    {
//        if(regionpos==NoRegion)
//            return true;
//        else
//            return false;
//    }
//};
//template<typename Label,int Dead>
//Label RestrictedSetWithoutALL2<Label,Dead>::NoRegion = Dead;
#endif // REGION_HPP
