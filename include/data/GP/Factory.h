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
#ifndef FACTORY_H
#define FACTORY_H
#include<map>
#include<string>
#include<vector>
#include"data/utility/Exception.h"
#include"data/GP/NullType.h"
#include"data/GP/Typelist.h"
#include"data/GP/TypelistMacros.h"
#include"data/GP/TypeTraitsTemplateTemplate.h"
#include"data/GP/Type2Id.h"
namespace pop
{
namespace Details {
template<typename Product>
struct CreatorMemberClone
{
    Product * createObject(Product * creator){
        return creator->clone();
    }
};
template<typename Product, typename Key>
struct RegisterInFactoryInInitiatilzation
{
    std::vector<std::pair<Key,Product*> > Register(){
        return std::vector<std::pair<Key,Product*> >();
    }
};
}
template
<
        class Product,
        typename Key,
        template <class> class CreatorMemberPolicy =Details::CreatorMemberClone,
        template <class,class> class RegisterData = Details::RegisterInFactoryInInitiatilzation
        >
class Factory : public CreatorMemberPolicy<Product>, RegisterData<Product,Key>
{
protected:
    std::map<Key,Product *> _map;
public:
    Factory()
    {
        std::vector<std::pair<Key,Product*> > vec= RegisterData<Product,Key>::Register();
        for(unsigned int i=0;i<vec.size();i++){
            Register(vec[i].first,vec[i].second);
        }
    }

    virtual ~Factory()
    {
        typename std::map<Key,Product *>::iterator it;
        for ( it=_map.begin() ; it != _map.end(); it++ )
            delete it->second;
        _map.clear();
    }
    bool Register(Key key, Product * productcreator)
    {
        _map[key]=productcreator;
        return true;
    }
    void Unregister(Key key)
    {
        _map.erase (key);
    }

    Product * createObject(Key key) throw(pexception)
    {
        if(_map.find(key)!=_map.end()){
            Product * p =    _map[key];
            return CreatorMemberPolicy<Product>::createObject(p);
        }
        else
        {
            throw(pexception("Factory: Unknown key\n"));
        }
        return CreatorMemberPolicy<Product>::createObject((_map.begin())->second);
    }
    template<typename Parm1>
    Product * createObject(Key key,Parm1 parm1) throw(pexception)
    {
        if(_map.find(key)!=_map.end())
            return CreatorMemberPolicy<Product>::createObject(_map[key],parm1);
        else
        {
            throw(pexception("Factory: Unknown key\n"));
        }
        return CreatorMemberPolicy<Product>::createObject((_map.begin())->second,parm1);
    }
    template<typename Parm1, typename Parm2>
    Product * createObject(Key key, Parm1 parm1, Parm2 parm2) throw(pexception)
    {
        if(_map.find(key)!=_map.end())
            return CreatorMemberPolicy<Product>::createObject(_map[key],parm1,parm2);
        else
        {
            throw(pexception("Factory: Unknown key\n"));
        }
        return CreatorMemberPolicy<Product>::createObject((_map.begin())->second,parm1,parm2);
    }
    //    template<typename Parm1, typename Parm2, typename Parm3>
    //    virtual Product * createObject(Key key, Parm1 parm1, Parm2 parm2, Parm3 parm3)
    //    {
    //        if(_map.find(key)!=_map.end())
    //            return CreatorMemberPolicy<Product>::createObject(_map[key],parm1,parm2,parm3);
    //        else
    //           throw(pexception("Factory: Unknown key\n"));
    //    }
    //    template<typename Parm1, typename Parm2, typename Parm3, typename Parm4>
    //    virtual Product * createObject(Key key, Parm1 parm1, Parm2 parm2, Parm3 parm3, Parm4 parm4)
    //    {
    //        if(_map.find(key)!=_map.end())
    //            return CreatorMemberPolicy<Product>::createObject(_map[key],parm1,parm2,parm3,parm4);
    //        else
    //           throw(pexception("Factory: Unknown key\n"));
    //    }
    //    template<typename Parm1, typename Parm2, typename Parm3, typename Parm4, typename Parm5>
    //    virtual Product * createObject(Key key, Parm1 parm1, Parm2 parm2, Parm3 parm3, Parm4 parm4, Parm5 parm5)
    //    {
    //        if(_map.find(key)!=_map.end())
    //            return CreatorMemberPolicy<Product>::createObject(_map[key],parm1,parm2,parm3,parm4,parm5);
    //        else
    //           throw(pexception("Factory: Unknown key\n"));
    //    }
};

template<typename T>
struct GPFactoryRegister;
template <>
struct GPFactoryRegister<Loki::NullType>
{

    template<typename Factory, typename SubClassBlank>
    static bool Register( Factory & , Loki::Type2Type<SubClassBlank>)
    {
        return true;
    }
};
template <class Head, class Tail>
struct GPFactoryRegister<Loki::Typelist< Head, Tail> >
{
    template<typename Factory, typename SubClassBlank>
    static bool  Register(Factory & factory, Loki::Type2Type<SubClassBlank> t)
    {
        typedef typename SubstituteTemplateParameter<Head,SubClassBlank>::Result SubClass;
        Type2Id<SubClass> type;
        for(unsigned int i=0;i<type.id.size();i++)
        {
            SubClass * c = new SubClass();
            factory.Register( type.id[i],c);
        }
        return GPFactoryRegister<Tail>::Register(factory,t);
    }
};
}
#endif // FACTORY_H
