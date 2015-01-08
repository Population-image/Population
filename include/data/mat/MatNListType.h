///******************************************************************************\
//|*                   Population library for C++ X.X.X                         *|
//|*----------------------------------------------------------------------------*|
//The Population License is similar to the MIT license in adding this clause:
//for any writing public or private that has resulted from the use of the
//software population, the reference of this book "Population library, 2012,
//Vincent Tariel" shall be included in it.

//So, the terms of the Population License are:

//Copyright © 2012-2015, Tariel Vincent

//Permission is hereby granted, free of charge, to any person obtaining a copy
//of this software and associated documentation files (the "Software"), to
//deal in the Software without restriction, including without limitation the
//rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
//sell copies of the Software, and to permit persons to whom the Software is
//furnished to do so, subject to the following conditions:

//The above copyright notice and this permission notice shall be included in
//all copies or substantial portions of the Software and for any writing
//public or private that has resulted from the use of the software population,
//the reference of this book "Population library, 2012, Vincent Tariel" shall
//be included in it.

//The Software is provided "as is", without warranty of any kind, express or
//implied, including but not limited to the warranties of merchantability,
//fitness for a particular purpose and noninfringement. In no event shall the
//authors or copyright holders be liable for any claim, damages or other
//liability, whether in an action of contract, tort or otherwise, arising
//from, out of or in connection with the software or the use or other dealings
//in the Software.
//\***************************************************************************/

//#ifndef FUNCTIONMatNLISTTYPE_HPP
//#define FUNCTIONMatNLISTTYPE_HPP
//#include<iostream>
//#include"data/typeF/TypeF.h"
//#include"data/GP/CartesianProduct.h"
//#include"data/GP/TypeTraitsTemplateTemplate.h"
//#include"data/GP/Type2Id.h"
//#include"data/mat/MatN.h"
//namespace pop{
//namespace Details{
//template<>
//struct POP_EXPORTS Type2Id<pop::MatN<2,UI8> >
//{
//    Type2Id();
//    std::vector<std::string> id;
//};

//template<>
//struct POP_EXPORTS Type2Id<pop::MatN<2,UI16> >
//{
//    Type2Id();
//    std::vector<std::string> id;
//};

//template<>
//struct POP_EXPORTS Type2Id<pop::MatN<2,UI32> >
//{
//    Type2Id();
//    std::vector<std::string> id;
//};

//template<>
//struct POP_EXPORTS Type2Id<pop::MatN<2,F64> >
//{
//    Type2Id();
//    std::vector<std::string> id;
//};

//template<>
//struct POP_EXPORTS Type2Id<pop::MatN<2,RGBUI8>  >
//{
//    Type2Id();
//    std::vector<std::string> id;
//};
//template<>
//struct POP_EXPORTS Type2Id<pop::MatN<2,RGBF64> >
//{
//    Type2Id();
//    std::vector<std::string> id;
//};
////template<>
////struct POP_EXPORTS Type2Id<pop::MatN<2,RGBAUI8>  >
////{
////    Type2Id();
////    std::vector<std::string> id;
////};

//template<>
//struct POP_EXPORTS Type2Id<pop::MatN<2,ComplexF64> >
//{
//    Type2Id();
//    std::vector<std::string> id;
//};
//template<>
//struct POP_EXPORTS Type2Id<pop::MatN<2,Vec2F64 > >
//{

//        Type2Id();
//    std::vector<std::string> id;
//};
//template<>
//struct POP_EXPORTS Type2Id<pop::MatN<3,UI8> >
//{

//        Type2Id();
//    std::vector<std::string> id;
//};

//template<>
//struct POP_EXPORTS Type2Id<pop::MatN<3,UI16> >
//{

//        Type2Id();
//    std::vector<std::string> id;
//};

//template<>
//struct POP_EXPORTS Type2Id<pop::MatN<3,UI32> >
//{

//        Type2Id();
//    std::vector<std::string> id;
//};

//template<>
//struct POP_EXPORTS Type2Id<pop::MatN<3,F64> >
//{

//        Type2Id();
//    std::vector<std::string> id;
//};

//template<>
//struct POP_EXPORTS Type2Id<pop::MatN<3,RGBUI8> >
//{

//        Type2Id();
//    std::vector<std::string> id;
//};
////template<>
////struct POP_EXPORTS Type2Id<pop::MatN<3,RGBAUI8> >
////{

////        Type2Id();
////    std::vector<std::string> id;
////};
//template<>
//struct POP_EXPORTS Type2Id<pop::MatN<3,RGBF64> >
//{

//        Type2Id();
//    std::vector<std::string> id;
//};
//template<>
//struct POP_EXPORTS Type2Id<pop::MatN<3,ComplexF64 > >
//{

//        Type2Id();
//    std::vector<std::string> id;
//};
//template<>
//struct POP_EXPORTS Type2Id<pop::MatN<3,Vec3F64 > >
//{

//        Type2Id();
//    std::vector<std::string> id;
//};

////1D
//template<>
//struct POP_EXPORTS Type2Id<pop::MatN<1,UI8> >
//{

//        Type2Id();
//    std::vector<std::string> id;
//};

//template<>
//struct POP_EXPORTS Type2Id<pop::MatN<1,UI16> >
//{

//        Type2Id();
//    std::vector<std::string> id;
//};

//template<>
//struct POP_EXPORTS Type2Id<pop::MatN<1,UI32> >
//{

//        Type2Id();
//    std::vector<std::string> id;
//};

//template<>
//struct POP_EXPORTS Type2Id<pop::MatN<1,F64> >
//{

//        Type2Id();
//    std::vector<std::string> id;
//};

//template<>
//struct POP_EXPORTS Type2Id<pop::MatN<1,RGBUI8> >
//{

//        Type2Id();
//    std::vector<std::string> id;
//};
////template<>
////struct POP_EXPORTS Type2Id<pop::MatN<1,RGBAUI8> >
////{

////        Type2Id();
////    std::vector<std::string> id;
////};
//template<>
//struct POP_EXPORTS Type2Id<pop::MatN<1,RGBF64> >
//{

//        Type2Id();
//    std::vector<std::string> id;
//};
//template<>
//struct POP_EXPORTS Type2Id<pop::MatN<1,ComplexF64 > >
//{

//        Type2Id();
//    std::vector<std::string> id;
//};
//template<>
//struct POP_EXPORTS Type2Id<pop::MatN<1,VecN<1,F64> > >
//{

//        Type2Id();
//    std::vector<std::string> id;
//};
//}
//}
//#endif // FUNCTIONMatNLISTTYPE_HPP
