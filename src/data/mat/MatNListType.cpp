#include"data/mat/MatN.h"
#include"data/mat/MatNListType.h"

namespace pop{


//1D
//template<>
Type2Id<MatN<1,UI8> >::Type2Id(){id.push_back("Pe");id.push_back("Pf");}
//template<>
Type2Id<MatN<1,UI16> >::Type2Id(){id.push_back("Pg");id.push_back("Ph");}
//template<>
Type2Id<MatN<1,UI32> >::Type2Id(){id.push_back("Pi");id.push_back("Pj");}
//template<>
Type2Id<MatN<1,F64> >::Type2Id(){id.push_back("Pk");id.push_back("Pm");}
//template<>
Type2Id<MatN<1,RGBUI8 > >::Type2Id(){id.push_back("Pn");id.push_back("Po");}
//template<>
Type2Id<MatN<1,RGBAUI8 > >::Type2Id(){id.push_back("Pp");id.push_back("Pq");}
//template<>
Type2Id<MatN<1,RGBF64 > >::Type2Id(){id.push_back("Pr");id.push_back("Ps");}
//template<>
Type2Id<MatN<1,ComplexF64 > >::Type2Id(){id.push_back("Pt");id.push_back("Pu");}
//template<>
Type2Id<MatN<1,VecN<1,F64> > >::Type2Id(){id.push_back("Pv");id.push_back("Pw");}

//2D
//template<>
Type2Id<MatN<2,UI8> >::Type2Id(){id.push_back("P2");id.push_back("P5");}
//template<>
Type2Id<MatN<2,UI16> >::Type2Id(){id.push_back("P3");id.push_back("PA");}
//template<>
Type2Id<MatN<2,UI32> >::Type2Id(){id.push_back("P4");id.push_back("PB");}
//template<>
Type2Id<MatN<2,F64> >::Type2Id(){id.push_back("P7");id.push_back("PC");}
//template<>
Type2Id<MatN<2,RGBUI8 > >::Type2Id(){id.push_back("P8");id.push_back("P6");}
//template<>
Type2Id<MatN<2,RGBAUI8 > >::Type2Id(){id.push_back("PG");id.push_back("PH");}
//template<>
Type2Id<MatN<2,RGBF64 > >::Type2Id(){id.push_back("PE");id.push_back("PF");}
//template<>
Type2Id<MatN<2,ComplexF64 > >::Type2Id(){id.push_back("PI");id.push_back("PJ");}
//template<>
Type2Id<MatN<2,Vec2F64 > >::Type2Id(){id.push_back("Pa");id.push_back("Pb");}

//3D
//template<>
Type2Id<MatN<3,UI8> >::Type2Id(){id.push_back("PK");id.push_back("PL");}
//template<>
Type2Id<MatN<3,UI16> >::Type2Id(){id.push_back("PM");id.push_back("PN");}
//template<>
Type2Id<MatN<3,UI32> >::Type2Id(){id.push_back("PO");id.push_back("PP");}
//template<>
Type2Id<MatN<3,F64> >::Type2Id(){id.push_back("PQ");id.push_back("PR");}
//template<>
Type2Id<MatN<3,RGBUI8 > >::Type2Id(){id.push_back("PS");id.push_back("PT");}
//template<>
Type2Id<MatN<3,RGBAUI8 > >::Type2Id(){id.push_back("PX");id.push_back("PY");}
//template<>
Type2Id<MatN<3,RGBF64 > >::Type2Id(){id.push_back("PU");id.push_back("PV");}
//template<>
Type2Id<MatN<3,ComplexF64 > >::Type2Id(){id.push_back("PZ");id.push_back("P1");}
//template<>
Type2Id<MatN<3,Vec3F64 > >::Type2Id(){id.push_back("Pc");id.push_back("Pd");}


//const bool _registerMatNsingleton = GPFactoryRegister<_TListImgGrid>::Register(*SingletonFactoryMatN::getInstance(),Loki::Type2Type<MatN<2,F64> >());
}
