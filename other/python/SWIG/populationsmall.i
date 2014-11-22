
%include "carrays.i"


%array_class(pop::UI8, ArrayUI8);
%array_class(pop::UI16, ArrayUI16);
%array_class(pop::UI32, ArrayUI32);
%array_class(pop::F64, ArrayF64);
%array_class(pop::RGBUI8, ArrayRGBUI8);
%array_class(pop::RGBF64, ArrayRGBF64);
%array_class(pop::ComplexF64, ArrayComplexF64);
%array_class(pop::Vec2F64, ArrayVec2F64);
//###basic types###
%include "../../../core/data/typeF/TypeF.h"
namespace pop{
typedef unsigned char UI8;      /* 8 bit unsigned */
typedef signed char I8;         /* 8 bit signed */
typedef unsigned short UI16;    /* 16 bit unsigned */
typedef short I16;              /* 16 bit signed */
typedef unsigned int UI32;      /* 32 bit unsigned */
typedef int I32;                /* 32 bit signed */
typedef float F32;
typedef double F64 ;
typedef long double F128;
}

namespace std
{

template<typename T>
struct numeric_limits;
}
namespace pop
{
template<typename Function, typename F>
struct FunctionTypeTraitsSubstituteF;
template< typename R, typename T>
struct ArithmeticsSaturation
{
    static R Range(T p)
    {
        if(p>=std::numeric_limits<R>::max())return std::numeric_limits<R>::max();
        else if(p<numeric_limits_perso<R>::min())return numeric_limits_perso<R>::min();
        else return static_cast<R>(p);
    }
};
template<typename F>
struct isVectoriel{
    enum { value =false};
};
}
//###RGB###
%include "../../../core/data/typeF/RGB.h"
%template(RGBUI8) pop::RGB<pop::UI8>;
typedef pop::RGB<pop::UI8> RGBUI8;



//###RGB###
%include "../../../core/data/typeF/Complex.h"
%template(ComplexF64) pop::Complex<pop::F64>;

//###VecN###
%include "../../../core/data/vec/VecN.h"

%template(Vec2I32) pop::VecN<2,pop::I32>;
%template(Vec3I32) pop::VecN<3,pop::I32>;
%template(Vec2F64) pop::VecN<2,pop::F64>;
%template(Vec3F64) pop::VecN<3,pop::F64>;



%include "../../../core/data/vec/Vec.h"
%template(VecI32) pop::Vec<pop::I32>;
%template(VecF64) pop::Vec<pop::F64>;
//### MatN###
%include "../../../core/data/mat/MatNBoundaryCondition.h"
%include "../../../core/data/mat/MatNIteratorE.h"
%include "../../../core/data/mat/MatN.h"
%include "../../../core/data/mat/Mat2x.h"
typedef pop::MatNIteratorEDomain<pop::Vec2I32>    Mat2IteratorEDomain;
%template(Mat2IteratorEDomain) pop::MatNIteratorEDomain<pop::Vec2I32>;
typedef pop::MatNIteratorEDomain<pop::Vec3I32>    Mat3IteratorEDomain;
%template(Mat3IteratorEDomain) pop::MatNIteratorEDomain<pop::Vec3I32>;

typedef pop::MatNIteratorENeighborhood<pop::Vec2I32,pop::BoundaryConditionBounded>    Mat2IteratorENeighborhood;
%template(Mat2IteratorENeighborhood) pop::MatNIteratorENeighborhood<pop::Vec2I32,pop::BoundaryConditionBounded>;
typedef pop::MatNIteratorENeighborhood<pop::Vec3I32,pop::BoundaryConditionBounded>    Mat3IteratorENeighborhood;
%template(Mat3IteratorENeighborhood) pop::MatNIteratorENeighborhood<pop::Vec3I32,pop::BoundaryConditionBounded>;




%template(Mat2UI8) pop::MatN<2,pop::UI8>;
%template(Mat2RGBUI8) pop::MatN<2,pop::RGBUI8>;
%include randomgeometry.i
