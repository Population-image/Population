%include "carrays.i"
%array_class(pop::UI8, ArrayUI8);
%array_class(pop::UI16, ArrayUI16);
%array_class(pop::UI32, ArrayUI32);
%array_class(pop::F32, ArrayF32);
%array_class(pop::RGBUI8, ArrayRGBUI8);
%array_class(pop::RGBF32, ArrayRGBF32);
%array_class(pop::ComplexF32, ArrayComplexF32);
%array_class(pop::Vec2F32, ArrayVec2F32);

//###basic types###
%include "../../include/data/typeF/TypeF.h"

namespace pop{
typedef unsigned char UI8;      /* 8 bit unsigned */
typedef signed char I8;         /* 8 bit signed */
typedef unsigned short UI16;    /* 16 bit unsigned */
typedef short I16;              /* 16 bit signed */
typedef unsigned int UI32;      /* 32 bit unsigned */
typedef int I32;                /* 32 bit signed */
typedef float F32;
typedef double F64 ;
template <int v>
struct Int2Type
{
    enum { value = v };
};
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
%include "../../include/data/typeF/RGB.h"
%template(RGBUI8) pop::RGB<pop::UI8>;
%template(RGBF32) pop::RGB<pop::F32>;


//###RGB###
%include "../../include/data/typeF/Complex.h"
%template(ComplexF32) pop::Complex<pop::F32>;

//###VecN###
%include "../../include/data/vec/VecN.h"

%template(Vec2I32) pop::VecN<2,pop::I32>;
%template(Vec3I32) pop::VecN<3,pop::I32>;
%template(Vec2F32) pop::VecN<2,pop::F32>;
%template(Vec3F32) pop::VecN<3,pop::F32>;


%include "../../include/data/vec/Vec.h"
%template(VecI32) pop::Vec<pop::I32>;
%template(VecF32) pop::Vec<pop::F32>;


%template(vectorRGBUI8) std::vector<pop::RGB<pop::UI8> >;
%template(vectorRGBF32) std::vector<pop::RGB<pop::F32> >;
%template(vectorComplexF32) std::vector<pop::Complex<pop::F32> >;
%template(vectorVec2F32) std::vector<pop::VecN<2,pop::F32> >;
%template(vectorVec3F32) std::vector<pop::VecN<3,pop::F32> >;


//### MatN###
%include "../../include/data/mat/MatNBoundaryCondition.h"
%include "../../include/data/mat/MatNIteratorE.h"
%include "../../include/data/mat/MatN.h"
%include "../../include/data/mat/Mat2x.h"
%template(Mat2IteratorEDomain) pop::MatNIteratorEDomain<pop::Vec2I32>;
%template(Mat3IteratorEDomain) pop::MatNIteratorEDomain<pop::Vec3I32>;
%template(Mat2IteratorENeighborhood) pop::MatNIteratorENeighborhood<pop::Vec2I32,pop::MatNBoundaryConditionBounded>;
%template(Mat3IteratorENeighborhood) pop::MatNIteratorENeighborhood<pop::Vec3I32,pop::MatNBoundaryConditionBounded>;




%template(Mat2UI8) pop::MatN<2,pop::UI8>;
%template(Mat2UI16) pop::MatN<2,pop::UI16>;
%template(Mat2UI32) pop::MatN<2,pop::UI32>;
%template(Mat2F32) pop::MatN<2,pop::F32>;
%template(Mat2RGBUI8) pop::MatN<2,pop::RGBUI8>;
%template(Mat2RGBF32) pop::MatN<2,pop::RGBF32>;
%template(Mat2ComplexF32) pop::MatN<2,pop::ComplexF32>;
%template(Mat2Vec2F32) pop::MatN<2,pop::Vec2F32 >;

%template(Mat3UI8) pop::MatN<3,pop::UI8>;
%template(Mat3F32) pop::MatN<3,pop::F32>;
%template(Mat3RGBUI8) pop::MatN<3,pop::RGBUI8>;
%template(Mat3RGBF32) pop::MatN<3,pop::RGBF32>;
%template(Mat3ComplexF32) pop::MatN<3,pop::ComplexF32>;
%template(Mat3UI16) pop::MatN<3,pop::UI16>;
%template(Mat3UI32) pop::MatN<3,pop::UI32>;
%template(Mat3Vec3F32) pop::MatN<3,pop::Vec3F32 >;

%include"../../include/data/mat/MatNDisplay.h"


%template(Mat2x22F32) pop::Mat2x<pop::F32,2,2>;
%template(Mat2x22ComplexF32) pop::Mat2x<pop::ComplexF32,2,2>;

%template(Mat2x33F32) pop::Mat2x<pop::F32,3,3>;
%template(Mat2x33ComplexF32) pop::Mat2x<pop::ComplexF32,3,3>;




//###Distribution###

%include"../../include/data/distribution/Distribution.h"
%include"../../include/data/distribution/DistributionAnalytic.h"
%include"../../include/data/distribution/DistributionFromDataStructure.h"
%include"../../include/data/distribution/DistributionMultiVariate.h"
%include"../../include/data/distribution/DistributionMultiVariateFromDataStructure.h"
namespace pop{
class  Scene3d
{
public:

    Scene3d& operator =(const Scene3d& g);
    Scene3d& merge(const Scene3d& g);
    void display(bool stopprocess=true);
    Scene3d();
    ~Scene3d();
    void clear();
    void lock();
    void unlock();
    void snapshot(const char * file);
    void rotateX(F32 angle);
    void rotateY(F32 angle);
    void rotateZ(F32 angle);
    void setColorAllGeometricalFigure(const RGBUI8 & value);
    void setTransparencyAllGeometricalFigure(UI8 value);
    void setAmbient(const pop::RGBF32 &ambient);
    pop::RGBF32 getAmbient()const;
    void setDiffuse(const pop::RGBF32 &diffuse);
    pop::RGBF32 getDiffuse()const;
    void setTransparentMode(bool istranspararent);
    bool getTransparentMode()const;

};
template<typename VertexType,typename EdgeType>
class GraphAdjencyList{
public:
    GraphAdjencyList();
};
}

%include"../../include/data/germgrain/Germ.h"
%template(Germ2) pop::Germ<2>;
%template(Germ3) pop::Germ<3>;
%template(OrientationEulerAngle2) pop::OrientationEulerAngle<2>;
%template(OrientationEulerAngle3) pop::OrientationEulerAngle<3>;

%include"../../include/data/germgrain/GermGrain.h"
%template(GrainSphere2) pop::GrainSphere<2>;
%template(GrainSphere3) pop::GrainSphere<3>;
%template(GrainEllipsoid2) pop::GrainEllipsoid<2>;
%template(GrainEllipsoid3) pop::GrainEllipsoid<3>;
%template(GrainBox2) pop::GrainBox<2>;
%template(GrainBox3) pop::GrainBox<3>;
%template(GrainPolyhedra2) pop::GrainPolyhedra<2>;
%template(GrainPolyhedra3) pop::GrainPolyhedra<3>;
%template(ModelGermGrain2) pop::ModelGermGrain<2>;
%template(ModelGermGrain3) pop::ModelGermGrain<3>;
%template(VecGerm2) std::vector<pop::Germ<2>* >;
%template(VecGerm3) std::vector<pop::Germ<3>* >;

%include "../../include/data/video/Video.h"



%template(minimum) pop::minimum<2,pop::UI8>;
%template(minimum) pop::minimum<3,pop::UI8>;
%template(minimum) pop::minimum<2,pop::I32>;
%template(minimum) pop::minimum<3,pop::I32>;
%template(minimum) pop::minimum<2,pop::F32>;
%template(minimum) pop::minimum<3,pop::F32>;

%template(maximum) pop::maximum<2,pop::UI8>;
%template(maximum) pop::maximum<3,pop::UI8>;
%template(maximum) pop::maximum<2,pop::I32>;
%template(maximum) pop::maximum<3,pop::I32>;
%template(maximum) pop::maximum<2,pop::F32>;
%template(maximum) pop::maximum<3,pop::F32>;

%template(normValue) pop::normValue<2,pop::I32>;
%template(normValue) pop::normValue<3,pop::I32>;
%template(normValue) pop::normValue<2,pop::F32>;
%template(normValue) pop::normValue<3,pop::F32>;

%template(round) pop::round<2,pop::F32>;
%template(round) pop::round<3,pop::F32>;

%template(productInner) pop::productInner<2,pop::I32>;
%template(productInner) pop::productInner<3,pop::I32>;
%template(productInner) pop::productInner<2,pop::F32>;
%template(productInner) pop::productInner<3,pop::F32>;

%template(productVectoriel) pop::productVectoriel<pop::F32>;


