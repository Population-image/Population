//###PDE###
//%include "../../../core/algorithm/PDE.h"
//typedef pop::PDE PDE;

namespace pop
{
class  PDE
{
public:
template<typename Function >
static Function 	nonLinearAnisotropicDiffusionDeriche (const Function &f, int Nstep=20, F64 K=50, F64 alpha=1);
template<typename Function >
static Function 	nonLinearAnisotropicDiffusionDericheFast (const Function &in, int Nstep=20, F64 K=50, F64 alpha=1);
template<typename Function >
static Function 	nonLinearAnisotropicDiffusionGaussian (const Function &in, int Nstep=20, F64 K=50, F64 sigma=1);
template<int DIM,typename Type>
static VecF64  ___permeability(const  MatN<DIM,Type>& bulk,  MatN<DIM,VecN<DIM,F64> >& velocity, int direction=0,F64 errorpressuremax=0.01  );
template<typename Function >
static Mat2F64 	randomWalk (const Function &bulk, int nbrwalkers=50000, F64 standard_deviation=0.5, F64 time_max=2000, F64 delta_time_write=10);
};
}

%template(nonLinearAnisotropicDiffusionDeriche) pop::PDE::nonLinearAnisotropicDiffusionDeriche<pop::Mat2UI8>;
%template(nonLinearAnisotropicDiffusionDeriche) pop::PDE::nonLinearAnisotropicDiffusionDeriche<pop::Mat3UI8>;
%template(nonLinearAnisotropicDiffusionDericheFast) pop::PDE::nonLinearAnisotropicDiffusionDericheFast<pop::Mat2UI8>;
%template(nonLinearAnisotropicDiffusionDericheFast) pop::PDE::nonLinearAnisotropicDiffusionDericheFast<pop::Mat3UI8>;
%template(nonLinearAnisotropicDiffusionGaussian) pop::PDE::nonLinearAnisotropicDiffusionGaussian<pop::Mat2UI8>;
%template(nonLinearAnisotropicDiffusionGaussian) pop::PDE::nonLinearAnisotropicDiffusionGaussian<pop::Mat3UI8>;

%template(randomWalk) pop::PDE::randomWalk<pop::Mat2UI8>;
%template(randomWalk) pop::PDE::randomWalk<pop::Mat3UI8>;
%template(permeability) pop::PDE::___permeability<2,pop::UI8>;
%template(permeability) pop::PDE::___permeability<3,pop::UI8>;

/*%template(nonLinearAnisotropicDiffusionDeriche) pop::PDE::nonLinearAnisotropicDiffusionDeriche<pop::Mat2RGBUI8>;
%template(nonLinearAnisotropicDiffusionDeriche) pop::PDE::nonLinearAnisotropicDiffusionDeriche<pop::Mat3RGBUI8>;
%template(nonLinearAnisotropicDiffusionDericheFast) pop::PDE::nonLinearAnisotropicDiffusionDericheFast<pop::Mat2RGBUI8>;
%template(nonLinearAnisotropicDiffusionDericheFast) pop::PDE::nonLinearAnisotropicDiffusionDericheFast<pop::Mat3RGBUI8>;
%template(nonLinearAnisotropicDiffusionGaussian) pop::PDE::nonLinearAnisotropicDiffusionGaussian<pop::Mat2RGBUI8>;
%template(nonLinearAnisotropicDiffusionGaussian) pop::PDE::nonLinearAnisotropicDiffusionGaussian<pop::Mat3RGBUI8>;*/
