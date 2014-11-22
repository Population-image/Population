//###PDE###
%include "../../include/algorithm/PDE.h"
typedef pop::PDE PDE;

ALL_IMAGE2_DIM_TYPE(PDE,nonLinearAnisotropicDiffusionDeriche)
ALL_IMAGE2_DIM_TYPE(PDE,nonLinearAnisotropicDiffusionDericheFast)
ALL_IMAGE2_DIM_TYPE(PDE,nonLinearAnisotropicDiffusionGaussian)


%template(randomWalk) pop::PDE::randomWalk<2>;
%template(randomWalk) pop::PDE::randomWalk<3>;

%template(permeability) pop::PDE::permeability<2>;
%template(permeability) pop::PDE::permeability<3>;

/*%template(nonLinearAnisotropicDiffusionDeriche) pop::PDE::nonLinearAnisotropicDiffusionDeriche<pop::Mat2RGBUI8>;
%template(nonLinearAnisotropicDiffusionDeriche) pop::PDE::nonLinearAnisotropicDiffusionDeriche<pop::Mat3RGBUI8>;
%template(nonLinearAnisotropicDiffusionDericheFast) pop::PDE::nonLinearAnisotropicDiffusionDericheFast<pop::Mat2RGBUI8>;
%template(nonLinearAnisotropicDiffusionDericheFast) pop::PDE::nonLinearAnisotropicDiffusionDericheFast<pop::Mat3RGBUI8>;
%template(nonLinearAnisotropicDiffusionGaussian) pop::PDE::nonLinearAnisotropicDiffusionGaussian<pop::Mat2RGBUI8>;
%template(nonLinearAnisotropicDiffusionGaussian) pop::PDE::nonLinearAnisotropicDiffusionGaussian<pop::Mat3RGBUI8>;*/
