//###PDE###
%include "../../include/algorithm/PDE.h"
typedef pop::PDE PDE;


ALL_IMAGE2_DIM_TYPE(PDE,nonLinearAnisotropicDiffusion)

%template(randomWalk) pop::PDE::randomWalk<2>;
%template(randomWalk) pop::PDE::randomWalk<3>;

%template(permeability) pop::PDE::permeability<2>;
%template(permeability) pop::PDE::permeability<3>;

