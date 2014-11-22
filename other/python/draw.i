%include "../../include/algorithm/Draw.h"
typedef pop::Draw Draw;

ALL_IMAGE2_DIM_TYPE(Draw,circle)
ALL_IMAGE2_DIM_TYPE(Draw,disk)
ALL_IMAGE2_DIM_TYPE(Draw,rectangle)
ALL_IMAGE2_DIM_TYPE(Draw,polygone)
ALL_IMAGE2_DIM_TYPE(Draw,line)
ALL_IMAGE2_DIM_TYPE(Draw,arrow)

%template(text) pop::Draw::text<pop::UI8>;
%template(text) pop::Draw::text<pop::RGBUI8>;

ALL_IMAGE2_DIM_TYPE(Draw,mergeTwoMatrixHorizontal)
ALL_IMAGE2_DIM_TYPE(Draw,mergeTwoMatrixVertical)

%template(insertMatrix) pop::Draw::insertMatrix<2,pop::UI8,pop::UI8>;
%template(insertMatrix) pop::Draw::insertMatrix<2,pop::RGBUI8,pop::UI8>;

ALL_IMAGE2_DIM_TYPE(Draw,setFace)
ALL_IMAGE2_DIM_TYPE(Draw,setBorder)
ALL_IMAGE2_DIM_TYPE(Draw,addBorder)
%template(axis) pop::Draw::axis<2,pop::UI8>;
%template(axis) pop::Draw::axis<2,pop::RGBUI8>;
%template(distribution) pop::Draw::distribution<2,pop::UI8>;

