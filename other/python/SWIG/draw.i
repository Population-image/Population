%include "../../../include/algorithm/Draw.h"
typedef pop::Draw Draw;

%template(insertMatrix) pop::Draw::insertMatrix<pop::Mat2UI8,pop::Mat2UI8>;
%template(insertMatrix) pop::Draw::insertMatrix<pop::Mat2RGBUI8,pop::Mat2UI8>;
%template(insertMatrix) pop::Draw::insertMatrix<pop::Mat2RGBUI8,pop::Mat2RGBUI8>;
ALL_IMAGE(Draw,setFace)
ALL_IMAGE(Draw,setBorder)
ALL_IMAGE(Draw,addBorder)
%template(circle) pop::Draw::circle<2,pop::UI8,pop::I32>;
%template(circle) pop::Draw::circle<2,pop::UI8,pop::F64>;
%template(circle) pop::Draw::circle<2,pop::RGBUI8,pop::I32>;
%template(circle) pop::Draw::circle<2,pop::RGBUI8,pop::F64>;

%template(disk) pop::Draw::disk<2,pop::UI8,pop::I32>;
%template(disk) pop::Draw::disk<2,pop::UI8,pop::F64>;
%template(disk) pop::Draw::disk<2,pop::RGBUI8,pop::I32>;
%template(disk) pop::Draw::disk<2,pop::RGBUI8,pop::F64>;



ALL_IMAGE(Draw,rectangle)
ALL_IMAGE(Draw,line)
ALL_IMAGE(Draw,arrow)
ALL_IMAGE(Draw,mergeTwoMatrixHorizontal)
ALL_IMAGE(Draw,mergeTwoMatrixVertical)
ALL_IMAGE(Draw,text)
