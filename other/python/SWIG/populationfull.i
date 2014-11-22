%include "data.i"


//#####################Algorithm ###################
%define ALL_IMAGE(CLASS,METHOD)
%template(METHOD) pop::CLASS::METHOD<pop::Mat2UI8>;
%template(METHOD) pop::CLASS::METHOD<pop::Mat2F64>;
%template(METHOD) pop::CLASS::METHOD<pop::Mat2RGBUI8>;
%template(METHOD) pop::CLASS::METHOD<pop::Mat2UI16>;
%template(METHOD) pop::CLASS::METHOD<pop::Mat2UI32>;
%template(METHOD) pop::CLASS::METHOD<pop::Mat3UI8>;
%template(METHOD) pop::CLASS::METHOD<pop::Mat3F64>;
%template(METHOD) pop::CLASS::METHOD<pop::Mat3RGBUI8>;
%template(METHOD) pop::CLASS::METHOD<pop::Mat3UI16>;
%template(METHOD) pop::CLASS::METHOD<pop::Mat3UI32>;
%enddef
%define ALL_IMAGE_SCALAR(CLASS,METHOD)
%template(METHOD) pop::CLASS::METHOD<pop::Mat2UI8>;
%template(METHOD) pop::CLASS::METHOD<pop::Mat2F64>;
%template(METHOD) pop::CLASS::METHOD<pop::Mat2UI16>;
%template(METHOD) pop::CLASS::METHOD<pop::Mat2UI32>;
%template(METHOD) pop::CLASS::METHOD<pop::Mat3UI8>;
%template(METHOD) pop::CLASS::METHOD<pop::Mat3F64>;
%template(METHOD) pop::CLASS::METHOD<pop::Mat3UI16>;
%template(METHOD) pop::CLASS::METHOD<pop::Mat3UI32>;
%enddef
%define ALL_IMAGE_SCALAR_SWIG(CLASS,METHODSWIG,METHODNAME)
%template(METHODNAME) pop::CLASS::METHODSWIG<pop::Mat2UI8>;
%template(METHODNAME) pop::CLASS::METHODSWIG<pop::Mat2F64>;
%template(METHODNAME) pop::CLASS::METHODSWIG<pop::Mat2UI16>;
%template(METHODNAME) pop::CLASS::METHODSWIG<pop::Mat2UI32>;
%template(METHODNAME) pop::CLASS::METHODSWIG<pop::Mat3UI8>;
%template(METHODNAME) pop::CLASS::METHODSWIG<pop::Mat3F64>;
%template(METHODNAME) pop::CLASS::METHODSWIG<pop::Mat3UI16>;
%template(METHODNAME) pop::CLASS::METHODSWIG<pop::Mat3UI32>;
%enddef
%define ALL_IMAGE_UNINT(CLASS,METHOD)
%template(METHOD) pop::CLASS::METHOD<pop::Mat2UI8>;
%template(METHOD) pop::CLASS::METHOD<pop::Mat2UI16>;
%template(METHOD) pop::CLASS::METHOD<pop::Mat2UI32>;
%template(METHOD) pop::CLASS::METHOD<pop::Mat3UI8>;
%template(METHOD) pop::CLASS::METHOD<pop::Mat3UI16>;
%template(METHOD) pop::CLASS::METHOD<pop::Mat3UI32>;
%enddef

%define ALL_IMAGE_UNINT_BINARY(CLASS,METHOD)
%template(METHOD) pop::CLASS::METHOD<pop::Mat2UI8,pop::Mat2UI8>;
%template(METHOD) pop::CLASS::METHOD<pop::Mat2UI16,pop::Mat2UI8>;
%template(METHOD) pop::CLASS::METHOD<pop::Mat2UI32,pop::Mat2UI8>;
%template(METHOD) pop::CLASS::METHOD<pop::Mat3UI8,pop::Mat3UI8>;
%template(METHOD) pop::CLASS::METHOD<pop::Mat3UI16,pop::Mat3UI8>;
%template(METHOD) pop::CLASS::METHOD<pop::Mat3UI32,pop::Mat3UI8>;
%enddef

%define ALL_IMAGE_DIM_TYPE(CLASS,METHODSWIG,METHOD)
%template(METHOD) pop::CLASS::METHODSWIG<2,pop::UI8>;
%template(METHOD) pop::CLASS::METHODSWIG<2,pop::UI16>;
%template(METHOD) pop::CLASS::METHODSWIG<2,pop::UI32>;
%template(METHOD) pop::CLASS::METHODSWIG<2,pop::F64>;
%template(METHOD) pop::CLASS::METHODSWIG<2,pop::RGBUI8>;
%template(METHOD) pop::CLASS::METHODSWIG<3,pop::UI8>;
%template(METHOD) pop::CLASS::METHODSWIG<3,pop::UI16>;
%template(METHOD) pop::CLASS::METHODSWIG<3,pop::UI32>;
%template(METHOD) pop::CLASS::METHODSWIG<3,pop::F64>;
%template(METHOD) pop::CLASS::METHODSWIG<3,pop::RGBUI8>;
%enddef


%define ALL_IMAGE_UINT_TYPE(CLASS,METHODSWIG,METHOD)
%template(METHOD) pop::CLASS::METHODSWIG<2,pop::UI8>;
%template(METHOD) pop::CLASS::METHODSWIG<2,pop::UI16>;
%template(METHOD) pop::CLASS::METHODSWIG<2,pop::UI32>;
%template(METHOD) pop::CLASS::METHODSWIG<3,pop::UI8>;
%template(METHOD) pop::CLASS::METHODSWIG<3,pop::UI16>;
%template(METHOD) pop::CLASS::METHODSWIG<3,pop::UI32>;
%enddef

%define ALL_IMAGE_BINARY(CLASS,METHOD)
%template(METHOD) pop::CLASS::METHOD<pop::Mat2UI8,pop::Mat2UI8>;
%template(METHOD) pop::CLASS::METHOD<pop::Mat2F64,pop::Mat2UI8>;
%template(METHOD) pop::CLASS::METHOD<pop::Mat2RGBUI8,pop::Mat2UI8>;
%template(METHOD) pop::CLASS::METHOD<pop::Mat2UI32,pop::Mat2UI8>;
%template(METHOD) pop::CLASS::METHOD<pop::Mat3UI8,pop::Mat3UI8>;
%template(METHOD) pop::CLASS::METHOD<pop::Mat3F64,pop::Mat3UI8>;
%template(METHOD) pop::CLASS::METHOD<pop::Mat3RGBUI8,pop::Mat3UI8>;
%template(METHOD) pop::CLASS::METHOD<pop::Mat3UI32,pop::Mat3UI8>;
%enddef
%define ALL_IMAGE_FLOAT(CLASS,METHOD)
%template(METHOD) pop::CLASS::METHOD<pop::Mat2UI8,pop::Mat2F64>;
%template(METHOD) pop::CLASS::METHOD<pop::Mat2F64,pop::Mat2F64>;
%template(METHOD) pop::CLASS::METHOD<pop::Mat2RGBUI8,pop::Mat2F64>;
%template(METHOD) pop::CLASS::METHOD<pop::Mat2UI32,pop::Mat2F64>;
%template(METHOD) pop::CLASS::METHOD<pop::Mat3UI8,pop::Mat3F64>;
%template(METHOD) pop::CLASS::METHOD<pop::Mat3F64,pop::Mat3F64>;
%template(METHOD) pop::CLASS::METHOD<pop::Mat3RGBUI8,pop::Mat3F64>;
%template(METHOD) pop::CLASS::METHOD<pop::Mat3UI32,pop::Mat3F64>;
%enddef


%define ALL_IMAGE_LABEL(CLASS,METHOD)
%template(METHOD) pop::CLASS::METHOD<pop::Mat2UI8,pop::Mat2UI8>;
%template(METHOD) pop::CLASS::METHOD<pop::Mat2UI8,pop::Mat2UI16>;
%template(METHOD) pop::CLASS::METHOD<pop::Mat2UI8,pop::Mat2UI32>;
%template(METHOD) pop::CLASS::METHOD<pop::Mat2UI16,pop::Mat2UI8>;
%template(METHOD) pop::CLASS::METHOD<pop::Mat2UI16,pop::Mat2UI16>;
%template(METHOD) pop::CLASS::METHOD<pop::Mat2UI16,pop::Mat2UI32>;
%template(METHOD) pop::CLASS::METHOD<pop::Mat2UI32,pop::Mat2UI8>;
%template(METHOD) pop::CLASS::METHOD<pop::Mat2UI32,pop::Mat2UI16>;
%template(METHOD) pop::CLASS::METHOD<pop::Mat2UI32,pop::Mat2UI32>;
%template(METHOD) pop::CLASS::METHOD<pop::Mat3UI8,pop::Mat3UI8>;
%template(METHOD) pop::CLASS::METHOD<pop::Mat3UI8,pop::Mat3UI16>;
%template(METHOD) pop::CLASS::METHOD<pop::Mat3UI8,pop::Mat3UI32>;
%template(METHOD) pop::CLASS::METHOD<pop::Mat3UI16,pop::Mat3UI8>;
%template(METHOD) pop::CLASS::METHOD<pop::Mat3UI16,pop::Mat3UI16>;
%template(METHOD) pop::CLASS::METHOD<pop::Mat3UI16,pop::Mat3UI32>;
%template(METHOD) pop::CLASS::METHOD<pop::Mat3UI32,pop::Mat3UI8>;
%template(METHOD) pop::CLASS::METHOD<pop::Mat3UI32,pop::Mat3UI16>;
%template(METHOD) pop::CLASS::METHOD<pop::Mat3UI32,pop::Mat3UI32>;
%enddef

%define ALL_IMAGE_LABEL_MASK(CLASS,METHOD)
%template(METHOD) pop::CLASS::METHOD<pop::Mat2UI8,pop::Mat2UI8,pop::Mat2UI8>;
%template(METHOD) pop::CLASS::METHOD<pop::Mat2UI8,pop::Mat2UI16,pop::Mat2UI8>;
%template(METHOD) pop::CLASS::METHOD<pop::Mat2UI8,pop::Mat2UI32,pop::Mat2UI8>;
%template(METHOD) pop::CLASS::METHOD<pop::Mat2UI16,pop::Mat2UI8,pop::Mat2UI8>;
%template(METHOD) pop::CLASS::METHOD<pop::Mat2UI16,pop::Mat2UI16,pop::Mat2UI8>;
%template(METHOD) pop::CLASS::METHOD<pop::Mat2UI16,pop::Mat2UI32,pop::Mat2UI8>;
%template(METHOD) pop::CLASS::METHOD<pop::Mat2UI32,pop::Mat2UI8,pop::Mat2UI8>;
%template(METHOD) pop::CLASS::METHOD<pop::Mat2UI32,pop::Mat2UI16,pop::Mat2UI8>;
%template(METHOD) pop::CLASS::METHOD<pop::Mat2UI32,pop::Mat2UI32,pop::Mat2UI8>;
%template(METHOD) pop::CLASS::METHOD<pop::Mat3UI8,pop::Mat3UI8,pop::Mat3UI8>;
%template(METHOD) pop::CLASS::METHOD<pop::Mat3UI8,pop::Mat3UI16,pop::Mat3UI8>;
%template(METHOD) pop::CLASS::METHOD<pop::Mat3UI8,pop::Mat3UI32,pop::Mat3UI8>;
%template(METHOD) pop::CLASS::METHOD<pop::Mat3UI16,pop::Mat3UI8,pop::Mat3UI8>;
%template(METHOD) pop::CLASS::METHOD<pop::Mat3UI16,pop::Mat3UI16,pop::Mat3UI8>;
%template(METHOD) pop::CLASS::METHOD<pop::Mat3UI16,pop::Mat3UI32,pop::Mat3UI8>;
%template(METHOD) pop::CLASS::METHOD<pop::Mat3UI32,pop::Mat3UI8,pop::Mat3UI8>;
%template(METHOD) pop::CLASS::METHOD<pop::Mat3UI32,pop::Mat3UI16,pop::Mat3UI8>;
%template(METHOD) pop::CLASS::METHOD<pop::Mat3UI32,pop::Mat3UI32,pop::Mat3UI8>;
%enddef




%include randomgeometry.i
%include processing.i
%include PDE.i
%include visualization.i
%include analysis.i
%include "../../../include/algorithm/Statistics.h"
%include convertor.i
%include "../../../include/algorithm/LinearAlgebra.h"
%include representation.i
%include draw.i
%include "../../../include/algorithm/Application.h"
%template(thresholdSelection) pop::Application::thresholdSelection<pop::Mat2UI8>;
%template(thresholdSelection) pop::Application::thresholdSelection<pop::Mat3UI8>;
%include geometricaltransformation.i
#%include vision.i
