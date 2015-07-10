%include "../../include/algorithm/GeometricalTransformation.h"
typedef pop::GeometricalTransformation GeometricalTransformation;


ALL_IMAGE2_DIM_TYPE(GeometricalTransformation,translate)



ALL_IMAGE2_DIM_TYPE(GeometricalTransformation,scale)
ALL_IMAGE2_DIM_TYPE(GeometricalTransformation,rotateMultPi_2)
%template(rotate) pop::GeometricalTransformation::rotate<pop::UI8>;
%template(rotate) pop::GeometricalTransformation::rotate<pop::UI16>;
%template(rotate) pop::GeometricalTransformation::rotate<pop::UI32>;
%template(rotate) pop::GeometricalTransformation::rotate<pop::F32>;
%template(rotate) pop::GeometricalTransformation::rotate<pop::RGBUI8>;
ALL_IMAGE2_DIM_TYPE(GeometricalTransformation,mirror)

%template(transformAffine2D) pop::GeometricalTransformation::transformAffine2D<pop::UI8>;
%template(transformAffine2D) pop::GeometricalTransformation::transformAffine2D<pop::F64>;
%template(transformAffine2D) pop::GeometricalTransformation::transformAffine2D<pop::RGBUI8>;
%template(transformAffine2D) pop::GeometricalTransformation::transformAffine2D<pop::UI16>;
%template(transformAffine2D) pop::GeometricalTransformation::transformAffine2D<pop::UI32>;

%template(transformHomogeneous2D) pop::GeometricalTransformation::transformHomogeneous2D<pop::UI8>;
%template(transformHomogeneous2D) pop::GeometricalTransformation::transformHomogeneous2D<pop::F64>;
%template(transformHomogeneous2D) pop::GeometricalTransformation::transformHomogeneous2D<pop::RGBUI8>;
%template(transformHomogeneous2D) pop::GeometricalTransformation::transformHomogeneous2D<pop::UI16>;
%template(transformHomogeneous2D) pop::GeometricalTransformation::transformHomogeneous2D<pop::UI32>;
