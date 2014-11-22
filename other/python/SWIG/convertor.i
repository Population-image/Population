%include "../../../include/algorithm/Convertor.h"
typedef pop::Convertor Convertor;

%template(toRGB) pop::Convertor::toRGB<pop::Mat2RGBUI8,pop::Mat2UI8>;
%template(fromRGB) pop::Convertor::fromRGB<pop::Mat2RGBUI8,pop::Mat2UI8>;

%template(toYUV) pop::Convertor::toYUV<pop::Mat2RGBUI8,pop::Mat2UI8>;
%template(fromYUV) pop::Convertor::fromYUV<pop::Mat2RGBUI8,pop::Mat2UI8>;

%template(toRealImaginary) pop::Convertor::toRealImaginary<pop::Mat2ComplexF64,pop::Mat2F64>;
%template(fromRealImaginary) pop::Convertor::fromRealImaginary<pop::Mat2ComplexF64,pop::Mat2F64>;

%template(toVecN) pop::Convertor::toVecN<pop::Mat2Vec2F64,pop::Mat2F64>;
%template(toVecN) pop::Convertor::toVecN<pop::Mat3Vec3F64,pop::Mat3F64>;
%template(fromVecN) pop::Convertor::fromVecN<pop::Mat2Vec2F64,pop::Mat2F64>;
%template(fromVecN) pop::Convertor::fromVecN<pop::Mat3Vec3F64,pop::Mat3F64>;

