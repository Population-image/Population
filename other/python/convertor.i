%include "../../include/algorithm/Convertor.h"
typedef pop::Convertor Convertor;

%template(toRGB) pop::Convertor::toRGB<2,pop::UI8>;
%template(toRGB) pop::Convertor::toRGB<2,pop::F32>;
%template(fromRGB) pop::Convertor::fromRGB<2,pop::UI8>;
%template(fromRGB) pop::Convertor::fromRGB<2,pop::F32>;

%template(toYUV) pop::Convertor::toYUV<2,pop::UI8>;
%template(toYUV) pop::Convertor::toYUV<2,pop::F32>;
%template(fromYUV) pop::Convertor::fromYUV<2,pop::UI8>;
%template(fromYUV) pop::Convertor::fromYUV<2,pop::F32>;

%template(toRealImaginary) pop::Convertor::toRealImaginary<2,pop::F32>;
%template(toRealImaginary) pop::Convertor::toRealImaginary<3,pop::F32>;
%template(fromRealImaginary) pop::Convertor::fromRealImaginary<2,pop::F32>;
%template(fromRealImaginary) pop::Convertor::fromRealImaginary<3,pop::F32>;

%template(toVecN) pop::Convertor::toVecN<pop::Mat2Vec2F32,pop::Mat2F32>;
%template(toVecN) pop::Convertor::toVecN<pop::Mat3Vec3F32,pop::Mat3F32>;
%template(fromVecN) pop::Convertor::fromVecN<pop::Mat2Vec2F32,pop::Mat2F32>;
%template(fromVecN) pop::Convertor::fromVecN<pop::Mat3Vec3F32,pop::Mat3F32>;

