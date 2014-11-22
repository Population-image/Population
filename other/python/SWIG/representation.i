%include "../../../include/algorithm/Representation.h"
typedef pop::Representation Representation;
%template(truncateMulitple2) pop::Representation::truncateMulitple2<pop::MatN<2,pop::ComplexF64> >;
%template(truncateMulitple2) pop::Representation::truncateMulitple2<pop::MatN<3,pop::ComplexF64> >;

%template(FFT) pop::Representation::FFT<pop::MatN<2,pop::ComplexF64> >;
%template(FFT) pop::Representation::FFT<pop::MatN<3,pop::ComplexF64> >;

%template(FFTDisplay) pop::Representation::___FFTDisplay<2>;
%template(FFTDisplay) pop::Representation::___FFTDisplay<3>;

%template(highPass) pop::Representation::highPass<2 >;
%template(highPass) pop::Representation::highPass<3 >;
%template(lowPass) pop::Representation::lowPass<2 >;
%template(lowPass) pop::Representation::lowPass<3 >;
