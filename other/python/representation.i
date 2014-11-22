%include "../../include/algorithm/Representation.h"
typedef pop::Representation Representation;


%template(FFT) pop::Representation::FFT<2>;
%template(FFT) pop::Representation::FFT<3>;

%template(FFTDisplay) pop::Representation::FFTDisplay<2>;
%template(FFTDisplay) pop::Representation::FFTDisplay<3>;

%template(highPass) pop::Representation::highPass<2 >;
%template(highPass) pop::Representation::highPass<3 >;
%template(lowPass) pop::Representation::lowPass<2 >;
%template(lowPass) pop::Representation::lowPass<3 >;
