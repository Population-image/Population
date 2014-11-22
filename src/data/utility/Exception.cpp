#include"data/utility/Exception.h"
#include"PopulationConfig.h"
#if defined(HAVE_CIMG)
#include"dependency/CImg.h"
#endif
namespace pop
{
pexception::pexception(){_message = "ERROR";}
pexception::pexception(const char * message) {_message = message;}
pexception::pexception(std::string message) {_message = message;}
const char * pexception::what() const throw() { return _message.c_str(); }
void pexception::display() const throw() {
#if defined(HAVE_CIMG)
    cimg_library::cimg::dialog("Population exception", _message.c_str(),"Abort");
#endif
}
pexception::~pexception() throw() {}

}
