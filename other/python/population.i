/* population.i
*/
%module population
%{
#include"Population.h"
%}
#define HAVE_SWIG
#define POP_EXPORTS
%include <std_string.i>
%include "std_vector.i"

//#####################Exception ###################
namespace std{
%template(vectorUI8) vector<unsigned char>;
%template(vectorI32) vector<int>;
%template(vectorUI16) vector<unsigned short>;
%template(vectorUI32) vector<unsigned int>;
%template(vectorF64) vector<double>;
struct exception{
};
  template <class T, class U > struct pair {
    typedef T first_type;
    typedef U second_type;
    pair();
    pair(T first, U second);
    pair(const pair& p);
    T first;
    U second;

  };
}
%template(pairi) std::pair<int,int>;
namespace pop{
class pexception : public std::exception
{
private:
  std::string _message;
public:
  pexception();
  pexception(const char *message);
  pexception(std::string message);
  //! Return a C-string containing the error message associated to the thrown exception.
  const char *what() const throw();
  void display()const throw();
  virtual ~pexception() throw();
};

typedef  pop::pexception pexception;

}
//###Processing###
%include populationfull.i

