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
namespace std{
%template(vectorUI8) vector<unsigned char>;
%template(vectorI32) vector<int>;
%template(vectorUI16) vector<unsigned short>;
%template(vectorUI32) vector<unsigned int>;
%template(vectorF32) vector<float>;
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
#pragma SWIG nowarn=302,314,317,362,389,509
//###Processing###
%include populationfull.i
