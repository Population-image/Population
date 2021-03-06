/******************************************************************************\
|*                   Population library for C++ X.X.X                         *|
|*----------------------------------------------------------------------------*|
The Population License is similar to the MIT license in adding this clause:
for any writing public or private that has resulted from the use of the
software population, the reference of this book "Population library, 2012,
Vincent Tariel" shall be included in it.

So, the terms of the Population License are:

Copyright © 2012-2015, Tariel Vincent
Copyright © 2015, Aublin Pierre Louis

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to
deal in the Software without restriction, including without limitation the
rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
sell copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software and for any writing
public or private that has resulted from the use of the software population,
the reference of this book "Population library, 2012, Vincent Tariel" shall
be included in it.

The Software is provided "as is", without warranty of any kind, express or
implied, including but not limited to the warranties of merchantability,
fitness for a particular purpose and noninfringement. In no event shall the
authors or copyright holders be liable for any claim, damages or other
liability, whether in an action of contract, tort or otherwise, arising
from, out of or in connection with the software or the use or other dealings
in the Software.
\***************************************************************************/

#ifndef BasicUtility_HPP
#define BasicUtility_HPP

#include <sstream>
#include <vector>
#include "data/typeF/TypeF.h"
#include "PopulationConfig.h"

namespace pop
{
/*! \ingroup Other
* \defgroup BasicUtility  BasicUtility
* \brief Some cross OS utilities

*/

const F32 PI = 3.141592f;
const F32 EPSILON = 0.0001f;
#define TOKENPASTE(x, y) x ## y
#define TOKENPASTE2(x, y) TOKENPASTE(x, y)
template<typename T>
struct Identity
{
    typedef T Result;
};

template <typename T>
int POP_EXPORTS sgn(T val)
{
    return (val >= T(0)) - (val < T(0));
}

class POP_EXPORTS BasicUtility
{
public:
    /*!
        \class pop::BasicUtility
        \brief  String functionnality
        \author Tariel Vincent
        \ingroup BasicUtility
      *
    */

    /*!
    \brief Get the OS separator
    \return The OS Separator: / on Linux, \ on Windows
    */
    static std::string getPathSeparator();
    /*!
    \brief Is the file_path a file ?
    \param filepath file path
    \return yes is file, false otherwise
    \brief check for a file's existence
    *
    * \code
    string path ="/home/vtariel/lena.pgm";
    if(BasicUtility::isFile(path)==true){
        //Process
    }
    * \endcode
    *
    */
    static bool isFile(std::string filepath);
    /*!
    \brief Is the file_path a directory ?
    \param dirpath directory path
    \return yes is directory, false otherwise
     \brief check for a directory's existence
    *
    * \code
    string path ="/home/vtariel/";
    if(BasicUtility::isDirectory(path)==true){
        //Process
    }
    * \endcode
    *
    */
    static bool isDirectory(std::string dirpath);
    /*!
    \brief make a directory
    \param dirpath directory path
    \return false if the directory already exists
    *
    * \code
    string path ="/home/vtariel/";
    if(BasicUtility::makeDirectory(path)==true){
        //Process
    }
    * \endcode
    *
    */
    static bool makeDirectory(std::string dirpath);

    /*!
    \brief return the base name
    \param file file name
    \return the base name
    *
    * \code
    string file ="/home/vtariel/toto.xml";
    cout<<BasicUtility::getBasefilename(file)<<endl;// toto
    * \endcode
    *
    */
    static  std::string getBasefilename(std::string file);

    /*!
    \brief return the directory path  n
    \param file file name
    \return the directory pat
    *
    * \code
    string file ="/home/vtariel/toto.xml";
    cout<<BasicUtility::getBasefilename(file)<<endl;// /home/vtariel/
    * \endcode
    *
    */
    static  std::string getPath(std::string file);
    static  std::string getExtension(std::string file);

    /*!
    \brief get string until the delimeter
    \param in input stream
    \param del delimeter
    \return string
    *
    * \code
    std::stringstream in("Data1$$Data2$$Data3$$Data4$$");
    while(!in.eof()){
        std::cout<<BasicUtility::getline(in,"$$")<<std::endl;
     }
    return 1;
    * \endcode
    *
    */
    static  std::string getline(std::istream& in,std::string del);
    /*!
    \brief get files in the given directory
    \param dir input stream
    \return file name in the directory
    *
    * \code
    string dir = "/home/vtariel/Bureau/DataBase/Acquisition/record/";
    std::vector<string> files = BasicUtility::getFilesInDirectory(dir);
    for(int i=0;i<files.size();i++){
        Mat2RGBUI8 img;
        string path = dir+files[i];
        img.load(path.c_str());
        //process
    }
    * \endcode
    *
    */
    static std::vector<std::string> getFilesInDirectory (std::string dir);
    /*!
    \brief convert a string to any type
    \param s input string
    \param Dest output object (The insertion operator (<<) should be surcharged for the template class)
    \return true success, false otherwise
    *
    * \code
    std::string str = "2.7";
    F32 d;
    BasicUtility::String2Any(str,d);
    std::cout<<d<<std::endl;
    * \endcode
    *
    */
    template<typename T>
    static bool String2Any(std::string s,  T & Dest )
    {
        std::istringstream iss(s);
        iss >> Dest;
        return true;
    }
    static bool String2Any(std::string s,  bool & Dest );
    static bool String2Float(std::string s,  F32 & Dest);
    /*!
    \brief convert a string representing an hexadecimal number to any type
    \param s input string (the 0x prefix is optional)
    \param Dest output object (The insertion operator (<<) should be surcharged for the template class)
    \return true success, false otherwise
    *
    * \code
    std::string str = "0xA";
    int i;
    BasicUtility::HexString2Any(str,i);
    std::cout<<i<<std::endl;
    * \endcode
    *
    */
    template<typename T>
    static bool HexString2Any(std::string s,  T & Dest )
    {
        std::stringstream ss;
        ss << std::hex << s;
        ss >> Dest;
        return true;
    }
    /*!
    \brief convert any type to a string
    \param Value input object (The extraction operator (>>) should be surcharged for the template class)
    \param s output string
    \return true success, false otherwise
    *
    * \code
    std::string str ;
    F32 d=2.7;
    BasicUtility::Any2String(d,str);
    std::cout<<str<<std::endl;
    * \endcode
    *
    */
    template<typename T>
    static bool Any2String(T Value,  std::string & s)
    {
        s.clear();
        std::ostringstream oss;
        std::ostream& os =  oss;

        bool temp = os << Value;
        s= oss.str();
        return temp;
    }
    static bool Any2String(bool Value,  std::string & s);
    /*!
    \brief convert any type to a string
    \param Value input object (The extraction operator (>>) should be surcharged for the template class)
    \return output string
    *
    * \code
    F32 d=2.7;
    std::cout<<BasicUtility::Any2String(d,str)<<std::endl;
    * \endcode
    *
    */
    template<typename T>
    static  std::string Any2String(T Value)
    {
        std::string s;
        s.clear();
        std::ostringstream oss;
        //        std::ostream &os =  oss;
        oss << Value;
        s= oss.str();
        return s;
    }
    static  std::string Any2String(bool Value);
    /*!
    \brief convert a possitive number to a string with a fixed digit number
    \param value unsigned integer value
    \param digitnumber number of digit
    \return output string
    *
    * \code
     std::cout<<BasicUtility::IntFixedDigit2String(35,4)<<std::endl;//0035
    * \endcode
    *
    */
    static  std::string IntFixedDigit2String(unsigned int value,int digitnumber);

    static  std::string replaceSlashByAntiSlash(std::string filepath);
    //edit-Distance
    static int editDistance(std::string s1,std::string s2);

    static void sleep_ms(int ms);
};

template<typename T>
static int power2(T x)
{
    return (x <= 0 ? 1 : 1 << x);
}
////////////////////////////////////////////////////////////////////////////////
// class template Int2Type
// Converts each integral constant into a unique type
// Invocation: Int2Type<v> where v is a compile-time constant integral
// Defines 'value', an enum that evaluates to v
////////////////////////////////////////////////////////////////////////////////

template <int v>
struct POP_EXPORTS Int2Type
{
    enum { value = v };
};

////////////////////////////////////////////////////////////////////////////////
// class template Type2Type
// Converts each type into a unique, insipid type
// Invocation Type2Type<T> where T is a type
// Defines the type OriginalType which maps back to T
////////////////////////////////////////////////////////////////////////////////

template <typename T>
struct Type2Type
{
    typedef T OriginalType;
};
class NullType {};
inline std::ostream& operator << (std::ostream& out, const NullType & ){
    return out;
}
inline std::istream& operator >> (std::istream& in, NullType & ){
    return in;
}
template<int Value,int Dim>
struct PowerGP{enum{value=Value*PowerGP<Value,Dim-1>::value};};
template<int Value>
struct PowerGP<Value,0>{enum{value=1};};
}
namespace std{

template<class _T1, class _T2>
std::ostream& operator << (std::ostream &out, const std::pair<_T1,_T2> &arg)
{

    out << arg.first << "<P>" << arg.second<<"<P>";
    return out;
}
template<class _T1, class _T2>
std::istream& operator >>(std::istream& in, std::pair<_T1,_T2> &arg)
{
    std::string str;
    str = pop::BasicUtility::getline( in, "<P>" );
    pop::BasicUtility::String2Any(str,arg.first );
    str = pop::BasicUtility::getline( in,"<P>" );
    pop::BasicUtility::String2Any(str,arg.second );
    return in;
}

template<class _T1>
std::ostream& operator << (std::ostream &out, const std::vector<_T1> &arg)
{
    out<<(int)arg.size()<<"<V>";
    for(unsigned int i=0;i<arg.size();i++)
        out << arg[i] << "<V>";
    return out;
}
template<class _T1>
std::istream& operator >> (std::istream &in,std::vector<_T1> &arg)
{
    arg.clear();
    std::string str;
    str = pop::BasicUtility::getline( in, "<V>" );
    int size;
    pop::BasicUtility::String2Any(str,size );
    for(int i=0;i<size;i++){
        _T1 t;
        str = pop::BasicUtility::getline( in, "<V>" );
        pop::BasicUtility::String2Any(str,t );
        arg.push_back(t);
    }
    return in;
}
}
#endif // BasicUtility_HPP
