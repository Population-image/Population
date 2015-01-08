/******************************************************************************\
|*                   Population library for C++ X.X.X                         *|
|*----------------------------------------------------------------------------*|
The Population License is similar to the MIT license in adding this clause:
for any writing public or private that has resulted from the use of the
software population, the reference of this book "Population library, 2012,
Vincent Tariel" shall be included in it.

So, the terms of the Population License are:

Copyright © 2012-2015, Tariel Vincent

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

#ifndef FUNCTIONMatNINOUT_HPP
#define FUNCTIONMatNINOUT_HPP


#define UNICODE 1

#include <cstring>
#include <cstring>
#include <ctype.h>
#include <queue>
#include <iostream>
#include <ostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <iterator>
#include <list>
#include <cmath>
#include <string>
#include <algorithm>
#include "data/typeF/TypeF.h"
#include "data/typeF/RGB.h"
#include "data/utility/CollectorExecutionInformation.h"
#include "data/mat/MatN.h"
#include "data/mat/MatNListType.h"
namespace pop
{


class POP_EXPORTS MatNInOutPgm
{    /*!
        \class pop::MatNInOut
        \brief The template MatN class represents a grid matrix
        \author Tariel Vincent
      \sa MatN
    */
public:

    /*!
    \fn static I32 read(std::istream &File,MatN<D,T> *in )
    * default destructor
    */
    template<I32 D,typename T>
    static void read(MatN<D,T> &in,std::istream &File );
    template<I32 D,typename T>
    static void writeAscii(const MatN<D,T>&in,std::ostream & out );
    template<I32 D,typename T>
    static void writeRaw(const MatN<D,T> &in,std::ostream & out );
    template<I32 D,typename T>
    static void writeHeader(const MatN<D,T> &in,std::ostream & out,bool ascii  );
    template<I32 D,typename T>
    static void readRaw(MatN<D,T> &in,std::istream &File);
    template<I32 D,typename T>
    static void readAscii(MatN<D,T> &in ,std::istream &File);
    template<I32 D,typename T>
    static void writeRawData(const MatN<D,T> &in ,std::ostream &File);
    template<I32 D,typename T>
    static void writeAsciiData(const MatN<D,T>&in,std::ostream &File);

};

class POP_EXPORTS MatNInOut
{
private:
    class impl;
    impl *_pImpl;
    template<int DIM,typename Type>
    static void  _save(const MatN<DIM, Type > &, const char * );
    template<int DIM,typename Type>
    static void  _load(const MatN<DIM, Type > &, const char * );

    static void  _save(const MatN<2, UI8 > &img, const char * filename);
    static void  _save(const MatN<2, UI16 > &img, const char * filename);
    static void  _save(const MatN<2, UI32 > &img, const char * filename);
    static void  _save(const MatN<2, I32 > &img, const char * filename);
    static void  _save(const MatN<2, F32 > &img, const char * filename);
    static void  _save(const MatN<2, F64 > &img, const char * filename);
    static void  _save(const MatN<2, RGBUI8 > &img, const char * filename);


    static void  _load( MatN<2, UI8 > &img, const char * filename);
    static void  _load( MatN<2, UI16 > &img, const char * filename);
    static void  _load( MatN<2, UI32 > &img, const char * filename);
    static void  _load( MatN<2, I32 > &img, const char * filename);
    static void  _load( MatN<2, F32 > &img, const char * filename);
    static void  _load( MatN<2, F64 > &img, const char * filename);
    static void  _load( MatN<2, RGBUI8 > &img, const char * filename);




    template<int DIM,typename Type>
    static void  _savePNG(const MatN<DIM, Type > &, const char * );

    template<int DIM,typename Type>
    static void  _loadPNG(const MatN<DIM, Type > &, const char * );
    static void  _savePNG(const MatN<2, UI8 > &img, const char * filename);
    static void  _savePNG(const MatN<2, RGBUI8 > &img, const char * filename);
    static void  _loadPNG( MatN<2, UI8 > &img, const char * filename);
    static void  _loadPNG( MatN<2, RGBUI8 > &img, const char * filename);

    template<int DIM,typename Type>
    static void  _loadBMP( MatN<DIM, Type > &, const char * );
    template<int DIM,typename Type>
    static void  _saveBMP(const MatN<DIM, Type > &, const char * );
    static void  _loadBMP( MatN<2, UI8 > &img, const char * filename);
    static void  _loadBMP( MatN<2, RGBUI8 > &img, const char * filename);
    static void  _saveBMP(const MatN<2, UI8 > &img, const char * filename);
    static void  _saveBMP(const MatN<2, RGBUI8 > &img, const char * filename);


    template<int DIM,typename Type>
    static void  _loadJPG( MatN<DIM, Type > &, const char * );
    template<int DIM,typename Type>
    static void  _saveJPG(const MatN<DIM, Type > &, const char * );
    static void  _loadJPG( MatN<2, UI8 > &img, const char * filename);
    static void  _loadJPG( MatN<2, RGBUI8 > &img, const char * filename);
    static void  _saveJPG(const MatN<2, UI8 > &img, const char * filename);
    static void  _saveJPG(const MatN<2, RGBUI8 > &img, const char * filename);


public:
    template<I32 D,typename T>
    static void save(const MatN<D,T> &in,const char * file );
    template<I32 D,typename T>
    static void saveRaw(const MatN<D,T> &in,const char * file );
    template<I32 DIM,typename Result>
    static void load(MatN<DIM,Result> &in,const char * file  );
    template<I32 DIM,typename Result>
    static void loadRaw(MatN<DIM,Result> &in,const char * file  );

    template<int DIM,typename Type>
    static void loadFromDirectory(MatN<DIM,Type> & in1cast,const char * pathdir, const char * basefilename, const char * extension);
    template<int DIM,typename Result>
    static void saveFromDirectory(const MatN<DIM,Result> & in1cast,const char * pathdir, const char * basefilename, const char * extension);
};

namespace Private{
inline void  headerPNG(std::ostream & out,Loki::Type2Type<pop::UI8>){
      out<<"255"<<std::endl;
}
inline void  headerPNG(std::ostream & out,Loki::Type2Type<pop::RGBUI8>){
      out<<"255"<<std::endl;
}
template<typename PixelType>
inline void  headerPNG(std::ostream & out,Loki::Type2Type<PixelType>){
      out<<NumericLimits<typename TypeTraitsTypeScalar<PixelType>::Result >::maximumRange()<<std::endl;
}
inline std::pair<std::string,std::string>  header2PNG(Loki::Type2Type<pop::MatN<2,UI8> >){return std::make_pair("P2","P5");}
inline std::pair<std::string,std::string>  header2PNG(Loki::Type2Type<pop::MatN<2,UI16> >){return std::make_pair("P3","PA");}
inline std::pair<std::string,std::string>  header2PNG(Loki::Type2Type<pop::MatN<2,UI32> >){return std::make_pair("P4","PB");}
inline std::pair<std::string,std::string>  header2PNG(Loki::Type2Type<pop::MatN<2,F64> >){return std::make_pair("P7","PC");}
inline std::pair<std::string,std::string>  header2PNG(Loki::Type2Type<pop::MatN<2,RGBUI8 > >){return std::make_pair("P8","P6");}
inline std::pair<std::string,std::string>  header2PNG(Loki::Type2Type<pop::MatN<2,RGBF64 > >){return std::make_pair("PE","PF");}
inline std::pair<std::string,std::string>  header2PNG(Loki::Type2Type<pop::MatN<2,ComplexF64 > >){return std::make_pair("PI","PJ");}
inline std::pair<std::string,std::string>  header2PNG(Loki::Type2Type<pop::MatN<2,Vec2F64 > >){return std::make_pair("Pa","Pb");}

inline std::pair<std::string,std::string>  header2PNG(Loki::Type2Type<pop::MatN<3,UI8> >){return std::make_pair("PK","PL");}
inline std::pair<std::string,std::string>  header2PNG(Loki::Type2Type<pop::MatN<3,UI16> >){return std::make_pair("PM","PN");}
inline std::pair<std::string,std::string>  header2PNG(Loki::Type2Type<pop::MatN<3,UI32> >){return std::make_pair("PO","PP");}
inline std::pair<std::string,std::string>  header2PNG(Loki::Type2Type<pop::MatN<3,F64> >){return std::make_pair("PQ","PR");}
inline std::pair<std::string,std::string>  header2PNG(Loki::Type2Type<pop::MatN<3,RGBUI8 > >){return std::make_pair("PS","PT");}
inline std::pair<std::string,std::string>  header2PNG(Loki::Type2Type<pop::MatN<3,RGBF64 > >){return std::make_pair("PU","PV");}
inline std::pair<std::string,std::string>  header2PNG(Loki::Type2Type<pop::MatN<3,ComplexF64 > >){return std::make_pair("PZ","P1");}
inline std::pair<std::string,std::string>  header2PNG(Loki::Type2Type<pop::MatN<3,Vec3F64 > >){return std::make_pair("Pc","Pd");}
}
template<int Dim, typename Type>
void MatN<Dim,Type>::loadFromDirectory(const char * pathdir,const char * basefilename,const char * extension)
{
    MatNInOut::loadFromDirectory(*this,pathdir,basefilename,extension);
}
template<int Dim, typename Type>
void MatN<Dim,Type>::load(const char * file)
{
    MatNInOut::load(*this,file);
}
template<int Dim, typename Type>
void MatN<Dim,Type>::loadRaw(const char * file,const Domain & d)
{
    this->resize(d);
    MatNInOut::loadRaw(*this,file);
}
template<int Dim, typename Type>
void MatN<Dim,Type>::saveFromDirectory(const char * pathdir,const char * basefilename,const char * extension)const 
{
    MatNInOut::saveFromDirectory(*this,pathdir,basefilename,extension);
}
template<int Dim, typename Type>
void MatN<Dim,Type>::save(const char * file)const 
{
    MatNInOut::save(*this,file);
}
template<int Dim, typename Type>
void MatN<Dim,Type>::saveRaw(const char * file)const 
{
    MatNInOut::saveRaw(*this,file);
}
template<int Dim, typename Type>
void MatN<Dim,Type>::saveAscii(const char * file,std::string header)const 
{
    std::ofstream  out(file);
    if (out.fail())
        std::cerr<<"In MatN::save, cannot open file: "+std::string(file);
    else{
        if(header!="")
            out<<header<<std::endl;
        MatNInOutPgm::writeAsciiData(*this ,out);
    }
}
template<I32 D,typename T>
void MatNInOutPgm::read(MatN<D,T> &in,std::istream &File )
{
    if (File.fail()){
        std::cerr<<"In MatN::read, cannot open the file";
    }
    std::pair<std::string,std::string> type=Private::header2PNG(Loki::Type2Type<MatN<D,T> >());
    std::string idascii = type.first;
    std::string buffer ;
    std::getline( File, buffer, '\n');
    bool _ascii;
    if(idascii[1]==buffer[1]){
        _ascii = true;
    }
    else{
        _ascii = false;
    }
    std::string str;
    bool firsttime=true;
    do{
        std::getline( File, buffer, '\n');
        if(!isdigit(buffer[0]))
            str+=buffer;
        if(firsttime==true){
            str+="\n";
            firsttime = false;
        }
    } while (!isdigit(buffer[0]));
    //        in.setInformation(str);

    std::istringstream iss( buffer );
    std::string mot;
    I32 i=0;
    typename MatN<D,T>::E  x;
    while ( std::getline( iss, mot, ' ' ) ){
        std::istringstream iss2( mot );
        if(i<D)
            iss2 >> x[i];
        else{
            int value;
            iss2 >> value;
        }
        i++;
    }

    std::getline( File, mot, '\n' );
    std::swap(x(0),x(1));
    in.resize(x);
    if(_ascii==true){
        readAscii(in,File);
    }
    else{
        readRaw(in,File);
    }
}
template<I32 D,typename T>
void MatNInOutPgm::writeAscii(const MatN<D,T>&in ,std::ostream & out)
{
    writeHeader(in,out,true );
    writeAsciiData(in,out);
}
template<I32 D,typename T>
void MatNInOutPgm::writeRaw(const MatN<D,T> &in,std::ostream & out )
{
    writeHeader(in,out,false );
    writeRawData(in,out);
}



template<I32 D,typename T>
void MatNInOutPgm::writeHeader(const MatN<D,T> &in,std::ostream & out,bool ascii  )
{

    std::string idascii;
    std::pair<std::string,std::string> type = Private::header2PNG(Loki::Type2Type<MatN<D,T> >());
    if(ascii==true)
    {
        idascii = type.first;
    }
    else
    {
        idascii = type.second;
    }
    out<<idascii<<std::endl;
    std::stringstream ss("");
    std::string item;
    while(std::getline(ss, item, '\n')) {
        out<<"#"<<item<<std::endl;
    }
    for(int i =0;i<D;i++){
        if(i==0)
            out<<in.getDomain()(1);
        else if(i==1)
            out<<in.getDomain()(0);
        else
            out<<in.getDomain()(i);
        if(i!=D-1)
            out<<" ";
    }
    out<<std::endl;

   Private::headerPNG(out,Loki::Type2Type<T>());

}
template<I32 D,typename T>
void MatNInOutPgm::readRaw(MatN<D,T> &in,std::istream &File )
{
    T * t =&( in.operator()(0));
    File.read(reinterpret_cast<char*>(t), sizeof(T)*in.getDomain().multCoordinate());
}


//void readRaw(std::istream &File,MatN<2,RGBConditionBounded > *in );
template<I32 D,typename T>
void MatNInOutPgm::readAscii(MatN<D,T> &in,std::istream &File )
{
    File>>in;
}


template<I32 D,typename T>
void MatNInOutPgm::writeRawData(const MatN<D,T> &in,std::ostream &File )
{
    const T * t =&( in.operator()(0));
    File.write(reinterpret_cast<const char*>(t), sizeof(T)*(in).getDomain().multCoordinate());
}

template<I32 D,typename T>
void MatNInOutPgm::writeAsciiData(const MatN<D,T>&in,std::ostream &File )
{
    File<<in;
}


template<int DIM,typename Type>
void  MatNInOut::_save(const MatN<DIM, Type > &, const char * )
{
    if(DIM==3){
        std::cerr<<"In MatN::save, cannot save 3d matrix if the matrix format is not pgm";
    }
    else
        std::cerr<<"In MatN::save, cannot save this matrix pixel voxel type";
}
template<int DIM,typename Type>
void  MatNInOut::_load(const MatN<DIM, Type > &, const char * )
{
    if(DIM==3){
        std::cerr<<"In MatN::load, cannot save 3d matrix if the matrix format is not pgm";
    }
    else
        std::cerr<<"In MatN::load, cannot save this matrix pixel voxel type";

}

template<int DIM,typename Type>
void  MatNInOut::_savePNG(const MatN<DIM, Type > &, const char * )
{
    if(DIM==3){
        std::cerr<<"In MatN::save, cannot save 3d matrix if the matrix format is not pgm";
    }
    else
        std::cerr<<"In MatN::save, cannot save this matrix pixel voxel type";
}
template<int DIM,typename Type>
void  MatNInOut::_loadPNG(const MatN<DIM, Type > &, const char * )
{
    if(DIM==3){
        std::cerr<<"In MatN::load, cannot save 3d matrix if the matrix format is not pgm";
    }
    else
        std::cerr<<"In MatN::load, cannot save this matrix pixel voxel type";

}
template<int DIM,typename Type>
void  MatNInOut::_loadBMP( MatN<DIM, Type > &, const char * )
{
    if(DIM==3){
        std::cerr<<"In MatN::load, cannot save 3d matrix if the matrix format is not pgm";
    }
    else
        std::cerr<<"In MatN::load, cannot save this matrix pixel voxel type";
}
template<int DIM,typename Type>
void  MatNInOut::_saveBMP(const MatN<DIM, Type > &, const char * )
{
    if(DIM==3){
        std::cerr<<"In MatN::save, cannot save 3d matrix if the matrix format is not pgm";
    }
    else
        std::cerr<<"In MatN::save, cannot save this matrix pixel voxel type";

}



template<int DIM,typename Type>
void  MatNInOut::_loadJPG( MatN<DIM, Type > &, const char * )
{
    if(DIM==3){
        std::cerr<<"In MatN::load, cannot save 3d matrix if the matrix format is not pgm";
    }
    else
        std::cerr<<"In MatN::load, cannot save this matrix pixel voxel type";
}
template<int DIM,typename Type>
void  MatNInOut::_saveJPG(const MatN<DIM, Type > &, const char * )
{
    if(DIM==3){
        std::cerr<<"In MatN::save, cannot save 3d matrix if the matrix format is not pgm";
    }
    else
        std::cerr<<"In MatN::save, cannot save this matrix pixel voxel type";

}
template<I32 D,typename T>
void MatNInOut::saveRaw(const MatN<D,T> &in ,const char * file){
    std::ofstream  out(file);
    if (out.fail())
        std::cerr<<"In MatN::save, cannot open file: "+std::string(file);
    else
        MatNInOutPgm::writeRawData(in ,out);
}
template<I32 D,typename T>
void MatNInOut::save(const MatN<D,T> &in ,const char * file){
    std::string ext( BasicUtility::getExtension(file));
    if(ext==".pgm")
    {
        if(in.getDomain().multCoordinate()<20){
            std::ofstream  out(file);
            if (out.fail()){
                std::cerr<<"In MatN::save, cannot open file: "+std::string(file);
            }
            else{
                MatNInOutPgm::writeAscii(in,out);
            }
            out.close();
        }
        else{
            std::ofstream  out(file,std::ios::binary);
            if (out.fail()){
                std::cerr<<"In MatN::save, Cannot open file: "+std::string(file);
            }
            else{
                MatNInOutPgm::writeRaw(in,out);
            }
            out.close();
        }
    }else if(ext==".png"||ext==".PNG"){
        _savePNG(in,file);
    }
    else if(ext==".bmp"||ext==".BMP"){
        _saveBMP(in,file);
    }
    else if(ext==".jpg"||ext==".jpeg"||ext==".JPG"||ext==".JPEG"){
        _saveJPG(in,file);
    }
    else{
        _save(in,file);
    }
}
template<I32 DIM,typename Result>
void MatNInOut::loadRaw(MatN<DIM,Result> &in, const char * file  ){

    std::ifstream  is(file,std::iostream::binary);
    if (is.fail()){
        std::cerr<<"In MatN::load, Cannot open file: "+std::string(file);
    }
    else{
        is.seekg (0, is.end);
        unsigned int length = is.tellg();
        is.seekg (0, is.beg);
        if(length>=sizeof(Result)*in.getDomain().multCoordinate())
            MatNInOutPgm::readRaw(in,is);
        else
            std::cerr<<"In MatN::loadRaw, the file should be equal or superior to sizeof(Type)*in.getDomain().multCoordinate()";
    }
}

template<I32 DIM,typename Result>
void MatNInOut::load(MatN<DIM,Result> &in,const char * file  )
{
    std::ifstream  is2(file,std::iostream::binary);
    if (is2.fail())
    {
        std::cerr<<"In MatN::load, Cannot open file: "+std::string(file);
    }
    is2.close();

    std::string ext( BasicUtility::getExtension(file));
    if(ext==".pgm")
    {
        std::ifstream  is(file,std::iostream::binary);
        if (is.fail())
        {
            std::cerr<<"In MatN::load, Cannot open file: "+std::string(file);
        }
        else
        {
            MatNInOutPgm::read(in,is);
        }
    }else if(ext==".png"||ext==".PNG"){
        _loadPNG(in,file);
    }else if(ext==".bmp"||ext==".BMP"){
        _loadBMP(in,file);
    }
    else if(ext==".jpg"||ext==".jpeg"||ext==".JPG"||ext==".JPEG"){
        _loadJPG(in,file);
    }
    else{
        _load(in,file);
    }


}
template<int DIM,typename Type>
void MatNInOut::loadFromDirectory(MatN<DIM,Type> & in1cast,const char * pathdir, const char * basefilename, const char * extension)
{
    std::vector<std::string> vec =BasicUtility::getFilesInDirectory(std::string(pathdir));
    std::string ext =extension;
    if(ext!=""&&ext[0]!='.')
        ext="."+ext;
    for(int i = 0;i<(int)vec.size();i++){
        if(ext!=""&& ext!=(BasicUtility::getExtension(vec[i])))
        {
            vec.erase(vec.begin()+i);
            i--;
        }else if(std::string(basefilename)!=""&&vec[i].find(std::string(basefilename))==std::string::npos)
        {
            vec.erase(vec.begin()+i);
            i--;
        }else
        {
            vec[i]=std::string(pathdir)+"/"+vec[i];
        }
    }
    std::sort (vec.begin(), vec.end());
    if(vec.size()==0)
        std::cerr<<"In MatN::loadFromDirectory, the directory is empty"+std::string(pathdir);

    MatN<2,Type> img;
    MatNInOut::load(img,vec[0].c_str());
    VecN<3,int> d;
    d(0)=img.getDomain()(0);
    d(1)=img.getDomain()(1);
    d(2)=vec.size();
    in1cast.resize(d);
    for(int i = 0;i<(int)vec.size();i++){
        if(i!=0)
            MatNInOut::load(img,vec[i].c_str());
        if(img.sizeI()!=in1cast.sizeI()||img.sizeJ()!=in1cast.sizeJ()){
            std::cerr<<std::string("In MatN::loadFromDirectory, all matrix must have the same domain");
        }
        typename MatN<2,Type>::IteratorEDomain it(img.getIteratorEDomain());
        while(it.next())
            in1cast.operator ()(it.x()(0),it.x()(1),i)=img(it.x()(0),it.x()(1));;
    }
}
template<int DIM,typename Result>
void MatNInOut::saveFromDirectory(const MatN<DIM,Result> & in1cast,const char * pathdir, const char * basefilename, const char * extension)
{
    if(BasicUtility::isDirectory(pathdir)==false)
        BasicUtility::makeDirectory(pathdir);
    std::string path =pathdir;
    std::string file =basefilename;
    std::string ext = extension;
    path = path+"/";
    Vec2I32 d;
    d(0)=in1cast.getDomain()(0);
    d(1)=in1cast.getDomain()(1);
    MatN<2, Result> plane(d);

    typename MatN<2, Result>::IteratorEDomain it(plane.getIteratorEDomain());


    int slice =in1cast.getDomain()(2);
    for(int i=0;i<slice;i++)
    {
        VecN<3,int> x;
        x(2)=i;
        it.init();
        while(it.next()){
            x(0) = it.x()(0);
            x(1) = it.x()(1);
            plane(it.x())=(in1cast )(x(0),x(1),x(2));
        }
        std::string str = BasicUtility::IntFixedDigit2String(i,4);
        std::string fileout = path+"/"+file+str+ext;
        MatNInOut::save(plane,fileout.c_str());
    }
}

}
#endif
