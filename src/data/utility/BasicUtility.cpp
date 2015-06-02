
#include<cmath>
#include"PopulationConfig.h"

#include <sys/types.h>  // For stat().
#include <sys/stat.h>   // For stat().
#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include "data/mat/MatN.h"
#include"data/utility/BasicUtility.h"

#if Pop_OS==2
#ifdef _MSC_VER
#include"3rdparty/direntvc.h"
#include <direct.h>
#else
#include <direct.h>
#include <dirent.h>
#endif
#else
#include <dirent.h>
#endif

namespace pop
{
bool BasicUtility::String2Any(std::string s,  bool & Dest ){
    Dest = (s=="true");
    return true;
}
bool BasicUtility::String2Float(std::string s,  F32 & Dest ){
    std::cout << "reading a float: " << s << std::endl;
    Dest = atof(s.c_str());
    return true;
}
bool BasicUtility::Any2String(bool Value,  std::string & s){
    s = (Value ? "true" : "false");
    return true;
}
std::string BasicUtility::Any2String(bool Value){
    if(Value){
        return "true";
    }
    else{
        return "false";
    }
}


std::string BasicUtility::IntFixedDigit2String(unsigned int value,int digitnumber)
{
    long int  number =(long int)std::pow (10.,digitnumber);
    long int tempvalue = value/number;
    value-=tempvalue*number;
    std::string s;
    for(int i =digitnumber-1;i>=0;i--)
    {
        number =(long int)std::pow (10.,i);
        tempvalue = value/number;
        value-=tempvalue*number;
        std::string str;
        Any2String(tempvalue,str);
        s+=str;
    }
    return s;
}
std::string BasicUtility::getline(std::istream& in,std::string del){
    std::string str;
    std::string temp;
    int index=0;
    char c;
    while(index<(int)del.size()){
        c = in.get();
        if (in.good()){
            if(c==del.operator [](index))
            {
                temp+=c;
                index++;
            }
            else
            {
                if(index!=0){
                    str+=temp;
                    temp.clear();
                    index=0;
                }
                if(c==del.operator [](index))
                {
                    temp+=c;
                    index++;
                }
                else
                    str+=c;
            }
        }
    }
    return str;
}
std::string BasicUtility::getExtension(std::string file){
    int size =0, index = 0;

    do{
        if(file[size] == '.') {
            index = size;
        }
        size ++;
    }while(size<(int)file.size());

    if(size && index)
        return file.assign(file.begin()+index,file.end());
    else
        return "";
    //std::cerr<<"BasicUtility::getExtension, no extension in your file :" +file));
}
std::string BasicUtility::getBasefilename(std::string file){
    size_t slash, dot;
    slash = -1;
    dot = file.size()-1;
    size_t size =0;
    while(size ++, file[size]) {
        if(file[size] == '.') {
            dot = size;
        }
        if(file[size] == '/'||file[size] == '\\') {
            slash = size;
        }
    }
    return file.assign(file.begin()+slash+1,file.begin()+dot);

}
std::string BasicUtility::getPath(std::string file){
    int slash;
    slash = -1;
    int size =0;
    while(size ++, file[size]) {
        if(file[size] == '/'||file[size] == '\\') {
            slash = size;
        }
    }
    return file.assign(file.begin(),file.begin()+slash);

}

std::vector<std::string> BasicUtility::getFilesInDirectory(std::string dir)
{
    DIR *dp;
    struct dirent *dirp;
    if((dp  = opendir(dir.c_str())) == NULL) {
        std::cerr<<"BasicUtility::getFilesInDirectory, Cannot open this directory"+std::string(dir);
        return std::vector<std::string>();
    }
    std::vector<std::string> files;
    while ((dirp = readdir(dp)) != NULL) {
        std::string str  = std::string(dirp->d_name);
        if(str!="."&&str!=".."&&*str.rbegin()!='~'){
            files.push_back(str);
        }
    }
    closedir(dp);
    std::sort(files.begin(),files.end());
    return files;

}
bool BasicUtility::isFile( std::string filepath){

    int status;
    struct stat st_buf;
    status = stat (filepath.c_str(), &st_buf);
    if (status != 0) {
        return false;
    }

    if (S_ISREG (st_buf.st_mode)) {
        return true;
    }
    else
        return false;
}

bool BasicUtility::isDirectory(std::string dirpath){
    //#if Pop_OS==1
    ////    if ( access( dirpath.c_str(), 0 ) == 0 )
    //#endif
    {
        struct stat status;
        stat( dirpath.c_str(), &status );

        if ( status.st_mode & S_IFDIR )
        {
            return true;
        }
        else
        {
            return false;
        }
    }
    //#if Pop_OS==1
    //    else
    //#endif
    //    {
    //        return false;
    //    }
}

bool BasicUtility::makeDirectory(std::string dirpath){

    if(isDirectory(dirpath)==false){
#if Pop_OS==1
        mkdir(dirpath.c_str(),S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
#endif
#if Pop_OS==2
        _mkdir(dirpath.c_str());
#endif
        return true;
    }
    else
        return false;
}

struct operatorReplaceSlashByAntiSlash
{
    char operator()(char c){
        if(c=='/')
            return '\\';
        else
            return c;
    }
};
std::string BasicUtility::replaceSlashByAntiSlash(std::string filepath){
    std::transform(filepath.begin(),filepath.end(),filepath.begin(),operatorReplaceSlashByAntiSlash());
    return filepath;
}
int BasicUtility::editDistance(std::string s1,std::string s2){
    if(s1.size()==0)
        return static_cast<int>(s2.size());
    if(s2.size()==0)
        return static_cast<int>(s1.size());
    pop::Mat2UI16 m(static_cast<int>(s1.size()+1),static_cast<int>(s2.size()+1));
    for(unsigned int i =0;i<m.sizeI();i++){
        m(i,0)=i;
    }
    for(unsigned int j =0;j<m.sizeJ();j++){
        m(0,j)=j;
    }
    int k =2;
    bool test=true;
    while(test==true){
        test=false;
        for(int i=1;i<=k-1;i++){
            int j=k-i;
            if(m.isValid(i,j)){
                test =true;
                int v=1;
                if(s1[i-1]==s2[j-1])
                    v=0;

                int v1;
                if(m(i,j-1)+1<m(i-1,j)+1)
                    v1 = m(i,j-1)+1;
                else
                    v1 = m(i-1,j)+1;
                if(v1<m(i-1,j-1)+v)
                    m(i,j)= v1;
                else
                    m(i,j)= m(i-1,j-1)+v;
            }
        }
        k++;
    }
    return m(m.sizeI()-1,m.sizeJ()-1);
}

std::string BasicUtility::getPathSeparator() {
#if Pop_OS==1
    return "/";
#elif Pop_OS==2
    return "\\";
#else
    return "???";
#endif
}
UI8 maximum(UI8 v1,UI8 v2){return (std::max)(v1,v2);}
UI16 maximum(UI16 v1,UI16 v2){return (std::max)(v1,v2);}
UI32 maximum(UI32 v1,UI32 v2){return (std::max)(v1,v2);}
I8 maximum(I8 v1,I8 v2){return (std::max)(v1,v2);}
I16 maximum(I16 v1,I16 v2){return (std::max)(v1,v2);}
I32 maximum(I32 v1,I32 v2){return (std::max)(v1,v2);}
F32 maximum(F32 v1,F32 v2){return (std::max)(v1,v2);}
F64 maximum(F64 v1,F64 v2){return (std::max)(v1,v2);}
UI8 minimum(UI8 v1,UI8 v2){return (std::min)(v1,v2);}
UI16 minimum(UI16 v1,UI16 v2){return (std::min)(v1,v2);}
UI32 minimum(UI32 v1,UI32 v2){return (std::min)(v1,v2);}
I8 minimum(I8 v1,I8 v2){return (std::min)(v1,v2);}
I16 minimum(I16 v1,I16 v2){return (std::min)(v1,v2);}
I32 minimum(I32 v1,I32 v2){return (std::min)(v1,v2);}
F32 minimum(F32 v1,F32 v2){return (std::min)(v1,v2);}
F64 minimum(F64 v1,F64 v2){return (std::min)(v1,v2);}


F32 absolute(UI8 v1){return (F32)v1;}
F32 absolute(UI16 v1){return (F32)v1;}
F32 absolute(UI32 v1){return (F32)v1;}
F32 absolute(I8 v1){return std::abs(static_cast<F32>(v1));}
F32 absolute(I16 v1){return std::abs(static_cast<F32>(v1));}
F32 absolute(I32 v1){return std::abs(static_cast<F32>(v1));}
F32 absolute(F32 v1){return std::abs(v1);}
F64 absolute(F64 v1){return std::abs(v1);}

F32 normValue(UI8 v1,int ){return (F32)v1;}
F32 normValue(UI16 v1,int ){return (F32)v1;}
F32 normValue(UI32 v1,int ){return (F32)v1;}
F32 normValue(I8 v1,int ){return std::abs(static_cast<F32>(v1));}
F32 normValue(I16 v1,int ){return std::abs(static_cast<F32>(v1));}
F32 normValue(I32 v1,int){return std::abs(static_cast<F32>(v1));}
F32 normValue(F32 v1,int ){return std::abs(v1);}
F64 normValue(F64 v1,int ){return std::abs(v1);}

F32 normPowerValue(UI8 v1,int p){
    if(p==0||p==1)
        return normValue(v1);
    else if(p==2)
        return (F32)v1*(F32)v1;
    else
        return std::pow(normValue(v1),p);
}
F32 normPowerValue(UI16 v1,int p){
    if(p==0||p==1)
        return normValue(v1);
    else if(p==2)
        return (F32)v1*(F32)v1;
    else
        return std::pow(normValue(v1),p);
}
F32 normPowerValue(UI32 v1,int p){
    if(p==0||p==1)
        return normValue(v1);
    else if(p==2)
        return (F32)v1*(F32)v1;
    else
        return std::pow(normValue(v1),p);
}
F32 normPowerValue(I8 v1,int p){
    if(p==0||p==1)
        return normValue(v1);
    else if(p==2)
        return (F32)v1*(F32)v1;
    else
        return std::pow(normValue(v1),p);
}
F32 normPowerValue(I16 v1,int p){
    if(p==0||p==1)
        return normValue(v1);
    else if(p==2)
        return (F32)v1*(F32)v1;
    else
        return std::pow(normValue(v1),p);
}
F32 normPowerValue(I32 v1,int p){
    if(p==0||p==1)
        return normValue(v1);
    else if(p==2)
        return (F32)v1*(F32)v1;
    else
        return std::pow(normValue(v1),p);
}
F32 normPowerValue(F32 v1,int p){
    if(p==0||p==1)
        return normValue(v1);
    else if(p==2)
        return v1*v1;
    else
        return std::pow(normValue(v1),p);
}
F64 normPowerValue(F64 v1,int p){
    if(p==0||p==1)
        return normValue(v1);
    else if(p==2)
        return v1*v1;
    else
        return std::pow(normValue(v1),p);
}

F32 distance(UI8 v1,UI8 v2, int){return std::abs(static_cast<F32>(v1) -static_cast<F32>(v2));}
F32 distance(UI16 v1,UI16 v2, int){return std::abs(static_cast<F32>(v1) -static_cast<F32>(v2));}
F32 distance(UI32 v1,UI32 v2, int){return std::abs(static_cast<F32>(v1) -static_cast<F32>(v2));}
F32 distance(I8 v1,I8 v2, int){return std::abs(static_cast<F32>(v1) -static_cast<F32>(v2));}
F32 distance(I16 v1,I16 v2, int){return std::abs(static_cast<F32>(v1) -static_cast<F32>(v2));}
F32 distance(I32 v1,I32 v2, int){return std::abs(static_cast<F32>(v1) -static_cast<F32>(v2));}
F32 distance(F32 v1,F32 v2, int){return std::abs(static_cast<F32>(v1) -static_cast<F32>(v2));}
F64 distance(F64 v1,F64 v2, int){return std::abs(static_cast<F32>(v1) -static_cast<F32>(v2));}

F32 productInner(UI8 v1,UI8 v2){return static_cast<F32>(v1)*static_cast<F32>(v2);}
F32 productInner(UI16 v1,UI16 v2){return static_cast<F32>(v1)*static_cast<F32>(v2);}
F32 productInner(UI32 v1,UI32 v2){return static_cast<F32>(v1)*static_cast<F32>(v2);}
F32 productInner(I8 v1,I8 v2){return static_cast<F32>(v1)*static_cast<F32>(v2);}
F32 productInner(I16 v1,I16 v2){return static_cast<F32>(v1)*static_cast<F32>(v2);}
F32 productInner(I32 v1,I32 v2){return static_cast<F32>(v1)*static_cast<F32>(v2);}
F32 productInner(F32 v1,F32 v2){return static_cast<F32>(v1)*static_cast<F32>(v2);}
F64 productInner(F64 v1,F64 v2){return v1*v2;}
F32 round(F32 v1){return std::floor(v1+0.5f);}
F64 round(F64 v1){return std::floor(v1+0.5);}

F32 squareRoot(F32 v1){return std::sqrt(v1);}
F64 squareRoot(F64 v1){return std::sqrt(v1);}
}
