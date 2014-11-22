#ifndef OPERATORSaveRawRAWMatN_H
#define OPERATORSaveRawRAWMatN_H

#include"COperator.h"
#include"data/mat/MatN.h"
#include"data/mat/MatNInOut.h"
using namespace pop;

class OperatorSaveRawMatN: public COperator
{
public:
    OperatorSaveRawMatN();
    void exec();
    COperator * clone();
    struct foo
    {
        template<int DIM,typename Type>
        void operator()(MatN<DIM,Type> * in1cast,string file)throw(pexception){

//            ofstream  out(file.c_str());
//            if (out.fail()){
//                throw(std::string("Cannot open file: "+file));
//            }
//            else{
//                 MatNInOutPgm::writeRaw(out,*in1cast);
//            }
        }
    };
};

#endif // OPERATORSaveRawRAWMatN_H
