#ifndef OPERATORCONVERTMatN2IMAGEQT_H
#define OPERATORCONVERTMatN2IMAGEQT_H

#include"COperator.h"
#include"data/mat/MatN.h"
#include"algorithm/Visualization.h"
using namespace pop;

class OperatorMatN2ImageQT : public COperator
{
public:
    OperatorMatN2ImageQT();
    void exec();
    COperator * clone();
    struct foo{
        template<typename Type>
        void operator()(MatN<2,Type> * ){

        }
    };

};

#endif // OPERATORCONVERTMatN2IMAGEQT_H
