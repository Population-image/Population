#ifndef OPERATORRANDOMSTRUCTUREMatN_H
#define OPERATORRANDOMSTRUCTUREMatN_H

#include"COperator.h"

class OperatorRandomStructureMatN : public COperator
{
public:
    OperatorRandomStructureMatN();
    void exec();
    COperator * clone();


};

#endif // OPERATORRANDOMSTRUCTUREMatN_H
