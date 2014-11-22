#ifndef OPERATORMEDIALAXISMatN_H
#define OPERATORMEDIALAXISMatN_H

#include"COperator.h"
#include"data/mat/MatN.h"
#include"algorithm/Analysis.h"
using namespace pop;
class OperatorMedialAxisMatN : public COperator
{
public:
    OperatorMedialAxisMatN();
    void exec();
    COperator * clone();
    void initState();
    struct foo
    {
        template<int DIM,typename Type>
        void operator()(MatN<DIM,Type> * in1cast,int norm,BaseMatN * &out)
        {

            MatN<DIM,unsigned char> *outcast = new MatN<DIM,unsigned char> (in1cast->getDomain());
            (*outcast) = Analysis::medialAxis(  *in1cast,norm);
            out = outcast;

        }
    };

};
#endif // OPERATORMEDIALAXISMatN_H
