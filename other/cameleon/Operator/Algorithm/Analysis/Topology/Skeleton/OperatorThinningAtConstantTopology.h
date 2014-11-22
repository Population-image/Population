#ifndef OPERATORTHINNINGATCONSTANTTOPOLOGY_H
#define OPERATORTHINNINGATCONSTANTTOPOLOGY_H
#include"COperator.h"
#include"data/mat/MatN.h"
#include"algorithm/Analysis.h"
using namespace pop;
class OperatorThinningAtConstantTopologyMatN : public COperator
{
public:
    OperatorThinningAtConstantTopologyMatN();
    void exec();
    COperator * clone();
    struct foo
    {
        template<typename Type>
        void operator()(MatN<2,Type> * in1cast,string file,BaseMatN * &out)
        {

            MatN<2,unsigned char> *outcast = new MatN<2,unsigned char> (in1cast->getDomain());
            *outcast    = Analysis::thinningAtConstantTopology2d(*in1cast);

            out = outcast;

        }
        template<typename Type>
        void operator()(MatN<3,Type> * in1cast,string file,BaseMatN * &out)
        {

            MatN<3,unsigned char> *outcast = new MatN<3,unsigned char> (in1cast->getDomain());
            *outcast    = Analysis::thinningAtConstantTopology3d(*in1cast,file.c_str());

            out = outcast;

        }
    };

};
#endif // OPERATORTHINNINGATCONSTANTTOPOLOGY_H
