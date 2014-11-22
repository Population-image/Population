#ifndef OPERATORCLUSTERMAXMatN_H
#define OPERATORCLUSTERMAXMatN_H

#include"COperator.h"
#include"data/mat/MatN.h"
#include"algorithm/Processing.h"
using namespace pop;
class OperatorClusterMaxMatN : public COperator
{
public:
    OperatorClusterMaxMatN();
    void exec();
    COperator * clone();
    void initState();
    struct foo{
        template<int DIM,typename Type>
        void operator()(MatN<DIM,Type> * binarycast,int norm,BaseMatN * &maxcluster)
        {
            MatN<DIM,Type> * maxclustercast = new MatN<DIM,Type>(binarycast->getDomain());
            *maxclustercast = Processing::clusterMax(*binarycast,  norm);
            maxcluster = maxclustercast;
        }
    };
};

#endif // OPERATORCLUSTERMAXMatN_H
