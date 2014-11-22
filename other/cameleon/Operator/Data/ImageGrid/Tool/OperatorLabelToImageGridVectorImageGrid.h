#ifndef OPERATORLABELTOMatNVECTORMatN_H
#define OPERATORLABELTOMatNVECTORMatN_H

#include"COperator.h"
#include"data/mat/MatN.h"
#include"algorithm/Analysis.h"
class OperatorLabelToMatNVectorMatN : public COperator
{
public:
    OperatorLabelToMatNVectorMatN();
    void exec();
    COperator * clone();
    struct foo
    {
        template<int DIM,typename Type>
        void operator()(pop::MatN<DIM,Type> * in1cast, vector<pop::BaseMatN *> &out)
        {
            vector<pop::VecN<DIM,int> >xmin,xmax;
            vector<pop::MatN<DIM,unsigned char> > outcast  = pop::Analysis::labelToMatrices(* in1cast,xmin,xmax);
            for(int i =0;i<(int)outcast.size();i++){
                out.push_back(new pop::MatN<DIM,unsigned char>((outcast[i])));
            }
        }
    };

};
#endif // OPERATORLABELTOMatNVECTORMatN_H
