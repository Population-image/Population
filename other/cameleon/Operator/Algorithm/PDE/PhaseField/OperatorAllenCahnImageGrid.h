#ifndef OPERATORALLENCAHNMatN_H
#define OPERATORALLENCAHNMatN_H

#include"COperator.h"
#include"algorithm/PDE.h"
using namespace pop;
class OperatorAllenCahnMatN : public COperator
{
public:
    OperatorAllenCahnMatN();
    void exec();
    COperator * clone();
    struct foo
    {
        template<int DIM,typename Type>
        void operator()(MatN<DIM,Type> * phaseinit,BaseMatN* bulk, int nbrsteps, BaseMatN * &label,BaseMatN * &phaseend)throw(pexception)
        {



            if(MatN<DIM,unsigned char> * bulkcast = dynamic_cast<MatN<DIM,unsigned char> *>(bulk))
            {

                MatN<DIM,pop::F64> * phasefield = new  MatN<DIM,pop::F64>;
                * phasefield = PDE::allenCahn(* phaseinit,* bulkcast,nbrsteps);
                MatN<DIM,Type> * phase = new MatN<DIM,Type>  (* phaseinit);
                label=phase;
                phaseend = phasefield;
            }
            else{
                throw(pexception("Pixel/voxel type of Bulk image must have 1 byte "));
            }


        }
    };

};
#endif // OPERATORALLENCAHNMatN_H
