#ifndef OPERATORPERMEABILITYMatN_H
#define OPERATORPERMEABILITYMatN_H

#include"COperator.h"
#include"data/mat/MatN.h"
#include"data/mat/MatN.h"
#include"algorithm/PDE.h"
using namespace pop;
class OperatorpermeabilityMatN : public COperator
{
private:
    int dim;
public:
    OperatorpermeabilityMatN();
    void exec();
    COperator * clone();
         void updateMarkingAfterExecution();
         void initState();
    struct foo
    {
        template<int DIM,typename Type>
        void operator()(MatN<DIM,Type> * in1cast,int direction,double error,VecF64& k,BaseMatN * &vx,BaseMatN * &vy,BaseMatN * &vz){
            if(DIM==2)
                vz=NULL;
            MatN<DIM,VecN<DIM,pop::F64> >  velocity;


            k = PDE::permeability(* in1cast,velocity,direction,error);

            for(int i =0;i<DIM;i++)
            {
                   MatN<DIM, pop::F64  >  * velocitydirection = new MatN<DIM, pop::F64  > (velocity.getDomain());
                   typename MatN<DIM, pop::F64  >::IteratorEDomain it (velocitydirection->getIteratorEDomain());
                   it.init();
                   while(it.next())
                   {
                       (*velocitydirection)(it.x()) =  (velocity)(it.x())(i);
                   }
                   if(i==0)
                      vx =velocitydirection;
                   else if(i==1)
                      vy =velocitydirection;
                   else
                      vz =velocitydirection;

            }
        }
    };

};
#endif // OPERATORPERMEABILITYMatN_H
