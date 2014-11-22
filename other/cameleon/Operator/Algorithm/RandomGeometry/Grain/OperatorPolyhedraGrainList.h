#ifndef OPERATORPOLYHEDRAGRAINLIST_H
#define OPERATORPOLYHEDRAGRAINLIST_H
#include"COperator.h"
#include"algorithm/RandomGeometry.h"
using namespace pop;
class OperatorPolyhedraGermGrain : public COperator
{
public:
    OperatorPolyhedraGermGrain();
    void exec();
    COperator * clone();
    void initState();
    struct foo
    {

        void operator()(GermGrain2 * in,vector<Distribution * >v_dist1, vector<Distribution * >v_dist2,Distribution * angle)
        {
            vector<Distribution> dradius;
            for(int i;i<(int)v_dist1.size();i++){
                dradius.push_back(*v_dist1[i]);
            }
            vector<Distribution>  dnormal;
            for(int i;i<(int)v_dist2.size();i++){
                dnormal.push_back(*v_dist2[i]);
            }
            vector<Distribution> vangle;
            vangle.push_back(*angle);
            RandomGeometry::polyhedra(*in,dradius,dnormal,vangle);
        }
        void operator()(GermGrain3 * in,vector<Distribution * >v_dist1, vector<Distribution * >v_dist2,Distribution * anglex,Distribution * angley,Distribution * anglez)
        {
            vector<Distribution> dradius;
            for(int i;i<(int)v_dist1.size();i++){
                dradius.push_back(*v_dist1[i]);
            }
            vector<Distribution>  dnormal;
            for(int i;i<(int)v_dist2.size();i++){
                dnormal.push_back(*v_dist2[i]);
            }
            vector<Distribution> vangle;
            vangle.push_back(*anglex);
            vangle.push_back(*angley);
            vangle.push_back(*anglez);
            RandomGeometry::polyhedra(*in,dradius,dnormal,vangle);
        }
    };

};
#endif // OPERATORPOLYHEDRAGRAINLIST_H
