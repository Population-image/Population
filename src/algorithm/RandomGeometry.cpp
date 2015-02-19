#include"algorithm/RandomGeometry.h"
#include"data/mat/MatNDisplay.h"
namespace pop{


void  RandomGeometry::rhombohedron( ModelGermGrain3  & grain,const Distribution&distradius,const Distribution& distangle, const DistributionMultiVariate&distorientation )
{
    if(distorientation.getNbrVariable() !=3)
    {
        std::cerr<<"In RandomGeometry::rhombohedron, for d = 3, the angle distribution std::vector must have 3 variables with d the space dimension";
    }
    for( std::vector<Germ<3> * >::iterator it= grain.grains().begin();it!=grain.grains().end();it++ )
    {
        Germ<3> * g = (*it);
        GrainEquilateralRhombohedron * box = new GrainEquilateralRhombohedron();
        box->setGerm(*g);
        box->radius= distradius.randomVariable();
        box->setAnglePlane( distangle.randomVariable());
        VecF32 v = distorientation.randomVariable();
        for(int i=0;i<3;i++)
            box->orientation.setAngle_ei(v(i),i);
        (*it) = box;
        delete g;
    }
}
void RandomGeometry::cylinder( pop::ModelGermGrain3  & grain,const Distribution&  distradius,const Distribution&  distheight,const DistributionMultiVariate& distorientation )
{

    if(distorientation.getNbrVariable()!=3)
    {
        std::cerr<<"In RandomGeometry::cylinder, for d = 3, the angle distribution std::vector must have 3 variables with d the space dimension";
    }
    for( std::vector<Germ<3> * >::iterator it= grain.grains().begin();it!=grain.grains().end();it++ )
    {
        Germ<3> * g = (*it);
        GrainCylinder * box = new GrainCylinder();
        box->setGerm(*g);
        box->radius = distradius.randomVariable();
        box->height = distheight.randomVariable();
        VecF32 v = distorientation.randomVariable();
        for(int i=0;i<3;i++)
            box->orientation.setAngle_ei(v(i),i);
        (*it) = box;
        delete g;
    }
}

MatN<2,UI8 > RandomGeometry::diffusionLimitedAggregation2D(int size,int nbrwalkers){
    MatN<2,UI8 > in(size,size);
    in(VecN<2,F32>(in.getDomain())*0.5)=255;

    DistributionUniformInt d(0,1);
    DistributionUniformInt dface(0,MatN<2,UI8 >::DIM-1);

    DistributionUniformInt dpos(0,in.sizeI()-1);

    MatN<2,UI8 >::IteratorENeighborhood N(in.getIteratorENeighborhood(1,0));

    MatNDisplay windows;

    for(int index_walker =0;index_walker<nbrwalkers;index_walker++){
        std::cout<<"walker "<<index_walker<<std::endl;
        MatN<2,UI8 >::E randomwalker;
        //GENERATE INIT POSITION
        int face= static_cast<int>(dface.randomVariable());
        int noindex= face/2;
        for(int j =0;j<MatN<2,UI8 >::DIM;j++){
            if(j==noindex){
                randomwalker(j)=(in.sizeI()-1)*(face%2);
            }
            else{
                randomwalker(j)=static_cast<int>(dpos.randomVariable());
            }
        }
        bool touch=false;
        while(touch==false){
            //MOUVEMENT
            MatN<2,UI8 >::E temp;
            do{
                temp=randomwalker;
                for(int i =0;i<MatN<2,UI8 >::DIM;i++){
                    temp(i)+=static_cast<int>((d.randomVariable()*2-1));
                }
            }while(in.isValid(temp)==false);
            randomwalker = temp;
            N.init(randomwalker);
            while(N.next()==true){
                if(in(N.x())!=0)
                {
                    touch=true;
                    in(randomwalker)=255;
                    if(index_walker%50==0)
                        windows.display(in);
                }
            }

        }
    }
    return in;
}
MatN<3,UI8 > RandomGeometry::diffusionLimitedAggregation3D(int size,int nbrwalkers){
    MatN<3,UI8 > in(size,size,size);
    in(VecN<3,F32>(in.getDomain())*0.5)=255;

    DistributionUniformInt d(0,1);
    DistributionUniformInt dface(0,MatN<3,UI8 >::DIM-1);

    DistributionUniformInt dpos(0,in.sizeI()-1);

    MatN<3,UI8 >::IteratorENeighborhood N(in.getIteratorENeighborhood(1,1));
    for(int index_walker =0;index_walker<nbrwalkers;index_walker++){
        std::cout<<"walker "<<index_walker<<std::endl;

        MatN<3,UI8 >::E randomwalker;
        //GENERATE INIT POSITION
        int face= static_cast<int>(dface.randomVariable());
        int noindex= face/2;
        for(int i =0;i<MatN<3,UI8 >::DIM;i++){
            if(i==noindex){
                randomwalker(i)=(in.sizeI()-1)*(face%2);
            }
            else{
                randomwalker(i)=static_cast<int>(dpos.randomVariable());
            }
        }
        bool touch=false;
        while(touch==false){
            //MOUVEMENT
            MatN<3,UI8 >::E temp;
            do{
                temp=randomwalker;
                for(int i =0;i<MatN<3,UI8 >::DIM;i++){
                    temp(i)+=static_cast<int>((d.randomVariable()*2-1));
                }
            }while(in.isValid(temp)==false);
            randomwalker = temp;
            N.init(randomwalker);
            while(N.next()==true){
                if(in(N.x())!=0)
                {
                    touch=true;
                    in(randomwalker)=255;
                }
            }

        }
    }
    return in;
}
}
