
#include "algorithm/Analysis.h"
#include "algorithm/Processing.h"
#include "data/notstable/CharacteristicCluster.h"
namespace pop {
CharacteristicClusterDistance::~CharacteristicClusterDistance(){

}

CharacteristicCluster::CharacteristicCluster()
    :_mass(0),_min(std::numeric_limits<int>::max(),std::numeric_limits<int>::max()),_max(std::numeric_limits<int>::min(),std::numeric_limits<int>::min())//,_barycentre(0,0)
{

}
void CharacteristicCluster::setLabel(int label){
    _label = label;
}
Vec2I32 CharacteristicCluster::size()const{
    return _max-_min;
}
Vec2I32 CharacteristicCluster::center()const{
    return (_max +_min)/2;
}

void CharacteristicCluster::addPoint(const Vec2I32 & x){
    _min = pop::minimum(_min,x);
    _max = pop::maximum(_max,x);
    //_barycentre=(_barycentre*_mass+ Vec2F32(x))/(_mass+1);
    _mass++;

}

F32 CharacteristicClusterDistanceMass::operator ()(const CharacteristicCluster& a,const CharacteristicCluster& b){
    if(a._mass!=0&&b._mass!=0)
        return std::abs(a._mass-b._mass)*1./std::min(a._mass,b._mass);
    else
        return std::numeric_limits<F32>::max();
}

F32 CharacteristicClusterDistanceHeight::operator ()(const CharacteristicCluster& a,const CharacteristicCluster& b){
    if(a.size()(0)!=0&&b.size()(0)!=0)
        return std::abs(a.size()(0)-b.size()(0))*1./std::min(a.size()(0),b.size()(0));
    else
        return std::numeric_limits<F32>::max();
}

F32 CharacteristicClusterDistanceWidth::operator ()(const CharacteristicCluster& a,const CharacteristicCluster& b){
    if(a.size()(1)!=0&&b.size()(1)!=0)
        return std::abs(a.size()(1)-b.size()(1))*1./std::min(a.size()(1),b.size()(1));
    else
        return std::numeric_limits<F32>::max();
}


F32 CharacteristicClusterDistanceWidthInterval::operator ()(const CharacteristicCluster& a,const CharacteristicCluster& b){
    if(a.size()(1)!=0&&b.size()(1)!=0)
        return std::abs(a.center()(1)-b.center()(1))*1./std::max(a.size()(1),b.size()(1));
    else
        return std::numeric_limits<F32>::max();
}

F32 CharacteristicClusterDistanceHeightInterval::operator ()(const CharacteristicCluster& a,const CharacteristicCluster& b){
    if(a.size()(0)!=0&&b.size()(0)!=0)
        return std::abs(a.center()(0)-b.center()(0))*1./std::max(a.size()(0),b.size()(0));
    else
        return std::numeric_limits<F32>::max();
}




CharacteristicClusterFilter::CharacteristicClusterFilter()
    :_min(0),_max(std::numeric_limits<int>::max())
{

}

bool CharacteristicClusterFilter::operator ()(const CharacteristicCluster& ){
    return true;
}

CharacteristicClusterFilter::~CharacteristicClusterFilter(){

}
bool CharacteristicClusterFilterMass::operator ()(const CharacteristicCluster& a){
    return (a._mass>=_min&&a._mass<_max);
}
bool CharacteristicClusterFilterHeight::operator ()(const CharacteristicCluster& a){
    return (a.size()(0)>=_min&&a.size()(0)<_max);
}
bool CharacteristicClusterFilterWidth::operator ()(const CharacteristicCluster& a){
    return (a.size()(1)>=_min&&a.size()(1)<_max);
}
bool CharacteristicClusterFilterAsymmetryHeightPerWidth::operator ()(const CharacteristicCluster& a){
    if(a.size()(0)!=0){
        F32 ratio =a.size()(0)*1.0/a.size()(1);
        return (ratio>=_min&&ratio<_max);
    }
    else
        return false;
}
Vec<CharacteristicCluster> applyCharacteristicClusterFilter(const Vec<CharacteristicCluster>& v_cluster, CharacteristicClusterFilter * filter){
    Vec<CharacteristicCluster> v_cluster_out;
    for(unsigned int i=0;i<v_cluster.size();i++){
        if(filter->operator ()(v_cluster[i]))
            v_cluster_out.push_back(v_cluster[i]);
    }
    return v_cluster_out;
}
pop::Mat2UI32 applyClusterFilter(const pop::Mat2UI32& labelled_image, Vec<CharacteristicClusterFilter*> v_filter  ){

    int max_value = Analysis::maxValue(labelled_image);
    Vec<CharacteristicCluster> v_cluster(max_value+1);
    for(unsigned int i =0;i<v_cluster.size();i++){
        v_cluster[i]._label=i;
    }
    ForEachDomain2D(x,labelled_image){
        if(labelled_image(x)>0)
            v_cluster[labelled_image(x)].addPoint(x);
    }
    Vec<CharacteristicCluster> v_filter_out(v_cluster);
    for(unsigned int i =0;i<v_filter.size();i++){
        CharacteristicClusterFilter * filter = v_filter[i];
        v_filter_out = applyCharacteristicClusterFilter(v_filter_out,filter);
    }
    Vec<int> v_hit(max_value+1,0);
    for(unsigned int i =0;i<v_filter_out.size();i++){
        v_hit(v_filter_out[i]._label)=1;
    }
    pop::Mat2UI32 labelled_image_out(labelled_image.getDomain());

    ForEachDomain2D(xx,labelled_image){
        if(labelled_image(xx)>0&& v_hit(labelled_image(xx))==1)
            labelled_image_out(xx)=labelled_image(xx);
    }
    return  Processing::greylevelRemoveEmptyValue(labelled_image_out);
}
namespace Private{
bool sortMyFunction (std::pair<F32,int> i,std::pair<F32,int> j) { return (i.first<j.first); }
bool sortMyFunctionLeft (std::pair<int,int>  i,std::pair<int,int> j) { return (i.first<j.first); }
}

Vec<Vec<Mat2UI8> > applyGraphCluster(const pop::Mat2UI32& labelled_image, Vec<CharacteristicClusterDistance*> v_dist, Vec<F32> v_weight ,F32 threshold ){

    int max_value = Analysis::maxValue(labelled_image);
    Vec<CharacteristicCluster> v_cluster(max_value+1);
    for(unsigned int i =0;i<v_cluster.size();i++){
        v_cluster[i]._label=i;
    }
    ForEachDomain2D(x,labelled_image){
        if(labelled_image(x)>0)
            v_cluster[labelled_image(x)].addPoint(x);
    }


    Vec<Vec<int> >v_v_neigh;
    v_v_neigh.push_back(Vec<int>());
    for(unsigned int i=1;i<v_cluster.size();i++){
        Vec<int> v_neight;
        for(unsigned int j=i+1;j<v_cluster.size();j++){
            F32 sum=0;
            for(unsigned int k=0;k<v_dist.size();k++){
                CharacteristicCluster & a = v_cluster[i];
                CharacteristicCluster & b = v_cluster[j];
                sum+=v_weight[k]*v_dist[k]->operator()(a,b);
            }
            if(sum<threshold){
                v_neight.push_back(j);
            }
        }
        v_v_neigh.push_back(v_neight);
    }
    pop::Vec<Vec2I32> v_xmin,v_xmax;
    Vec<Mat2UI8> labels= Analysis::labelToMatrices(labelled_image,v_xmin,v_xmax);






    //    Mat2RGBUI8 graph= Visualization::labelToRandomRGB(labelled_image);
    //    for(unsigned int i=0;i<v_v_neigh.size();i++){
    //        for(unsigned int j=0;j<v_v_neigh[i].size();j++){

    //            int index = v_v_neigh[i][j];
    //            //            std::cout<<i<<"<->"<<index <<std::endl;
    //            Draw::circle(graph,v_cluster[i].center(),20,RGBUI8(255,255,255),1);
    //            Draw::circle(graph,v_cluster[index].center(),20,RGBUI8(255,255,255),1);
    //            Draw::line(graph,v_cluster[i].center(),v_cluster[index].center(),RGBUI8(255,0,0),3);
    //        }
    //    }
    //    graph.display();
    Vec<Vec<Mat2UI8> > v_word;
    Vec<Vec<std::pair<int,int> > > v_word_label;
    Vec<int> v_hit(v_v_neigh.size(),-1);
    for(unsigned int i=0;i<v_v_neigh.size();i++){
        if(v_v_neigh(i).size()>0){
            int index;
            if(v_hit(i)==-1){
                v_hit(i)=v_word_label.size();
                index = v_word_label.size();
                v_word_label.push_back(Vec<std::pair<int,int> >());
                v_word_label(index).push_back(std::make_pair(v_cluster[i].center()(1) , i));
                //                v_word.push_back(Vec<Mat2UI8>());
                //                v_word(index).push_back(labels(i-1));
                //                std::cout<<i<<std::endl;
                //                labels(i-1).display();
            }
            else{
                index = v_hit(i);
            }
            for(unsigned int j=0;j<v_v_neigh[i].size();j++){
                if(v_hit(v_v_neigh(i)(j))==-1){
                    v_hit(v_v_neigh(i)(j))=index;

                    v_word_label(index).push_back(std::make_pair(v_cluster[v_v_neigh(i)(j)].center()(1) , v_v_neigh(i)(j)) );
                    //                    v_word(index).push_back(labels(v_v_neigh(i)(j)-1));
                    //                    std::cout<<v_v_neigh(i)(j)<<std::endl;
                    //                    labels(v_v_neigh(i)(j)-1).display();
                }
            }
        }
    }
    for(unsigned int i=0;i<v_word_label.size();i++){
        v_word.push_back(Vec<Mat2UI8>());
        Vec<std::pair<int,int> > & _v_temp = v_word_label(i);
        std::sort(_v_temp.begin(),_v_temp.end(),Private::sortMyFunctionLeft);
        for(unsigned int j=0;j<_v_temp.size();j++){
            //            labels(_v_temp(j).second-1).display();
            v_word(i).push_back(labels(_v_temp(j).second-1) );
        }
    }

    return v_word;
}


//version 2
CharacteristicMass::CharacteristicMass()
    :_mass(0)
{}

int CharacteristicMass::getMass()const{
    return _mass;
}

void CharacteristicLabel::setLabel(int label){
    _label = label;
}

int CharacteristicLabel::getLabel()const{
    return _label;
}
}
