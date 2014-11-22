#include"treillisnumeric.h"

namespace pop{


bool TreillisNumeric::sortMyFunctionLeft2 (std::pair<int,int>  i,std::pair<int,int> j) { return (i.first<j.first); }


Vec<Vec<Mat2UI8> > TreillisNumeric::applyGraphClusterTT(const pop::Mat2UI32& labelled_image, Vec<CharacteristicClusterDistance*> v_dist, Vec<Distribution> v_distribution ,double threshold ){

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
            double sum=0;
            for(unsigned int k=0;k<v_dist.size();k++){
                CharacteristicCluster & a = v_cluster[i];
                CharacteristicCluster & b = v_cluster[j];
                sum+=v_distribution[k](v_dist[k]->operator()(a,b));
            }
            if(sum<threshold){
                v_neight.push_back(j);
            }
        }
        v_v_neigh.push_back(v_neight);
    }
    pop::Vec<Vec2I32> v_xmin,v_xmax;
    Vec<Mat2UI8> labels= Analysis::labelToMatrices(labelled_image,v_xmin,v_xmax);





    Mat2RGBUI8 graph= Visualization::labelToRandomRGB(labelled_image);
    for(unsigned int i=0;i<v_v_neigh.size();i++){
        for(unsigned int j=0;j<v_v_neigh[i].size();j++){

            int index = v_v_neigh[i][j];
            //            std::cout<<i<<"<->"<<index <<std::endl;
            Draw::circle(graph,v_cluster[i].center(),20,RGBUI8(255,255,255),1);
            Draw::circle(graph,v_cluster[index].center(),20,RGBUI8(255,255,255),1);
            Draw::line(graph,v_cluster[i].center(),v_cluster[index].center(),RGBUI8(255,0,0),3);
        }
    }
    disp.display(graph);

  graph.save("test_link.png");


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
        std::sort(_v_temp.begin(),_v_temp.end(),TreillisNumeric::sortMyFunctionLeft2);
        for(unsigned int j=0;j<_v_temp.size();j++){
            //            labels(_v_temp(j).second-1).display();
            v_word(i).push_back(labels(_v_temp(j).second-1) );
        }
    }
    return v_word;
}


void TreillisNumeric::treillisNumerique(Mat2UI8 m,int nbr_pixels_width_caracter){
    //    disp2.display(m);
    //    std::cout<<m.getDomain()<<std::endl;

    clock_t start_global2, end_global2;
    start_global2 = clock();
    Mat2UI8 threhold =        Processing::thresholdNiblackMethod(m,0.2,15*nbr_pixels_width_caracter,-10);
    end_global2 = clock();
    std::cout<<"threshold : "<<(double) (end_global2 - start_global2) / CLOCKS_PER_SEC<<std::endl;
    threhold.save("test_seg.png");
//    threhold.display();

    //    threhold.display();
    clock_t start_global, end_global;
    start_global = clock();
    Mat2UI32 label = Processing::clusterToLabel(threhold,0);


    CharacteristicClusterFilterMass filter_mass;
    filter_mass._min = nbr_pixels_width_caracter*30;
    filter_mass._max = nbr_pixels_width_caracter*1000;

    CharacteristicClusterFilterAsymmetryHeightPerWidth filter_asymmetry;
    filter_asymmetry._min =0.5;
    filter_asymmetry._max = 20;
    Vec<CharacteristicClusterFilter*> v_filter;
    v_filter.push_back(&filter_mass);
    v_filter.push_back(&filter_asymmetry);

    label =  applyClusterFilter(label,v_filter );
    end_global = clock();
    //Visualization::labelToRandomRGB(label).display();
    Visualization::labelToRandomRGB(label).save("test_label.png");

    std::cout<<"unit : "<<(double) (end_global - start_global) / CLOCKS_PER_SEC<<std::endl;


    pop::Vec<CharacteristicClusterDistance*> v_dist;
    pop::Vec<Distribution> v_weight;


    CharacteristicClusterDistanceHeight dist_height;
    CharacteristicClusterDistanceWidth dist_width;
    CharacteristicClusterDistanceWidthInterval dist_interval_width;
    CharacteristicClusterDistanceHeightInterval dist_interval_height;


    v_dist.push_back(&dist_height);
    v_dist.push_back(&dist_width);
    v_dist.push_back(&dist_interval_width);
    v_dist.push_back(&dist_interval_height);

    v_weight.push_back(Distribution("(8*x)^2"));
    v_weight.push_back(Distribution("(x)^2"));
    v_weight.push_back(Distribution("(0.5*x)^2"));
    v_weight.push_back(Distribution("(4*x)^2"));



    clock_t start_global3, end_global3;
    start_global3 = clock();
    Vec<Vec<Mat2UI8> > v_v_img = applyGraphClusterTT(label,v_dist,v_weight,4);
    end_global3 = clock();
    std::cout<<"filter : "<<(double) (end_global3 - start_global3) / CLOCKS_PER_SEC<<std::endl;

    clock_t start_global4, end_global4;
    start_global4 = clock();
    std::string str;
    for(unsigned int i=0;i<v_v_img.size();i++){
        std::string str2;
        bool parse=false;
        if(v_v_img(i).size()>=5){
            for(unsigned int j=0;j<v_v_img(i).size();j++){

                char c =  ocr.parseMatrix(v_v_img(i)(j));
//                std::cout<<i<<" "<<c<<" "<<ocr.characterConfidence() <<std::endl;
                //                v_v_img(i)(j).display();
                if( ocr.characterConfidence()>-20){
                    str2.push_back(c);
                    parse=true;
                }
            }
            //        std::cout<<str2<<std::endl;
            if(str.size()!=0&&parse==true){

                str=str+" ";
            }
            str=str+str2;
        }
    }
    std::cout<<str<<std::endl;
    end_global4 = clock();
    std::cout<<"OCR : "<<(double) (end_global4- start_global4) / CLOCKS_PER_SEC<<std::endl;
}
void TreillisNumeric::treillisNumeriqueVersionText(Mat2UI32 label, int nbr_pixels_width_caracter){




    CharacteristicClusterFilterMass filter_mass;
    filter_mass._min = nbr_pixels_width_caracter*20;
    filter_mass._max = nbr_pixels_width_caracter*1000;

    CharacteristicClusterFilterAsymmetryHeightPerWidth filter_asymmetry;
    filter_asymmetry._min =0.5;
    filter_asymmetry._max = 20;
    Vec<CharacteristicClusterFilter*> v_filter;
    v_filter.push_back(&filter_mass);
    v_filter.push_back(&filter_asymmetry);

    label =  applyClusterFilter(label,v_filter );

    Visualization::labelToRandomRGB(label).display("label");

    pop::Vec<CharacteristicClusterDistance*> v_dist;
    pop::Vec<Distribution> v_weight;


    CharacteristicClusterDistanceHeight dist_height;
    CharacteristicClusterDistanceWidth dist_width;
    CharacteristicClusterDistanceWidthInterval dist_interval_width;
    CharacteristicClusterDistanceHeightInterval dist_interval_height;


    v_dist.push_back(&dist_height);
    v_dist.push_back(&dist_width);
    v_dist.push_back(&dist_interval_width);
    v_dist.push_back(&dist_interval_height);
    v_weight.push_back(Distribution("(8*x)^2"));
    v_weight.push_back(Distribution("(x)^2"));
    v_weight.push_back(Distribution("(0.5*x)^2"));
    v_weight.push_back(Distribution("(4*x)^2"));


    clock_t start_global3, end_global3;
    start_global3 = clock();
    Vec<Vec<Mat2UI8> > v_v_img = applyGraphClusterTT(label,v_dist,v_weight,2);
    end_global3 = clock();
    std::cout<<"filter : "<<(double) (end_global3 - start_global3) / CLOCKS_PER_SEC<<std::endl;

    clock_t start_global4, end_global4;
    start_global4 = clock();
    std::string str;
    for(unsigned int i=0;i<v_v_img.size();i++){
        std::string str2;
        bool parse=false;
        if(v_v_img(i).size()>=5){
            for(unsigned int j=0;j<v_v_img(i).size();j++){

                char c =  ocr.parseMatrix(v_v_img(i)(j));
//                std::cout<<i<<" "<<c<<" "<<ocr.characterConfidence() <<std::endl;
                //                v_v_img(i)(j).display();
                if( ocr.characterConfidence()>-20){
                    str2.push_back(c);
                    parse=true;
                }
            }
            //        std::cout<<str2<<std::endl;
            if(str.size()!=0&&parse==true){

                str=str+" ";
            }
            str=str+str2;
        }
    }
    std::cout<<str<<std::endl;
    end_global4 = clock();
    std::cout<<"OCR : "<<(double) (end_global4- start_global4) / CLOCKS_PER_SEC<<std::endl;
}
}
