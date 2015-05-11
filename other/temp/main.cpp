#include"Population.h"
#include<iostream>
#include"neuralnetworkmatrix.h"
#include"data/notstable/graph/Graph.h"
#include"popconfig.h"
#include"data/notstable/MatNReference.h"
using namespace pop;

void segmentation(){

    OCRNeuralNetwork ocr;
    ocr.setDictionnary("/home/vincent/DEV2/DEV/LAPI-ACCES/usr/bin/dictionaries/neuralnetwork.xml");
    Mat2UI8 m_init;
    std::string dir="/home/vincent/DEV2/DEV/LAPI-API/images/F_FRANCE/";
    std::vector<std::string> v_files =BasicUtility::getFilesInDirectory(dir);
    MatNDisplay disp,disp2;
    for(unsigned int i=0;i<v_files.size();i++){
        if(BasicUtility::getExtension(v_files[i])==".png"){
            std::cout<<i<<std::endl;
            m_init.load( dir+v_files[i]);

            F32 scale_init =1200./m_init.sizeJ();
            m_init = GeometricalTransformation::scale(m_init,Vec2F32(scale_init,scale_init));
            disp.display(m_init);
            //    m_init.load("/home/vincent/Desktop/coutin.jpg");
            F32 scalefactor =400./m_init.sizeJ();
            pop::Vec2F32 border_extra=pop::Vec2F32(0.01f,0.1f);
            Mat2UI8 m = GeometricalTransformation::scale(m_init,Vec2F32(scalefactor,scalefactor));
            std::cout<<m.getDomain()<<std::endl;

            Mat2UI8 tophat_init =   Processing::closing(m,2)-m;
            //            tophat_init.display("top hat",false);
            Mat2UI8 elt(3,3);
            elt(1,0)=1;
            elt(1,1)=1;
            elt(1,2)=1;

            Mat2UI8 tophat = Processing::closingStructuralElement(tophat_init,elt  ,8);
            tophat = Processing::openingStructuralElement(tophat,elt  ,2  );
            elt = elt.transpose();
            tophat = Processing::openingStructuralElement(tophat,elt  ,2 );
            //            tophat.display("filter",false);

            int value;
            Mat2UI8 binary = Processing::thresholdOtsuMethod(tophat,value);
            Mat2UI8 binary2 =  Processing::threshold(tophat,20);
            binary = pop::minimum(binary,binary2);
            //            binary.display("binary",false);



            Mat2UI32 imglabel =ProcessingAdvanced::clusterToLabel(binary, binary.getIteratorENeighborhood(1,0),binary.getIteratorEOrder(1));
            //            Visualization::labelToRandomRGB(imglabel).display("label",false);
            pop::Vec<pop::Vec2I32> v_xmin;
            pop::Vec<pop::Vec2I32> v_xmax;
            pop::Vec<Mat2UI8> v_img = Analysis::labelToMatrices(imglabel,v_xmin,v_xmax);

            disp2.display(binary);

            for(unsigned int index_label =0;index_label<v_img.size();index_label++){
                Vec2I32 domain = v_img[index_label].getDomain();
                if(domain(1)>0.5*binary.sizeJ()){

                    Vec2I32 xmin =  v_xmin[index_label];
                    xmin(1)+=10;
                    Vec2I32 xmax =  v_xmax[index_label];
                    xmax(1)-=10;
                    int sum_0=0;
                    int sum_1=0;
                    int bary_i_0=0;
                    int bary_i_1=0;
                    int index_j_0 = xmin(1)+10;
                    int index_j_1 = xmax(1)-10;
                    unsigned int index_i_0_min=10000;
                    unsigned int index_i_0_max=0;
                    unsigned int index_i_1_min=10000;
                    unsigned int index_i_1_max=0;
                    for(unsigned int i=xmin(0);i<xmax(0);i++){
                        if(binary(i,index_j_0)>0){
                            index_i_0_min=(std::min)(i,index_i_0_min);
                            index_i_0_max=(std::max)(i,index_i_0_max);
                            sum_0++;
                            bary_i_0+=i;
                        }
                        if(binary(i,index_j_1)>0){
                            index_i_1_min=(std::min)(i,index_i_1_min);
                            index_i_1_max=(std::max)(i,index_i_1_max);
                            sum_1++;
                            bary_i_1+=i;
                        }
                    }
                    bary_i_0/=sum_0;
                    bary_i_1/=sum_1;
                    double delta_y = index_j_1-index_j_0;
                    double delta_x = bary_i_1-bary_i_0;
                    double rot = atan2(delta_x,delta_y);
                    xmin =  v_xmin[index_label];
                    xmin(1)=(std::max)(0,xmin(1)-10);
                    xmin(0)=(std::max)(0,xmin(0)-3);
                    xmax =  v_xmax[index_label];
                    xmax(1)=(std::min)(tophat_init.getDomain()(1)-1,xmax(1)+25);
                    xmax(0)=(std::min)(tophat_init.getDomain()(0)-1,xmax(0)+3);
                    std::cout<<rot<<std::endl;
                    Mat2UI8 m_lign = GeometricalTransformation::rotate(m_init(Vec2F32(xmin)/scalefactor,Vec2F32(xmax)/scalefactor),-rot,MATN_BOUNDARY_CONDITION_MIRROR);
                    //GeometricalTransformation::scale(,Vec2F32(3,3)).display("rot",true,false);

                    xmin(0)=std::abs(delta_x/2)*1./scalefactor;

                    xmin(1)=0;
                    xmax(0)=m_lign.getDomain()(0)-1-std::abs(delta_x/2)*1./scalefactor;
                    xmax(1)=m_lign.getDomain()(1)-1;
                    std::cout<<xmin<<std::endl;
                    std::cout<<xmax<<std::endl;
                    std::cout<<m_lign.getDomain()<<std::endl;

                    //                F32 ratio = m_init.sizeJ()/1000;
                    //                std::cout<<ratio<<std::endl;
                    m_lign = m_lign(xmin,xmax);
                    m_lign = Processing::median(m_lign,2);
                    Mat2UI8 tophat_letter = Processing::closing(m_lign,7)-m_lign;

                    int value = 0;
                    //                    m_lign.display("init",false,false);
                    Processing::thresholdOtsuMethod(tophat_letter,value);
                    tophat_letter = Processing::threshold(tophat_letter,value-5);
                    //                    tophat_letter.display("rot",false,false);

                    //                    GeometricalTransformation::scale(tophat_letter,Vec2F32(3,3)).display("rot",false,false);

                    MatN<1,UI16> m1(VecN<1,int>(tophat_letter.sizeJ())),m2(VecN<1,int>(tophat_letter.sizeJ()));
                    Mat2UI16 m(tophat_letter.sizeJ(),2);
                    int maxi=0,mini=NumericLimits<int>::maximumRange();
                    for(unsigned int j=0;j<tophat_letter.sizeJ();j++){
                        int sum=0;
                        for(unsigned int i=0;i<tophat_letter.sizeI();i++){
                            sum+=tophat_letter(i,j);
                        }
                        m(j,0)=j;
                        m(j,1)=sum;
                        m1(j)=sum;
                        maxi=(std::max)(maxi,sum);
                        mini=(std::min)(mini,sum);
                    }
                    for(unsigned int i=0;i<m1.getDomain()(0);i++){
                        m1(i)= (m1(i)-mini)*100/(maxi-mini);

                    }

                    m2= Processing::smoothGaussian(m1,2);
                    m2= Processing::dynamic(m2,5);
                    m2 = Processing::minimaRegional(m2,1);


                    MatN<1,UI8> thhinning = Analysis::thinningAtConstantTopology(MatN<1,UI8>(m2));
                    int min_previous=-1;
                    for(unsigned int j=0;j<m2.getDomain()(0);j++){
                        if(thhinning(j)>0&&m1(j)<10){
                            if(min_previous==-1)
                                min_previous =j;
                            else{
                                int value;

                                Mat2UI8 m_letter = Processing::clusterMax(Processing::thresholdOtsuMethod(tophat_letter(Vec2I32(0,min_previous),Vec2I32(tophat_letter.getDomain()(0)-1,j)),value));//.display();
                                ;
                                std::cout<<ocr.parseMatrix(m_letter)<<" ";
                                min_previous =j;
                            }
                            for(unsigned int i=0;i<m_lign.sizeI();i++){
                                m_lign(i,j)=255;
                            }

                        }
                    }
                    std::cout<<std::endl;
                    //                    GeometricalTransformation::scale(m_lign,Vec2F32(3,3)).display("rot",true,false);

                }
            }
        }
    }
}
template<int Dim, typename PixelType>
MatN<Dim,PixelType>  mult(const MatN<Dim,PixelType> &other,const MatN<Dim,PixelType> &m)
{
    MatN<Dim,PixelType> mtrans = m.transpose();
    MatN<Dim,PixelType> mout(other.sizeI(),m.sizeJ());
#if defined(HAVE_OPENMP)
#pragma omp parallel for
#endif
    for(unsigned int i=0;i<(other.sizeI());i++){
        for(unsigned  j=0;j<(m.sizeJ());j++){
            PixelType sum = 0;
            typename MatN<Dim,PixelType>::const_iterator this_it  = other.begin() +  i*other.sizeJ();
            typename MatN<Dim,PixelType>::const_iterator mtrans_it= mtrans.begin() + j*mtrans.sizeJ();
            for(unsigned int k=0;k<other.sizeJ();k++){
                sum+=(* this_it) * (* mtrans_it);
                this_it++;
                mtrans_it++;
            }
            mout(i,j)=sum;
        }
    }
    return mout;
}
int main(){

    omp_set_num_threads(4);
    {

        //omp_set_num_threads(1);
    pop::Mat2UI8 m(1000,1000);


    int time1 = time(NULL);
    for(unsigned int i=0;i<100;i++){
        m=Processing::thresholdNiblackMethod(m);
    }
    int time2 = time(NULL);
    std::cout<<time2-time1<<std::endl;
        int size_i=200;
//        Mat2F32 m1(size_i,size_i);
//        Mat2F32 m2(size_i,size_i);
//        int time1=time(NULL);
//        mult(m1,m2);
//        int time1=time(NULL);
        return 1;
    }

    int size=600;
    NeuralNet n1;
    n1.addLayerLinearInput(size);
    n1.addLayerLinearFullyConnected(size);
    n1.addLayerLinearFullyConnected(size);

    VecF32 v_in(size),v_out;

    Mat2F32 m1(2000,2000);
    Mat2F32 m2(m1.getDomain());
    //    m1 = m1*m2;

    OCRNeuralNetwork ocr;
    ocr.setDictionnary("D:/Users/vtariel/Desktop/ANV/DEV/LAPI-ACCES/usr/bin/dictionaries/neuralnetwork.xml");
    Mat2UI8 m("D:/Users/vtariel/Desktop/C.jpg");
    int time1=time(NULL);
    std::string str;
//    std::chrono::time_point<std::chrono::system_clock> start, end;
//    start = std::chrono::system_clock::now();
    for(unsigned int i=0;i<1;i++)
        std::cout<<ocr.parseMatrix(m)<<std::endl;
//        m1 = m1*m2;
//    end = std::chrono::system_clock::now();

//    std::chrono::duration<double> elapsed_seconds = end-start;
//    std::time_t end_time = std::chrono::system_clock::to_time_t(end);

//    std::cout << "finished computation at " << std::ctime(&end_time)
//              << "elapsed time: " << elapsed_seconds.count() << "s\n";
//        //        n1.forwardCPU(v_in,v_out);
//        m1 = mult(m1,m2);
    //        str+=ocr.parseMatrix(m);
    int time2=time(NULL);
    std::cout<<time2-time1<<std::endl;
    return 1;

    //    int size_i=5;
    //    int size_j=5;
    //    NeuralNetworkFeedForward n_ref;
    //    n_ref.addInputLayerMatrix(size_i,size_j);
    //    n_ref.addLayerConvolutionalPlusSubScaling(2,3,2);
    //    //        n_ref.addLayerFullyConnected(20);
    //    //        n_ref.addLayerFullyConnected(10);
    //    n_ref.addLayerFullyConnected(2);
    //    //        n_ref.addLayerConvolutionalPlusSubScaling(2,3,2);
    //    n_ref.setLearningRate(0.1);


    //    NeuralNet neural;
    //    neural.addLayerMatrixInput(size_i,size_i,1);
    //    neural.addLayerMatrixConvolutionSubScaling(2,2,1);
    //    //        neural.addLayerLinearFullyConnected(20);
    //    //        neural.addLayerLinearFullyConnected(10);
    //    neural.addLayerLinearFullyConnected(2);
    //    //        neural.addLayerMatrixConvolutionSubScaling(2,1,1);
    //    neural.setTrainable(true);
    //    neural.setLearnableParameter(0.1);

    //    DistributionNormal d(0,1);
    //    VecF32 v_in(size_i*size_j);
    //    for(unsigned int i=0;i<v_in.size();i++){
    //        if(i%2==0)
    //            v_in(i)=1;
    //        else
    //            v_in(i)=-1;
    //    }
    //    //        std::cout<<v_in<<std::endl;
    //    //        return 0;

    //    for(unsigned int i=1;i<n_ref.layers().size();i++){
    //        NNLayer* layer_neural = n_ref.layers()(i);
    //        if(NeuralLayerLinearFullyConnected* layer_new = dynamic_cast<NeuralLayerLinearFullyConnected *>(neural._v_layer(i))){
    //            //        std::cout<<layer_neural->_weights.size()<<std::endl;
    //            //        std::cout<<test._v_layer(i)._W.size()<<std::endl;
    //            //fully connected
    //            for(unsigned int i=0;i<layer_new->_W.sizeI();i++){
    //                for(unsigned int j=0;j<layer_new->_W.sizeJ();j++){
    //                    if(j<layer_new->_W.sizeJ()-1){
    //                        layer_new->_W(i,j)=layer_neural->_weights(j+i*layer_new->_W.sizeJ()+1)->_Wn;
    //                    }else{
    //                        layer_new->_W(i,j)=layer_neural->_weights(i*layer_new->_W.sizeJ())->_Wn;
    //                    }
    //                }
    //            }
    //        }else if(NeuralLayerMatrixConvolutionSubScaling* layer_new = dynamic_cast<NeuralLayerMatrixConvolutionSubScaling *>(neural._v_layer(i))){

    //            int index_weight_old=0;
    //            for(unsigned int i=0;i<layer_new->_W_kernels.size();i++){
    //                for(unsigned int j=0;j<layer_new->_d_E_W_kernels(i).size();j++,index_weight_old++){
    //                    if(j==0){
    //                        layer_new->_W_biais(i)=layer_neural->_weights(index_weight_old)->_Wn;
    //                        index_weight_old++;
    //                    }
    //                    layer_new->_W_kernels(i)(j)=layer_neural->_weights(index_weight_old)->_Wn;
    //                }
    //            }
    //        }
    //    }
    //    for(unsigned int i=2;i>=1;i--){
    //        NNLayer* layer_neural = n_ref.layers()(i);
    //        if(NeuralLayerLinearFullyConnected* layer_new = dynamic_cast<NeuralLayerLinearFullyConnected *>(neural._v_layer(i))){
    //            std::cout<<"WEIGHT FULLY"<<std::endl;
    //            std::cout<<layer_new->_W<<std::endl;
    //            //                std::cout<<layer_new->_d_E_X<<std::endl;
    //            //                std::cout<<layer_new->_d_E_Y<<std::endl;
    //            //                for(unsigned int j=0;j<layer_neural->.size();j++){
    //            //                    std::cout<<layer_neural->_weights[j]->_Wn<<" ";
    //            //                }

    //            for(unsigned int j=0;j<layer_neural->_weights.size();j++){
    //                std::cout<<layer_neural->_weights[j]->_Wn<<" ";
    //            }
    //            std::cout<<std::endl;

    //        }else if(NeuralLayerMatrixConvolutionSubScaling* layer_new = dynamic_cast<NeuralLayerMatrixConvolutionSubScaling *>(neural._v_layer(i))){

    //            std::cout<<"WEIGHT CONV"<<std::endl;
    //            std::cout<<layer_new->_W_kernels<<std::endl;
    //            std::cout<<layer_new->_W_biais<<std::endl;
    //            for(unsigned int j=0;j<layer_neural->_weights.size();j++){
    //                std::cout<<layer_neural->_weights[j]->_Wn<<" ";
    //            }
    //            std::cout<<std::endl;

    //        }
    //    }

    //    VecF32 v_out(10);
    //    n_ref.propagateFront(v_in,v_out);
    //    std::cout<<v_out<<std::endl;
    //    v_out(0)=0;
    //    neural.forwardCPU(v_in,v_out);
    //    std::cout<<v_out<<std::endl;
    //    //        return 0;

    //    v_out(0)=1;v_out(1)=1;v_out(2)=1;
    //    n_ref.propagateBackFirstDerivate(v_out);
    //    n_ref.learningFirstDerivate();

    //    neural.backwardCPU(v_out);
    //    neural.learn();
    //    std::cout<<"############### LEARN #############"<<std::endl;
    //    for(unsigned int i=2;i>=1;i--){
    //        NNLayer* layer_neural = n_ref.layers()(i);
    //        if(NeuralLayerLinearFullyConnected* layer_new = dynamic_cast<NeuralLayerLinearFullyConnected *>(neural._v_layer(i))){
    //            std::cout<<layer_new->_d_E_W<<std::endl;
    //            //                std::cout<<layer_new->_d_E_X<<std::endl;
    //            //                std::cout<<layer_new->_d_E_Y<<std::endl;
    //            //                for(unsigned int j=0;j<layer_neural->.size();j++){
    //            //                    std::cout<<layer_neural->_weights[j]->_Wn<<" ";
    //            //                }
    //            for(unsigned int j=0;j<layer_neural->_weights.size();j++){
    //                std::cout<<layer_neural->_weights[j]->_dE_dWn<<" ";
    //            }
    //            std::cout<<std::endl;
    //            std::cout<<"WEIGHT FULLY"<<std::endl;
    //            std::cout<<layer_new->_W<<std::endl;
    //            //                std::cout<<layer_new->_d_E_X<<std::endl;
    //            //                std::cout<<layer_new->_d_E_Y<<std::endl;
    //            //                for(unsigned int j=0;j<layer_neural->.size();j++){
    //            //                    std::cout<<layer_neural->_weights[j]->_Wn<<" ";
    //            //                }

    //            for(unsigned int j=0;j<layer_neural->_weights.size();j++){
    //                std::cout<<layer_neural->_weights[j]->_Wn<<" ";
    //            }

    //        }else if(NeuralLayerMatrixConvolutionSubScaling* layer_new = dynamic_cast<NeuralLayerMatrixConvolutionSubScaling *>(neural._v_layer(i))){
    //            std::cout<<std::endl;
    //            std::cout<<std::endl;
    //            std::cout<<std::endl;
    //            std::cout<<layer_new->_d_E_W_kernels<<std::endl;
    //            std::cout<<layer_new->_d_E_W_biais<<std::endl;
    //            for(unsigned int j=0;j<layer_neural->_weights.size();j++){
    //                std::cout<<layer_neural->_weights[j]->_dE_dWn<<" ";
    //            }
    //            std::cout<<std::endl;
    //            std::cout<<std::endl;
    //            std::cout<<std::endl;
    //            std::cout<<"WEIGHT CONV"<<std::endl;
    //            std::cout<<layer_new->_W_kernels<<std::endl;
    //            std::cout<<layer_new->_W_biais<<std::endl;
    //            for(unsigned int j=0;j<layer_neural->_weights.size();j++){
    //                std::cout<<layer_neural->_weights[j]->_Wn<<" ";
    //            }

    //        }
    //    }
    //    std::cout<<std::endl;
    //    std::cout<<std::endl;
    //    std::cout<<std::endl;

    //    n_ref.propagateFront(v_in,v_out);
    //    std::cout<<v_out<<std::endl;
    //    std::cout<<"zut"<<std::endl;
    //    neural.forwardCPU(v_in,v_out);
    //    std::cout<<v_out<<std::endl;
}
