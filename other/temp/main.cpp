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
                            index_i_0_min=std::min(i,index_i_0_min);
                            index_i_0_max=std::max(i,index_i_0_max);
                            sum_0++;
                            bary_i_0+=i;
                        }
                        if(binary(i,index_j_1)>0){
                            index_i_1_min=std::min(i,index_i_1_min);
                            index_i_1_max=std::max(i,index_i_1_max);
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
                    xmin(1)=std::max(0,xmin(1)-10);
                    xmin(0)=std::max(0,xmin(0)-3);
                    xmax =  v_xmax[index_label];
                    xmax(1)=std::min(tophat_init.getDomain()(1)-1,xmax(1)+25);
                    xmax(0)=std::min(tophat_init.getDomain()(0)-1,xmax(0)+3);
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
                        maxi=std::max(maxi,sum);
                        mini=std::min(mini,sum);
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
struct POP_EXPORTS FunctorNiblackMethod
{
    const Mat2UI32* _integral;
    const Mat2UI32* _integral_power_2;
    F32 area_minus1;
    F32 _k;
    F32 _radius;
    F32 _offset_value;
    FunctorNiblackMethod(const Mat2UI32 & integral,const Mat2UI32& integral_power_2,F32 _area_minus1,F32 k,int radius,F32 offset_value)
        :_integral(&integral),_integral_power_2(&integral_power_2),area_minus1(_area_minus1),_k(k),_radius(radius),_offset_value(offset_value){

    }

    template<typename PixelType>
    UI8 operator()(const MatN<2,PixelType > & f,const  typename MatN<2,PixelType>::E & x){
        Vec2I32 xadd1=x+Vec2I32(_radius);
        Vec2I32 xadd2=x+Vec2I32(-_radius);
        Vec2I32 xsub1=x-Vec2I32(_radius,-_radius);
        Vec2I32 xsub2=x-Vec2I32(-_radius,_radius);
        F32 mean = (*_integral)(xadd1)+(*_integral)(xadd2)-(*_integral)(xsub1)-(*_integral)(xsub2);
        mean*=area_minus1;

        F32 standartdeviation =(*_integral_power_2)(xadd1)+(*_integral_power_2)(xadd2)-(*_integral_power_2)(xsub1)-(*_integral_power_2)(xsub2);
        standartdeviation*=area_minus1;
        standartdeviation =standartdeviation-mean*mean;

        if(standartdeviation>0)
            standartdeviation = std::sqrt( standartdeviation);
        else
            standartdeviation =1;
        if(f(x-_radius)>ArithmeticsSaturation<PixelType,F32>::Range( mean+_k*standartdeviation)-_offset_value)
            return 255;
        else
            return  0;
    }
};

template<typename PixelType>
static MatN<2,UI8>  thresholdNiblackMethod(const MatN<2,PixelType> & f,F32 k=0.2,int radius=5,F32 offset_value=0  ){
    MatN<2,PixelType> fborder(f);
    Draw::addBorder(fborder,radius,typename MatN<2,PixelType>::F(0),MATN_BOUNDARY_CONDITION_MIRROR);
    MatN<2,UI32> f_F32(fborder);
    MatN<2,UI32> integral = Processing::integral(f_F32);
    MatN<2,UI32> integralpower2 = Processing::integralPower2(f_F32);
    typename MatN<2,UI32>::IteratorERectangle it(fborder.getIteratorERectangle(Vec2I32(radius),f_F32.getDomain()-1-Vec2I32(radius)));

    F32 area_minus1 = 1.f/((2*radius+1)*(2*radius+1));
    FunctorNiblackMethod func(integral,integralpower2,area_minus1,k, radius, offset_value);
    forEachFunctorBinaryFunctionE(f,fborder,func,it);
    return fborder( Vec2I32(radius) , fborder.getDomain()-Vec2I32(radius));
}

struct Func
{
    template<typename PixelType>
    PixelType operator()(const MatN<2,PixelType> &f,const Vec2I32 &x){
        return f(x);
    }

};
template<typename PixelType>
static MatN<2,PixelType>  integral(const MatN<2,PixelType> & f)
{
    MatN<2,PixelType> s (f.getDomain());
    MatN<2,PixelType> out(f.getDomain());

    //    typename MatN<2,PixelType>::IteratorEDomain it = f.getIteratorEDomain();
    //    Func func;
    //    forEachFunctorBinaryFunctionE(f,out,func);
    //        forEachFunctorBinaryFunctionE(f,out,func);
    //    return out;
#if 1

#pragma omp parallel for
    for(int i=0;i<f.getDomain()(0);i++){
        for(int j=0;j<f.getDomain()(1);j++){
            s(i,j)=f(i,j) + (j==0?0:s(i,j-1));
        }
    }

#pragma omp parallel for
    for(int j=0;j<f.getDomain()(1);j++){
        for(int i=0;i<f.getDomain()(0);i++){
            out(i,j)=s(i,j) + (i==0?0:out(i-1,j));
        }
    }
#else

    for(int i=0;i<f.getDomain()(0);i++){
        for(int j=0;j<f.getDomain()(1);j++){
            if(j==0){
                s(i,j)=f(i,j);
            }
            else{
                s(i,j)=f(i,j)+s(i,j-1);
            }
        }
    }
    for(int i=0;i<f.getDomain()(0);i++){
        for(int j=0;j<f.getDomain()(1);j++){
            if(i==0){
                out(i,j)=s(i,j);
            }
            else{
                out(i,j)=s(i,j)+out(i-1,j);
            }
        }
    }
    //        std::cout<<out<<std::endl;
    //        exit(0);
#endif

    return out;
}
int main(){
    {
        //        Mat2UI32 m(10000,10000);
//        Mat2UI32 m(10,10);
//        for(unsigned int i=0;i<m.size();i++){
//            m(i)=i;
//        }
//        //        while(1==1)
//        int time1 = time(NULL);
//        for(unsigned int i=0;i<10;i++)
//            integral(m);
//        int time2 = time(NULL);
//        std::cout<<time2-time1<<std::endl;
//        return 0;
        Mat2UI8 m(10000,10000);
//        m.load(POP_PROJECT_SOURCE_DIR+std::string("/image/Lena.bmp"));

        int time1 = time(NULL);
        Mat2UI8 m2;

        for(unsigned int i=0;i<10;i++)
             m = thresholdNiblackMethod(m);
        int time2 = time(NULL);
        std::cout<<time2-time1<<std::endl;
//        m.display();
        return 1;
        //        segmentation2();
        int size_i=29;
        int size_j=29;
        VecF32 v_in(size_i*size_j);
        for(unsigned int i=0;i<size_j*size_j;i++){
            v_in(i)=0.5;
        }
        VecF32 v_out;
        NeuralNet n2;
        {
            NeuralNet neural_conv;
            neural_conv.addLayerMatrixInput(size_i,size_j,1);
            neural_conv.addLayerMatrixConvolutionSubScaling(5,2,2);
            neural_conv.addLayerMatrixConvolutionSubScaling(60,2,2);
            neural_conv.addLayerLinearFullyConnected(20);
            neural_conv.addLayerLinearFullyConnected(20);
            n2=neural_conv;

            neural_conv.forwardCPU(v_in,v_out);
            std::cout<<v_out<<std::endl;
        }
        n2.forwardCPU(v_in,v_out);
        std::cout<<v_out<<std::endl;
        return 1;
    }


    {
        std::cout<<"toto"<<std::endl;
        OCRNeuralNetwork ocr;
        int size=10;
        Vec2I32 domain(29,29);
        NeuralNet neural_conv;
        neural_conv.addLayerMatrixInput(domain(0),domain(1),1);
        neural_conv.addLayerMatrixConvolutionSubScaling(5,2,2);
        neural_conv.addLayerMatrixConvolutionSubScaling(60,2,2);
        neural_conv.addLayerLinearFullyConnected(100);
        neural_conv.addLayerLinearFullyConnected(size);
        neural_conv.setTrainable(true);
        neural_conv.setLearnableParameter(0.01);

        NeuralNetworkFeedForward n;
        n.addInputLayerMatrix(domain(0),domain(1));
        n.addLayerConvolutionalPlusSubScaling(6,5,2,1);
        n.addLayerConvolutionalPlusSubScaling(50,5,2,1);
        n.addLayerFullyConnected(100,1);
        n.addLayerFullyConnected(size,1);

        Vec<Vec<Mat2UI8> > number_training =  TrainingNeuralNetwork::loadMNIST("/home/vincent/train-images.idx3-ubyte","/home/vincent/train-labels.idx1-ubyte");


        {
            Mat2UI8 m=GeometricalTransformation::scale(number_training(0)(0),Vec2F32(4,4));
            VecF32 v_in = NNLayerMatrix::inputMatrixToInputNeuron(m,domain);
            VecF32 v_out(10);
            int time1=time(NULL);
            for(unsigned int i=0;i<10000;i++){
                //                 v_in = NNLayerMatrix::inputMatrixToInputNeuron(m,domain);
                neural_conv.forwardCPU(v_in,v_out);
            }
            int time2=time(NULL);
            std::cout<<time2-time1<<std::endl;

            time1=time(NULL);
            for(unsigned int i=0;i<10000;i++){
                //                 v_in = NNLayerMatrix::inputMatrixToInputNeuron(m,domain);
                n.propagateFront(v_in,v_out);
            }
            time2=time(NULL);
            std::cout<<time2-time1<<std::endl;
        }


        number_training.resize(size);
        for(unsigned int i=0;i<size;i++)
            number_training(i).resize(200);
        //        number_training(1).resize(500);
        Vec<VecF32> trainingins;
        Vec<VecF32> trainingouts;
        for(unsigned int i=0;i<number_training.size();i++){
            for(unsigned int j=0;j<number_training(i).size();j++){
                Mat2UI8 binary = number_training(i)(j);
                VecF32 vin = NNLayerMatrix::inputMatrixToInputNeuron(binary,domain);
                trainingins.push_back(vin);
                VecF32 v_out(static_cast<int>(number_training.size()),-1);
                v_out(i)=1;
                trainingouts.push_back(v_out);

            }
        }
        F32 eta = 0.001f;
        int nbr_epoch =0;
        neural_conv.setLearnableParameter(eta);
        n.setLearningRate(eta);
        std::vector<int> v_global_rand(trainingins.size());
        for(unsigned int i=0;i<v_global_rand.size();i++)
            v_global_rand[i]=i;
        std::cout<<"iter_epoch\t error_train\t error_test\t learning rate"<<std::endl;

        for(unsigned int i=0;i<nbr_epoch;i++){
            std::random_shuffle ( v_global_rand.begin(), v_global_rand.end() ,Distribution::irand());
            int error_training=0,error_training_n=0;
            for(unsigned int j=0;j<v_global_rand.size();j++){
                VecF32 vout(10),v_out2(10);
                neural_conv.forwardCPU(trainingins(v_global_rand[j]),vout);
                n.propagateFront(trainingins(v_global_rand[j]),v_out2);
                n.propagateBackFirstDerivate(trainingouts(v_global_rand[j]));
                neural_conv.backwardCPU(trainingouts(v_global_rand[j]));
                neural_conv.learn();
                n.learningFirstDerivate();
                //                neural_conv.forwardCPU(trainingins(v_global_rand[j]),vout);
                //                n.propagateFront(trainingins(v_global_rand[j]),v_out2);

                int label1 = std::distance(vout.begin(),std::max_element(vout.begin(),vout.end()));
                int label_n = std::distance(v_out2.begin(),std::max_element(v_out2.begin(),v_out2.end()));
                int label2 = std::distance(trainingouts(v_global_rand[j]).begin(),std::max_element(trainingouts(v_global_rand[j]).begin(),trainingouts(v_global_rand[j]).end()));
                if(label1!=label2)
                    error_training++;
                if(label_n!=label2)
                    error_training_n++;
                //                if(j*10%v_global_rand.size()==0)
                //                    if(j%100==0)
                //                    std::cout<<error_training*1./j<<std::endl;
            }
            std::cout<<i<<"\t"<<error_training*1./trainingins.size()<<"\t"<<error_training_n*1./trainingins.size()<<std::endl;
            eta *=0.9f;
            neural_conv.setLearnableParameter(eta);
            n.setLearningRate(eta);
        }
    }

    //        //        VecF32 v_in(index_i*index_j);
    //        //        DistributionNormal dist(0,1);
    //        //        for(unsigned int i=0;i<v_in.size();i++){
    //        //            v_in(i)=dist.randomVariable();
    //        //        }
    //        //        VecF32 v_out(1);
    //        //        neural_conv.forwardCPU(v_in,v_out);
    //        //        std::cout<<v_out<<std::endl;
    //        //        v_out(0)=1;
    //        //        neural_conv.backwardCPU(v_out);
    //        //        neural_conv.learn();
    //        //        neural_conv.forwardCPU(v_in,v_out);
    //        //        std::cout<<v_out<<std::endl;
    //        return 1;
    //    }

    {
        int index_i=29;
        int index_j=29;
        NeuralNet neural_conv;
        neural_conv.addLayerMatrixInput(index_i,index_j,1);
        neural_conv.addLayerMatrixConvolutionSubScaling(6,2,2);
        neural_conv.addLayerMatrixConvolutionSubScaling(50,2,2);
        neural_conv.addLayerLinearFullyConnected(100);
        neural_conv.addLayerLinearFullyConnected(10);


        VecF32 v_in(index_i*index_j);
        for(unsigned int i=0;i<index_i*index_j;i++){
            v_in(i)=0.5;
        }
        VecF32 v_out(100000);

        int time1=time(NULL);
        for(unsigned int i=0;i<30000;i++)
            neural_conv.forwardCPU(v_in,v_out);
        int time2=time(NULL);
        std::cout<<time2-time1<<std::endl;

        NeuralNetworkFeedForward n;
        n.addInputLayerMatrix(index_i,index_j);
        n.addLayerConvolutionalPlusSubScaling(6,5,2,1);
        n.addLayerConvolutionalPlusSubScaling(50,5,2,1);
        n.addLayerFullyConnected(100,1);
        n.addLayerFullyConnected(10,1);
        time1=time(NULL);
        for(unsigned int i=0;i<30000;i++)
            n.propagateFront(v_in,v_out);
        time2=time(NULL);
        std::cout<<time2-time1<<std::endl;

        return 0;
    }

    //    neural.setTrainable(true);
    //    neural.setLearnableParameter(0.1);

    //    Vec<F32*> v;
    //    F32 v1;
    //    v.push_back(&v1);
    //    *v(0)=20;
    //    std::cout<<v1<<std::endl;
    return 0;
    {
        int size_i=5;
        int size_j=5;
        NeuralNetworkFeedForward n_ref;
        n_ref.addInputLayerMatrix(size_i,size_j);
        n_ref.addLayerConvolutionalPlusSubScaling(2,3,2);
        //        n_ref.addLayerFullyConnected(20);
        //        n_ref.addLayerFullyConnected(10);
        n_ref.addLayerFullyConnected(2);
        //        n_ref.addLayerConvolutionalPlusSubScaling(2,3,2);
        n_ref.setLearningRate(0.1);


        NeuralNet neural;
        neural.addLayerMatrixInput(size_i,size_i,1);
        neural.addLayerMatrixConvolutionSubScaling(2,2,1);
        //        neural.addLayerLinearFullyConnected(20);
        //        neural.addLayerLinearFullyConnected(10);
        neural.addLayerLinearFullyConnected(2);
        //        neural.addLayerMatrixConvolutionSubScaling(2,1,1);
        neural.setTrainable(true);
        neural.setLearnableParameter(0.1);

        DistributionNormal d(0,1);
        VecF32 v_in(size_i*size_j);
        for(unsigned int i=0;i<v_in.size();i++){
            if(i%2==0)
                v_in(i)=1;
            else
                v_in(i)=-1;
        }
        //        std::cout<<v_in<<std::endl;
        //        return 0;

        for(unsigned int i=1;i<n_ref.layers().size();i++){
            NNLayer* layer_neural = n_ref.layers()(i);
            if(NeuralLayerLinearFullyConnected* layer_new = dynamic_cast<NeuralLayerLinearFullyConnected *>(neural._v_layer(i))){
                //        std::cout<<layer_neural->_weights.size()<<std::endl;
                //        std::cout<<test._v_layer(i)._W.size()<<std::endl;
                //fully connected
                for(unsigned int i=0;i<layer_new->_W.sizeI();i++){
                    for(unsigned int j=0;j<layer_new->_W.sizeJ();j++){
                        if(j<layer_new->_W.sizeJ()-1){
                            layer_new->_W(i,j)=layer_neural->_weights(j+i*layer_new->_W.sizeJ()+1)->_Wn;
                        }else{
                            layer_new->_W(i,j)=layer_neural->_weights(i*layer_new->_W.sizeJ())->_Wn;
                        }
                    }
                }
            }else if(NeuralLayerMatrixConvolutionSubScaling* layer_new = dynamic_cast<NeuralLayerMatrixConvolutionSubScaling *>(neural._v_layer(i))){

                int index_weight_old=0;
                for(unsigned int i=0;i<layer_new->_W_kernels.size();i++){
                    for(unsigned int j=0;j<layer_new->_d_E_W_kernels(i).size();j++,index_weight_old++){
                        if(j==0){
                            layer_new->_W_biais(i)=layer_neural->_weights(index_weight_old)->_Wn;
                            index_weight_old++;
                        }
                        layer_new->_W_kernels(i)(j)=layer_neural->_weights(index_weight_old)->_Wn;
                    }
                }
            }
        }
        for(unsigned int i=2;i>=1;i--){
            NNLayer* layer_neural = n_ref.layers()(i);
            if(NeuralLayerLinearFullyConnected* layer_new = dynamic_cast<NeuralLayerLinearFullyConnected *>(neural._v_layer(i))){
                std::cout<<"WEIGHT FULLY"<<std::endl;
                std::cout<<layer_new->_W<<std::endl;
                //                std::cout<<layer_new->_d_E_X<<std::endl;
                //                std::cout<<layer_new->_d_E_Y<<std::endl;
                //                for(unsigned int j=0;j<layer_neural->.size();j++){
                //                    std::cout<<layer_neural->_weights[j]->_Wn<<" ";
                //                }

                for(unsigned int j=0;j<layer_neural->_weights.size();j++){
                    std::cout<<layer_neural->_weights[j]->_Wn<<" ";
                }
                std::cout<<std::endl;

            }else if(NeuralLayerMatrixConvolutionSubScaling* layer_new = dynamic_cast<NeuralLayerMatrixConvolutionSubScaling *>(neural._v_layer(i))){

                std::cout<<"WEIGHT CONV"<<std::endl;
                std::cout<<layer_new->_W_kernels<<std::endl;
                std::cout<<layer_new->_W_biais<<std::endl;
                for(unsigned int j=0;j<layer_neural->_weights.size();j++){
                    std::cout<<layer_neural->_weights[j]->_Wn<<" ";
                }
                std::cout<<std::endl;

            }
        }

        VecF32 v_out(10);
        n_ref.propagateFront(v_in,v_out);
        std::cout<<v_out<<std::endl;
        v_out(0)=0;
        neural.forwardCPU(v_in,v_out);
        std::cout<<v_out<<std::endl;
        //        return 0;

        v_out(0)=1;v_out(1)=1;v_out(2)=1;
        n_ref.propagateBackFirstDerivate(v_out);
        n_ref.learningFirstDerivate();

        neural.backwardCPU(v_out);
        neural.learn();
        std::cout<<"############### LEARN #############"<<std::endl;
        for(unsigned int i=2;i>=1;i--){
            NNLayer* layer_neural = n_ref.layers()(i);
            if(NeuralLayerLinearFullyConnected* layer_new = dynamic_cast<NeuralLayerLinearFullyConnected *>(neural._v_layer(i))){
                std::cout<<layer_new->_d_E_W<<std::endl;
                //                std::cout<<layer_new->_d_E_X<<std::endl;
                //                std::cout<<layer_new->_d_E_Y<<std::endl;
                //                for(unsigned int j=0;j<layer_neural->.size();j++){
                //                    std::cout<<layer_neural->_weights[j]->_Wn<<" ";
                //                }
                for(unsigned int j=0;j<layer_neural->_weights.size();j++){
                    std::cout<<layer_neural->_weights[j]->_dE_dWn<<" ";
                }
                std::cout<<std::endl;
                std::cout<<"WEIGHT FULLY"<<std::endl;
                std::cout<<layer_new->_W<<std::endl;
                //                std::cout<<layer_new->_d_E_X<<std::endl;
                //                std::cout<<layer_new->_d_E_Y<<std::endl;
                //                for(unsigned int j=0;j<layer_neural->.size();j++){
                //                    std::cout<<layer_neural->_weights[j]->_Wn<<" ";
                //                }

                for(unsigned int j=0;j<layer_neural->_weights.size();j++){
                    std::cout<<layer_neural->_weights[j]->_Wn<<" ";
                }

            }else if(NeuralLayerMatrixConvolutionSubScaling* layer_new = dynamic_cast<NeuralLayerMatrixConvolutionSubScaling *>(neural._v_layer(i))){
                std::cout<<std::endl;
                std::cout<<std::endl;
                std::cout<<std::endl;
                std::cout<<layer_new->_d_E_W_kernels<<std::endl;
                std::cout<<layer_new->_d_E_W_biais<<std::endl;
                for(unsigned int j=0;j<layer_neural->_weights.size();j++){
                    std::cout<<layer_neural->_weights[j]->_dE_dWn<<" ";
                }
                std::cout<<std::endl;
                std::cout<<std::endl;
                std::cout<<std::endl;
                std::cout<<"WEIGHT CONV"<<std::endl;
                std::cout<<layer_new->_W_kernels<<std::endl;
                std::cout<<layer_new->_W_biais<<std::endl;
                for(unsigned int j=0;j<layer_neural->_weights.size();j++){
                    std::cout<<layer_neural->_weights[j]->_Wn<<" ";
                }

            }
        }
        std::cout<<std::endl;
        std::cout<<std::endl;
        std::cout<<std::endl;

        n_ref.propagateFront(v_in,v_out);
        std::cout<<v_out<<std::endl;
        std::cout<<"zut"<<std::endl;
        neural.forwardCPU(v_in,v_out);
        std::cout<<v_out<<std::endl;


        return 0;

        //        Vec<int> v_global_rand(4);
        //        for(unsigned int i=0;i<v_global_rand.size();i++)
        //            v_global_rand(i)=i;

        //        for(unsigned int i=0;i<100;i++){
        //            std::random_shuffle ( v_global_rand.begin(), v_global_rand.end() ,Distribution::irand());
        //            for(unsigned int j=0;j<v_global_rand.size();j++){

        //                n_ref.propagateFront(v_in(v_global_rand(j)),v_out_net);
        //                std::cout<<v_out_net<<std::endl;
        //                neural.forwardCPU(v_in(v_global_rand(j)),v_out_net);
        //                std::cout<<v_out_net<<std::endl;
        //                //            exit(0);
        //                neural.backwardCPU(v_out(v_global_rand(j)));
        //                n_ref.propagateBackFirstDerivate(v_out(v_global_rand(j)));
        //                neural.learn();
        //                n_ref.learningFirstDerivate();
        //                neural.forwardCPU(v_in(v_global_rand(j)),v_out_net);
        //                std::cout<<"neural1 "<<v_out_net<<std::endl;
        //                n_ref.propagateFront(v_in(v_global_rand(j)),v_out_net);
        //                std::cout<<"neural2 "<<v_out_net<<std::endl;
        //                std::cout<<"expected "<<v_out(v_global_rand(j))<<std::endl;
        //            }
        //        }
        //        return 0;
    }



    NeuralNetworkFeedForward n_ref;
    n_ref.addInputLayer(2);
    n_ref.addLayerFullyConnected(3);
    n_ref.addLayerFullyConnected(1);
    n_ref.setLearningRate(0.1);


    NeuralNet neural;
    neural.addLayerLinearInput(2);
    neural.addLayerLinearFullyConnected(3);
    neural.addLayerLinearFullyConnected(1);
    neural.setTrainable(true);
    neural.setLearnableParameter(0.1);



    Vec<VecF32> v_in(4,VecF32(2)),v_out(4,VecF32(1));
    v_in(0)(0)=-1;v_in(0)(1)=-1;v_out(0)(0)= 1;
    v_in(1)(0)= 1;v_in(1)(1)=-1;v_out(1)(0)=-1;
    v_in(2)(0)=-1;v_in(2)(1)= 1;v_out(2)(0)=-1;
    v_in(3)(0)= 1;v_in(3)(1)= 1;v_out(3)(0)= 1;

    VecF32 v_out_net(1);


    for(unsigned int i=1;i<=2;i++){
        NNLayer* layer_neural = n_ref.layers()(i);
        if(NeuralLayerLinearFullyConnected* layer_new = dynamic_cast<NeuralLayerLinearFullyConnected *>(neural._v_layer(i))){
            //        std::cout<<layer_neural->_weights.size()<<std::endl;
            //        std::cout<<test._v_layer(i)._W.size()<<std::endl;
            //fully connected
            for(unsigned int i=0;i<layer_new->_W.sizeI();i++){
                for(unsigned int j=0;j<layer_new->_W.sizeJ();j++){
                    if(j<layer_new->_W.sizeJ()-1){
                        layer_new->_W(i,j)=layer_neural->_weights(j+i*layer_new->_W.sizeJ()+1)->_Wn;
                    }else{
                        layer_new->_W(i,j)=layer_neural->_weights(i*layer_new->_W.sizeJ())->_Wn;
                    }
                }
            }
        }


        //        for(unsigned int index_weight=0;index_weight<layer_neural->_weights.size();index_weight++){
        //            if(index_weight==0){
        //                layer_new->_W(layer_neural->_weights.size()-1)=layer_neural->_weights(index_weight)->_Wn;
        //            }else{
        //                 layer_new->_W(index_weight-1)=layer_neural->_weights(index_weight)->_Wn;
        //            }
        //        }

    }


    Vec<int> v_global_rand(4);
    for(unsigned int i=0;i<v_global_rand.size();i++)
        v_global_rand(i)=i;

    for(unsigned int i=0;i<100;i++){
        std::random_shuffle ( v_global_rand.begin(), v_global_rand.end() ,Distribution::irand());
        for(unsigned int j=0;j<v_global_rand.size();j++){

            n_ref.propagateFront(v_in(v_global_rand(j)),v_out_net);
            std::cout<<v_out_net<<std::endl;
            neural.forwardCPU(v_in(v_global_rand(j)),v_out_net);
            std::cout<<v_out_net<<std::endl;
            //            exit(0);
            neural.backwardCPU(v_out(v_global_rand(j)));
            n_ref.propagateBackFirstDerivate(v_out(v_global_rand(j)));
            neural.learn();
            n_ref.learningFirstDerivate();
            neural.forwardCPU(v_in(v_global_rand(j)),v_out_net);
            std::cout<<"neural1 "<<v_out_net<<std::endl;
            n_ref.propagateFront(v_in(v_global_rand(j)),v_out_net);
            std::cout<<"neural2 "<<v_out_net<<std::endl;
            std::cout<<"expected "<<v_out(v_global_rand(j))<<std::endl;
        }
    }
    return 0;
    //    {
    //        Mat2UI8 m;
    //        m.load("plate1.pgm");
    //        //        m.display();
    //        findLine(m);
    //       return 0;

    //    }
    {

        Mat2F32 m(2,2);
        m(0,0)=1;m(0,1)=1;
        m(1,0)=3;m(1,1)=5;
        Mat2F32 m_lign(2,1);
        m_lign(0,0)=3;
        m_lign(1,0)=2;
        VecF32 & v = m_lign;
        std::cout<<m*v<<std::endl;
        return 1;


        //        Vec<LinearLeastSquareRANSACModel::Data> data;
        //        VecF32 x(2);F32 y;
        //        x(0)=1;x(1)=1;y=5.2;data.push_back(LinearLeastSquareRANSACModel::Data(x,y));
        //        x(0)=1;x(1)=2;y=5.8;data.push_back(LinearLeastSquareRANSACModel::Data(x,y));
        //        x(0)=1;x(1)=3;y=6.8;data.push_back(LinearLeastSquareRANSACModel::Data(x,y));
        //        x(0)=1;x(1)=4;y=8.2;data.push_back(LinearLeastSquareRANSACModel::Data(x,y));
        //        x(0)=1;x(1)=5;y=13; data.push_back(LinearLeastSquareRANSACModel::Data(x,y));

        //        LinearLeastSquareRANSACModel m;
        //        Vec<LinearLeastSquareRANSACModel::Data> dataconsencus;
        //        ransac(data,10,1,2,m,dataconsencus);
        //        std::cout<<m.getBeta()<<std::endl;
        //        std::cout<<m.getError()<<std::endl;
        //        std::cout<<dataconsencus<<std::endl;

        //                Vec<LinearLeastSquareRANSACModel::Data> data;
        //                VecF32 x(2);F32 y;
        //                x(0)=1;x(1)=1;y=5.2;data.push_back(LinearLeastSquareRANSACModel::Data(x,y));
        //                x(0)=1;x(1)=2;y=5.8;data.push_back(LinearLeastSquareRANSACModel::Data(x,y));
        //                x(0)=1;x(1)=3;y=6.8;data.push_back(LinearLeastSquareRANSACModel::Data(x,y));
        //                x(0)=1;x(1)=4;y=8.2;data.push_back(LinearLeastSquareRANSACModel::Data(x,y));
        //                x(0)=1;x(1)=5;y=13; data.push_back(LinearLeastSquareRANSACModel::Data(x,y));

        //                LinearLeastSquareRANSACModel m;
        //                Vec<LinearLeastSquareRANSACModel::Data> dataconsencus;
        //                ransac(data,10,1,3,m,dataconsencus);
        //                std::cout<<m.getBeta()<<std::endl;
        //                std::cout<<m.getError()<<std::endl;
        //                std::cout<<dataconsencus<<std::endl;
        //        return 1;
    }
    //    {
    //        Mat2UI8 m;
    //        m.load(POP_PROJECT_SOURCE_DIR+std::string("/image/barriere.png"));
    //        Mat2UI8 edge = Processing::edgeDetectorCanny(m,2,0.5,5);
    //        edge.display("edge",false);
    //        Mat2F32 hough = Feature::transformHough(edge);
    //        hough.display("hough",false);
    //        std::vector< std::pair<Vec2F32, Vec2F32 > > v_lines = Feature::HoughToLines(hough,edge ,0.5);
    //        Mat2RGBUI8 m_hough(m);
    //        for(unsigned int i=0;i<v_lines.size();i++){
    //            Draw::line(m_hough,v_lines[i].first,v_lines[i].second,  RGBUI8(255,0,0),2);
    //        }
    //        m_hough.display();
    //    }


    return 0;


    //    if(display)temp =Draw::mergeTwoMatrixVertical(temp,tophat);
    //    Mat2UI8 elt(3,3);
    //    elt(1,0)=1;
    //    elt(1,1)=1;
    //    elt(1,2)=1;

    //    tophat = Processing::closingStructuralElement(tophat,elt  ,std::ceil(pixel_width_letter));
    //    if(display)temp =Draw::mergeTwoMatrixVertical(temp,tophat);
    //    tophat = Processing::openingStructuralElement(tophat,elt  ,std::ceil(pixel_width_letter)/4  );
    //    elt = elt.transpose();
    //    tophat = Processing::openingStructuralElement(tophat,elt  ,std::ceil(pixel_width_letter)/factor_opening_vertical );
    //    if(display)temp =Draw::mergeTwoMatrixVertical(temp,tophat);
    //    int value;
    //    Mat2UI8 binary = Processing::thresholdOtsuMethod(tophat,value);
    //    Mat2UI8 binary2 =  Processing::threshold(tophat,10);



    neuralnetwortest2();
    return 1;
}
