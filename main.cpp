#include"Population.h"//Single header
#include"include/algorithm/Arithmetic.h"
using namespace pop;//Population namespace
#if __cplusplus > 199711L // c++11
#include <chrono>
#endif
#include<map>
VecF32 normalizedImageToNeuralNet( const Mat2UI8& f,Vec2I32 domain ,MatNInterpolation interpolation=MATN_INTERPOLATION_BILINEAR) {
    F32 mean     = Analysis::meanValue(f);
    F32 standart = Analysis::standardDeviationValue(f);

    VecF32 v_in(domain.multCoordinate());
    int k=0;
    Vec2F32 alpha(f.sizeI()*1.f/domain(0),f.sizeJ()*1.f/domain(1));
    for(unsigned int i=0;i<domain(0);i++){
        for(unsigned int j=0;j<domain(1);j++,k++){

            Vec2F32 x( (i+0.5)*alpha(0),(j+0.5)*alpha(1));
            if(interpolation.isValid(f.getDomain(),x)){
                v_in[k] = (interpolation.apply(f,x)-mean)/standart;
            }else{
                std::cerr<<"errror normalized"<<std::endl;
            }
        }
    }
    return v_in;
}
VecF32 normalizedImageToNeuralNet( const Mat2RGBUI8& f,Vec2I32 domain ,MatNInterpolation interpolation=MATN_INTERPOLATION_BILINEAR) {
    Mat2UI8 f_r,f_g,f_b;
    Convertor::toRGB(f,f_r,f_g,f_b);
    VecF32 v_r,v_g,v_b;

    v_r = normalizedImageToNeuralNet(f_r,domain,interpolation);
    v_g = normalizedImageToNeuralNet(f_g,domain,interpolation);
    v_b = normalizedImageToNeuralNet(f_b,domain,interpolation);

    VecF32 v_in;
    v_in.insert( v_in.end(), v_r.begin(), v_r.end() );
    v_in.insert( v_in.end(), v_g.begin(), v_g.end() );
    v_in.insert( v_in.end(), v_b.begin(), v_b.end() );
    return v_in;
}

VecF32 normalizedImageToOutputNet( const Mat2UI8& binary,Vec2I32 domain,MatNInterpolation interpolation=MATN_INTERPOLATION_BILINEAR) {


    VecF32 v_in(domain.multCoordinate());


    int k=0;
    Vec2F32 alpha(binary.sizeI()*1.f/domain(0),binary.sizeJ()*1.f/domain(1));

    //    Mat2UI8 m(domain);
    for(unsigned int i=0;i<domain(0);i++){
        for(unsigned int j=0;j<domain(1);j++,k++){
            Vec2F32 x( (i+0.5)*alpha(0),(j+0.5)*alpha(1));
            if(interpolation.isValid(binary.getDomain(),x)){
                v_in[k] = (interpolation.apply(binary,x)-127.5)/127.5;
                //                m(i,j)=(v_in[k]+1)*127.5;
            }else{
                std::cerr<<"errror normalized"<<std::endl;
            }
        }
    }
    //    std::cout<<domain<<std::endl;
    //    m.display();
    return v_in;
}

#define N2 8
struct ConvolutionFourier
{

    static Mat2ComplexF32 weightReal2Fourrier(const Mat2F32& W , Vec2I32 domain_mult_2){
        Mat2ComplexF32 m_W(domain_mult_2);
        MatNBoundaryConditionPeriodic c;
        Vec2I32 radius(W.getDomain()(0)/2,W.getDomain()(1)/2);
        Vec2I32 x;
        for(x(0)=0;x(0)<W.sizeI();x(0)++){
            for(x(1)=0;x(1)<W.sizeJ();x(1)++){
                Vec2I32 y =x -radius;
                c.apply(m_W.getDomain(),y);
                m_W(y)=ComplexF32(W(x),0);
            }
        }
        Mat2ComplexF32 m_w = Representation::FFT(m_W);
        for(unsigned int i=0;i<m_w.size();i++){
            m_w(i)=m_w(i).conjugate();
        }
        return m_w;
    }
    static Mat2F32 weightFourrier2Real( Mat2ComplexF32 m_W , Vec2I32 domain_W){
        m_W = Representation::FFT(m_W,FFT_BACKWARD);
        Representation::scale(m_W);
        Mat2F32 W(domain_W);
        MatNBoundaryConditionPeriodic c;
        Vec2I32 radius(W.getDomain()(0)/2,W.getDomain()(1)/2);
        Vec2I32 x;
        for(x(0)=0;x(0)<W.sizeI();x(0)++){
            for(x(1)=0;x(1)<W.sizeJ();x(1)++){
                Vec2I32 y =x -radius;
                c.apply(m_W.getDomain(),y);
                W(x)= m_W(y).real();
            }
        }
        return W;
    }


    void convolution(Mat2F32& m,const Mat2F32& W ,F32  W_biais=0){
        Mat2ComplexF32 m_i;
        Convertor::fromRealImaginary(m,m_i);
        Mat2ComplexF32 m_W = weightReal2Fourrier(W,m_i.getDomain());
        m_i = Representation::FFT(m_i);
        m_i= m_i.multTermByTerm(m_W);
        m_i(0,0) +=W_biais*m_i.getDomain().multCoordinate();
        m_i = Representation::FFT(m_i,FFT_BACKWARD);
        Representation::scale(m_i);
        Convertor::toRealImaginary(m_i,m);
    }
};

class NeuralLayerMatrixConvolutionFFT : public NeuronSigmoid,public NeuralLayerMatrix
{
public:

    int power2(int i){
        return 1<<(int)std::ceil(Arithmetic::exposant(i,2));
    }

    NeuralLayerMatrixConvolutionFFT(unsigned int nbr_map,unsigned int sizei,unsigned int sizej,int kernel_radius,unsigned int nbr_map_previous)
        :NeuralLayerMatrix(sizei-2*kernel_radius,sizej-2*kernel_radius,nbr_map_previous),
          //          :NeuralLayerMatrix(sizei,sizej,nbr_map_previous),
          _W_kernels(nbr_map*nbr_map_previous,Mat2F32(power2(sizei),power2(sizej))),
          _W_biais(nbr_map*nbr_map_previous),
          _X_complex(power2(sizei),power2(sizej)),
          _Y_complex(nbr_map,Mat2F32(power2(sizei),power2(sizej))),
          _radius(kernel_radius)
    {

    }

    void setTrainable(bool istrainable){



    }
    // {
    //        Mat2F32 f(2,3);
    //        f(0,0)=2;f(0,1)=5;f(0,2)=3;
    //        f(1,0)=1;f(1,1)=4;f(1,2)=1;

        void setWeight(Vec<Mat2F32> v_weights,Vec<F32> v_weighs_biais){
            for(unsigned int i=0;i<v_weights.size();i++){
                this->_W_kernels(i) = ConvolutionFourier::weightReal2Fourrier(v_weights(i),this->_W_kernels(i).getDomain());
            }
            _W_biais= v_weighs_biais;
        }

        void convertReal2Complex(const MatNReference<2,F32> & map,Mat2ComplexF32 & map_complex){
            map_complex.fill(ComplexF32(0,0));
            for(unsigned int i=0;i<map.sizeI();i++){
                for(unsigned int j=0;j<map.sizeJ();j++){
                    map_complex(i,j)=ComplexF32(map(i,j),0);
                }
            }

        }
        void FFTInverse(){

        }
        virtual void forwardCPU(const NeuralLayer& layer_previous){

            //init output maps
            for(unsigned int index_map=0;index_map<this->_Y_complex.size();index_map++){
                this->_Y_complex(index_map).fill(ComplexF32(0,0));
            }

            if(const NeuralLayerMatrix * neural_matrix = dynamic_cast<const NeuralLayerMatrix *>(&layer_previous)){
                //iterate over the input maps
                for(unsigned int index_map_previous=0;index_map_previous<neural_matrix->X_map().size();index_map_previous++){
                    const MatNReference<2,F32> & map =neural_matrix->X_map()(index_map_previous);

                    convertReal2Complex(map,_X_complex);
                    //apply the forward FFT to the previous map
                    _X_complex = Representation::FFT(_X_complex);

                    //iterate of the output map
                    for(unsigned int index_map=0;index_map<this->_Y_complex.size();index_map++){
                        //get the index of the kernel
                        int index = index_map*neural_matrix->X_map().size()+index_map_previous;

                        //apply the convolution (matrix multiplication term of term in frequency domain) and add the result to the output map
                        _Y_complex(index_map) += _X_complex.multTermByTerm(_W_kernels[index]);
                        //add the biais weight
                        _Y_complex(index_map)(0,0).real() +=_W_biais(index)*_X_complex.getDomain().multCoordinate();
                    }
                }
            }

            for(unsigned int index_map=0;index_map<this->_Y_complex.size();index_map++){
                //apply the backward FFT ot the  map
                _Y_complex(index_map) = Representation::FFT(_Y_complex(index_map),FFT_BACKWARD);
                //scale the result
                Representation::scale(_Y_complex(index_map));
                for(unsigned int i=0;i<this->_X_reference(index_map).sizeI();i++)
                    for(unsigned int j=0;j<this->_X_reference(index_map).sizeJ();j++){
                        this->_X_reference(index_map)(i,j) = NeuronSigmoid::activation(_Y_complex(index_map)(i,j).real());
                    }
            }

        }

        virtual void backwardCPU(NeuralLayer& layer_previous){}
        void learn(){}
        virtual NeuralLayer * clone(){}
        Mat2ComplexF32 _X_complex;
        Vec<Mat2ComplexF32> _W_kernels;
        Vec<Mat2ComplexF32> _Y_complex;
        Vec<F32> _W_biais;
        int _radius;
        unsigned int _sub_resolution_factor;
    };





void neuralNetworkForRecognitionForHandwrittenDigits()
{
    std::string path1="D:/Users/vtariel/Downloads/Demo-Mnist/train-images.idx3-ubyte";
    std::string path2="D:/Users/vtariel/Downloads/Demo-Mnist/train-labels.idx1-ubyte";
    Vec<Vec<Mat2UI8> > number_training =  TrainingNeuralNetwork::loadMNIST(path1,path2);


    NeuralNet net;
//    net.addLayerMatrixInput(32,32,1);
//    net.addLayerMatrixConvolutionSubScaling(6,1,2);
//    net.addLayerMatrixMaxPool(2);
//    net.addLayerMatrixConvolutionSubScaling(16,1,2);
//    net.addLayerMatrixMaxPool(2);
//    net.addLayerLinearFullyConnected(120);
//    net.addLayerLinearFullyConnected(84);
//    net.addLayerLinearFullyConnected(static_cast<unsigned int>(number_training.size()));

    Vec2I32 domain(29,29);
    net.addLayerMatrixInput(domain(0),domain(1),1);
    net.addLayerMatrixConvolutionSubScaling(6,2,2);
    net.addLayerMatrixConvolutionSubScaling(50,2,2);
    net.addLayerLinearFullyConnected(100);
    net.addLayerLinearFullyConnected(static_cast<unsigned int>(number_training.size()));

    Vec<std::string> label_digit;
    for(int i=0;i<10;i++)
        label_digit.push_back(BasicUtility::Any2String(i));
    net.label2String() = label_digit;


    Vec<VecF32> vtraining_in;
    Vec<VecF32> vtraining_out;

    for(unsigned int i=0;i<number_training.size();i++){
        for(unsigned int j=0;j<number_training(i).size();j++){
            Mat2UI8 binary = number_training(i)(j);
            VecF32 vin = net.inputMatrixToInputNeuron(binary);
            vtraining_in.push_back(vin);
            VecF32 v_out(static_cast<int>(number_training.size()),-1);
            v_out(i)=1;
            vtraining_out.push_back(v_out);

        }
    }
    F32 eta=0.01;
    net.setLearnableParameter(eta);
    net.setTrainable(true);
    std::vector<int> v_global_rand(vtraining_in.size());
    for(unsigned int i=0;i<v_global_rand.size();i++)
        v_global_rand[i]=i;
    std::cout<<"iter_epoch\t error_train\t error_test\t learning rate"<<std::endl;
    int nbr_epoch=100;
    for(unsigned int i=0;i<nbr_epoch;i++){
        std::random_shuffle ( v_global_rand.begin(), v_global_rand.end() ,Distribution::irand());
        int error_training=0;
        for(unsigned int j=0;j<v_global_rand.size();j++){
            VecF32 vout;
            net.forwardCPU(vtraining_in(v_global_rand[j]),vout);
            net.backwardCPU(vtraining_out(v_global_rand[j]));
            net.learn();
            int label1 = std::distance(vout.begin(),std::max_element(vout.begin(),vout.end()));
            int label2 = std::distance(vtraining_out(v_global_rand[j]).begin(),std::max_element(vtraining_out(v_global_rand[j]).begin(),vtraining_out(v_global_rand[j]).end()));
            if(label1!=label2)
                error_training++;
        }

        std::cout<<i<<"\t"<<error_training*1./vtraining_in.size()<<"\t"<<eta<<std::endl;
        eta *=0.9f;
        eta = std::max(eta,0.001f);
        net.setLearnableParameter(eta);
    }
}

void neuralNetworkForRecognitionForHandwrittenDigits2()
{
    std::string path1="D:/Users/vtariel/Downloads/Demo-Mnist/train-images.idx3-ubyte";
    std::string path2="D:/Users/vtariel/Downloads/Demo-Mnist/train-labels.idx1-ubyte";
    Vec<Vec<Mat2UI8> > number_training =  TrainingNeuralNetwork::loadMNIST(path1,path2);


    NeuralNetworkFeedForward net;
    Vec2I32 domain(29,29);
    net.addInputLayerMatrix(domain(0),domain(1));
    net.addLayerConvolutionalPlusSubScaling(6,5,2,1);
    net.addLayerConvolutionalPlusSubScaling(50,5,2,1);
    net.addLayerFullyConnected(100,1);
    net.addLayerFullyConnected(static_cast<unsigned int>(number_training.size()));


    Vec<std::string> label_digit;
    for(int i=0;i<10;i++)
        label_digit.push_back(BasicUtility::Any2String(i));
    net.label2String() = label_digit;


    Vec<VecF32> vtraining_in;
    Vec<VecF32> vtraining_out;

    for(unsigned int i=0;i<number_training.size();i++){
        for(unsigned int j=0;j<number_training(i).size();j++){
            Mat2UI8 binary = number_training(i)(j);
            VecF32 vin = net.inputMatrixToInputNeuron(binary);
            vtraining_in.push_back(vin);
            VecF32 v_out(static_cast<int>(number_training.size()),-1);
            v_out(i)=1;
            vtraining_out.push_back(v_out);

        }
    }
    F32 eta=0.01;
    net.setLearningRate(eta);
    std::vector<int> v_global_rand(vtraining_in.size());
    for(unsigned int i=0;i<v_global_rand.size();i++)
        v_global_rand[i]=i;
    std::cout<<"iter_epoch\t error_train\t error_test\t learning rate"<<std::endl;
    int nbr_epoch=100;
    for(unsigned int i=0;i<nbr_epoch;i++){
        std::random_shuffle ( v_global_rand.begin(), v_global_rand.end() ,Distribution::irand());
        int error_training=0;
        for(unsigned int j=0;j<v_global_rand.size();j++){
            VecF32 vout;
            net.propagateFront(vtraining_in(v_global_rand[j]),vout);
            net.propagateBackFirstDerivate(vtraining_out(v_global_rand[j]));
            net.learningFirstDerivate();
            int label1 = std::distance(vout.begin(),std::max_element(vout.begin(),vout.end()));
            int label2 = std::distance(vtraining_out(v_global_rand[j]).begin(),std::max_element(vtraining_out(v_global_rand[j]).begin(),vtraining_out(v_global_rand[j]).end()));
            if(label1!=label2)
                error_training++;
        }

        std::cout<<i<<"\t"<<error_training*1./vtraining_in.size()<<"\t"<<eta<<std::endl;
        eta *=0.9f;
        eta = std::max(eta,0.001f);
        net.setLearningRate(eta);
    }
}



int main()
{



//    neuralNetworkForRecognitionForHandwrittenDigits2();
    neuralNetworkForRecognitionForHandwrittenDigits();
    NeuralNet net;
    int sizei=4,sizej=4,nbr_map=2;
    net.addLayerMatrixInput(sizei,sizej,nbr_map);
    net.addLayerMatrixConvolutionSubScaling(2,1,1);
    net.addLayerMatrixMaxPool(2);
    net.addLayerLinearFullyConnected(1);
    net.setTrainable(true);
    net.setLearnableParameter(0.1);

    VecF32 v(sizei*sizej*nbr_map);
    DistributionUniformReal d(-1,1);
    for(unsigned int i=0;i<v.size();i++){
        v(i)=d.randomVariable();
    }
    VecF32 v_out;
    net.forwardCPU(v,v_out);
    std::cout<<"out " <<v_out(0)<<std::endl;
    v_out(0)=1;
    net.backwardCPU(v_out);
    net.learn();
    net.forwardCPU(v,v_out);
    std::cout<<"out " <<v_out(0)<<std::endl;
    return 0;

}
//    {

//        Vec<ComplexF32> v(SIZE);
//        v(0)=ComplexF32(2,-0.55);v(1)=ComplexF32(-0.5,0.255);v(2)=ComplexF32(4,-0.2);v(3)=ComplexF32(-3,1);
//        FFTDanielsonLanczosTest<SIZE> d2;


//        int nbr_iteration = 50;
//        auto start_global =  std::chrono::high_resolution_clock::now();
//        for(unsigned int i=0;i<nbr_iteration;i++){
//            for(unsigned int i=0;i<SIZE*SIZE;i++){
//                d2.apply(v.data()->data(),FFT_BACKWARD);
//            }
//        }
//        auto end_global= std::chrono::high_resolution_clock::now();
//        std::cout<<"processing : "<<std::chrono::duration<double, std::milli>(end_global-start_global).count()<<std::endl;
//        std::cout<<"processing : "<<std::chrono::duration<double, std::milli>(end_global-start_global).count()/nbr_iteration<<std::endl;

//        //        d.apply(v.data()->data(),FFT_BACKWARD);
//        //        std::cout<<v<<std::endl;
//        //        std::cout<<v1<<std::endl;
//        return 0;

//    }
//    {
//        Mat2F32 m(6,6);
//        m(0,0)=0;
//        m(2,2)=10;
//        m(4,4)=0;

//        Mat2F32 m2(6,6);
//        m2(3,1)=10;
//        NeuralLayerMatrixInput input_layer(6,6,2);
//        std::copy(m.begin(),m.end(),input_layer.X().begin());
//        std::copy(m2.begin(),m2.end(),input_layer.X().begin()+6*6);
//        NeuralLayerMatrixConvolutionFFT convolution_layer(2,6,6,1,2);

//        Vec<Mat2F32> v_W;
//        Mat2F32 W(3,3);
//        W(1,0)=-1;W(1,2)=1;
//        v_W.push_back(W);
//        W.fill(0);
//        W(0,1)=-1;W(2,1)=1;
//        v_W.push_back(W);
//        W.fill(0);
//        W(1,0)=-1;W(1,2)=0;
//        v_W.push_back(W);
//        W.fill(0);
//        W(0,1)=-1;W(2,1)=0;
//        v_W.push_back(W);

//        Vec<F32> v_W_biais(4,0);
//        convolution_layer.setWeight(v_W,v_W_biais);


//        convolution_layer.forwardCPU(input_layer);
//        std::cout<<convolution_layer.X_map()(0)<<std::endl;
//        std::cout<<convolution_layer.X_map()(1)<<std::endl;
//    }
//    {

//        Vec<ComplexF32> v(SIZE*SIZE);
//        v(0)=ComplexF32(2,-0.55);v(1)=ComplexF32(-0.5,0.255);v(2)=ComplexF32(4,-0.2);v(3)=ComplexF32(-3,1);
//        Private::FFTDanielsonLanczos<SIZE> d2;

//        int nbr_iteration = 6*50;
//        auto start_global =  std::chrono::high_resolution_clock::now();


//        Vec<ComplexF32>::iterator it_other =   v.begin();
//        Vec<ComplexF32>::iterator it;
//        int j=0,k=0;
//        for(unsigned int i=0;i<nbr_iteration;i++){

//            for(k=0;k<SIZE;k++){
//                for(unsigned int j=0;j<SIZE;j++){
//                    Vec<ComplexF32>::iterator it = it_other+j*SIZE;
//                    d2.apply( (it)->data(),FFT_BACKWARD);
//                }
//            }
//        }
//        auto end_global= std::chrono::high_resolution_clock::now();
//        std::cout<<"processing : "<<std::chrono::duration<double, std::milli>(end_global-start_global).count()<<std::endl;
//        std::cout<<"processing : "<<std::chrono::duration<double, std::milli>(end_global-start_global).count()/nbr_iteration<<std::endl;

//        //        d.apply(v.data()->data(),FFT_BACKWARD);
//        //        std::cout<<v<<std::endl;
//        //        std::cout<<v1<<std::endl;
//        return 1;

//    }
//    {

//        Private::FFTDanielsonLanczos<32 > fft;
//        Mat2F32 m1(16,16);

//    }



//    Mat2UI8 img(8,8);
//    img(0,0)=100;
//    img(2,2)=100;

//    Mat2F32 biais(8,8);
//    biais.fill(10);
//    //        img.load((std::string(POP_PROJECT_SOURCE_DIR)+"/image/eutel.bmp"));
//    Mat2F32 W(3,3);
//    W(0,0)=0.23;W(0,1)=0.14;W(0,2)=0.14;
//    W(1,0)=0.45;W(1,1)=0.24;W(1,2)=0.84;
//    W(2,0)=0.47;W(2,1)=0.21;W(2,2)=0.14;




//    Mat2F32 imgf(img);
//    Mat2F32 img2 = Processing::convolution(imgf,W,MatNBoundaryConditionPeriodic())+biais;
//    std::cout<<img2<<std::endl;
//    ConvolutionFourier c;


//    c.convolution(imgf,W,10);
//    //        std::cout<<imgf<<std::endl;



//    return 1;
//    //        ConvolutionFourier c;
//    //        Mat2F32 imgf(img);
//    //        //imgf.display();
//    //        MatNDisplay disp;
//    //        while(1==1){
//    //            c.convolution(imgf,W);
//    ////            std::cout<<imgf<<std::endl;
//    ////
//    //            disp.display(imgf);
//    //        }
//    //
//    //imgf.display();

//    //        Representation::correlationDirectionByFFT(img).display();
//    //        Mat2ComplexF32 imgc;
//    //        Convertor::fromRealImaginary(Mat2F32(img),imgc);
//    //        imgc = Representation::FFT(imgc);
//    //        img = Representation::FFTDisplay(imgc);
//    //        img.display();

//    {
//        Mat2ComplexF32 m(8,8);
//        DistributionUniformInt d(0,7);
//        for(unsigned int i=0;i<m.size();i++)
//            m(i)=ComplexF32(d.randomVariable(),0);
//        std::cout<<m<<std::endl;

//        //        std::cout<<out<<std::endl;
//        auto start_global =  std::chrono::high_resolution_clock::now();
//        m=Representation::FFT(m);
//        m=Representation::FFT(m,FFT_BACKWARD);
//        Representation::scale(m);
//        auto end_global= std::chrono::high_resolution_clock::now();
//        std::cout<<"processing : "<<std::chrono::duration<double, std::milli>(end_global-start_global).count()<<std::endl;

//        std::cout<<m<<std::endl;
//        return 1;
//    }
//    int number_iteration=1000;
//    {
//        Mat2F32 m(N2,N2);
//        /* prepare a cosine wave */
//        for (int i = 0; i < N2; i++) {
//            for (int j = 0; j < N2; j++) {
//                m(i,j) = sin(3 * 2*M_PI*(i+j)/N2);
//            }
//        }
//        int radius=3;
//        Mat2F32 W(radius,radius);
//        W(0,0)=-1;W(0,2)=1;
//        W(1,0)=-2;W(1,2)=2;
//        W(2,0)=-1;W(2,2)=1;
//        //        auto start_global =  std::chrono::high_resolution_clock::now();
//        //        for(unsigned int i=0;i<number_iteration;i++)
//        //            Processing::convolution(m,W,MatNBoundaryConditionPeriodic());
//        //        auto end_global= std::chrono::high_resolution_clock::now();
//        //        std::cout<<"processing : "<<std::chrono::duration<double, std::milli>(end_global-start_global).count()/number_iteration<<std::endl;
//        //        std::cout<<std::endl;


//        //        start_global =  std::chrono::high_resolution_clock::now();
//        //        for(unsigned int i=0;i<number_iteration;i++)
//        //            c.convolution(m,W);
//        //        end_global= std::chrono::high_resolution_clock::now();
//        //        std::cout<<"processing : "<<std::chrono::duration<double, std::milli>(end_global-start_global).count()/number_iteration<<std::endl;
//        //        //        std::cout<<m<<std::endl;


//        return 1;

//        h.resizeInformation(size_i_output,size_j_output);
//        std::cout<<h<<std::endl;
//        return 1;
//  }

//{
//NeuralNet net;
//net.load("/home/vincent/DEV2/DEV/CVSA/bin/dictionaries/neuralnetwork.xml");
//Mat2UI8 m("/home/vincent/Desktop/_.jpg");

//VecF32 vin= net.inputMatrixToInputNeuron(m);
//VecF32 vout;
//int number_iteration = 10000;
//#if __cplusplus > 199711L // c++11
//auto start_global =  std::chrono::high_resolution_clock::now();
//#endif
//{
//for(unsigned int i=0;i<number_iteration;i++)
//net.forwardCPU(vin,vout);
////            vin(0)=vout(0);
//}
//std::cout<<vout<<std::endl;
//#if __cplusplus > 199711L // c++11
//auto end_global= std::chrono::high_resolution_clock::now();
//std::cout<<"processing : "<<std::chrono::duration<double, std::milli>(end_global-start_global).count()/number_iteration<<std::endl;
//#endif
//VecF32::iterator itt = std::max_element(vout.begin(),vout.end());
//int label_max = std::distance(vout.begin(),itt);
//std::cout<<label_max<<std::endl;
//return 1;

//        NeuralNet net;
//        int size_windows=9;
//        net.addLayerMatrixInput(size_windows,size_windows,1);
//        net.add

//}


//    {
//        NeuralNet net;
//        net.addLayerMatrixInput(2,2,2);
//        net.addLayerMatrixConvolutionSubScaling(3,1,0);
//        net.addLayerMatrixMergeConvolution();
//        net.setTrainable(true);
//        net.setLearnableParameter(0.1);
//        VecF32 v(2*2*2);
//        v(0)=1;v(1)=-1;v(2)=-1;v(3)= 1;
//        v(4)=1;v(5)=-1;v(6)= 1;v(7)=-1;

//        VecF32 v_exp(2*2);
//        v_exp(0)=1;v_exp(1)=1;v_exp(2)=-1;v_exp(3)=-1;
////        //        v_exp(4)=1;v_exp(5)=-1;v_exp(6)= 1;v_exp(7)=-1;

//        while(1==1){
//            VecF32 v_out;
//            net.forwardCPU(v,v_out);
//            std::cout<<v<<std::endl;
//            std::cout<<v_out<<std::endl;
//            std::cout<<v_exp<<std::endl;
//            net.backwardCPU(v_exp);
//            net.learn();
//        }
//////        net.forwardCPU(v,v_out);

//////        std::cout<<v_out<<std::endl;
////        return 1;


//    }

//{
//NeuralNet net;
//net.addLayerMatrixInput(7,7,1);

//net.addLayerMatrixConvolutionSubScaling(2,1,1);
//net.addLayerMatrixConvolutionSubScaling(4,1,1);
////        net.addLayerMatrixConvolutionSubScaling(20,1,1);
//net.addLayerMatrixMergeConvolution();
//{
//Mat2F32 m(7,7);
//m.fill(-1);
//m(1,2)=1;m(2,2)=1;m(3,2)=1;
//std::cout<<m<<std::endl;
//Mat2F32 v_out;
//net.forwardCPU(m,v_out);
//std::cout<<net.getMatrixOutput(0)<<std::endl;
//m.fill(-1);
//m(1,4)=1;m(2,4)=1;m(3,4)=1;
////            m(3,1)=1;
////            m(3,2)=1;
////            m(3,3)=1;
//std::cout<<m<<std::endl;
//net.forwardCPU(m,v_out);
//std::cout<<net.getMatrixOutput(0)<<std::endl;
//}
////net.addLayerMatrixConvolutionSubScaling(2,1,1);
////        net.addLayerMatrixConvolutionSubScaling(2,2,1);
////                     net.addLayerLinearFullyConnected(9);
////    std::cout<<net.getDomainMatrixOutput().first<<std::endl;
//Vec<Mat2F32> v_inputs;
//Vec<Mat2F32> v_outputs;
//for(unsigned int i=0;i<=1;i++)
//for(unsigned int j=0;j<=1;j++){
//    std::pair<Mat2F32,Mat2F32> m= matrixOrientation(i,j,false);
//    v_inputs.push_back(m.first);
//    v_outputs.push_back(m.second);
//    m= matrixOrientation(i,j,true);
//    v_inputs.push_back(m.first);
//    v_outputs.push_back(m.second);
//}
////        std::pair<Mat2F32,Mat2F32> m1 = matrixOrientation(1,0,false);
////        std::pair<Mat2F32,Mat2F32> m2 = matrixOrientation(0,0,true);
////        std::pair<Mat2F32,Mat2F32> m3 = matrixOrientation(2,2,false);
////        std::pair<Mat2F32,Mat2F32> m4 = matrixOrientation(1,1,tur);
////        std::pair<Mat2F32,Mat2F32> m5 = matrixOrientation(2,2,false);

////        v_inputs.push_back(m1.first);
////        v_inputs.push_back(m2.first);
////        v_inputs.push_back(m3.first);

////        v_outputs.push_back(m1.second);
////        v_outputs.push_back(m2.second);
////        v_outputs.push_back(m3.second);

//VecF32 v_out;

//net.setTrainable(true);
//double eta=0.001;
//net.setLearnableParameter(eta);


//Vec<int> v_global_rand(v_inputs.size());
//for(unsigned int i=0;i<v_global_rand.size();i++)
//v_global_rand[i]=i;

//for(unsigned int i=0;i<1000;i++){
//    int error =0;
//    std::random_shuffle ( v_global_rand.begin(), v_global_rand.end() ,Distribution::irand());
//    for(unsigned int j=0;j<v_global_rand.size();j++){
//        net.forwardCPU(v_inputs(v_global_rand(j)),v_out);
//        net.backwardCPU(v_outputs(v_global_rand(j)));
//        net.learn();
//        int label1 = std::distance(v_out.begin(),std::max_element(v_out.begin(),v_out.end()));
//        int label2 = std::distance(v_outputs(v_global_rand[j]).begin(),std::max_element(v_outputs(v_global_rand[j]).begin(),v_outputs(v_global_rand[j]).end()));
//        if(label1!=label2){
//            error++;
//        }
//        if(i>80){
//            Mat2F32 m(7,7);
//            m.fill(-1);
//            m(1,2)=1;m(2,2)=1;m(3,2)=1;
//            net.forwardCPU(m,v_out);
//            std::cout<<net.getMatrixOutput(0)<<std::endl;
//            m.fill(-1);
//            m(3,1)=1;
//            m(3,2)=1;
//            m(3,3)=1;
//            net.forwardCPU(m,v_out);
//            std::cout<<net.getMatrixOutput(0)<<std::endl;
//        }
//    }
//    eta*=0.95;
//    eta=std::max(eta,0.0001);
//    net.setLearnableParameter(eta);
//    std::cout<<error<<" "<<eta <<" "<<i<<std::endl;
//}
//return 1;

////        std::cout<<v_out<<std::endl;

//return 1;
//}



//std::string plaque = "/home/vincent/Desktop/plate.jpeg";
//std::string plaque_mask = "/home/vincent/Desktop/plate_mask.jpeg";
//Mat2RGBUI8 plate;
//plate.load(plaque);
//Mat2UI8 plate_mask;
//plate_mask.load(plaque_mask);


//Vec2I32 domain(100,100*800./322);

//NeuralNet net;
//int size_i=domain(0);
//int size_j=domain(1);
//int nbr_map=3;
//net.addLayerMatrixInput(size_i,size_j,nbr_map);
//net.addLayerMatrixConvolutionSubScaling(20,2,2);
//net.addLayerMatrixConvolutionSubScaling(30,2,2);
//net.addLayerMatrixConvolutionSubScaling(40,2,2);

//Vec2I32 domain_out = net.getDomainMatrixOutput().first;
//VecF32 v_out_expected;
//v_out_expected = normalizedImageToOutputNet(plate_mask,domain_out);

//std::cout<<domain_out.multCoordinate()<<std::endl;
//std::cout<<v_out_expected.size()<<std::endl;
//VecF32 v_in;
//v_in = normalizedImageToNeuralNet(plate,domain);

//VecF32 v_out;
//for(unsigned int i=0;i<100;i++){
//#if __cplusplus > 199711L // c++11
//    auto start_global = std::chrono::high_resolution_clock::now();
//#else
//    unsigned int start_global = time(NULL);
//#endif
//    net.setTrainable(true);
//    net.setLearnableParameter(0.001);
//    F32 sum=0;
//    for(unsigned int i=0;i<20;i++){
//        net.forwardCPU(v_in,v_out);
//        sum=0;
//        for(unsigned int j=0;j<v_out.size();j++){
//            sum+=std::abs(  v_out(j)  - v_out_expected(j)  );
//        }
//        std::cout<<sum<<std::endl;
//        net.backwardCPU(v_out_expected);
//        net.learn();
//        net.forwardCPU(v_in,v_out);
//        sum=0;
//        for(unsigned int j=0;j<v_out.size();j++){
//            sum+=std::abs(  v_out(j)  - v_out_expected(j)  );
//        }
//        std::cout<<sum<<std::endl;
//    }
//    displayOutput(v_out,domain_out).display();
//#if __cplusplus > 199711L // c++11
//    auto end_global= std::chrono::high_resolution_clock::now();
//    std::cout<<"processing : "<<std::chrono::duration<double, std::milli>(end_global-start_global).count()<<std::endl;
//#else
//    unsigned int end_global = time(NULL);
//    std::cout << "processing: " << (start_global-end_global) << "s" << std::endl;
//#endif
//    return 1;
//}

////        clusterToLabel(m,m.getIteratorENeighborhood(1,1),m.getIteratorEDomain());


////    omp_set_num_threads(6);
////    pop::Mat2UI8 m(1200,1600);
////    //auto start_global = std::chrono::high_resolution_clock::now();

////    m=thresholdNiblackMethod(m);
////    //auto end_global = std::chrono::high_resolution_clock::now();
////    std::cout<<"processing nimblack1 : "<<std::chrono::duration<double, std::milli>(end_global-start_global).count()<<std::endl;

////    int time1 = time(NULL);
////     //start_global = std::chrono::high_resolution_clock::now();

////    m=Processing::thresholdNiblackMethod(m);
////     //end_global = std::chrono::high_resolution_clock::now();
////    std::cout<<"processing nimblack2 : "<<std::chrono::duration<double, std::milli>(end_global-start_global).count()<<std::endl;

////    return 1;

////    //	m = m*m;
////    int time2 = time(NULL);
////    std::cout<<time2-time1<<std::endl;
////    Mat2UI8 img;//2d grey-level image object
////    img.load(POP_PROJECT_SOURCE_DIR+std::string("/image/iex.png"));//replace this path by those on your computer
////    img = PDE::nonLinearAnisotropicDiffusion(img);//filtering
////    int value;
////    Mat2UI8 threshold = Processing::thresholdOtsuMethod(img,value);//threshold segmentation
////    threshold.save("iexthreshold.pgm");
////    Mat2RGBUI8 color = Visualization::labelForeground(threshold,img);//Visual validation
////    color.display();

//return 0;
//}
//        //        std::cout<<m<<std::endl;
//        //        FFT2D<N2,N2,F32> fft;

//        //        //    auto start_global =  std::chrono::high_resolution_clock::now();
//        //        //    for(unsigned int i=0;i<number_iteration;i++){
//        //        //    fft.apply(m);
//        //        //    }
//        //        //    auto end_global= std::chrono::high_resolution_clock::now();
//        //        //    std::cout<<"processing : "<<std::chrono::duration<double, std::milli>(end_global-start_global).count()/number_iteration<<std::endl;
//        //        //    return 0;
//        //        //    std::cout<<m<<std::endl;
//        //        fft.apply(m);
//        //        fft.apply(m,FFT_BACKWARD);
//        //        std::cout<<m<<std::endl;
//    }
//    //int N=16;
//    //    fftw_complex in[N], out[N], in2[N]; /* double [2] */
//    //    fftw_plan p, q;
//    //    int i;

//    //    /* prepare a cosine wave */
//    //    for (i = 0; i < N; i++) {
//    //        in[i][0] = sin(3 * 2*M_PI*i/N)+cos(5 * 2*M_PI*i/N);
//    //        in[i][1] = 0;
//    //    }

//    //    /* forward Fourier transform, save the result in 'out' */
//    //    p = fftw_plan_dft_1d(N, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
//    //    fftw_execute(p);
//    //    //    for (i = 0; i < N; i++)
//    //    //        printf("freq: %3d %+9.5f %+9.5f I\n", i, out[i][0], out[i][1]);
//    //    fftw_destroy_plan(p);

//    //    /* backward Fourier transform, save the result in 'in2' */
//    //    //    printf("\nInverse transform:\n");
//    //    q = fftw_plan_dft_1d(N, out, in2, FFTW_BACKWARD, FFTW_ESTIMATE);

//    //    {
//    //        auto start_global =  std::chrono::high_resolution_clock::now();
//    //        for(unsigned int i=0;i<number_iteration;i++){
//    //            fftw_execute(q);
//    //        }
//    //        auto end_global= std::chrono::high_resolution_clock::now();
//    //        std::cout<<"processing : "<<std::chrono::duration<double, std::milli>(end_global-start_global).count()/number_iteration<<std::endl;
//    //    }

//    //    /* normalize */
//    //    for (i = 0; i < N; i++) {
//    //        in2[i][0] *= 1./N;
//    //        in2[i][1] *= 1./N;
//    //    }
//    //    //    for (i = 0; i < N; i++)
//    //    //        printf("recover: %3d %+9.5f %+9.5f I vs. %+9.5f %+9.5f I\n",
//    //    //               i, in[i][0], in[i][1], in2[i][0], in2[i][1]);
//    //    fftw_destroy_plan(q);

//    //    fftw_cleanup();

//    //    Vec<ComplexF32> data2(N*2);


//    //    for (int i = 0; i < N; i++) {
//    //        //        data[i*2]= sin(3 * 2*M_PI*i/N);
//    //        //        data[i*2+1]=0;
//    //        data2(i)=ComplexF32(sin(3 * 2*M_PI*i/N),0);
//    //    }
//    //    F32 * data = data2.data()->data();
//    //    for(unsigned int i=0;i<2*N;i++){
//    //        if(std::abs(data[i])>0.01)
//    //            std::cout<<data[i]<<" ";
//    //        else
//    //            std::cout<<"0 ";
//    //    }
//    //    std::cout<<std::endl;
//    //    FFTDanielsonLanczos<N,F32> d;
//    //    d.apply(data);
//    //    d.apply(data,FFT_BACKWARD);
//    //    for(unsigned int i=0;i<2*N;i++){
//    //        if(std::abs(data[i])>0.01)
//    //            std::cout<<data[i]<<" ";
//    //        else
//    //            std::cout<<"0 ";
//    //    }
//    //    auto start_global =  std::chrono::high_resolution_clock::now();
//    //    for(unsigned int i=0;i<number_iteration;i++){

//    //        d.apply(data);
//    //    }
//    //    auto end_global= std::chrono::high_resolution_clock::now();
//    //    std::cout<<"processing : "<<std::chrono::duration<double, std::milli>(end_global-start_global).count()/number_iteration<<std::endl;


//    //    std::cout<<"data 2"<<std::endl;
//    //    for(unsigned int i=0;i<2*N;i++){
//    //        std::cout<<data2[i]<<" ";
//    //    }
//    //    std::cout<<std::endl;

//    //    four1(data.data(), N);
//    //    for(unsigned int i=0;i<2*N;i++){
//    //        std::cout<<data[i]<<" ";
//    //    }
//    //    std::cout<<std::endl;
//    //    for(unsigned int i=0;i<2*N;i++){
//    //        std::cout<<data2[i]<<" ";
//    //    }

//    //    std::cout<<std::endl;
//    //    std::cout<<Representation::FFT(f1,1)<<std::endl;

//    return 1;
//    //    Mat2F32 m(29,29);
//    //    for(unsigned int i=0;i<m.sizeI();i++)
//    //        for(unsigned int j=0;j<m.sizeJ();j++){
//    //            m(i,j)=i+j;
//    //        }
//    //    std::cout<<m<<std::endl;
//    //    Mat2F32 kernel(3,3);
//    //    kernel(0,0)=-1;
//    //    kernel(1,0)=-2;
//    //    kernel(2,0)=-1;

//    //    kernel(0,2)=1;
//    //    kernel(1,2)=2;
//    //    kernel(2,2)=3;

//    //    //    Mat2F32


//    //    std::cout<<Processing::convolution(m,kernel,MatNBoundaryConditionMirror())<<std::endl;
//    //    Mat2F32 H = toeplitzMatrix(m.getDomain(),kernel);
//    //    H = toeplitzMatrixRemoveBorder(m.getDomain(),kernel,H,2);
//    ////    std::cout<<H<<std::endl;
//    //    std::cout<<H.getDomain()<<std::endl;
//    //    VecF32 & m_v= m;
//    //    std::cout<<Mat2F32(Vec2I32(3,3),H*m_v)<<std::endl;



//    return 1;



//    {
//        NeuralNet net;
//        net.load("/home/vincent/DEV2/DEV/CVSA/bin/dictionaries/neuralnetwork.xml");
//        Mat2UI8 m("/home/vincent/Desktop/_.jpg");

//        VecF32 vin= net.inputMatrixToInputNeuron(m);
//        VecF32 vout;
//        int number_iteration = 10000;
//        auto start_global =  std::chrono::high_resolution_clock::now();{
//            for(unsigned int i=0;i<number_iteration;i++)
//                net.forwardCPU(vin,vout);
//            //            vin(0)=vout(0);
//        }
//        std::cout<<vout<<std::endl;
//        auto end_global= std::chrono::high_resolution_clock::now();
//        std::cout<<"processing : "<<std::chrono::duration<double, std::milli>(end_global-start_global).count()/number_iteration<<std::endl;
//        VecF32::iterator itt = std::max_element(vout.begin(),vout.end());
//        int label_max = std::distance(vout.begin(),itt);
//        std::cout<<label_max<<std::endl;
//        return 1;

//        //        NeuralNet net;
//        //        int size_windows=9;
//        //        net.addLayerMatrixInput(size_windows,size_windows,1);




//        //        net.add

//    }


//    //    {
//    //          NeuralNet net;
//    //        net.addLayerMatrixInput(2,2,2);
//    //            net.addLayerMatrixConvolutionSubScaling(3,1,0);
//    //        net.addLayerMatrixMergeConvolution();
//    //        net.setTrainable(true);
//    //        net.setLearnableParameter(0.1);
//    //        VecF32 v(2*2*2);
//    //        v(0)=1;v(1)=-1;v(2)=-1;v(3)= 1;
//    //        v(4)=1;v(5)=-1;v(6)= 1;v(7)=-1;

//    //        VecF32 v_exp(2*2);
//    //        v_exp(0)=1;v_exp(1)=1;v_exp(2)=-1;v_exp(3)=-1;
//    ////        //        v_exp(4)=1;v_exp(5)=-1;v_exp(6)= 1;v_exp(7)=-1;

//    //        while(1==1){
//    //            VecF32 v_out;
//    //            net.forwardCPU(v,v_out);
//    //            std::cout<<v<<std::endl;
//    //            std::cout<<v_out<<std::endl;
//    //            std::cout<<v_exp<<std::endl;
//    //            net.backwardCPU(v_exp);
//    //            net.learn();
//    //        }
//    //////        net.forwardCPU(v,v_out);

//    //////        std::cout<<v_out<<std::endl;
//    ////        return 1;


//    //    }

//    {
//        NeuralNet net;
//        net.addLayerMatrixInput(7,7,1);

//        net.addLayerMatrixConvolutionSubScaling(2,1,1);
//        net.addLayerMatrixConvolutionSubScaling(4,1,1);
//        //        net.addLayerMatrixConvolutionSubScaling(20,1,1);
//        net.addLayerMatrixMergeConvolution();
//        {
//            Mat2F32 m(7,7);
//            m.fill(-1);
//            m(1,2)=1;m(2,2)=1;m(3,2)=1;
//            std::cout<<m<<std::endl;
//            Mat2F32 v_out;
//            net.forwardCPU(m,v_out);
//            std::cout<<net.getMatrixOutput(0)<<std::endl;
//            m.fill(-1);
//            m(1,4)=1;m(2,4)=1;m(3,4)=1;
//            //            m(3,1)=1;
//            //            m(3,2)=1;
//            //            m(3,3)=1;
//            std::cout<<m<<std::endl;
//            net.forwardCPU(m,v_out);
//            std::cout<<net.getMatrixOutput(0)<<std::endl;
//        }
//        //net.addLayerMatrixConvolutionSubScaling(2,1,1);
//        //        net.addLayerMatrixConvolutionSubScaling(2,2,1);
//        //                     net.addLayerLinearFullyConnected(9);
//        //    std::cout<<net.getDomainMatrixOutput().first<<std::endl;
//        Vec<Mat2F32> v_inputs;
//        Vec<Mat2F32> v_outputs;
//        for(unsigned int i=0;i<=1;i++)
//            for(unsigned int j=0;j<=1;j++){
//                //                std::pair<Mat2F32,Mat2F32> m= matrixOrientation(i,j,false);
//                //                v_inputs.push_back(m.first);
//                //                v_outputs.push_back(m.second);
//                //                m= matrixOrientation(i,j,true);
//                //                v_inputs.push_back(m.first);
//                //                v_outputs.push_back(m.second);
//            }
//        //        std::pair<Mat2F32,Mat2F32> m1 = matrixOrientation(1,0,false);
//        //        std::pair<Mat2F32,Mat2F32> m2 = matrixOrientation(0,0,true);
//        //        std::pair<Mat2F32,Mat2F32> m3 = matrixOrientation(2,2,false);
//        //        std::pair<Mat2F32,Mat2F32> m4 = matrixOrientation(1,1,tur);
//        //        std::pair<Mat2F32,Mat2F32> m5 = matrixOrientation(2,2,false);

//        //        v_inputs.push_back(m1.first);
//        //        v_inputs.push_back(m2.first);
//        //        v_inputs.push_back(m3.first);

//        //        v_outputs.push_back(m1.second);
//        //        v_outputs.push_back(m2.second);
//        //        v_outputs.push_back(m3.second);

//        VecF32 v_out;

//        net.setTrainable(true);
//        double eta=0.001;
//        net.setLearnableParameter(eta);


//        Vec<int> v_global_rand(v_inputs.size());
//        for(unsigned int i=0;i<v_global_rand.size();i++)
//            v_global_rand[i]=i;

//        for(unsigned int i=0;i<1000;i++){
//            int error =0;
//            std::random_shuffle ( v_global_rand.begin(), v_global_rand.end() ,Distribution::irand());
//            for(unsigned int j=0;j<v_global_rand.size();j++){
//                net.forwardCPU(v_inputs(v_global_rand(j)),v_out);
//                net.backwardCPU(v_outputs(v_global_rand(j)));
//                net.learn();
//                int label1 = std::distance(v_out.begin(),std::max_element(v_out.begin(),v_out.end()));
//                int label2 = std::distance(v_outputs(v_global_rand[j]).begin(),std::max_element(v_outputs(v_global_rand[j]).begin(),v_outputs(v_global_rand[j]).end()));
//                if(label1!=label2){
//                    error++;
//                }
//                if(i>80){
//                    Mat2F32 m(7,7);
//                    m.fill(-1);
//                    m(1,2)=1;m(2,2)=1;m(3,2)=1;
//                    net.forwardCPU(m,v_out);
//                    std::cout<<net.getMatrixOutput(0)<<std::endl;
//                    m.fill(-1);
//                    m(3,1)=1;
//                    m(3,2)=1;
//                    m(3,3)=1;
//                    net.forwardCPU(m,v_out);
//                    std::cout<<net.getMatrixOutput(0)<<std::endl;
//                }
//            }
//            eta*=0.95;
//            eta=std::max(eta,0.0001);
//            net.setLearnableParameter(eta);
//            std::cout<<error<<" "<<eta <<" "<<i<<std::endl;
//        }
//        return 1;

//        //        std::cout<<v_out<<std::endl;

//        return 1;
//    }



//    std::string plaque = "/home/vincent/Desktop/plate.jpeg";
//    std::string plaque_mask = "/home/vincent/Desktop/plate_mask.jpeg";
//    Mat2RGBUI8 plate;
//    plate.load(plaque);
//    Mat2UI8 plate_mask;
//    plate_mask.load(plaque_mask);


//    Vec2I32 domain(100,100*800./322);

//    NeuralNet net;
//    int size_i=domain(0);
//    int size_j=domain(1);
//    int nbr_map=3;
//    net.addLayerMatrixInput(size_i,size_j,nbr_map);
//    net.addLayerMatrixConvolutionSubScaling(20,2,2);
//    net.addLayerMatrixConvolutionSubScaling(30,2,2);
//    net.addLayerMatrixConvolutionSubScaling(40,2,2);

//    Vec2I32 domain_out = net.getDomainMatrixOutput().first;
//    VecF32 v_out_expected;
//    v_out_expected = normalizedImageToOutputNet(plate_mask,domain_out);

//    std::cout<<domain_out.multCoordinate()<<std::endl;
//    std::cout<<v_out_expected.size()<<std::endl;
//    VecF32 v_in;
//    v_in = normalizedImageToNeuralNet(plate,domain);

//    VecF32 v_out;
//    for(unsigned int i=0;i<100;i++){
//#if __cplusplus > 199711L // c++11
//        auto start_global = std::chrono::high_resolution_clock::now();
//#else
//        unsigned int start_global = time(NULL);
//#endif
//        net.setTrainable(true);
//        net.setLearnableParameter(0.001);
//        F32 sum=0;
//        for(unsigned int i=0;i<20;i++){
//            net.forwardCPU(v_in,v_out);
//            sum=0;
//            for(unsigned int j=0;j<v_out.size();j++){
//                sum+=std::abs(  v_out(j)  - v_out_expected(j)  );
//            }
//            std::cout<<sum<<std::endl;
//            net.backwardCPU(v_out_expected);
//            net.learn();
//            net.forwardCPU(v_in,v_out);
//            sum=0;
//            for(unsigned int j=0;j<v_out.size();j++){
//                sum+=std::abs(  v_out(j)  - v_out_expected(j)  );
//            }
//            std::cout<<sum<<std::endl;
//        }
//        //        displayOutput(v_out,domain_out).display();
//#if __cplusplus > 199711L // c++11
//        auto end_global= std::chrono::high_resolution_clock::now();
//        std::cout<<"processing : "<<std::chrono::duration<double, std::milli>(end_global-start_global).count()<<std::endl;
//#else
//        unsigned int end_global = time(NULL);
//        std::cout << "processing: " << (start_global-end_global) << "s" << std::endl;
//#endif
//        return 1;
//    }

//    //        clusterToLabel(m,m.getIteratorENeighborhood(1,1),m.getIteratorEDomain());


//    //    omp_set_num_threads(6);
//    //    pop::Mat2UI8 m(1200,1600);
//    //    //auto start_global = std::chrono::high_resolution_clock::now();

//    //    m=thresholdNiblackMethod(m);
//    //    //auto end_global = std::chrono::high_resolution_clock::now();
//    //    std::cout<<"processing nimblack1 : "<<std::chrono::duration<double, std::milli>(end_global-start_global).count()<<std::endl;

//    //    int time1 = time(NULL);
//    //     //start_global = std::chrono::high_resolution_clock::now();

//    //    m=Processing::thresholdNiblackMethod(m);
//    //     //end_global = std::chrono::high_resolution_clock::now();
//    //    std::cout<<"processing nimblack2 : "<<std::chrono::duration<double, std::milli>(end_global-start_global).count()<<std::endl;

//    //    return 1;

//    //    //	m = m*m;
//    //    int time2 = time(NULL);
//    //    std::cout<<time2-time1<<std::endl;
//    //    Mat2UI8 img;//2d grey-level image object
//    //    img.load(POP_PROJECT_SOURCE_DIR+std::string("/image/iex.png"));//replace this path by those on your computer
//    //    img = PDE::nonLinearAnisotropicDiffusion(img);//filtering
//    //    int value;
//    //    Mat2UI8 threshold = Processing::thresholdOtsuMethod(img,value);//threshold segmentation
//    //    threshold.save("iexthreshold.pgm");
//    //    Mat2RGBUI8 color = Visualization::labelForeground(threshold,img);//Visual validation
//    //    color.display();

//    return 0;
//}

