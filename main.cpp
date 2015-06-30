#include"Population.h"//Single header
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
Mat2UI8 displayOutput(VecF32 v_out, Vec2I32 domain){
    Mat2UI8 m(domain(0),domain(1));
    int k=0;
    for(unsigned int i=0;i<domain(0);i++){
        for(unsigned int j=0;j<domain(1);j++,k++){
            m(i,j)=ArithmeticsSaturation<UI8,F64>::Range( (v_out[k]+1)*127.5);
        }
    }
    return m;
}
std::pair<Mat2F32,Mat2F32> matrixOrientation(int i,int j,bool orientation){
    Mat2F32 m_out_expected(3,3);
    m_out_expected.fill(-1);
    m_out_expected(i,j)=1;





    //    std::cout<<m_out_expected<<std::endl;

    Mat2F32 m_in(7,7);
    for(unsigned int l=0;l<m_in.size();l++){
        m_in(l)=-1;//(i%2)*2.-1;
    }
    for( int k=-1;k<=1;k++){
        if(orientation==true)
            m_in(i+2+k,j+2  )=1;
        else
            m_in(i+2  ,j+2+k)=1;
    }



    std::cout<<m_in<<std::endl;
    return std::make_pair(m_in,m_out_expected);
}


int main()
{
    //    {//AND
    //        // creation du réseau
    //        NeuralNet net;
    //        net.addLayerLinearInput(2);
    //        net.addLayerLinearFullyConnected(3);
    //        net.addLayerLinearFullyConnected(2);
    //        net.addLayerLinearFullyConnected(1);

    //        Vec<VecF32> v_in(4,VecF32(2));
    //        v_in(0)(0)=1; v_in(0)(1)=1;
    //        v_in(1)(0)=-1;v_in(1)(1)=1;
    //        v_in(2)(0)=1; v_in(2)(1)=-1;
    //        v_in(3)(0)=-1;v_in(3)(1)=-1;
    //        Vec<VecF32> v_exp(4,VecF32(1));
    //        v_exp(0)(0)=1;
    //        v_exp(1)(0)=-1;
    //        v_exp(2)(0)=-1;
    //        v_exp(3)(0)=-1;

    //        net.setTrainable(true);
    //        float mu = 0.9;
    //        float q = 0.6; //raison
    //        int c = 1; //itération
    //        net.setLearnableParameter(mu);

    //        // pour chaque itération on veut calculer aléatoirement les sorties du vecteur v_in
    //        // shuffle contient les indices du vect v_in
    //        Vec<int> v_shuffle(4);
    //        v_shuffle(0)=0;v_shuffle(1)=1;v_shuffle(2)=2;v_shuffle(3)=3;//mélange des indices du vect v_in

    //        while(1==1){


    //            std::random_shuffle(v_shuffle.begin(),v_shuffle.end());

    //            std::cout<<"ITERATION "<<c<<std::endl;
    //            std::cout<<"mu "<<mu<<std::endl;

    //            for(unsigned int i=0;i<v_shuffle.size();i++){
    //                VecF32 v_out;
    //                net.forwardCPU(v_in(v_shuffle(i)),v_out);
    //                std::cout<<"vect in : "<<v_in(v_shuffle(i))(0)<<" "<<v_in(v_shuffle(i))(1)<<std::endl;
    //               std::cout<<"vect expected : "<<v_exp(v_shuffle(i))(0)<<" "<<"vect obtenu : "<<v_out(0)<<std::endl<<std::endl;
    //                v_out(0)=1;
    //                net.backwardCPU(v_exp(v_shuffle(i)));
    //                net.learn();
    //            }
    //            mu=pow(q,c)*mu;
    //            net.setLearnableParameter(mu);
    //            c++;
    //        }

        //CONVOLUTION NEURAL NETWORK
        NeuralNet net;
        net.addLayerMatrixInput(7,7,1);
        net.addLayerMatrixConvolutionSubScaling(3,1,1);//nb map, subscaling factor, (radius kernel=1 ==> 3*3)
        net.addLayerMatrixConvolutionSubScaling(4,1,1);
        net.addLayerMatrixMergeConvolution();

        Mat2F32 m_in(7,7);
        m_in.fill(-1);
        m_in(1,2)=1;m_in(2,2)=1;m_in(3,2)=1;
        std::cout<<"m_in "<<m_in<<std::endl;
        VecF32 v_in= net.inputMatrixToInputNeuron(m_in);//conversion de la matrice d'entrée en vecteur

        Vec <float> v_exp(2);
        v_exp(0)=1;//
        v_exp(1)=1;

        net.setTrainable(true);
        net.setLearnableParameter(0.1);

        Vec<int> v_shuffle(49);
        for(unsigned int i=0;i<v_shuffle.size();i++){
            v_shuffle(i)=i;
        }
        int c = 1; //itération
        while(1==1){


            std::random_shuffle(v_shuffle.begin(),v_shuffle.end());//mélange des indices du vect v_in
            std::cout<<"ITERATION "<<c<<std::endl;
            for(unsigned int i=0;i<v_shuffle.size();i++){
                VecF32 v_out;
                net.forwardCPU(v_in(v_shuffle(i)),v_out);
                std::cout<<"vect expected : "<<v_exp(0)<<" "<<v_exp(1)<<" "<<"vect obtenu : "<<v_out(0)<<v_out(1)<<std::endl<<std::endl;
                // v_out(0)=1;
                net.backwardCPU(v_exp);
                net.learn();
            }

            c++;
        }



    return 0;
}

// {
//        Mat2F32 f(2,3);
//        f(0,0)=2;f(0,1)=5;f(0,2)=3;
//        f(1,0)=1;f(1,1)=4;f(1,2)=1;

//        Mat2F32 h(2,2);
//        h(0,0)=1;h(0,1)=-1;
//        h(1,0)=1;h(1,1)=1;
//        int size_i_output=f.sizeI()+h.sizeI()-1;
//        int size_j_output=f.sizeJ()+h.sizeJ()-1;

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


