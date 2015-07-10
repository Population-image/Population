#include"Population.h"//Single header
#include<chrono>
using namespace pop;//Population namespace

//void neuralNetworkForRecognitionForHandwrittenDigits()
//{
//    std::string path1="/home/olivia/workspace/MNIST/train-images-idx3-ubyte";
//    std::string path2="/home/olivia/workspace/MNIST/train-labels-idx1-ubyte";
//    Vec<Vec<Mat2UI8> > number_training =  TrainingNeuralNetwork::loadMNIST(path1,path2);


//    NeuralNet net;
//    net.addLayerMatrixInput(32,32,1);
//    net.addLayerMatrixConvolutionSubScaling(6,1,2);
//    net.addLayerMatrixMaxPool(2);
//    net.addLayerMatrixConvolutionSubScaling(16,1,2);
//    net.addLayerMatrixMaxPool(2);
//    net.addLayerLinearFullyConnected(120);
//    net.addLayerLinearFullyConnected(84);
//    net.addLayerLinearFullyConnected(static_cast<unsigned int>(number_training.size()));


//    Vec<std::string> label_digit;
//    for(int i=0;i<10;i++)
//        label_digit.push_back(BasicUtility::Any2String(i));
//    net.label2String() = label_digit;


//    Vec<VecF32> vtraining_in;
//    Vec<VecF32> vtraining_out;

//    for(unsigned int i=0;i<number_training.size();i++){
//        for(unsigned int j=0;j<number_training(i).size();j++){
//            Mat2UI8 binary = number_training(i)(j);
//            VecF32 vin = net.inputMatrixToInputNeuron(binary);
//            vtraining_in.push_back(vin);
//            VecF32 v_out(static_cast<int>(number_training.size()),-1);
//            v_out(i)=1;
//            vtraining_out.push_back(v_out);

//        }
//    }
//    F32 eta=0.01;
//    net.setLearnableParameter(eta);
//    net.setTrainable(true);
//    std::vector<int> v_global_rand(vtraining_in.size());
//    for(unsigned int i=0;i<v_global_rand.size();i++)
//        v_global_rand[i]=i;
//    std::cout<<"iter_epoch\t error_train\t error_test\t learning rate"<<std::endl;
//    int nbr_epoch=100;
//    for(unsigned int i=0;i<nbr_epoch;i++){
//        std::random_shuffle ( v_global_rand.begin(), v_global_rand.end() ,Distribution::irand());
//        int error_training=0;
//        for(unsigned int j=0;j<v_global_rand.size();j++){
//            VecF32 vout;
//            net.forwardCPU(vtraining_in(v_global_rand[j]),vout);
//            net.backwardCPU(vtraining_out(v_global_rand[j]));
//            net.learn();
//            int label1 = std::distance(vout.begin(),std::max_element(vout.begin(),vout.end()));
//            int label2 = std::distance(vtraining_out(v_global_rand[j]).begin(),std::max_element(vtraining_out(v_global_rand[j]).begin(),vtraining_out(v_global_rand[j]).end()));
//            if(label1!=label2)
//                error_training++;
//        }

//        std::cout<<i<<"\t"<<error_training*1./vtraining_in.size()<<"\t"<<eta<<std::endl;
//        eta *=0.9f;
//        eta = std::max(eta,0.001f);
//        net.setLearnableParameter(eta);
//    }
//}



Mat2UI8 normalize (Mat2UI8 m,F32 ratio=0.2){
    int imin=m.sizeI()*(1-ratio)/2;
    int imax=m.sizeI()*(1+ratio)/2-1;
    int jmin=m.sizeJ()*(1-ratio)/2;
    int jmax=m.sizeJ()*(1+ratio)/2-1;


    pop::Mat2UI32 m_label = Processing::clusterToLabel(m);
    Vec<int> v_count;
    for(unsigned int i=imin;i<imax;i++)
        for(unsigned int j=jmin;j<jmax;j++){
            int label = m_label(i,j);
            if(v_count.size()<=label){
                v_count.resize(label,0);
            }
            v_count(label-1)++;
        }

    int label = std::distance(v_count.begin(),std::max_element(v_count.begin(),v_count.end()))-1;
    return Processing::greylevelRange(m_label,label,label);
}

void fill_vtin_vtout(Vec<VecF32> &v_in,Vec<VecF32> &v_out,std::string root_dir,NeuralNet net,bool test,int nb_dir){
    // 6 dossiers '1' '2' '3' '4' '10A' '10B'
    std::vector<std::string> path_dirs(nb_dir);

    for (int i =1;i<path_dirs.size()+1;i++) {

        std::string s;
        if (i==1||i==2||i==3||i==4){
            pop::BasicUtility::Any2String(i, s);
        }
        if (i==5){
            s="random/";
        }
        //            if (i==6){
        //                s="10B";
        //            }

        std::string str_test;
        if(test){
            str_test = "test";
        }else {
            str_test = "";
        }

        path_dirs[i-1] = root_dir + s + str_test +"/" ;
        Vec<std::string> v_images=BasicUtility::getFilesInDirectory(path_dirs[i-1]);

        for(unsigned int j=0;j<v_images.size();j++){
            std::string path = path_dirs[i-1] + v_images(j);
            //            std::cout<<path<<std::endl;
            Mat2UI8 img;
            img.load(path);
            int value;
            img= Processing::thresholdOtsuMethod(img,value);
            net._method = NNLayerMatrix::Mass;
            try
            {
                VecF32 vin = net.inputMatrixToInputNeuron(img);

                v_in.push_back(vin);
                VecF32 v(2,-1);//VecF32=Vec<F32>
                if(i==5)
                    v(1)= 1;
                else
                    v(0)= 1;
                v_out.push_back(v);
            }
            catch(std::string ){
                std::cout<<"catch error"<<path<<std::endl;
            }
        }


    }


}

void algoSearch(Mat2UI8 m,NeuralNet & net,int size=20,int step=8,F32 fluctuation=60){
    MatNDisplay disp;
    Mat2UI8 m_test(m.getDomain());

    FunctorMeanStandardDeviationIntegral f;
    f.setImage(m);
    for(unsigned int i =10;i<m.sizeI()-size;i+=step){
        //    for(unsigned int i =10;i<20;i+=step){
        std::cout<<i<<std::endl;
        for(unsigned int j =10;j<m.sizeJ()-size;j+=step){

            F32 value_max=-1;
            int label_max=0;
            if(f.standartDeviation(Vec2I32(i+size/2,j+size/2),size/4)>fluctuation){
                Mat2UI8 patch = m(Vec2I32(i,j),Vec2I32(i+size,j+size));
                //            disp.display(temp);
                int value;
                patch = Processing::thresholdOtsuMethod(patch ,value);
                //                patch.display();
                //            temp.display();

                try
                {
                    VecF32 vin = net.inputMatrixToInputNeuron(patch );
                    VecF32 vouttest;
                    net.forwardCPU(vin,vouttest);
                    label_max = std::distance(vouttest.begin(),std::max_element(vouttest.begin(),vouttest.end()));
                    value_max =*std::max_element(vouttest.begin(),vouttest.end());
                    //                std::cout<<label_max<<std::endl;
                    if(label_max==0)
                        disp.display(GeometricalTransformation::scale(patch,Vec2F32(4,4)));
                }
                catch(std::string ){

                }

                label_max++;

                //                if(value_max<0.7)
                //                    label_max=0;
            }
            for(unsigned int i1 =i;i1<i+step;i1++){
                for(unsigned int j1 =j;j1<j+step;j1++){
                    m_test(i1+size/2,j1+size/2)=label_max;
                }
            }
            //            std::cout<<j<<" "<< value_max<<std::endl;
        }
    }
    m.display("neural",false);
    Processing::greylevelRange(m_test,0,255).display();
    //    Visualization::labelToRandomRGB(m_test).display();
}


Vec<Mat2UI8> createrandom(Mat2UI8 m,F32 fluctuation, int size,int number){
    Vec<Mat2UI8>  v;
    DistributionUniformInt di(0,m.sizeI()-1-size);
    DistributionUniformInt dj(0,m.sizeJ()-1-size);
    FunctorMeanStandardDeviationIntegral f;
    f.setImage(m);

    int iter=0;
    while(iter<number){
        Vec2I32 x(di.randomVariable(),dj.randomVariable());
        if(f.standartDeviation(x+size/2,size/4)> fluctuation){
            //            std::cout<<fluctuation<<std::endl;
            //            std::cout<< m(x,x+Vec2I32(size,size))<<std::endl;
            iter++;
            Mat2UI8 patch = m(x,x+Vec2I32(size,size));
            //             patch.display();
            v.push_back(patch);
        }
    }

    return v;
}


int main()
{
    Mat2UI8 mm ("D:/Users/vtariel/Desktop/02.png");






    mm.display();
    F32 fluctuation=40;
    int size = 20;
    //    {
    //        Vec<Mat2UI8>  v =createrandom(mm,fluctuation,size,250);
    //        for(unsigned int i=0;i<v.size();i++){
    //            v(i).save("D:/Users/vtariel/Desktop/ANV/Population/database/random/"+BasicUtility::IntFixedDigit2String(i,4)+".png");
    //        }
    //    }


    NeuralNet net;
    int sizei=32,sizej=32,nbr_map=1; //nb_map correspond à une image en niveaux de gris si rgb 3 maps
    int nbr_dir=5;
    net.addLayerMatrixInput(sizei,sizej,nbr_map);
    net.addLayerMatrixConvolutionSubScaling(6,1,2);
    net.addLayerMatrixMaxPool(2);
    net.addLayerMatrixConvolutionSubScaling(16,1,2);
    net.addLayerMatrixMaxPool(2);
    net.addLayerLinearFullyConnected(120);
    net.addLayerLinearFullyConnected(84);
    //    net.addLayerLinearFullyConnected(200);
    net.addLayerLinearFullyConnected(2);


    std::string root_dir ="D:/Users/vtariel/Desktop/ANV/Population/database/";

    Vec<VecF32> vtraining_in ;
    // vecteur de sortie attendu
    Vec<VecF32> vtraining_out ;
    //indique que je suis dans la partie test et non apprentissage
    bool test = false;
    fill_vtin_vtout(vtraining_in,vtraining_out,root_dir,net,test,nbr_dir);

    //    return 0;


    F32 eta=0.01;
    net.setLearnableParameter(eta);
    net.setTrainable(true);





    std::cout << "ok"<<std::endl;


    std::vector<int> v_global_rand((vtraining_in).size());
    for(unsigned int i=0;i<v_global_rand.size();i++)
        v_global_rand[i]=i;
    std::cout<<"iter_epoch\t error_train\t error_test\t learning rate"<<std::endl;
    int nbr_epoch=10;
    for(unsigned int i=0;i<nbr_epoch;i++){
        std::random_shuffle ( v_global_rand.begin(), v_global_rand.end() ,Distribution::irand());
        int error_training=0;
        for(unsigned int j=0;j<v_global_rand.size();j++){
            //            auto start_global =  std::chrono::high_resolution_clock::now();
            VecF32 vout;
            net.forwardCPU((vtraining_in)(v_global_rand[j]),vout);
            net.backwardCPU((vtraining_out)(v_global_rand[j]));
            net.learn();
            //            auto end_global= std::chrono::high_resolution_clock::now();
            //            std::cout<<std::chrono::duration<double, std::milli>(end_global-start_global).count()<<std::endl;

            int label1 = std::distance(vout.begin(),std::max_element(vout.begin(),vout.end()));
            int label2 = std::distance((vtraining_out)(v_global_rand[j]).begin(),std::max_element((vtraining_out)(v_global_rand[j]).begin(),(vtraining_out)(v_global_rand[j]).end()));
            if(label1!=label2)
                error_training++;

            //vérification des valeurs de vout pour voir si ça converge bien
            //                                                    std::cout<<"vout("<<j<<")0= "<<vout(0) << "and expected "<< (*v_training_out)(v_global_rand[j])(0)<<std::endl;
            //                                                    std::cout<<"vout("<<j<<")1= "<<vout(1) << "and expected "<< (*v_training_out)(v_global_rand[j])(1)<<std::endl;

            //std::cout<<i<<"\t"<<error_training*1./vtraining_in.size()<<"\t"<<eta<<std::endl;
        }


        std::cout<<i<<"\t"<<error_training*1./(vtraining_in).size()<<"\t"<<eta<<std::endl;
        eta *=0.9f;
        eta = std::max(eta,0.001f);
        net.setLearnableParameter(eta);
    }

    {
        //        NeuralNet net;
        //        net.load("net.xml");
        algoSearch(mm,net,size,6,fluctuation);
    }
    net.save("net.xml");
    return 0;
    //    //---------------------------------------------------TEST--------------------------------------------------------------

    test = true;
    Vec<VecF32> vtest_in ;
    // vecteur de sortie attendu
    Vec<VecF32> v_test_out ;

    fill_vtin_vtout(vtest_in,v_test_out,root_dir,net,test,nbr_dir);

    for(unsigned int j=0;j<vtest_in.size();j++){
        VecF32 vouttest;
        net.forwardCPU((vtest_in)(j),vouttest);

        int label1 = std::distance(vouttest.begin(),std::max_element(vouttest.begin(),vouttest.end()));
        int label2 = std::distance(v_test_out(j).begin(),std::max_element(v_test_out(j).begin(),v_test_out(j).end()));
        std::cout<<vouttest<<std::endl;
        std::cout<<v_test_out(j)<<std::endl;
        //        std::cout<<"test "<<label1<<" "<<label2<<std::endl;
    }

    return 0;

}
