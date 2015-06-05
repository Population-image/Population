#ifndef NEURAL_NET_TEST_H
#define NEURAL_NET_TEST_H

#include "popconfig.h"

void test_neural_net_cpu(const int nb_epoch);
void test_neural_net_cpu_mnist(const int nb_epoch);
void test_neural_net_conv_cpu(const int nb_epoch);
void test_neural_net_conv_cpu_mnist(const int nb_epoch);

#if defined(HAVE_CUDA)
void test_neural_net_gpu(const int nb_epoch);
void test_neural_net_gpu_mnist(const int nb_epoch);
void test_neural_net_gpu_augmented_database(const int max_files_per_folder, const int network_for_training, std::string database_training, std::string database_test, const int nb_epoch);
void bench_propagate_front_gpu_augmented_database(const int max_files_per_folder, std::string network_path, std::string database_training, std::string database_test, const int nb_epoch);
void test_neural_net_conv_gpu(const int nb_epoch);
void test_neural_net_conv_gpu_mnist(const int nb_epoch);

void test_neural_net();
#endif

#endif
