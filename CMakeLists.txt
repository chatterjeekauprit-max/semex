cmake_minimum_required(VERSION 3.18)
project(sem_drift_registration LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)

find_package(Python 3.8 COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 REQUIRED)
find_package(FFTW3 REQUIRED)
find_package(CUDAToolkit REQUIRED)

pybind11_add_module(sem_drift_core sem_drift_core.cpp)
target_link_libraries(sem_drift_core PRIVATE FFTW3::fftw3)

pybind11_add_module(sem_drift_cuda sem_drift_cuda.cu)
target_link_libraries(sem_drift_cuda PRIVATE CUDA::cufft CUDA::cudart)
set_target_properties(sem_drift_cuda PROPERTIES 
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "native"
)
