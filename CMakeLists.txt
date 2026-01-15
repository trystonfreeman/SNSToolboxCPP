cmake_minimum_required(VERSION 4.1)
project(SNSToolboxCPP)

set(CMAKE_CXX_STANDARD 20)
find_package(Eigen3 REQUIRED NO_MODULE)
set(EXTRA_COMPONENT_DIRS include)
add_library(SNSToolboxCPP STATIC
        Network.cpp
        include/cjsontoEigen.cpp
        )
target_link_libraries(SNSToolboxCPP Eigen3::Eigen)
