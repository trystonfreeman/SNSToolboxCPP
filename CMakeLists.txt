
cmake_minimum_required(VERSION 3.22.1)


project(SNSToolboxCPP)


set(CMAKE_CXX_STANDARD 20)
#find_package(Eigen3 REQUIRED NO_MODULE)
#set_property(TARGET Eigen3::Eigen PROPERTY IMPORTED_NO_SONAME TRUE)
set(EXTRA_COMPONENT_DIRS include)
add_library(SNSToolboxCPP STATIC
        src/Network.cpp
        src/cjsontoEigen.cpp
        )
#target_link_libraries(SNSToolboxCPP Eigen3::Eigen)
