cmake_minimum_required(VERSION 3.20)
project(histmap LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(pylene REQUIRED CONFIG)

add_executable(histmapseg src/main.cpp)
target_link_libraries(histmapseg PRIVATE pylene::pylene)