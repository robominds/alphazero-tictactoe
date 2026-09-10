cmake_minimum_required(VERSION 3.16)
project(alphazero_tictactoe CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

add_library(az
    src/board.cpp
)
target_include_directories(az PUBLIC include)

enable_testing()

add_executable(test_board tests/test_board.cpp)
target_link_libraries(test_board az)
add_test(NAME test_board COMMAND test_board)
