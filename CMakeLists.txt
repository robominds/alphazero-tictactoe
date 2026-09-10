cmake_minimum_required(VERSION 3.16)
project(alphazero_tictactoe CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

add_library(az
    src/board.cpp
    src/minimax.cpp
    src/network.cpp
    src/mcts.cpp
    src/replay_buffer.cpp
    src/selfplay.cpp
    src/eval.cpp
)
target_include_directories(az PUBLIC include)

enable_testing()

add_executable(test_board tests/test_board.cpp)
target_link_libraries(test_board az)
add_test(NAME test_board COMMAND test_board)

add_executable(test_minimax tests/test_minimax.cpp)
target_link_libraries(test_minimax az)
add_test(NAME test_minimax COMMAND test_minimax)

add_executable(test_network tests/test_network.cpp)
target_link_libraries(test_network az)
add_test(NAME test_network COMMAND test_network)

add_executable(test_mcts tests/test_mcts.cpp)
target_link_libraries(test_mcts az)
add_test(NAME test_mcts COMMAND test_mcts)

add_executable(test_replay_buffer tests/test_replay_buffer.cpp)
target_link_libraries(test_replay_buffer az)
add_test(NAME test_replay_buffer COMMAND test_replay_buffer)

add_executable(test_selfplay tests/test_selfplay.cpp)
target_link_libraries(test_selfplay az)
add_test(NAME test_selfplay COMMAND test_selfplay)

add_executable(test_eval tests/test_eval.cpp)
target_link_libraries(test_eval az)
add_test(NAME test_eval COMMAND test_eval)

add_executable(evaluate apps/evaluate.cpp)
target_link_libraries(evaluate az)

add_executable(train apps/train.cpp)
target_link_libraries(train az)
