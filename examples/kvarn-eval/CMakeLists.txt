set(TARGET kvarn-eval)

add_executable(${TARGET} kvarn-eval.cpp)

target_link_libraries(${TARGET} PRIVATE llama)

install(TARGETS ${TARGET} RUNTIME)
