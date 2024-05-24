file(GLOB RVTV_SRC "*.cpp")
add_executable(rvtv ${RVTV_SRC})

llvm_map_components_to_libnames(LLVM_LIBS support core analysis passes RISCV)
target_link_libraries(rvtv PRIVATE ${LLVM_LIBS})
