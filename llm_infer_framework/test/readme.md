cd /mnt/e/work-project/infra/llm_infer_framework

cmake -S . -B build
cmake --build build -j
ctest --test-dir build --output-on-failure

直接跑这个测试程序
./build/test_buffer

只跑这一个 case
./build/test_buffer --gtest_filter=test_buffer.allocate