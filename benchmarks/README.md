# Benchmarks

Benchmark harness will be added in the benchmark phase.

## Running benchmarks

```bash
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
./benchmark
```
