#include "Helper_Operations/concurrency_operations.hpp"

using namespace MachineLearning;

ConcurrentLoops::ConcurrentLoops(const size_t &start, const size_t &finish,
                                 const size_t &maxIterationsPerThread)
    : start(start), finish(finish),
      maxIterationsPerThread(maxIterationsPerThread),
      maxThreads(thread::hardware_concurrency()) {}
