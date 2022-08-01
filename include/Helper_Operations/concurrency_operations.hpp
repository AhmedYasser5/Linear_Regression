#pragma once

#include "defaults.hpp"
#include <thread>

namespace MachineLearning {

using std::thread;

class ConcurrentLoops {
private:
  const size_t maxThreads;

  template <typename Function, typename... Args>
  static void runFunctionWithoutReturns(size_t start, size_t finish,
                                        const Function &func,
                                        Args... parameters);

public:
  size_t start, finish, maxIterationsPerThread;

  ConcurrentLoops(const size_t &start, const size_t &finish,
                  const size_t &maxIterationsPerThread);

  template <typename Function, typename... Args>
  void initiateLoopsWithoutReturns(Function func, Args... parameters) const;
};

template <typename Function, typename... Args>
void ConcurrentLoops::runFunctionWithoutReturns(size_t start, size_t finish,
                                                const Function &func,
                                                Args... parameters) {
  while (start < finish) {
    func(start, parameters...);
    start++;
  }
}

// Run a function using multi-threading (finish - start) times
// Similar to running a loop that calls the function multiple times, but with
// threading
// The callable function should:
//		1. Expect that the first parameter is the current loop index
//		2. Be a function returning void

template <typename Function, typename... Args>
void ConcurrentLoops::initiateLoopsWithoutReturns(Function func,
                                                  Args... parameters) const {
  size_t iterations = finish - start;
  size_t neededThreads =
      (iterations + maxIterationsPerThread - 1) / maxIterationsPerThread;
  neededThreads = std::min(neededThreads, maxThreads);
  size_t portionPerThread = iterations / neededThreads;
  size_t remainder = iterations % neededThreads;
  neededThreads--;
  vector<thread> allThreads;
  allThreads.reserve(neededThreads);
  size_t currentStart = start;
  for (size_t i = 0; i < neededThreads; i++) {
    size_t currentFinish = currentStart + portionPerThread;
    if (remainder > 0) {
      currentFinish++;
      remainder--;
    }
    allThreads.emplace_back(runFunctionWithoutReturns<Function, Args...>,
                            currentStart, currentFinish, func, parameters...);
    currentStart = currentFinish;
  }
  runFunctionWithoutReturns(currentStart, finish, func, parameters...);
  for (size_t i = 0; i < neededThreads; i++)
    allThreads[i].join();
}

} // namespace MachineLearning
