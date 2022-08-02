#pragma once

#include "defaults.hpp"
#include <thread>

namespace MachineLearning {

using std::thread;

class ConcurrentLoops {
public:
  size_t start, finish, maxIterationsPerThread;

  ConcurrentLoops(const size_t &start, const size_t &finish,
                  const size_t &maxIterationsPerThread);

private:
  const size_t maxThreads;

  template <typename Function, typename... Args>
  static void runFunctionWithoutReturns(size_t start, size_t finish,
                                        Function func, Args... parameters);

public:
  // Run a function using multi-threading (finish - start) times
  // Similar to running a loop that calls the function multiple times, starting
  // from start and up to (but not including) finish (such that start <= finish)
  // The callable function should:
  //		1. Expect that the first parameter is the current loop index
  //		2. Be a function returning void

  template <typename Function, typename... Args>
  void initiateLoopsWithoutReturns(Function func, Args... parameters) const;

  // TODO: Those functions reduce the performance of the code
  //
  // private:
  //  template <typename T, typename MergerFunction, typename Function,
  //            typename... Args>
  //  static void runFunctionWithReturns(size_t start, size_t finish,
  //                                     T &mergedResult,
  //                                     MergerFunction returnsMerger,
  //                                     Function func, Args... parameters);
  //
  // public:
  //  template <typename T, typename MergerFunction, typename Function,
  //            typename... Args>
  //  T initiateLoopsWithReturns(MergerFunction returnsMerger, Function func,
  //                             Args... parameters) const;
};

/////////////////////////////////////////////////////////////////////////////
//////////////////////////////IMPLEMENTATION/////////////////////////////////
/////////////////////////////////////////////////////////////////////////////

template <typename Function, typename... Args>
void ConcurrentLoops::runFunctionWithoutReturns(size_t start, size_t finish,
                                                Function func,
                                                Args... parameters) {
  while (start < finish) {
    func(start, parameters...);
    start++;
  }
}

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

// TODO
//
// template <typename T, typename MergerFunction, typename Function,
//          typename... Args>
// void ConcurrentLoops::runFunctionWithReturns(size_t start, size_t finish,
//                                             T &mergedResult,
//                                             MergerFunction returnsMerger,
//                                             Function func,
//                                             Args... parameters) {
//  while (start < finish) {
//    mergedResult = returnsMerger(mergedResult, func(start, parameters...));
//    start++;
//  }
//}
//
// template <typename T, typename MergerFunction, typename Function,
//          typename... Args>
// T ConcurrentLoops::initiateLoopsWithReturns(MergerFunction returnsMerger,
//                                            Function func,
//                                            Args... parameters) const {
//  size_t iterations = finish - start;
//  size_t neededThreads =
//      (iterations + maxIterationsPerThread - 1) / maxIterationsPerThread;
//  neededThreads = std::min(neededThreads, maxThreads);
//  size_t portionPerThread = iterations / neededThreads;
//  size_t remainder = iterations % neededThreads;
//  neededThreads--;
//  vector<thread> allThreads;
//  allThreads.reserve(neededThreads);
//  vector<T> results(neededThreads);
//  size_t currentStart = start;
//  for (size_t i = 0; i < neededThreads; i++) {
//    size_t currentFinish = currentStart + portionPerThread;
//    if (remainder > 0) {
//      currentFinish++;
//      remainder--;
//    }
//    allThreads.emplace_back(
//        runFunctionWithReturns<T, MergerFunction, Function, Args...>,
//        currentStart, currentFinish, std::ref(results[i]), returnsMerger,
//        func, parameters...);
//    currentStart = currentFinish;
//  }
//  T finalResult{};
//  runFunctionWithReturns(currentStart, finish, finalResult, returnsMerger,
//  func,
//                         parameters...);
//  for (size_t i = 0; i < neededThreads; i++) {
//    allThreads[i].join();
//    finalResult = returnsMerger(finalResult, results[i]);
//  }
//  return finalResult;
//}

} // namespace MachineLearning
