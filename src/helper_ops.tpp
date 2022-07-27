#include "helper_ops.hpp"

#define number_of_threads() thread::hardware_concurrency()

template <typename Function, typename... Args>
static void runFunction(size_t start, size_t finish, const Function &func,
                        Args... parameters) {
  while (start < finish) {
    func(start, parameters...);
    start++;
  }
}

template <typename Function, typename... Args>
void ML::runLoopConcurrently(size_t start, size_t finish, Function func,
                             Args... parameters) {
  size_t num_threads = number_of_threads();
  size_t portion = (finish - start) / num_threads;
  size_t rem = (finish - start) % num_threads;
  num_threads--;
  vector<thread> vec_t;
  vec_t.reserve(num_threads);
  size_t curStart = start;
  for (size_t i = 0, curFinish; i < num_threads; i++) {
    curFinish = curStart + portion;
    if (rem) {
      curFinish++;
      rem--;
    }
    vec_t.emplace_back(runFunction<Function, Args...>, curStart, curFinish,
                       func, parameters...);
    curStart = curFinish;
  }
  runFunction(curStart, finish, func, parameters...);
  for (size_t i = 0; i < num_threads; i++)
    vec_t[i].join();
}
