#include "vector_ops.hpp"
#include <cassert>
#include <thread>

#define number_of_threads() thread::hardware_concurrency()

using namespace ML;
using std::size_t;
using std::thread;

template <typename Function, typename... Args>
static void runConcurrentlyWithoutReturns(Function &&func, const size_t &size,
                                          Args &&...parameters) {
  size_t num_threads = number_of_threads();
  size_t portion = size / num_threads;
  num_threads--;
  vector<thread> vec_t;
  vec_t.reserve(num_threads);
  for (size_t i = 0; i < num_threads; i++)
    vec_t.emplace_back(func, parameters..., i * portion, (i + 1) * portion);
  func(parameters..., num_threads * portion, size);
  for (size_t i = 0; i < num_threads; i++)
    vec_t[i].join();
}

template <typename Function, typename... Args>
static vector<data> runConcurrentlyWithReturns(Function &&func,
                                               const size_t &size,
                                               Args &&...parameters) {
  size_t num_threads = number_of_threads();
  size_t portion = size / num_threads;
  vector<data> results(num_threads);
  num_threads--;
  vector<thread> vec_t;
  vec_t.reserve(num_threads);
  for (size_t i = 0; i < num_threads; i++)
    vec_t.emplace_back(func, parameters..., std::ref(results[i]), i * portion,
                       (i + 1) * portion);
  func(parameters..., std::ref(results[num_threads]), num_threads * portion,
       size);
  for (size_t i = 0; i < num_threads; i++)
    vec_t[i].join();
  return results;
}

static void mulThread(vector<data> &a, data value, size_t start,
                      size_t finish) {
  while (start < finish) {
    a[start] *= value;
    start++;
  }
}

void ML::mul(vector<data> &a, data value) {
  runConcurrentlyWithoutReturns(mulThread, a.size(), std::ref(a), value);
}

static void dotThread(const vector<data> &a, const vector<data> &b,
                      data &result, size_t start, size_t finish) {
  while (start < finish) {
    result += a[start] * b[start];
    start++;
  }
}

data ML::dotProduct(const vector<data> &a, const vector<data> &b) {
  size_t m = a.size();
  assert(m == b.size());
  auto results =
      runConcurrentlyWithReturns(dotThread, m, std::ref(a), std::ref(b));
  data result = 0;
  for (auto &res : results)
    result += res;
  return result;
}

static void addThread(const vector<data> &a, data &result, size_t start,
                      size_t finish) {
  while (start < finish) {
    result += a[start];
    start++;
  }
}

data ML::add(const vector<data> &a) {
  auto results = runConcurrentlyWithReturns(addThread, a.size(), std::ref(a));
  data result = 0;
  for (auto &res : results)
    result += res;
  return result;
}

static void addVectorThread(vector<data> &a, data value, size_t start,
                            size_t finish) {
  while (start < finish) {
    a[start] += value;
    start++;
  }
}

void ML::add(vector<data> &a, data value) {
  runConcurrentlyWithoutReturns(addVectorThread, a.size(), std::ref(a), value);
}
