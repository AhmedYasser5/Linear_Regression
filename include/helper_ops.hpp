#pragma once

#include <thread>
#include <vector>

namespace ML {

using std::size_t;
using std::thread;
using std::vector;

typedef double data;

// Run a function using multi-threading (finish - start) times
// Similar to running a loop that calls the function multiple times, but with
// threading
// The callable function should:
//		1. Expect that the first parameter is the current loop index
//		2. Be a function returning void
template <typename Function, typename... Args>
void runLoopConcurrently(size_t start, size_t finish, Function func,
                         Args... parameters);

data dotProduct(const vector<data> &a, const vector<data> &b);
void mul(vector<data> &a, data value);

data add(const vector<data> &a);
void add(vector<data> &a, data value);

} // namespace ML

#include "helper_ops.tpp"
