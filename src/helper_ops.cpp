#include "helper_ops.hpp"
#include <cassert>

using namespace ML;

void ML::mul(vector<data> &a, data value) {
  size_t n = a.size();
  for (size_t i = 0; i < n; i++)
    a[i] *= value;
}

data ML::dotProduct(const vector<data> &a, const vector<data> &b,
                    const data scale) {
  size_t n = a.size();
  assert(n == b.size());
  data result = 0;
  for (size_t i = 0; i < n; i++)
    result += scale * a[i] * b[i];
  return result;
}

data ML::sum(const vector<data> &a, const data scale) {
  size_t n = a.size();
  data result = 0;
  for (size_t i = 0; i < n; i++)
    result += a[i] * scale;
  return result;
}

void ML::add(vector<data> &a, data value) {
  size_t n = a.size();
  for (size_t i = 0; i < n; i++)
    a[i] += value;
}
