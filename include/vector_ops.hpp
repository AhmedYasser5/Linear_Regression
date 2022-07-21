#pragma once

#include <vector>

namespace ML {

using std::vector;

typedef double data;

data dotProduct(const vector<data> &a, const vector<data> &b);

void mul(vector<data> &a, data value);

data add(const vector<data> &a);

void add(vector<data> &a, data value);

} // namespace ML
