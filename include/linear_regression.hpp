#pragma once

#include <cstddef>
#include <vector>

namespace ML {

using std::size_t;
using std::vector;

typedef double data;

data dotProduct(const vector<data> &a, const vector<data> &b);
void mul(vector<data> &a, data value);
data add(const vector<data> &a);
void add(vector<data> &a, data value);

class LinearRegression {
private:
  const double EPSILON = 1e-3;
  vector<data> weights, avg, std_dev;
  data base;
  double alpha;

  vector<vector<data>> transform(const vector<vector<data>> &x);
  vector<data> normalize(const vector<data> &x) const;
  bool processData(const vector<vector<data>> &dotX, const vector<data> &sumX,
                   const data &sumY, const vector<data> &sumXY);

public:
  LinearRegression(const double &alpha = 0.1);
  ~LinearRegression() = default;
  bool train(const vector<vector<data>> &x, const vector<data> &y);
  data predict(const vector<data> &x) const;
};

} // namespace ML
