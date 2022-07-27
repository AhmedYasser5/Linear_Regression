#pragma once

#include <cstddef>
#include <vector>

namespace ML {

using std::size_t;
using std::vector;

typedef long double data;

class LinearRegression {
private:
  const data EPSILON = 1e-6;
  vector<data> weights, avg, std_dev;
  data base;
  data alpha;

  vector<vector<data>> transform(const vector<vector<data>> &x);
  vector<data> normalize(const vector<data> &x) const;
  bool processData(const vector<vector<data>> &dotX, const vector<data> &sumX,
                   const data &sumY, const vector<data> &sumXY);

public:
  LinearRegression(const data &alpha = 0.1);
  ~LinearRegression() = default;
  bool train(const vector<vector<data>> &x, const vector<data> &y);
  data predict(const vector<data> &x) const;
};

} // namespace ML
