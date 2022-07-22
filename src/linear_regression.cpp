#include "linear_regression.hpp"
#include "helper_ops.hpp"
#include <cmath>

using namespace ML;

LinearRegression::LinearRegression(const double &alpha) : alpha(alpha) {}

static void initialize(const size_t &i, vector<vector<data>> &X,
                       const vector<vector<data>> &x, const size_t &features) {
  for (size_t j = 0; j < features; j++)
    X[j][i] = x[i][j];
}

static void getStats(const size_t &i, vector<data> &avg,
                     vector<vector<data>> &X, vector<data> &std_dev,
                     const size_t &data_size) {
  avg[i] = add(X[i]) / data_size;
  add(X[i], -avg[i]);
  std_dev[i] = sqrt(dotProduct(X[i], X[i]) / data_size);
  mul(X[i], 1 / std_dev[i]);
}

vector<vector<data>>
LinearRegression::transform(const vector<vector<data>> &x) {
  size_t features = x[0].size(), data_size = x.size();
  vector<vector<data>> X(features, vector<data>(data_size));

  runLoopConcurrently(0, data_size, initialize, std::ref(X), std::cref(x),
                      std::cref(features));

  avg.resize(features);
  std_dev.resize(features);

  runLoopConcurrently(0, features, getStats, std::ref(avg), std::ref(X),
                      std::ref(std_dev), std::cref(data_size));

  return X;
}

vector<data> LinearRegression::normalize(const vector<data> &x) const {
  vector<data> res(x);
  size_t features = avg.size();
  for (size_t i = 0; i < features; i++)
    res[i] = (res[i] - avg[i]) / std_dev[i];
  return res;
}

bool LinearRegression::processData(const vector<vector<data>> &dotX,
                                   const vector<data> &sumX, const data &sumY,
                                   const vector<data> &sumXY) {
  size_t features = sumX.size();
  weights.resize(features);
  std::fill(weights.begin(), weights.end(), 0);
  base = 0;

  vector<data> newWeights(features);
  bool converged = false;
  while (!converged) {
    converged = true;

    for (size_t i = 0; i < features; i++) {
      newWeights[i] = weights[i] - (dotProduct(weights, dotX[i]) +
                                    base * sumX[i] - sumXY[i]);
      if (fabs(weights[i] - newWeights[i]) > EPSILON)
        converged = false;
    }

    data newBase = base - (dotProduct(weights, sumX) + alpha * base - sumY);

    if (fabs(base - newBase) > EPSILON)
      converged = false;

    weights = newWeights;
    base = newBase;
  }
  return true;
}

static void precalculate(const size_t &i, const size_t &features,
                         const vector<vector<data>> &X, const vector<data> &y,
                         vector<data> &sumX, vector<data> &sumXY,
                         vector<vector<data>> &dotX, const data &multiplicand) {
  sumX[i] = add(X[i]);
  sumXY[i] = dotProduct(X[i], y);

  for (size_t j = i; j < features; j++)
    dotX[i][j] = dotProduct(X[i], X[j]);

  mul(dotX[i], multiplicand);
}

bool LinearRegression::train(const vector<vector<data>> &x,
                             const vector<data> &y) {
  auto X = transform(x);
  size_t features = X.size(), data_size = y.size();

  vector<vector<data>> dotX(features, vector<data>(features));
  vector<data> sumX(features), sumXY(features);

  runLoopConcurrently(0, features, precalculate, std::cref(features),
                      std::cref(X), std::cref(y), std::ref(sumX),
                      std::ref(sumXY), std::ref(dotX), alpha / data_size);
  for (size_t i = 0; i < features; i++)
    for (size_t j = 0; j < i; j++)
      dotX[i][j] = dotX[j][i];

  mul(sumX, alpha / data_size);
  mul(sumXY, alpha / data_size);
  data sumY = add(y) * alpha / data_size;

  return processData(dotX, sumX, sumY, sumXY);
}

data LinearRegression::predict(const vector<data> &x) const {
  return dotProduct(weights, normalize(x)) + base;
}
