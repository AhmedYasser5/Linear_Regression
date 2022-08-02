#pragma once

#include "Helper_Operations/defaults.hpp"
#include <cstddef>

namespace MachineLearning {

// TODO: make an interface for the DataType
// TODO: separate the z-score normalizer from the linear regression model

class LinearRegression {
private:
  const DataType EPSILON = 1e-6;
  vector<DataType> weights, avg, std_dev;
  DataType base;
  DataType alpha;

  vector<vector<DataType>> normalize(const vector<vector<DataType>> &x);
  vector<DataType> normalize(const vector<DataType> &x) const;
  bool processData(const vector<vector<DataType>> &dotX,
                   const vector<DataType> &sumX, const DataType &sumY,
                   const vector<DataType> &sumXY);

public:
  LinearRegression(const DataType &alpha = 0.1);
  ~LinearRegression() = default;
  // TODO: add optional iterations
  bool train(const vector<vector<DataType>> &x, const vector<DataType> &y);
  DataType predict(const vector<DataType> &x) const;
};

} // namespace MachineLearning
