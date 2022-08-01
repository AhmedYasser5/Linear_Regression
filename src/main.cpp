#include "Linear_Regression/linear_regression.hpp"
#include <chrono>
#include <fstream>
#include <iostream>
using namespace std;
int main() {

  MachineLearning::LinearRegression module(0.1);

  ifstream in("data.csv");
  int features = 6;
  vector<vector<MachineLearning::DataType>> x;
  vector<MachineLearning::DataType> y;
  while (true) {
    double tmp;
    in >> tmp;
    if (!in)
      break;
    y.emplace_back(tmp);
    x.emplace_back(features);
    for (auto &it : x.back())
      in >> it;
  }

  auto cur = chrono::steady_clock::now();
  bool state = module.train(x, y);
  if (!state) {
    cout << "Didn't converge..." << endl;
    return 0;
  }
  chrono::duration<MachineLearning::DataType> dur =
      chrono::steady_clock::now() - cur;
  cout << "Took " << dur.count() * 1e3 << " ms to train" << endl;
  vector<MachineLearning::DataType> test({1500, 3, 2, 4, 0, 0});
  cout << module.predict(test) << endl;

  return 0;
}
