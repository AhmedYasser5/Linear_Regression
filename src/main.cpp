#include "linear_regression.hpp"
#include <fstream>
#include <iostream>
#include <thread>

using namespace std;

int main() {

  ML::LinearRegression mod(0.001);

  ifstream in("data.csv");
  int features = 6;
  // in >> features;
  vector<vector<double>> x;
  vector<double> y;
  while (true) {
    double tmp;
    in >> tmp;
    if (!in)
      break;
    y.emplace_back(tmp);
    x.emplace_back(features);
    for (auto &cur : x.back())
      in >> cur;
  }

  // for (auto &it : x) {
  //   for (auto &x : it)
  //     cout << x << ' ';
  //   cout << endl;
  // }

  cout << boolalpha << mod.train(x, y) << endl;
  vector<double> test({1790, 2, 2, 2, 0, 0});
  cout << mod.predict(test) << endl;

  return 0;
}
