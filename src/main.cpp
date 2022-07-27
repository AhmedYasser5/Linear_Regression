#include "linear_regression.hpp"
#include <chrono>
#include <fstream>
#include <iostream>
using namespace std;
int main() {

  ML::LinearRegression mod(0.1);

  ifstream in("data.csv");
  int features = 6;
  // in >> features;
  vector<vector<ML::data>> x;
  vector<ML::data> y;
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
  bool state = mod.train(x, y);
  if (!state) {
    cout << "Didn't converge..." << endl;
    return 0;
  }
  chrono::duration<ML::data> dur = chrono::steady_clock::now() - cur;
  cout << "Took " << dur.count() * 1e3 << " ms to train" << endl;
  vector<ML::data> test({1500, 3, 2, 4, 0, 0});
  cout << mod.predict(test) << endl;

  return 0;
}
