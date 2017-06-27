// In this quiz you'll implement the global kinematic model.
#include <math.h>
#include <iostream>
#include "Eigen/Dense"
#include "MPC.h"
#include "matplotlibcpp.h"

namespace plt = matplotlibcpp;

//
// Helper functions
//
double pi() { return M_PI; }
double deg2rad(double x) { return x * pi() / 180; }
double rad2deg(double x) { return x * 180 / pi(); }

const double Lf = 2;

void testGlobalKinematic(MPC mpc) {
  // [x, y, psi, v]
  Eigen::VectorXd state(4);
  // [delta, v]
  Eigen::VectorXd actuators(2);

  state << 0, 0, deg2rad(45), 1;
  actuators << deg2rad(5), 1;

  Eigen::VectorXd next_state = mpc.globalKinematic(state, actuators, 0.3);

  std::cout << next_state << std::endl;
}

void testPolyFit(MPC mpc) {
  Eigen::VectorXd xvals(6);
  Eigen::VectorXd yvals(6);
  // x waypoint coordinates
  xvals << 9.261977, -2.06803, -19.6663, -36.868, -51.6263, -66.3482;
  // y waypoint coordinates
  yvals << 5.17, -2.25, -15.306, -29.46, -42.85, -57.6116;

  Eigen::VectorXd coeffs = mpc.polyfit(xvals, yvals, 3);

  for (double x = 0; x <= 20; x += 1.0) {
    std::cout << mpc.polyeval(coeffs, x) << std::endl;
  }
}

int main() {
  MPC mpc = MPC();

  testGlobalKinematic(mpc);
  testPolyFit(mpc);

  int iters = 50;

  Eigen::VectorXd ptsx(2);
  Eigen::VectorXd ptsy(2);
  ptsx << -100, 100;
  ptsy << -1, -1;

  // The polynomial is fitted to a straight line so a polynomial with
  // order 1 is sufficient.
  auto coeffs = mpc.polyfit(ptsx, ptsy, 1);

  // NOTE: free feel to play around with these
  double x = -1;
  double y = 10;
  double psi = 0;
  double v = 10;
  // The cross track error is calculated by evaluating at polynomial at x, f(x)
  // and subtracting y.
  double cte = mpc.polyeval(coeffs, x) - y;
  // Due to the sign starting at 0, the orientation error is -f'(x).
  // derivative of coeffs[0] + coeffs[1] * x -> coeffs[1]
  double epsi = psi - atan(coeffs[1]);

  Eigen::VectorXd state(6);
  state << x, y, psi, v, cte, epsi;

  std::vector<double> x_vals = {state[0]};
  std::vector<double> y_vals = {state[1]};
  std::vector<double> psi_vals = {state[2]};
  std::vector<double> v_vals = {state[3]};
  std::vector<double> cte_vals = {state[4]};
  std::vector<double> epsi_vals = {state[5]};
  std::vector<double> delta_vals = {};
  std::vector<double> a_vals = {};

  for (size_t i = 0; i < iters; i++) {
    std::cout << "Iteration " << i << std::endl;

    auto vars = mpc.solve(state, coeffs);

    x_vals.push_back(vars[0]);
    y_vals.push_back(vars[1]);
    psi_vals.push_back(vars[2]);
    v_vals.push_back(vars[3]);
    cte_vals.push_back(vars[4]);
    epsi_vals.push_back(vars[5]);

    delta_vals.push_back(vars[6]);
    a_vals.push_back(vars[7]);

    state << vars[0], vars[1], vars[2], vars[3], vars[4], vars[5];
    std::cout << "x = " << vars[0] << std::endl;
    std::cout << "y = " << vars[1] << std::endl;
    std::cout << "psi = " << vars[2] << std::endl;
    std::cout << "v = " << vars[3] << std::endl;
    std::cout << "cte = " << vars[4] << std::endl;
    std::cout << "epsi = " << vars[5] << std::endl;
    std::cout << "delta = " << vars[6] << std::endl;
    std::cout << "a = " << vars[7] << std::endl;
    std::cout << std::endl;
  }

  // Plot values
  // NOTE: feel free to play around with this.
  // It's useful for debugging!
  printf("Creating plot\n");
  plt::subplot(3, 1, 1);
  plt::title("CTE");
  plt::plot(cte_vals);
  plt::subplot(3, 1, 2);
  plt::title("Delta (Radians)");
  plt::plot(delta_vals);
  plt::subplot(3, 1, 3);
  plt::title("Velocity");
  plt::plot(v_vals);

  printf("Saving plot.png\n");
  plt::save("plot");
}
