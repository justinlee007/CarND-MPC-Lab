#ifndef MPC_H
#define MPC_H

#include <vector>
#include "Eigen-3.3/Eigen/Core"

class MPC {
 public:
  MPC();

  virtual ~MPC();

  // Solve the model given an initial state.
  // Return the next state and actuations as a vector.
  std::vector<double> solve(Eigen::VectorXd x0, Eigen::VectorXd coeffs);

  double polyeval(Eigen::VectorXd coeffs, double x);

  Eigen::VectorXd polyfit(Eigen::VectorXd xvals, Eigen::VectorXd yvals, int order);

  Eigen::VectorXd globalKinematic(Eigen::VectorXd state, Eigen::VectorXd actuators, double dt);
};

#endif /* MPC_H */
