#include <iostream>
#include <Eigen/Dense>
#include <fmt/core.h>

#define ITERATIONS 100
#define TOLERANCE 1E-06

using namespace Eigen;
using namespace std;

class SuccessiveOverRelaxationMethod {
public:
    virtual VectorXd solve(const MatrixXd& A, const VectorXd& b, const double w);
};

VectorXd SuccessiveOverRelaxationMethod::solve(const MatrixXd& A, const VectorXd& b, const double w) {
    int n = A.rows();
    VectorXd x = VectorXd::Zero(n);
    VectorXd x_old;
    double error = 1.0;

    for (int iter = 0; iter < ITERATIONS; ++iter) {
        x_old = x;
        for (int i = 0; i < n; ++i) {
            double sum = 0.0;
            for (int j = 0; j < n; ++j) {
                if (j != i) {
                    sum += A(i, j) * x(j);
                }
            }
            x(i) = (1.0 - w) * x(i) + (w / A(i, i)) * (b(i) - sum);
        }
        error = (x - x_old).norm();
        if (error < TOLERANCE) {
            fmt::print("Converged at iteration {}\n", iter + 1);
            break;
        }
    }

    return x;
}

class GaussSeidelMethod : public SuccessiveOverRelaxationMethod {
public:
    VectorXd solve(const MatrixXd& A, const VectorXd& b, const double w) override;
};

VectorXd GaussSeidelMethod::solve(const MatrixXd& A, const VectorXd& b, const double w) {
    return SuccessiveOverRelaxationMethod::solve(A, b, 1.0);
}

int main() {
    MatrixXd A(4, 4);
    VectorXd b(4);

    A << 10, -1, 2, 0,
         -1, 11, -1, 3,
         2, -1, 10, -1,
         0, 3, -1, 8;
    b << 6, 25, -11, 15;

    SuccessiveOverRelaxationMethod sor_solver;
    VectorXd x = sor_solver.solve(A, b, 1.25);
    fmt::print("x = [{}, {}, {}, {}]\n", x[0], x[1], x[2], x[3]);

    VectorXd y = sor_solver.solve(A, b, 0.25);
    fmt::print("x = [{}, {}, {}, {}]\n", y[0], y[1], y[2], y[3]);

    GaussSeidelMethod gauss_seidel_solver;
    VectorXd z = gauss_seidel_solver.solve(A, b, 1.0);
    fmt::print("x = [{}, {}, {}, {}]\n", z[0], z[1], z[2], z[3]);

    return 0;
}
