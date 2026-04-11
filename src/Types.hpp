/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   Types.hpp                                          :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: RuhanGi <mohammedruhan.goltay@kaust.edu    +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2026/04/08 14:15:50 by RuhanGi           #+#    #+#             */
/*   Updated: 2026/04/08 14:15:50 by RuhanGi          ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#pragma once

#include <algorithm>
#include <execution>
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <numeric>
#include <random>
#include <vector>
#include <string>
#include <cmath>
#include <set>
#include <map>

#define GREY	"\x1B[30m"
#define RED		"\x1B[31m"
#define GREEN	"\x1B[32m"
#define YELLOW	"\x1B[33m"
#define BLUE	"\x1B[34m"
#define PURPLE	"\x1B[35m"
#define CYAN	"\x1B[36m"
#define RESET	"\x1B[0m"

using Row = std::vector<double>;
using Matrix = std::vector<Row>;

const double LR = 0.01;
const int MAX_EPOCHS = 1500;
const double TRAIN_RATIO = 0.8;
const int BATCH_SIZE = 32;
const double TOLERANCE = 1e-6;
const int MAX_NODES = 24;

enum class Activation {
    SIGMOID,
    TANH,
    RELU,
    LEAKY_RELU,
    SOFTMAX,
    LINEAR,
};

typedef struct s_metrics {
    bool classif;

    Row train_losses;
    Row train_metrics;
    Row val_losses;
    Row val_metrics;

    Matrix train_preds;
    Matrix train_truth;
    Matrix val_preds;
    Matrix val_truth;

    std::vector<std::vector<int>> confus;
} t_metrics;

struct AdamParams {
    Matrix m;
    Matrix v;
    int t = 0;

    double beta1 = 0.9;
    double beta2 = 0.999;
    double eps = 1e-8;

    void init(int rows, int cols) {
        m = Matrix(rows, Row(cols, 0.0));
        v = Matrix(rows, Row(cols, 0.0));
        t = 0;
    }
};

Matrix initMatrix(size_t rows, size_t cols, double min, double max);
bool checkRectangle(const Matrix& A, bool Square=false);
Matrix transpose(const Matrix& m);
double det(Matrix m);
Matrix invert(const Matrix& m);

double dot(const Row& A, const Row& B);
std::ostream& operator<<(std::ostream& os, const Matrix& m);
std::ostream& operator<<(std::ostream& os, const Row& row);
Matrix operator+(const Matrix& A, const Matrix& B);
Matrix operator-(const Matrix& A, const Matrix& B);
Matrix operator*(const double k, const Matrix& m);
Matrix operator*(const Matrix& A, const Matrix& B);


#include "Data.hpp"
#include "Layer.hpp"
#include "NN.hpp"

void exportTrainingData(const t_metrics& m);
void evaluateComplexity(Dataset& data);