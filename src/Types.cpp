/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   Types.cpp                                          :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: RuhanGi <mohammedruhan.goltay@kaust.edu    +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2026/04/08 19:28:57 by RuhanGi           #+#    #+#             */
/*   Updated: 2026/04/08 19:28:57 by RuhanGi          ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#include "Types.hpp"

Matrix initMatrix(size_t rows, size_t cols, double min, double max)
{
    static std::random_device rd;  // Seed
    static std::mt19937 gen(rd()); // Generator

    std::uniform_real_distribution<double> dis(min, max);
    Matrix res(rows, Row(cols));
    for (size_t i = 0; i < rows; i++)
        for (size_t j = 0; j < cols; j++)
            res[i][j] = dis(gen);
    return res;
}


bool checkRectangle(const Matrix& A, bool Square)
{
    if (A.empty())
        return false;

    size_t rows = A.size();
    size_t cols = A[0].size();
    if (cols == 0 || (Square && rows != cols))
        return false;
    for (size_t i = 1; i < rows; i++)
        if (A[i].size() != cols)
            return false;    
    return true;
}


Matrix transpose(const Matrix& m)
{
    if (m.empty())
        return {};

    Matrix res;
    for (size_t j = 0; j < m[0].size(); j++)
    {
        Row row;
        for (size_t i = 0; i < m.size(); i++)
            row.push_back(m[i][j]);
        res.push_back(row);
    }
    return res;
}



double swap(Matrix& m, size_t row_1, size_t row_2)
{
    if (row_1 >= m.size() || row_2 >= m.size())
        throw std::invalid_argument("Invalid Index");
    if (row_1 == row_2)
        return 1;

    std::swap(m[row_1], m[row_2]);
    return -1;
}


double scale(Matrix& m, double k, size_t row)
{
    if (row >= m.size())
        throw std::invalid_argument("Invalid Index");

    for (size_t i = 0; i < m[row].size(); i++)
        m[row][i] *= k;
    return k;
}


double replace(Matrix& m, size_t row_1, double k, size_t row_2)
{
    if (row_1 >= m.size() || row_2 >= m.size())
        throw std::invalid_argument("Invalid Index");
    if (m[row_1].size() != m[row_2].size())
        throw std::invalid_argument("Row Dimensions Mismatch");

    for (size_t i = 0; i < m[row_1].size(); i++)
        m[row_1][i] += k * m[row_2][i];
    return 1;
}


double gaussJordan(Matrix& aug, size_t n) 
{
    double det_val = 1.0;

    for (size_t j = 0; j < n; j++)
    {
        size_t pivot = j;
        for (size_t i = j + 1; i < n; i++)
            if (std::abs(aug[i][j]) > std::abs(aug[pivot][j]))
                pivot = i;
        det_val *= swap(aug, j, pivot);
        if (std::abs(aug[j][j]) < 1e-15) 
            return 0.0;
        det_val *= 1.0 / scale(aug, 1.0/aug[j][j], j);
        for (size_t i = 0; i < n; i++)
            if (i != j)
                det_val *= replace(aug, i, -aug[i][j], j);
    }
    return det_val;
}


double det(Matrix m)
{
    if (m.empty())
        return 1.0;
    size_t n = m.size();
    if (!checkRectangle(m, true))
        throw std::invalid_argument("Non-Square Matrix");

    return gaussJordan(m, n);
}


Matrix invert(const Matrix& m)
{
    if (m.empty())
        return {};

    size_t n = m.size();
    if (!checkRectangle(m, true))
        throw std::invalid_argument("Can't Invert a non-Square Matrix");

    Matrix aug = m;
    for (size_t i = 0; i < n; i++)
        for (size_t j = 0; j < n; j++)
            aug[i].push_back(i==j);

    double d = gaussJordan(aug, n);
    if (std::abs(d) < 1e-15) 
        throw std::invalid_argument("Singular Matrix has no Inverse");

    Matrix res(n, Row(n));
    for (size_t i = 0; i < n; i++)
        for (size_t j = 0; j < n; j++)
            res[i][j] = aug[i][n + j];
    return res;
}


double dot(const Row& A, const Row& B)
{
    if (A.size() != B.size())
        throw std::invalid_argument("Dimension Mismatch: Vectors must be same size for Dot Product");

    double sum = 0.0;
    for (size_t i = 0; i < A.size(); i++)
        sum += A[i] * B[i];
    return sum;
}


std::string smartFormat(double val) {
    if (std::abs(val) < 1e-9)
        return "0";
    
    if (std::abs(val) > 1e6 || std::abs(val) < 1e-4) {
        std::ostringstream oss;
        oss << std::scientific << std::setprecision(1) << val;
        return oss.str();
    }
    if (std::abs(val - std::round(val)) < 1e-9)
        return std::to_string((long long) std::round(val));

    std::ostringstream oss;
    oss << std::fixed << std::setprecision(3) << val;
    std::string s = oss.str();
    s.erase(s.find_last_not_of('0') + 1, std::string::npos);
    if (s.back() == '.')
        s.pop_back();
    return s;
}


std::ostream& operator<<(std::ostream& os, const Matrix& m)
{
    const int W_NUM  = 10;
    for (size_t i = 0; i < m.size(); i++)
    {
        os << "[";
        for (size_t j = 0; j < m[i].size(); j++)
            os << std::setw(W_NUM) << smartFormat(m[i][j]);
        os << "]\n";
    }
    return os;
}


std::ostream& operator<<(std::ostream& os, const Row& row)
{
    const int W_NUM  = 10;
    os << "[";
    for (size_t i = 0; i < row.size(); i++)
        os << std::setw(W_NUM) << smartFormat(row[i]);
    os << "]\n";
    return os;
}

Matrix operator+(const Matrix& A, const Matrix& B)
{
    if (A.empty() || B.empty())
        return {};
    if (A.size() != B.size() || A[0].size() != B[0].size())
        throw std::invalid_argument("Matrix Mismath in Addition");

    Matrix res = A;
    for (size_t i = 0; i < A.size(); i++)
        for (size_t j = 0; j < A[0].size(); j++)
            res[i][j] += B[i][j];
    return res;
}


Matrix operator-(const Matrix& A, const Matrix& B)
{
    if (A.empty() || B.empty())
        return {};
    if (A.size() != B.size() || A[0].size() != B[0].size())
        throw std::invalid_argument("Matrix Mismath in Addition");

    Matrix res = A;
    for (size_t i = 0; i < A.size(); i++)
        for (size_t j = 0; j < A[0].size(); j++)
            res[i][j] -= B[i][j];
    return res;
}


Matrix operator*(const double k, const Matrix& m)
{
    if (m.empty())
        return {};

    Matrix res(m.size(), Row(m[0].size()));
    for (size_t i = 0; i < m.size(); i++)
        for (size_t j = 0; j < m[0].size(); j++)
            res[i][j] = k * m[i][j];
    return res;
}


Matrix operator*(const Matrix& A, const Matrix& B)
{
    if (A.empty() || B.empty())
        return {};
    if (A[0].size() != B.size())
        throw std::invalid_argument("Matrix Mismath Mid Multiplication");

    Matrix res(A.size(), Row(B[0].size(), 0.0));
    std::for_each(std::execution::par, res.begin(), res.end(),
        [&](Row& res_row) {
            size_t i = &res_row - &res[0];
            for (size_t m = 0; m < A[0].size(); m++)
                for (size_t j = 0; j < B[0].size(); j++)
                    res_row[j] += A[i][m] * B[m][j];
        }
    );
    return res;
}
