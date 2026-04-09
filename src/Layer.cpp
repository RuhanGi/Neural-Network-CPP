/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   Layer.cpp                                          :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: RuhanGi <mohammedruhan.goltay@kaust.edu    +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2026/04/08 18:42:24 by RuhanGi           #+#    #+#             */
/*   Updated: 2026/04/08 18:42:24 by RuhanGi          ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#include "Layer.hpp"

Layer::Layer(int num_inputs, int num_nodes, Activation actType) 
    : actType(actType), num_inputs(num_inputs), num_nodes(num_nodes) 
{
    double limit;
    if (actType == Activation::RELU)
        limit = std::sqrt(2.0 / num_inputs);
    else
        limit = std::sqrt(6.0 / (num_inputs + num_nodes));
    weights = initMatrix(num_inputs+1, num_nodes, -limit, limit);
}


void softmax(Row &r)
{
    if (r.empty())
        return;

    double max_val = *std::max_element(r.begin(), r.end());

    double sum = 0.0;
    for (double &val : r) {
        val = std::exp(val - max_val);
        sum += val;
    }

    if (sum > 1e-15)
        for (double &val : r)
            val /= sum;
}


void    Layer::act(Matrix &z)
{
    if (actType == Activation::SOFTMAX)
        for (Row& r : z)
            softmax(r);
    else
        for (size_t i = 0; i < z.size(); i++)
            for (size_t j = 0; j < z[i].size(); j++)
                switch (actType) {
                    case Activation::SIGMOID:
                        z[i][j] = 1.0 / (1.0 + std::exp(-z[i][j]));
                        break;
                    case Activation::TANH:
                        z[i][j] = std::tanh(z[i][j]);
                        break;
                    case Activation::RELU:
                        z[i][j] = (z[i][j] > 0) ? z[i][j] : 0.0;
                        break;
                    case Activation::SOFTMAX:
                        break;
                    case Activation::LINEAR:
                        break;
                    default:
                        break;
                }
}


void addBiasColumn(Matrix& inputs) {
    for (auto& row : inputs)
        row.push_back(1.0);
}


Matrix Layer::forward(Matrix in)
{
    inputs = in;
    addBiasColumn(inputs);
    z = inputs * weights;
    act(z);
    return z;
}

double Layer::getDeriv(double a) 
{
    switch (actType) {
        case Activation::SIGMOID:
            return a * (1.0 - a);
        case Activation::TANH:
            return 1.0 - (a * a);
        case Activation::RELU:
            return (a > 0) ? 1.0 : 0.0;
        case Activation::SOFTMAX:
            return 1.0;
        default:
            return 1.0;
    }
}

Matrix Layer::backprop(Matrix errors)
{
    Matrix delta = errors;
    for (size_t i = 0; i < delta.size(); i++)
        for (size_t j = 0; j < delta[i].size(); j++)
            delta[i][j] *= getDeriv(z[i][j]);

    for (size_t i = 0; i < weights.size(); i++)
        for (size_t j = 0; j < weights[i].size(); j++)
        {
            double grad = 0;
            for (size_t k = 0; k < inputs.size(); k++)
                grad += inputs[k][i] * delta[k][j];
            weights[i][j] -= LR * (grad / inputs.size());
        }

    Matrix prev_error(inputs.size(), Row(num_inputs, 0.0));
    for (size_t i = 0; i < inputs.size(); ++i)
        for (int j = 0; j < num_inputs; ++j)
            for (int k = 0; k < num_nodes; ++k)
                prev_error[i][j] += delta[i][k] * weights[j][k];
    return prev_error;
}
