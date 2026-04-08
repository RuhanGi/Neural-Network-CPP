/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   NN.cpp                                             :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: RuhanGi <mohammedruhan.goltay@kaust.edu    +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2026/04/08 18:39:42 by RuhanGi           #+#    #+#             */
/*   Updated: 2026/04/08 18:39:42 by RuhanGi          ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#include "NN.hpp"

NN::NN(const Dataset& data) : data(data)
{

}


double NN::calcLoss(Matrix preds, Matrix actual)
{
    double totalLoss = 0.0;
    size_t n = preds.size();
    size_t k = preds[0].size();

    if (data.classif) 
    {
        for (size_t i = 0; i < n; i++) {
            for (size_t j = 0; j < k; j++) {
                double p = std::max(1e-15, std::min(1.0 - 1e-15, preds[i][j]));
                totalLoss -= actual[i][j] * std::log(p);
            }
        }
    }
    else
    {
        for (size_t i = 0; i < n; i++) {
            for (size_t j = 0; j < k; j++) {
                double error = actual[i][j] - preds[i][j];
                totalLoss += error * error;
            }
        }
    }
    return totalLoss / (n * k);
}


int getIndexMax(const Row& row) {
    if (row.empty())
        return -1; // Safety check

    auto maxIt = std::max_element(row.begin(), row.end());
    return std::distance(row.begin(), maxIt);
}

double getAccuracy(Matrix preds, Matrix actual)
{
    int count = 0;
    for (size_t i = 0; i < preds.size(); i++)
        if (getIndexMax(preds[i]) == getIndexMax(actual[i]))
            count++;
    return (double) count / preds.size();
}


void NN::addLayer(int num_nodes, Activation actType)
{
    int num_inputs = (layers.size()) ? (layers.back().num_nodes) : (data.X[0].size());    
    layers.push_back(Layer(num_inputs, num_nodes, actType));
}


void NN::addOutputLayer()
{
    int numOutputs = data.Y[0].size();

    if (data.classif)
        addLayer(numOutputs, Activation::SOFTMAX); 
    else
        addLayer(numOutputs, Activation::LINEAR);
}


Matrix NN::forward(Matrix passer)
{
    for (Layer &l : layers)
        passer = l.forward(passer);
    return passer;
}

Matrix NN::backprop(Matrix passer)
{
    for (int i = layers.size()-1; i >= 0; i--)
        passer = layers[i].backprop(passer);
    return passer;
}

void NN::epochPrint(size_t e, Matrix preds)
{
    std::cout << GREY "Epoch [" << e << "\t"<< MAX_EPOCHS << "]";
    if (data.classif)
    {
        std::cout << GREY " | Train Acc: "  RED << getAccuracy(preds, data.Y);
        std::cout << GREY " | Val Acc: " RED << getAccuracy(forward(data.valX), data.valY);
    }
    else
    {
        std::cout << GREY " | Train Loss: "  RED << calcLoss(preds, data.Y);
        std::cout << GREY " | Val Loss: " RED << calcLoss(forward(data.valX), data.valY);
    }
    std::cout << RESET "\r" << std::flush;
}

void NN::fit()
{
    addOutputLayer();

    int patience = 10;
    int wait = 0;
    double bestValLoss = std::numeric_limits<double>::max();
    for (size_t e = 1; e <= MAX_EPOCHS; e++)
    {
        Matrix preds = forward(data.X);
        Matrix error = preds - data.Y;
        backprop(error);
        epochPrint(e, preds);

        double currentValLoss = calcLoss(forward(data.valX), data.valY);
        if (currentValLoss < bestValLoss) {
            bestValLoss = currentValLoss;
            wait = 0;
        } else
            wait++;
        
        if (wait >= patience)
            break;
    }
    std::cout << RESET "\n";
}
