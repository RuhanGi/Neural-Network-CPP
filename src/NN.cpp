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

NN::NN(Dataset& data) : data(data)
{

}


double NN::calcLoss(const Matrix &preds, const Matrix &actual)
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

double getAccuracy(const Matrix &preds, const Matrix &actual)
{
    int count = 0;
    for (size_t i = 0; i < preds.size(); i++)
        if (getIndexMax(preds[i]) == getIndexMax(actual[i]))
            count++;
    return (double) count / preds.size();
}


double rSqr(const Matrix& actual, const Matrix& pred)
{
	size_t n = actual.size();
	if (n != pred.size())
        throw std::invalid_argument("Difference in size");

	double mean = 0;
	for (size_t i = 0; i < n; i++)
		mean += actual[i][0];
	mean /= n;

	double sumRes = 0;
	double sumSqr = 0;
	for (size_t i = 0; i < n; i++)
	{
		sumRes += std::pow(actual[i][0] - pred[i][0], 2);
		sumSqr += std::pow(actual[i][0] - mean, 2);
	}
	return 1 - (sumRes / sumSqr);
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

void NN::epochPrint(size_t e, const t_metrics &m)
{
    std::cout << GREY "Epoch [" << e << "\t"<< MAX_EPOCHS << "]"
              << GREY " | Train Loss: ["  RED << m.train_losses.back()
              << GREY "] | Val Loss: [" RED << m.val_losses.back()
              << GREY "]\r" << std::flush;
}

void NN::metricPrint(const t_metrics &m)
{
    if (data.classif)
    {
        std::cout << GREY " Train Acc = "  GREEN << m.train_metrics.back()
                  << GREY " | Val Acc = " GREEN << m.val_metrics.back();
    }
    else
    {
        std::cout << GREY " Train R^2 = "  GREEN << m.train_metrics.back()
                  << GREY " | Val R^2 = " GREEN << m.val_metrics.back();
    }
    std::cout << std::string(80, ' ') << RESET "\n";
}

void NN::calcMetrics(t_metrics& m)
{
    Matrix train_preds = forward(data.X);
    Matrix val_preds = forward(data.valX);

    m.train_losses.push_back(calcLoss(train_preds, data.Y));
    m.val_losses.push_back(calcLoss(val_preds, data.valY));
    if (m.classif)
    {
        m.train_metrics.push_back(getAccuracy(train_preds, data.Y));
        m.val_metrics.push_back(getAccuracy(val_preds, data.valY));
    }
    else
    {
        m.train_metrics.push_back(rSqr(train_preds, data.Y));
        m.val_metrics.push_back(rSqr(val_preds, data.valY));
    }
}


void NN::setMetrics(t_metrics& m)
{
   m.val_preds = forward(data.valX);

    if (m.classif)
    {
        int numClasses = data.valY[0].size();
    
        m.confus.assign(numClasses, std::vector<int>(numClasses, 0));
        for (size_t i = 0; i < m.val_preds.size(); i++)
            m.confus[getIndexMax(data.valY[i])][getIndexMax(m.val_preds[i])]++;
    }
    else
    {
        m.train_preds = forward(data.X);
        m.train_truth = data.Y;
        m.val_truth = data.valY;
    }
}


t_metrics   NN::fit()
{
    addOutputLayer();

    int patience = 10;
    int wait = 0;
    double bestLoss = std::numeric_limits<double>::max();
    t_metrics m;
    m.classif = data.classif;
    std::vector<Layer> bestLayers = layers;
    for (size_t e = 1; e <= MAX_EPOCHS; e++)
    {
        for (size_t i = 0; i < data.X.size(); i += BATCH_SIZE)
        {
            size_t end = std::min(i + BATCH_SIZE, data.X.size());
            Matrix batchX(data.X.begin() + i, data.X.begin() + end);
            Matrix batchY(data.Y.begin() + i, data.Y.begin() + end);
            Matrix preds = forward(batchX);
            Matrix error = preds - batchY;
            backprop(error);
        }
        calcMetrics(m);
        epochPrint(e, m);
        if (m.val_losses.back() < (bestLoss - TOLERANCE)) {
            bestLoss = m.val_losses.back();
            wait = 0;
            bestLayers = layers;
        } else
            wait++;
    
        if (wait >= patience)
            break;
        data.shuffle();
    }
    layers = bestLayers;
    setMetrics(m);
    std::cout << RESET "\n";
    metricPrint(m);
    return m;
}
