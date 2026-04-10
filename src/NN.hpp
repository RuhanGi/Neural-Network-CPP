/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   NN.hpp                                             :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: RuhanGi <mohammedruhan.goltay@kaust.edu    +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2026/04/08 18:39:40 by RuhanGi           #+#    #+#             */
/*   Updated: 2026/04/08 18:39:40 by RuhanGi          ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#pragma once

#include "Types.hpp"

class Dataset;
class Layer;

class NN {
private:
    Dataset& data;
    std::vector<Layer> layers;
    void addOutputLayer();
    double calcLoss(const Matrix &preds, const Matrix &actual);
    void epochPrint(size_t e, const t_metrics &m);
    void metricPrint(const t_metrics &m);
    void calcMetrics(t_metrics& m);
    void setMetrics(t_metrics& m);

public:
    NN(Dataset& data);

    void addLayer(int num_nodes, Activation actType = Activation::SIGMOID);
    t_metrics fit();
    Matrix forward(Matrix passer);
    Matrix backprop(Matrix passer);
};
