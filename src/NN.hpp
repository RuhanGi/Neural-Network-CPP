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
    const Dataset& data;
    std::vector<Layer> layers;
    void addOutputLayer();
    double calcLoss(Matrix preds, Matrix actual);
    void epochPrint(size_t e);

public:
    NN(const Dataset& data);

    void addLayer(int num_nodes, Activation actType = Activation::SIGMOID);
    void fit();
    Matrix forward(Matrix passer);
    Matrix backprop(Matrix passer);
};
