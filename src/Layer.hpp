/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   Layer.hpp                                          :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: RuhanGi <mohammedruhan.goltay@kaust.edu    +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2026/04/08 18:42:22 by RuhanGi           #+#    #+#             */
/*   Updated: 2026/04/08 18:42:22 by RuhanGi          ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#pragma once

#include "Types.hpp"

// TODO optimizers?
// ? ADAM and RMSprop?

class Layer {
private:
    Matrix weights; // num_inputs+1  x  num_nodes
    Activation actType;
    Matrix inputs;
    Matrix z;
    void act(Matrix &z);
    double getDeriv(double a);

public:
    int num_inputs;
    int num_nodes;

    Layer(int num_inputs, int num_nodes, Activation actType);

    Matrix forward(Matrix in);
    Matrix backprop(Matrix errors);
};
