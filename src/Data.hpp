/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   Data.hpp                                           :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: RuhanGi <mohammedruhan.goltay@kaust.edu    +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2026/04/08 11:26:21 by RuhanGi           #+#    #+#             */
/*   Updated: 2026/04/08 11:26:21 by RuhanGi          ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#pragma once

#include "Types.hpp"

class Dataset {
private:
    void parse(std::ifstream& file);
    void oneHotEncode();
    void split();

public:
    Matrix X;
    Matrix Y;
    Matrix valX;
    Matrix valY;
    bool classif;
    std::vector<std::string> headers;
    std::string labelName;
    Row means;
    Row stds;
    std::map<double, int> classMap;

    Dataset(const std::string& file, bool classif=false);
    void normalize();
    void printStats();
    void shuffle();
};
