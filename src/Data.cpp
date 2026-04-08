/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   Data.cpp                                           :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: RuhanGi <mohammedruhan.goltay@kaust.edu    +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2026/04/08 11:26:18 by RuhanGi           #+#    #+#             */
/*   Updated: 2026/04/08 11:26:18 by RuhanGi          ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#include "Data.hpp"


Dataset::Dataset(const std::string& file, bool classif) : classif(classif) {
    std::ifstream ifile(file);
    if (!ifile.is_open())
        throw std::runtime_error("Could not open file: " + file);

    parse(ifile);
    if (classif)
        oneHotEncode();

    shuffle();
    split();
    normalize();
}

void Dataset::parse(std::ifstream& file) {
    std::string line;
    if (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string col;
        while (std::getline(ss, col, ','))
            headers.push_back(col);
        if (!headers.empty()) {
            labelName = headers.back();
            headers.pop_back();
        }
    }

    while (std::getline(file, line)) {
        if (line.empty())
            continue;
        std::stringstream ss(line);
        std::string token;
        Row row;
        while (std::getline(ss, token, ',')) {
            try {
                row.push_back(std::stod(token));
            } catch (...) {
                row.push_back(0.0);
            }
        }
        if (row.size() > 0) {
            double targetVal = row.back();
            row.pop_back();
            X.push_back(row);
            Y.push_back({targetVal});
        }
    }
}

void Dataset::oneHotEncode() {
    std::set<double> classes;
    for (const auto& targetRow : Y)
        classes.insert(targetRow[0]);

    int numClasses = classes.size();
    int idx = 0;
    for (double c : classes)
        classMap[c] = idx++;

    Matrix oneHotY(Y.size(), Row(numClasses, 0.0));
    for (size_t i = 0; i < Y.size(); ++i) 
        oneHotY[i][classMap[Y[i][0]]] = 1.0;
    Y = std::move(oneHotY);
}

void Dataset::normalize() {
    if (X.empty() || !means.empty())
        return;
    size_t nFeatures = X[0].size();
    size_t nSamples = X.size();

    means.assign(nFeatures + (classif ? 0 : 1), 0.0);
    stds.assign(means.size(), 0.0);

    // Calculate Means
    for (size_t i = 0; i < nSamples; i++) {
        for (size_t j = 0; j < nFeatures; j++)
            means[j] += X[i][j];
        if (!classif)
            means[nFeatures] += Y[i][0];
    }
    for (double& m : means)
        m /= nSamples;

    // Calculate Stds
    for (size_t i = 0; i < nSamples; i++) {
        for (size_t j = 0; j < nFeatures; j++) 
            stds[j] += std::pow(X[i][j] - means[j], 2);
        if (!classif)
            stds[nFeatures] += std::pow(Y[i][0] - means[nFeatures], 2);
    }
    for (double& s : stds)
        s = std::sqrt(s / nSamples);

    // Apply Normalization
    std::vector<size_t> indices(nSamples);
    std::iota(indices.begin(), indices.end(), 0);
    std::for_each(std::execution::par, indices.begin(), indices.end(), [this, nFeatures](size_t i) {
        for (size_t j = 0; j < nFeatures; j++)
            if (stds[j] > 1e-9) 
                X[i][j] = (X[i][j] - means[j]) / stds[j];
        if (!classif && stds[nFeatures] > 1e-9)
            Y[i][0] = (Y[i][0] - means[nFeatures]) / stds[nFeatures];
    });

    if (!valX.empty()) {
        std::vector<size_t> valIndices(valX.size());
        std::iota(valIndices.begin(), valIndices.end(), 0);
        std::for_each(std::execution::par, valIndices.begin(), valIndices.end(), [this, nFeatures](size_t i) {
            for (size_t j = 0; j < nFeatures; j++)
                if (stds[j] > 1e-9) 
                    valX[i][j] = (valX[i][j] - means[j]) / stds[j];
            if (!classif && stds[nFeatures] > 1e-9)
                valY[i][0] = (valY[i][0] - means[nFeatures]) / stds[nFeatures];
        });
    }
}

void Dataset::shuffle() {
    if (X.empty())
        return;

    std::vector<size_t> indices(X.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(indices.begin(), indices.end(), g);

    Matrix shuffledX;
    Matrix shuffledY;
    shuffledX.reserve(X.size());
    shuffledY.reserve(Y.size());

    for (size_t i : indices) {
        shuffledX.push_back(std::move(X[i]));
        shuffledY.push_back(std::move(Y[i]));
    }

    X = std::move(shuffledX);
    Y = std::move(shuffledY);
}

void Dataset::split() {
    if (X.empty())
        return;

    size_t trainSize = static_cast<size_t>(X.size() * TRAIN_RATIO);

    valX.assign(X.begin() + trainSize, X.end());
    valY.assign(Y.begin() + trainSize, Y.end());
    X.erase(X.begin() + trainSize, X.end());
    Y.erase(Y.begin() + trainSize, Y.end());
}

void Dataset::printStats() {
    normalize();

    const int W_NAME = 25;
    const int W_NUM  = 15;

    std::cout << YELLOW << (classif ? "Classification" : "Regression") << " Summary" RESET << "\n";
    std::cout << PURPLE << "Train Samples: " << GREEN << X.size()
              << PURPLE << " | Test Samples: " << GREEN << valX.size()
              << PURPLE << " | Features: " << GREEN << (X.empty() ? 0 : X[0].size()) << RESET << "\n";
    std::cout << std::string(W_NAME + 2 * W_NUM, '-') << "\n";
    
    // Header Row
    std::cout << YELLOW << std::left 
              << std::setw(W_NAME) << "Feature" 
              << std::setw(W_NUM)  << "Mean" 
              << std::setw(W_NUM)  << "Std Dev" << RESET << "\n";

    // Feature Rows
    for (size_t j = 0; j < headers.size(); j++) {
        std::cout << CYAN << std::left << std::setw(W_NAME) << headers[j] << GREY;
        std::cout << std::fixed << std::setprecision(4)
                  << std::setw(W_NUM) << means[j]
                  << std::setw(W_NUM) << stds[j] << RESET << "\n";
    }

    // Target Row for Regression
    std::cout << CYAN << std::left << std::setw(W_NAME) << labelName << GREEN;
    if (!classif) {
        std::cout << std::fixed << std::setprecision(4)
                  << std::setw(W_NUM) << means.back()
                  << std::setw(W_NUM) << stds.back() << RESET << "\n";
    } else {
        std::cout << std::fixed << std::setprecision(4)
                  << std::setw(W_NUM) << "Classes   ="
                  << std::setw(W_NUM) << classMap.size() << RESET << "\n";
    }
}