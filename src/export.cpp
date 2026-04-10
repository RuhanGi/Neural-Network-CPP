/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   export.cpp                                         :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: RuhanGi <mohammedruhan.goltay@kaust.edu    +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2026/04/10 18:39:12 by RuhanGi           #+#    #+#             */
/*   Updated: 2026/04/10 18:39:12 by RuhanGi          ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#include "Types.hpp"

void exportTrainingData(const t_metrics& m) {

    std::ofstream historyFile("history.csv");
    historyFile << "epoch,train_loss,val_loss,train_metric,val_metric\n";
    for (size_t i = 0; i < m.train_losses.size(); ++i)
        historyFile << i << "," << m.train_losses[i] << "," << m.val_losses[i] << ","
                    << m.train_metrics[i] << "," << m.val_metrics[i] << "\n";
    historyFile.close();

    if (m.classif) {
        std::ofstream cfFile("graph.csv");
        for (const auto& row : m.confus)
        {
            for (size_t i = 0; i < row.size(); ++i) 
                cfFile << row[i] << (i == row.size() - 1 ? "" : ",");
            cfFile << "\n";
        }
        cfFile.close();
    } else {
        std::ofstream regFile("graph.csv");
        regFile << "truth,pred\n";
        for (size_t i = 0; i < m.val_truth.size(); ++i)
            regFile << m.val_truth[i][0] << "," << m.val_preds[i][0] << "\n";
        regFile.close();
    }
    std::cout << "\n" << GREEN << "Metrics exported!" << RESET << std::endl;
}

void evaluateComplexity(Dataset& data) {
    std::ofstream file("complexity.csv");

    file << "nodes,train_loss,val_loss,train_metric,val_metric\n";
    for (int n = 1; n <= MAX_NODES; n++) {
        std::cout << YELLOW << "Architecture Check: [" CYAN << n << YELLOW "] hidden nodes..." RESET "\n";
        NN net(data);
        net.addLayer(n, Activation::LEAKY_RELU);
        t_metrics results = net.fit();
    
        file << n << "," 
             << results.train_losses.back() << "," 
             << results.val_losses.back()   << ","
             << results.train_metrics.back() << "," 
             << results.val_metrics.back()  << "\n";
    }
    file.close();
    std::cout << "\n" << GREEN << "Complexity report saved!" RESET "\n";
}
