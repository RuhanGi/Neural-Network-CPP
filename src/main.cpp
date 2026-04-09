/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   main.cpp                                           :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: RuhanGi <mohammedruhan.goltay@kaust.edu    +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2026/01/28 18:35:50 by RuhanGi           #+#    #+#             */
/*   Updated: 2026/01/28 18:35:50 by RuhanGi          ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#include "Types.hpp"

int	main(int argc, char *argv[])
{
	if (argc != 3 || !argv || (argv[2][0] != 'R' && argv[2][0] != 'C'))
		return (std::cerr << RED "Usage: neural.exe {data} {R or C}" RESET "\n", 1);

	try
	{
		Dataset data = Dataset(argv[1], argv[2][0] == 'C');
		data.printStats();

		NN net = NN(data);
		// net.addLayer(10, Activation::RELU);
		net.addLayer(16, Activation::RELU);
		net.addLayer(8, Activation::RELU);
		net.fit();
	}
	catch (std::exception & e)
	{
		std::cerr << RED "ERROR: " << e.what() << RESET "\n";
		return (1);
	}

	return (0);
}
