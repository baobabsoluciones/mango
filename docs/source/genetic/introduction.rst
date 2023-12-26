Introduction
-------------

Genetic Algorithms (GAs) represent a powerful and innovative approach to problem-solving inspired by the principles of natural selection and genetics. Developed by `John Holland <https://en.wikipedia.org/wiki/John_Henry_Holland>`_ in the 1960s, genetic algorithms are a subset of evolutionary algorithms that emulate the process of natural selection to find optimal solutions to complex problems.

At their core, genetic algorithms draw inspiration from the mechanisms of biological evolution, incorporating concepts such as selection, crossover, and mutation. The fundamental idea is to mimic the process of natural selection in order to evolve a population of potential solutions (individuals) over successive generations. This population undergoes genetic operations, such as crossover (recombination of solutions) and mutation (introduction of small, random changes), simulating the genetic diversity seen in nature.

Genetic algorithms have proven to be versatile and effective in solving optimization and search problems across various domains, ranging from engineering and computer science to finance and biology. Their ability to explore vast solution spaces and adapt to changing environments makes them particularly suitable for problems with multiple, interacting variables or when a direct analytical solution is difficult to obtain.

This introduction will delve into the key components of genetic algorithms, their underlying principles, and explore real-world applications that showcase their adaptability and efficiency in addressing complex challenges. As we navigate through the intricacies of genetic algorithms, we will uncover how these algorithms have become a valuable tool in the computational toolkit, providing innovative solutions to some of the most intricate problems faced in science, engineering, and beyond.

Key concepts on metaheuristic search
====================================

**Exploration** and **exploitation** are fundamental concepts in the context of metaheuristic search algorithms, including genetic algorithms, particle swarm optimization, simulated annealing, and others. These two aspects represent a delicate balance that these algorithms must strike to effectively navigate the solution space and converge towards optimal or near-optimal solutions.

Exploration involves the search for new regions in the solution space, aiming to discover novel and potentially better solutions. In the early stages of the algorithm, a focus on exploration is crucial to ensure that a diverse set of solutions is considered. This helps prevent premature convergence to suboptimal solutions and allows the algorithm to discover different regions of the solution space that may contain more promising candidates.

On the other hand, exploitation is the process of intensifying the search in regions that have shown promise in terms of solution quality. As the algorithm progresses, an increased emphasis on exploitation becomes essential to refine and improve the current solutions. Exploitation allows the algorithm to zoom in on areas of the solution space that are likely to contain optimal solutions, leveraging the information gained during the exploration phase.

The challenge for metaheuristic search algorithms lies in effectively balancing exploration and exploitation throughout the search process. Striking this balance ensures that the algorithm explores a sufficiently large portion of the solution space early on while gradually shifting focus towards refining and exploiting the most promising solutions as the search progresses. This dynamic interplay between exploration and exploitation is vital for the success of metaheuristic search algorithms in efficiently finding high-quality solutions to complex optimization problems.

Structure of a genetic algorithm
=================================

Most genetic algorithms follow a similar structure that can be seen in the following figure, but some may include different operators based on the individuals or the population.

.. figure:: ../static/ga.webp
    :alt: Structure of a genetic algorithm
    :width: 75%
    :align: center

    Höschel, K.; Lakshminarayanan, V. Genetic algorithms for lens design: A review. J. Opt. 2018, 48, 134–144.

As it can be seen all genetic algorithms start from a similar point in which we need to generate an initial random population of solutions (individuals). This population is then evaluated and their fittness is calculated to see how they adapt to the problem that we are trying to solve. The quality of this initial population will determine the convergence progress of the population and have a crucial impact on the initial phases of exploration done by the algorithm.

After this the genetic algorithm starts iterating and repeating the following steps or genetic operators.

- **Selection**: In this step we select the individuals that will be used to generate the next generation. This selection is based on the fitness of the individuals, so the individuals with the highest fitness will have a higher probability of being selected. This is done to simulate the natural selection process in which the individuals with the highest fitness are more likely to survive and reproduce.
- **Crossover**: In this step we generate new individuals by combining the genes of two (or more) individuals. This is done to simulate the reproduction process in which the genes of two individuals are combined to generate a new individual. In this case as we are not restricted to the species reproduction the crossover can be done from more than two "parents".
- **Mutation**: In this step we introduce random changes in the genes of the individuals. This is done to simulate the mutation process in which the genes of an individual can change randomly. This is done to introduce new genes in the population and to avoid the algorithm to get stuck in a local minimum. Usually the mutation rate is very low and this will help with the exploration part of the algorithm, even, in some cases, being the only source of exploration.
- **Stopping criteria**: In this step we check if the algorithm has reached a stopping criteria. This can be a maximum number of iterations, a maximum number of generations without improvement, a maximum number of generations, etc. This is done to avoid the algorithm to run forever and to stop it when it has reached a good enough solution.
- **Replacement**: In this step we replace the individuals of the current generation with the individuals of the next generation. This is done to simulate the death of the individuals that are not fit enough to survive and reproduce.

Some of these steps have different versions that can be used in different situations, impacting how the population evolves and thus, how the population converges to a solution. The different versions of this operators that are implemented on mango will be explained on their given sections in order to enter into much detail as possible and explain the origin of each one of these operators.

Glossary
=========

- **Crossover**: The process of combining the genes of two (or more) individuals to generate a new individual. This is done to simulate the reproduction process in which the genes of two individuals are combined to generate a new individual.
- **Encoding**: The process of representing the individuals of the population as a set of genes. This is done to simulate the genes of an individual in nature.
- **Fitness**: A measure of how well an individual solves the problem that we are trying to solve. In the case of genetic algorithms this is usually a number that represents how close the individual is to the optimal solution.
- **Gene**: A part of the individual that is used to represent a characteristic of the individual. In the case of genetic algorithms this is usually a number that represents a characteristic of the individual.
- **Genome**: The set of genes that are used to represent an individual. In the case of genetic algorithms this is usually a set of numbers that represent the characteristics of the individual.
- **Genotype**: The set of genes that are used to represent an individual. In the case of genetic algorithms this is usually a set of numbers that represent the characteristics of the individual.
- **Individual**: A solution to the problem that we are trying to solve. In the case of genetic algorithms this solution is represented by a set of genes.
- **Mutation**: The process of introducing random changes in the genes of the individuals. This is done to simulate the mutation process in which the genes of an individual can change randomly.
- **Phenotype**: The solution represented by the individual through their genotype.
- **Population**: A set of individuals that are used to solve the problem that we are trying to solve. In the case of genetic algorithms this set of individuals is called a population because it simulates a population of individuals in nature.
- **Replacement**: The process of replacing the individuals of the current generation with the individuals of the next generation. This is done to simulate the death of the individuals that are not fit enough to survive and reproduce.
- **Selection**: The process of selecting the individuals that will be used to generate the next generation. This is done to simulate the natural selection process in which the individuals with the highest fitness are more likely to survive and reproduce.
