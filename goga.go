package main

import (
	"fmt"
	"math/rand"
	"sort"
	"time"
)

// Calculates the fitness score for each member in the population
func (g *Goga) calculatePopulationFitness(population []Chromosome) []Chromosome {
	for i, member := range population {
		member.CalculateFitness(g.target)
		population[i] = member
	}
	return population
}

// Function to determine if the population has converged. This is defined by
// whether or not the best, or most fit, chromosome from the population has a level
// of fitness above a specified threshold.
func isGoodEnough(population []Chromosome) bool {
	return getBestChromosome(population).IsGoodEnough()
}

// Returns the chromosome in the population with the highest fitness level
func getBestChromosome(population []Chromosome) Chromosome {
	best := population[0]

	for _, individual := range population {
		if individual.GetFitness() > best.GetFitness() {
			best = individual
		}
	}

	return best
}

type Chromosome interface {
	AccNormalize(float64)                         // Accumulative normalize a chromosome's fitness value, takes current accumulated total
	Breed(interface{}) (interface{}, interface{}) // Creates two member children from two parents using a single crossover site
	CalculateFitness(interface{})                 // Calculate and set a chromosome's fitness value
	GetFitness() float64                          // Returns the chomosome's fitness score
	IsGoodEnough() bool                           // Returns true if the Chromosome has a fitness score above a certain threshold
	Mutate()                                      // Mutate a single chromosome
	Normalize(float64)                            // Normalize a chromosome's fitness value, takes total fitness of population
	String() string                               // Returns a string representation of the chromosome
}

// Sortable interface
type Population []Chromosome

func (a Population) Len() int           { return len(a) }
func (a Population) Swap(i, j int)      { a[i], a[j] = a[j], a[i] }
func (a Population) Less(i, j int) bool { return a[i].GetFitness() < a[j].GetFitness() }

// This randomly generates the population of strings
func (g *Goga) GeneratePopulation(populationSize int, generator func() Chromosome) []Chromosome {
	population := make([]Chromosome, 0)
	for i := 0; i < populationSize; i++ {
		population = append(population, generator())
	}
	return population
}

// Calculates the fitness score for each chromosome in the population
func (g *Goga) CalculatePopulationFitness(population []Chromosome) []Chromosome {
	for i, chromosome := range population {
		chromosome.CalculateFitness(g.target)
		population[i] = chromosome
	}
	return population
}

// Picks a single parent candidate from the population. Taken straight from
// wikipedia, needs to be optimized.
func selectParent(population []Chromosome) (Chromosome, []Chromosome) {
	// 1. The fitness function is evaluated for each individual, providing fitness values, which are
	//    then normalized. Normalization means dividing the fitness value of each individual by the
	//    sum of all fitness values, so that the sum of all resulting fitness values equals 1.
	sum := 0.0
	for _, chromosome := range population {
		sum += chromosome.GetFitness()
	}

	for i := range population {
		population[i].Normalize(sum)
	}

	// 2. The population is sorted by descending fitness values.
	sort.Sort(Population(population))
	sort.Reverse(Population(population)) // Cause sort.Sort is ascending order

	// 3. Accumulated normalized fitness values are computed (the accumulated fitness value of an
	//    individual is the sum of its own fitness value plus the fitness values of all the previous
	//    individuals). The accumulated fitness of the last individual should be 1 (otherwise
	//    something went wrong in the normalization step).
	accumulation := 0.0
	for i, chromosome := range population {
		accumulation += chromosome.GetFitness()
		population[i].AccNormalize(accumulation)
	}

	// 4. A random number R between 0 and 1 is chosen.
	r := rand.Float64()

	// 5. The selected individual is the first one whose accumulated normalized value is greater than R.
	for i, chromosome := range population {
		if chromosome.GetFitness() > r {
			return chromosome, append(population[:i], population[i+1:]...)
		}
	}

	return population[len(population)-1], population[:len(population)-1]
}

func (g *Goga) Selection(population []Chromosome) []Chromosome {
	parentsCount := len(population) / 2
	parents := make([]Chromosome, 0)
	for len(parents) < parentsCount {
		parent, restOfPopulation := selectParent(population)
		parents = append(parents, parent)
		population = restOfPopulation
	}
	return parents
}

// Creates two offspring chromosomes to be added to the population essentially
// replacing the given parents
func (g *Goga) Crossover(parents []Chromosome) []Chromosome {
	nextGeneration := make([]Chromosome, 0)
	for i := 0; i < len(parents); i += 2 {
		child1, child2 := parents[i].Breed(parents[i+1])
		nextGeneration = append(nextGeneration, g.Converter(child1), g.Converter(child2))
	}
	return nextGeneration
}

// Randomly changes chromosomes in the population to prevent premature convergence
func (g *Goga) Mutation(generation []Chromosome) []Chromosome {
	for i := range generation {
		generation[i].Mutate()
		generation[i].CalculateFitness(g.target)
	}
	return generation
}

type Goga struct {
	target        Chromosome
	maxIterations int
	Result        Chromosome
	Status        string
	Converter     func(interface{}) Chromosome
}

func Init(chromosomeConverter func(interface{}) Chromosome) *Goga {
	return &Goga{
		Converter:     chromosomeConverter,
		maxIterations: 1000,
	}
}

func (g *Goga) Run(target Chromosome, population []Chromosome) {
	g.target = target

	rand.Seed(time.Now().UnixNano())

	population = g.calculatePopulationFitness(population)
	g.Status = fmt.Sprintf("Failed to converge after %d iterations.", g.maxIterations)

	for i := 0; i < g.maxIterations; i++ {
		parents := g.Selection(population)
		parents = g.calculatePopulationFitness(parents)
		nextGeneration := g.Crossover(parents)
		nextGeneration = g.Mutation(nextGeneration)

		// Next generation
		population = append(parents, nextGeneration...)

		g.Result = getBestChromosome(nextGeneration)
		if isGoodEnough(nextGeneration) {
			g.Status = fmt.Sprintf("Took %d generations.", i)
			break
		}
	}
}
