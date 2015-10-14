package goga

import (
	"fmt"
	"math"
	"math/rand"
	"runtime"
	"sort"
	"sync"
	"time"
)

// Calculates the fitness score for each member in the population
func (g *Goga) calculatePopulationFitness(population []Chromosome) []Chromosome {
	var wg sync.WaitGroup
	for i := range population {
		wg.Add(1)
		go func(j int) {
			defer wg.Done()
			population[j].CalculateFitness(g.target)
		}(i)
	}
	wg.Wait()
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
		if individual.GetFitness() > best.GetFitness() || math.IsNaN(best.GetFitness()) {
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
	GetKey() string                               // Returns the chomosome's key
	IsGoodEnough() bool                           // Returns true if the Chromosome has a fitness score above a certain threshold
	Mutate()                                      // Mutate a single chromosome
	Learn()                                       // Learn a single chromosome
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
	length := len(parents)
	if length%4 != 0 {
		parents = parents[:length-(length%4)]
		length = len(parents)
	}

	c := make(chan []interface{})
	for i := 0; i < length; i += 2 {
		go func(j int) {
			child1, child2 := parents[j].Breed(parents[j+1])
			c <- []interface{}{child1, child2}
		}(i)
	}

	nextGeneration := make([]Chromosome, 0)
	count := 0
	for children := range c {
		nextGeneration = append(nextGeneration, g.Converter(children[0]), g.Converter(children[1]))
		count += 2
		if count >= length {
			close(c)
		}
	}

	return nextGeneration
}

// Randomly changes chromosomes in the population to prevent premature convergence
func (g *Goga) Mutation(generation []Chromosome) []Chromosome {
	var wg sync.WaitGroup

	for i := range generation {
		wg.Add(1)
		go func(j int) {
			defer wg.Done()
			generation[j].Mutate()
			generation[j].CalculateFitness(g.target)
		}(i)
	}

	wg.Wait()

	return generation
}

// Randomly changes chromosomes in the population to prevent premature convergence
func (g *Goga) Learn(generation []Chromosome) []Chromosome {
	var wg sync.WaitGroup

	for i := range generation {
		wg.Add(1)
		go func(j int) {
			defer wg.Done()
			generation[j].Learn()
			generation[j].CalculateFitness(g.target)
		}(i)
	}

	wg.Wait()

	return generation
}

type Goga struct {
	target        Chromosome
	MaxIterations int
	Result        Chromosome
	Status        string
	Converter     func(interface{}) Chromosome
}

func Init(chromosomeConverter func(interface{}) Chromosome) *Goga {
	return &Goga{
		Converter:     chromosomeConverter,
		MaxIterations: 1000,
	}
}

func (g *Goga) Run(target Chromosome, population []Chromosome) {
	fmt.Println("\tSetting GOMAXPROCS to 16. Previous setting was", runtime.GOMAXPROCS(16))
	g.target = target

	rand.Seed(time.Now().UnixNano())

	var checkpoint time.Time
	fmt.Print("\tcalculatePopulationFitness(population)")
	checkpoint = time.Now()
	population = g.calculatePopulationFitness(population)
	fmt.Println(" // Took:", time.Since(checkpoint))

	g.Status = fmt.Sprintf("Failed to converge after %d iterations.", g.MaxIterations)

	for i := 0; i < g.MaxIterations; i++ {
		fmt.Println("\tGeneration:", i)
		fmt.Print("\t\tSelection()")
		checkpoint = time.Now()
		parents := g.Selection(population)
		fmt.Println(" // Took:", time.Since(checkpoint))
		fmt.Print("\t\tcalculatePopulationFitness(parents)")
		checkpoint = time.Now()
		parents = g.calculatePopulationFitness(parents)
		fmt.Println(" // Took:", time.Since(checkpoint))
		fmt.Print("\t\tCrossover()")
		checkpoint = time.Now()
		nextGeneration := g.Crossover(parents)
		fmt.Println(" // Took:", time.Since(checkpoint))
		// fmt.Print("\t\tcalculatePopulationFitness(nextGeneration)")
		// checkpoint = time.Now()
		// nextGeneration = g.calculatePopulationFitness(nextGeneration)
		// fmt.Println(" // Took:", time.Since(checkpoint))
		fmt.Print("\t\tMutation()")
		checkpoint = time.Now()
		nextGeneration = g.Mutation(nextGeneration)
		fmt.Println(" // Took:", time.Since(checkpoint))
		fmt.Print("\t\tLearning()")
		checkpoint = time.Now()
		nextGeneration = g.Learn(nextGeneration)
		fmt.Println(" // Took:", time.Since(checkpoint))

		// Next generation
		population = append(parents, nextGeneration...)

		g.Result = getBestChromosome(nextGeneration)
		if isGoodEnough(nextGeneration) {
			g.Status = fmt.Sprintf("Took %d generations.", i)
			break
		} else {
			fmt.Println("\tBest Chromosome:", g.Result)
		}
	}
}
