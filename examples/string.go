package main

import (
	"github.com/renquinn/goga"

	"fmt"
	"math/rand"
	"strings"
)

// Levenshtein Distance for comparing strings
func ld(s, t string) int {
	d := make([][]int, len(s)+1)
	for i := range d {
		d[i] = make([]int, len(t)+1)
	}
	for i := range d {
		d[i][0] = i
	}
	for j := range d[0] {
		d[0][j] = j
	}
	for j := 1; j <= len(t); j++ {
		for i := 1; i <= len(s); i++ {
			if s[i-1] == t[j-1] {
				d[i][j] = d[i-1][j-1]
			} else {
				min := d[i-1][j]
				if d[i][j-1] < min {
					min = d[i][j-1]
				}
				if d[i-1][j-1] < min {
					min = d[i-1][j-1]
				}
				d[i][j] = min + 1
			}
		}

	}
	return d[len(s)][len(t)]
}

type Member struct {
	Fitness float64
	Value   string
}

func (m *Member) IsGoodEnough() bool {
	return m.Fitness > 7.0
}

func (m *Member) CalculateFitness(t interface{}) {
	target, _ := t.(*Member)
	edits := ld(m.Value, target.Value)
	m.Fitness = float64(len(m.Value)) - float64(edits)
}

func (m *Member) Normalize(total float64) {
	m.Fitness = m.Fitness / total
}

func (m *Member) AccNormalize(accumulation float64) {
	m.Fitness = accumulation
}

func (m *Member) Mutate() {
	alphabet := []string{"a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z"}
	value := m.Value
	for _, c := range value {
		if rand.Float64() > .8 {
			m.Value = strings.Replace(value, string(c), alphabet[rand.Intn(len(alphabet))], 1)
		}
	}
}

func (m *Member) GetFitness() float64 {
	return m.Fitness
}

func (mom *Member) Breed(daddy interface{}) (interface{}, interface{}) {
	dad, _ := daddy.(*Member)
	crossoverSite := rand.Intn(len(dad.Value))

	child1 := &Member{Value: mom.Value[:crossoverSite] + dad.Value[crossoverSite:]}
	child2 := &Member{Value: dad.Value[:crossoverSite] + mom.Value[crossoverSite:]}

	return child1, child2
}

func (m *Member) String() string {
	return fmt.Sprintf("{%d %s}", int(m.Fitness), m.Value)
}

func main() {
	target := "renquinn"
	g := goga.Init(func(member interface{}) goga.Chromosome {
		mem, _ := member.(*Member)
		return mem
	})
	population := g.GeneratePopulation(100, func() goga.Chromosome {
		alphabet := []string{"a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z"}
		member := ""
		for i := 0; i < len(target); i++ {
			member += alphabet[rand.Intn(len(alphabet))]
		}
		return &Member{Value: member}
	})

	g.Run(&Member{Value: target}, population)
	fmt.Println(g.Result)
	fmt.Println(g.Status)
}
