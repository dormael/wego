package pairwise

import (
	"context"
	"fmt"
	"io"
	"math"
	"runtime"
	"sync"

	"golang.org/x/sync/semaphore"

	"github.com/pkg/errors"
	"github.com/ynqa/wego/pkg/clock"
	"github.com/ynqa/wego/pkg/corpus"
	"github.com/ynqa/wego/pkg/corpus/pairwise/encode"
	"github.com/ynqa/wego/pkg/model"
	"github.com/ynqa/wego/pkg/verbose"
)

type Pairwise struct {
	opts   Options
	mopts  model.Options
	corpus *corpus.Corpus

	colloc map[uint64]float64
	mu     sync.Mutex

	verbose *verbose.Verbose
}

func New(
	opts Options,
	copts corpus.Options,
	mopts model.Options,
	verbose *verbose.Verbose,
) *Pairwise {
	return &Pairwise{
		opts:   opts,
		mopts:  mopts,
		corpus: corpus.New(copts, verbose),

		colloc: make(map[uint64]float64),

		verbose: verbose,
	}
}

func (p *Pairwise) Colloc() map[uint64]float64 {
	return p.colloc
}

func (p *Pairwise) Corpus() *corpus.Corpus {
	return p.corpus
}

func (p *Pairwise) Read(r io.Reader) error {
	if err := p.corpus.Read(r); err != nil {
		return err
	}
	return p.build()
}

func (p *Pairwise) build() error {
	counted, clk := make(chan struct{}), clock.New()
	go p.observe(counted, clk)

	docCh := make(chan []int)
	go p.corpus.BatchDocument(docCh, p.mopts.BatchSize)

	sem := semaphore.NewWeighted(int64(runtime.NumCPU()))
	wg := &sync.WaitGroup{}

	for d := range docCh {
		wg.Add(1)
		go p.count(d, counted, sem, wg)
	}

	wg.Wait()
	close(counted)
	return nil
}

func (p *Pairwise) count(
	doc []int,
	counted chan struct{},
	sem *semaphore.Weighted,
	wg *sync.WaitGroup,
) error {
	defer func() {
		wg.Done()
		sem.Release(1)
	}()

	if err := sem.Acquire(context.Background(), 1); err != nil {
		return err
	}

	for i := 0; i < len(doc); i++ {
		for j := i + 1; j < len(doc) && j <= i+p.mopts.Window; j++ {
			l1, l2 := doc[i], doc[j]
			f, err := p.cntval(l1, l2)
			if err != nil {
				return err
			}
			dec := encode.EncodeBigram(uint64(l1), uint64(l2))
			p.mu.Lock()
			p.colloc[dec] += f
			p.mu.Unlock()
		}
		counted <- struct{}{}
	}
	return nil
}

func (p *Pairwise) cntval(left, right int) (float64, error) {
	switch p.opts.CountType {
	case Increment:
		return 1., nil
	case Distance:
		div := left - right
		if div == 0 {
			return 0, errors.Errorf("Divide by zero on counting co-occurrence")
		}
		return 1. / math.Abs(float64(div)), nil
	default:
		return 0, invalidCountTypeError(p.opts.CountType)
	}
}

func (p *Pairwise) observe(counted chan struct{}, clk *clock.Clock) {
	var cnt int
	for range counted {
		p.verbose.Do(func() {
			cnt++
			if cnt%100000 == 0 {
				fmt.Printf("count %d words %v\r", cnt, clk.AllElapsed())
			}
		})
	}
	p.verbose.Do(func() {
		fmt.Printf("count %d words %v\r\n", cnt, clk.AllElapsed())
	})
}
