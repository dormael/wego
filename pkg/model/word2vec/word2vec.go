package word2vec

import (
	"bufio"
	"bytes"
	"context"
	"fmt"
	"io"
	"math"
	"math/rand"
	"sync"

	"golang.org/x/sync/semaphore"

	"github.com/pkg/errors"
	"github.com/ynqa/wego/pkg/clock"
	"github.com/ynqa/wego/pkg/corpus"
	"github.com/ynqa/wego/pkg/model"
	"github.com/ynqa/wego/pkg/model/modelutil/matrix"
	"github.com/ynqa/wego/pkg/model/modelutil/save"
	"github.com/ynqa/wego/pkg/verbose"
)

type word2vec struct {
	opts Options

	corpus     *corpus.Corpus
	param      *matrix.Matrix
	subSamples []float64
	currentlr  float64
	mod        mod
	optimizer  optimizer

	verbose *verbose.Verbose
}

func New(opts ...ModelOption) (model.Model, error) {
	options := Options{
		CorpusOptions: corpus.DefaultOption(),
		ModelOptions:  model.DefaultOptions(),

		MaxDepth:           defaultMaxDepth,
		ModelType:          defaultModelType,
		NegativeSampleSize: defaultNegativeSampleSize,
		OptimizerType:      defaultOptimizerType,
		SubsampleThreshold: defaultSubsampleThreshold,
		Theta:              defaultTheta,
	}

	for _, fn := range opts {
		fn(&options)
	}

	return NewForOptions(options)
}

func NewForOptions(opts Options) (model.Model, error) {
	v := verbose.New(opts.ModelOptions.Verbose)
	return &word2vec{
		opts: opts,

		corpus: corpus.New(v),

		currentlr: opts.ModelOptions.Initlr,

		verbose: v,
	}, nil
}

func (w *word2vec) preTrain(r io.Reader) error {
	if err := w.corpus.Read(r, w.opts.CorpusOptions); err != nil {
		return errors.Wrap(err, "failed to read corpus")
	}

	dic := w.corpus.Dictionary()
	dim := w.opts.ModelOptions.Dim
	w.param = matrix.New(dic.Len(), dim, func(vec []float64) {
		for i := 0; i < dim; i++ {
			vec[i] = (rand.Float64() - 0.5) / float64(dim)
		}
	})

	w.subSamples = make([]float64, dic.Len())
	for i := 0; i < dic.Len(); i++ {
		z := float64(dic.IDFreq(i)) / float64(w.corpus.Len())
		w.subSamples[i] = (math.Sqrt(z/w.opts.SubsampleThreshold) + 1.0) *
			w.opts.SubsampleThreshold / z
	}

	switch w.opts.ModelType {
	case SkipGram:
		w.mod = newSkipGram(w.opts)
	case Cbow:
		w.mod = newCbow(w.opts)
	default:
		return invalidModelTypeError(w.opts.ModelType)
	}

	switch w.opts.OptimizerType {
	case NegativeSampling:
		w.optimizer = newNegativeSampling(
			w.corpus.Dictionary(),
			w.opts,
		)
	case HierarchicalSoftmax:
		w.optimizer = newHierarchicalSoftmax(
			w.corpus.Dictionary(),
			w.opts,
		)
	default:
		return invalidOptimizerTypeError(w.opts.OptimizerType)
	}
	return nil
}

func (w *word2vec) Train(r io.Reader) error {
	if err := w.preTrain(r); err != nil {
		return err
	}

	for i := 1; i <= w.opts.ModelOptions.Iter; i++ {
		trained := make(chan struct{})
		go w.observeLearningRate(trained)

		docCh := make(chan []int)
		sem := semaphore.NewWeighted(int64(w.opts.ModelOptions.ThreadSize))
		wg := &sync.WaitGroup{}

		go w.corpus.Load(docCh, w.opts.CorpusOptions)
		for d := range docCh {
			wg.Add(1)
			go w.trainPerThread(d, trained, sem, wg)
		}

		wg.Wait()
		close(trained)
	}
	return nil
}

func (w *word2vec) trainPerThread(
	doc []int,
	trained chan struct{},
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

	for pos, id := range doc {
		bernoulliTrial := rand.Float64()
		p := w.subSamples[id]
		if p < bernoulliTrial {
			continue
		}
		w.mod.trainOne(doc, pos, w.currentlr, w.param, w.optimizer)
		trained <- struct{}{}
	}

	return nil
}

func (w *word2vec) observeLearningRate(trained chan struct{}) {
	cnt, clk := 0, clock.New()
	for range trained {
		cnt++
		if cnt%10000 == 0 {
			lower := w.opts.ModelOptions.Initlr * w.opts.Theta
			if w.currentlr < lower {
				w.currentlr = lower
			} else {
				w.currentlr = w.opts.ModelOptions.Initlr * (1.0 - float64(cnt)/float64(w.corpus.Len()))
			}
			w.verbose.Do(func() {
				fmt.Printf("trained %d words %v\r", cnt, clk.AllElapsed())
			})
		}
	}
	w.verbose.Do(func() {
		fmt.Printf("trained %d words %v\r\n", cnt, clk.AllElapsed())
	})
}

func (w *word2vec) Save(f io.Writer, typ save.VectorType) error {
	writer := bufio.NewWriter(f)
	defer writer.Flush()

	dic := w.corpus.Dictionary()
	var ctx *matrix.Matrix
	ng, ok := w.optimizer.(*negativeSampling)
	if ok {
		ctx = ng.ctx
	}

	var buf bytes.Buffer
	clk := clock.New()
	for i := 0; i < dic.Len(); i++ {
		word, _ := dic.Word(i)
		fmt.Fprintf(&buf, "%v ", word)
		for j := 0; j < w.opts.ModelOptions.Dim; j++ {
			var v float64
			switch {
			case typ == save.AggregatedVector && ctx.Row() > i:
				v = w.param.Slice(i)[j] + ctx.Slice(i)[j]
			case typ == save.SingleVector:
				v = w.param.Slice(i)[j]
			default:
				return save.InvalidVectorTypeError(typ)
			}
			fmt.Fprintf(&buf, "%f ", v)
		}
		fmt.Fprintln(&buf)
		w.verbose.Do(func() {
			fmt.Printf("save %d words %v\r", i, clk.AllElapsed())
		})
	}
	writer.WriteString(fmt.Sprintf("%v", buf.String()))
	return nil
}
