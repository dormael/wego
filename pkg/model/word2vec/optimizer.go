package word2vec

import (
	"math/rand"

	"github.com/ynqa/wego/pkg/corpus/dictionary"
	"github.com/ynqa/wego/pkg/corpus/dictionary/node"
	"github.com/ynqa/wego/pkg/model/modelutil"
	"github.com/ynqa/wego/pkg/model/modelutil/matrix"
)

type optimizer interface {
	optim(id int, lr float64, ctx, tmp []float64)
}

type negativeSampling struct {
	ctx        *matrix.Matrix
	sigtable   *sigmoidTable
	sampleSize int
}

func newNegativeSampling(dic *dictionary.Dictionary, opts Options) optimizer {
	dim := opts.ModelOptions.Dim
	return &negativeSampling{
		ctx: matrix.New(
			dic.Len(),
			dim,
			func(vec []float64) {
				for i := 0; i < dim; i++ {
					vec[i] = (rand.Float64() - 0.5) / float64(dim)
				}
			},
		),
		sigtable:   newSigmoidTable(),
		sampleSize: opts.NegativeSampleSize,
	}
}

func (opt *negativeSampling) optim(
	id int,
	lr float64,
	ctx, tmp []float64,
) {
	var (
		label  int
		picked int
	)
	dim := len(ctx)
	for n := -1; n < opt.sampleSize; n++ {
		if n == -1 {
			label = 1
			picked = id
		} else {
			label = 0
			picked = modelutil.NextRandom(opt.ctx.Row())
			if id == picked {
				continue
			}
		}
		rnd := opt.ctx.Slice(picked)
		var inner float64
		for i := 0; i < dim; i++ {
			inner += rnd[i] * ctx[i]
		}
		var g float64
		if inner <= -opt.sigtable.maxExp {
			g = (float64(label - 0)) * lr
		} else if inner >= opt.sigtable.maxExp {
			g = (float64(label - 1)) * lr
		} else {
			g = (float64(label) - opt.sigtable.sigmoid(inner)) * lr
		}
		for i := 0; i < dim; i++ {
			tmp[i] += g * rnd[i]
			rnd[i] += g * ctx[i]
		}
	}
}

type hierarchicalSoftmax struct {
	sigtable *sigmoidTable
	nodeset  []*node.Node
	maxDepth int
}

func newHierarchicalSoftmax(dic *dictionary.Dictionary, opts Options) optimizer {
	return &hierarchicalSoftmax{
		sigtable: newSigmoidTable(),
		nodeset:  dic.HuffnamTree(opts.ModelOptions.Dim),
		maxDepth: opts.MaxDepth,
	}
}

func (opt *hierarchicalSoftmax) optim(
	id int,
	lr float64,
	ctx, tmp []float64,
) {
	path := opt.nodeset[id].GetPath(opt.maxDepth)
	for i := 0; i < len(path)-1; i++ {
		p := path[i]
		childCode := path[i+1].Code
		var inner float64
		for j := 0; j < len(p.Vector); j++ {
			inner += ctx[j] * p.Vector[j]
		}
		if inner <= -opt.sigtable.maxExp || inner >= opt.sigtable.maxExp {
			return
		}
		g := (1.0 - float64(childCode) - opt.sigtable.sigmoid(inner)) * lr
		for j := 0; j < len(p.Vector); j++ {
			tmp[j] += g * p.Vector[j]
			p.Vector[j] += g * ctx[j]
		}
	}
}
