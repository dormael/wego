// Copyright © 2017 Makoto Ito
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package glove

import (
	"github.com/spf13/cobra"

	"github.com/ynqa/wego/pkg/corpus"
	"github.com/ynqa/wego/pkg/model"
)

type SolverType string

const (
	Stochastic SolverType = "sgd"
	AdaGrad    SolverType = "adagrad"
)

func (t *SolverType) String() string {
	return string(*t)
}

func (t *SolverType) Set(string) error {
	return nil
}

func (t *SolverType) Type() string {
	return t.String()
}

const (
	defaultAlpha      = 0.75
	defaultSolverType = Stochastic
	defaultXmax       = 100
)

type Options struct {
	CorpusOptions corpus.Options
	ModelOptions  model.Options

	Alpha      float64
	SolverType SolverType
	Xmax       int
}

func LoadForCmd(cmd *cobra.Command, opts *Options) {
	cmd.Flags().Float64Var(&opts.Alpha, "alpha", defaultAlpha, "exponent of weighting function")
	cmd.Flags().Var(&opts.SolverType, "solver", "solver for GloVe objective. One of: sgd|adagrad")
	cmd.Flags().IntVar(&opts.Xmax, "xmax", defaultXmax, "specifying cutoff in weighting function")
}

type ModelOption func(*Options)

// corpus options
func ToLower() ModelOption {
	return ModelOption(func(opts *Options) {
		opts.CorpusOptions.ToLower = true
	})
}

func WithMinCount(v int) ModelOption {
	return ModelOption(func(opts *Options) {
		opts.CorpusOptions.MinCount = v
	})
}

// model options
func WithDimension(v int) ModelOption {
	return ModelOption(func(opts *Options) {
		opts.ModelOptions.Dim = v
	})
}

func WithInitLearningRate(v float64) ModelOption {
	return ModelOption(func(opts *Options) {
		opts.ModelOptions.Initlr = v
	})
}

func WithIteration(v int) ModelOption {
	return ModelOption(func(opts *Options) {
		opts.ModelOptions.Iter = v
	})
}

func WithThreadSize(v int) ModelOption {
	return ModelOption(func(opts *Options) {
		opts.ModelOptions.ThreadSize = v
	})
}

func WithWindow(v int) ModelOption {
	return ModelOption(func(opts *Options) {
		opts.ModelOptions.Window = v
	})
}

func Verbose() ModelOption {
	return ModelOption(func(opts *Options) {
		opts.ModelOptions.Verbose = true
	})
}

// for glove options
func WithAlpha(v float64) ModelOption {
	return ModelOption(func(opts *Options) {
		opts.Alpha = v
	})
}

func WithSolver(typ SolverType) ModelOption {
	return ModelOption(func(opts *Options) {
		opts.SolverType = typ
	})
}

func WithXmax(v int) ModelOption {
	return ModelOption(func(opts *Options) {
		opts.Xmax = v
	})
}

func New(opts ...ModelOption) (model.Model, error) {
	options := Options{
		CorpusOptions: corpus.DefaultOption(),
		ModelOptions:  model.DefaultOptions(),

		Alpha:      defaultAlpha,
		SolverType: defaultSolverType,
		Xmax:       defaultXmax,
	}

	for _, fn := range opts {
		fn(&options)
	}

	return NewForOptions(options)
}

func NewForOptions(opts Options) (model.Model, error) {
	return nil, nil
}
