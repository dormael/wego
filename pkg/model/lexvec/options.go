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

package lexvec

import (
	"github.com/spf13/cobra"

	"github.com/ynqa/wego/pkg/corpus"
	"github.com/ynqa/wego/pkg/model"
)

type RelationType string

const (
	PPMI           RelationType = "ppmi"
	PMI            RelationType = "pmi"
	Collocation    RelationType = "co"
	LogCollocation RelationType = "logco"
)

func (t *RelationType) String() string {
	return string(*t)
}

func (t *RelationType) Set(string) error {
	return nil
}

func (t *RelationType) Type() string {
	return t.String()
}

const (
	defaultNegativeSampleSize = 5
	defaultRelationType       = PPMI
	defaultSmooth             = 0.75
	defaultSubsampleThreshold = 1.0e-3
	defaultTheta              = 1.0e-4
)

type Options struct {
	CorpusOptions corpus.Options
	ModelOptions  model.Options

	NegativeSampleSize int
	RelationType       RelationType
	Smooth             float64
	SubsampleThreshold float64
	Theta              float64
}

func LoadForCmd(cmd *cobra.Command, opts *Options) {
	cmd.Flags().IntVar(&opts.NegativeSampleSize, "sample", defaultNegativeSampleSize, "negative sample size")
	cmd.Flags().Var(&opts.RelationType, "rel", "relation type for counting co-occurrence. One of ppmi|pmi|co|logco")
	cmd.Flags().Float64Var(&opts.Smooth, "smooth", defaultSmooth, "smoothing value for co-occurence value")
	cmd.Flags().Float64Var(&opts.SubsampleThreshold, "threshold", defaultSubsampleThreshold, "threshold for subsampling")
	cmd.Flags().Float64Var(&opts.Theta, "theta", defaultTheta, "lower limit of learning rate (lr >= initlr * theta)")
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

// for lexvec options
func WithNegativeSampleSize(v int) ModelOption {
	return ModelOption(func(opts *Options) {
		opts.NegativeSampleSize = v
	})
}

func WithRelation(typ RelationType) ModelOption {
	return ModelOption(func(opts *Options) {
		opts.RelationType = typ
	})
}

func WithSmooth(v float64) ModelOption {
	return ModelOption(func(opts *Options) {
		opts.Smooth = v
	})
}

func WithSubsampleThreshold(v float64) ModelOption {
	return ModelOption(func(opts *Options) {
		opts.SubsampleThreshold = v
	})
}

func WithTheta(v float64) ModelOption {
	return ModelOption(func(opts *Options) {
		opts.Theta = v
	})
}

func New(opts ...ModelOption) (model.Model, error) {
	options := Options{
		CorpusOptions: corpus.DefaultOption(),
		ModelOptions:  model.DefaultOptions(),

		NegativeSampleSize: defaultNegativeSampleSize,
		RelationType:       defaultRelationType,
		Smooth:             defaultSmooth,
		SubsampleThreshold: defaultSubsampleThreshold,
		Theta:              defaultTheta,
	}

	for _, fn := range opts {
		fn(&options)
	}

	return NewForOptions(options)
}

func NewForOptions(opts Options) (model.Model, error) {
	return nil, nil
}
