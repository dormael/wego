package corpus

import (
	"github.com/spf13/cobra"
)

const (
	defaultBatchSize = 10000
	defaultMinCount  = 5
	defaultToLower   = false
)

type Options struct {
	BatchSize int
	MinCount  int
	ToLower   bool
}

func DefaultOption() Options {
	return Options{
		BatchSize: 0,
		MinCount:  defaultMinCount,
		ToLower:   defaultToLower,
	}
}

func LoadOptionsForCmd(cmd *cobra.Command, opts *Options) {
	cmd.Flags().IntVar(&opts.BatchSize, "batch", defaultBatchSize, "batch size to train")
	cmd.Flags().IntVar(&opts.MinCount, "min-count", defaultMinCount, "lower limit to filter rare words")
	cmd.Flags().BoolVar(&opts.ToLower, "lower", defaultToLower, "whether the words on corpus convert to lowercase or not")
}
