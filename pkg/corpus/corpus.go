package corpus

import (
	"bufio"
	"fmt"
	"io"
	"strings"

	"github.com/pkg/errors"
	"github.com/ynqa/wego/pkg/clock"
	"github.com/ynqa/wego/pkg/corpus/dictionary"
	"github.com/ynqa/wego/pkg/verbose"
)

type Corpus struct {
	dic *dictionary.Dictionary
	doc []int

	verbose *verbose.Verbose
}

func New(verbose *verbose.Verbose) *Corpus {
	return &Corpus{
		dic: dictionary.New(),
		doc: make([]int, 0),

		verbose: verbose,
	}
}

func (c *Corpus) Len() int {
	return len(c.doc)
}

func (c *Corpus) Dictionary() *dictionary.Dictionary {
	return c.dic
}

func (c *Corpus) Read(r io.Reader, opts Options) error {
	scanner := bufio.NewScanner(r)
	scanner.Split(bufio.ScanWords)

	var pos int
	clock := clock.New()
	for scanner.Scan() {
		word := scanner.Text()
		if opts.ToLower {
			word = strings.ToLower(word)
		}

		c.dic.Add(word)
		id, _ := c.dic.ID(word)
		c.doc = append(c.doc, id)

		c.verbose.Do(func() {
			if pos%10000 == 0 {
				fmt.Printf("read %d words %v\r", pos, clock.AllElapsed())
			}
			pos++
		})
	}
	if err := scanner.Err(); err != nil && err != io.EOF {
		return errors.Wrap(err, "Unable to complete scanning")
	}

	c.verbose.Do(func() {
		fmt.Printf("read %d words %v\r\n", pos, clock.AllElapsed())
	})
	return nil
}

func (c *Corpus) Load(ch chan []int, opts Options) {
	tmp, idx := make([]int, opts.BatchSize), 0
	for _, id := range c.doc {
		if c.dic.IDFreq(id) > opts.MinCount {
			tmp[idx] = id
			idx++
		}
		if idx == opts.BatchSize {
			ch <- tmp
			tmp, idx = make([]int, opts.BatchSize), 0
		}
	}
	ch <- tmp
	close(ch)
}
