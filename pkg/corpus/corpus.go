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
	opts   Options
	dic    *dictionary.Dictionary
	maxLen int

	doc []int

	verbose *verbose.Verbose
}

func New(
	opts Options,
	verbose *verbose.Verbose,
) *Corpus {
	return &Corpus{
		opts: opts,
		dic:  dictionary.New(),

		doc: make([]int, 0),

		verbose: verbose,
	}
}

func (c *Corpus) Dictionary() *dictionary.Dictionary {
	return c.dic
}

func (c *Corpus) Len() int {
	return c.maxLen
}

func (c *Corpus) Read(r io.Reader) error {
	scanner := bufio.NewScanner(r)
	scanner.Split(bufio.ScanWords)

	cnt, clk := 0, clock.New()
	for scanner.Scan() {
		word := scanner.Text()
		if c.opts.ToLower {
			word = strings.ToLower(word)
		}

		c.dic.Add(word)
		id, _ := c.dic.ID(word)
		c.doc = append(c.doc, id)
		c.maxLen++

		c.verbose.Do(func() {
			if cnt%100000 == 0 {
				fmt.Printf("read %d words %v\r", cnt, clk.AllElapsed())
			}
			cnt++
		})
	}
	if err := scanner.Err(); err != nil && err != io.EOF {
		return errors.Wrap(err, "failed to scan")
	}

	c.verbose.Do(func() {
		fmt.Printf("read %d words %v\r\n", cnt, clk.AllElapsed())
	})
	return nil
}

func (c *Corpus) BatchDocument(closable chan []int, batchSize int) {
	var tmp []int
	for _, id := range c.doc {
		if len(tmp) == batchSize {
			closable <- tmp
			tmp = make([]int, 0)
		}
		tmp = append(tmp, id)
	}
	closable <- tmp
	close(closable)
}
