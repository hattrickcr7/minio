package lrc

import (
	"math/rand"
	"testing"
)

func testOptions(o ...Option) []Option {
	return o
}

func TestEncoding(t *testing.T) {
	t.Run("default", func(t *testing.T) {
		testEncoding(t, testOptions()...)
	})
}

func testEncoding(t *testing.T, o ...Option) {
	data, parity := 4, 4
	rng := rand.New(rand.NewSource(0xabadc0cac01a))
	perShard := 100003

	r, err := New(data,2 , parity, testOptions(o...)...)
	if err != nil {
		t.Fatal(err)
	}
	shards := make([][]byte, data+parity+2)
	for s := range shards {
		shards[s] = make([]byte, perShard)
	}

	for s := 0; s < data; s++ {
		rng.Read(shards[s])
	}

	err = r.Encode(shards)
	if err != nil {
		t.Fatal(err)
	}
	ok, err := r.Verify(shards)
	if err != nil {
		t.Fatal(err)
	}
	if !ok {
		t.Fatal("Verification failed")
	}
}
