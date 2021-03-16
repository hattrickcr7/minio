package lrc

import (
	"bytes"
	"github.com/minio/minio/cmd/crypto"
	"io"
	"runtime"
	"sync"
	"errors"

	"github.com/klauspost/cpuid/v2"
)


// ErrTooFewShards is returned if too few shards where given to
// Encode/Verify/Reconstruct/Update. It will also be returned from Reconstruct
// if there were too few shards to reconstruct the missing data.
var ErrTooFewShards = errors.New("too few shards given")

// ErrShortData will be returned by Split(), if there isn't enough data
// to fill the number of shards.
var ErrShortData = errors.New("not enough data to fill the number of requested shards")

// ErrTooFewShards is returned if too few shards where given to
// Encode/Verify/Reconstruct/Update. It will also be returned from Reconstruct
// if there were too few shards to reconstruct the missing data.
var ErrTooFewShards = errors.New("too few shards given")

// Encoder is an interface to encode Reed-Salomon parity sets for your data.
type Encoder interface {
	// Encode parity for a set of data shards.
	// Input is 'shards' containing data shards followed by parity shards.
	// The number of shards must match the number given to New().
	// Each shard is a byte array, and they must all be the same size.
	// The parity shards will always be overwritten and the data shards
	// will remain the same, so it is safe for you to read from the
	// data shards while this is running.
	Encode(shards [][]byte) error

	// Verify returns true if the parity shards contain correct data.
	// The data is the same format as Encode. No data is modified, so
	// you are allowed to read from data while this is running.
	Verify(shards [][]byte) (bool, error)

	// Reconstruct will recreate the missing shards if possible.
	//
	// Given a list of shards, some of which contain data, fills in the
	// ones that don't have data.
	//
	// The length of the array must be equal to the total number of shards.
	// You indicate that a shard is missing by setting it to nil or zero-length.
	// If a shard is zero-length but has sufficient capacity, that memory will
	// be used, otherwise a new []byte will be allocated.
	//
	// If there are too few shards to reconstruct the missing
	// ones, ErrTooFewShards will be returned.
	//
	// The reconstructed shard set is complete, but integrity is not verified.
	// Use the Verify function to check if data set is ok.
	Reconstruct(shards [][]byte) error

	// ReconstructData will recreate any missing data shards, if possible.
	//
	// Given a list of shards, some of which contain data, fills in the
	// data shards that don't have data.
	//
	// The length of the array must be equal to Shards.
	// You indicate that a shard is missing by setting it to nil or zero-length.
	// If a shard is zero-length but has sufficient capacity, that memory will
	// be used, otherwise a new []byte will be allocated.
	//
	// If there are too few shards to reconstruct the missing
	// ones, ErrTooFewShards will be returned.
	//
	// As the reconstructed shard set may contain missing parity shards,
	// calling the Verify function is likely to fail.
	ReconstructData(shards [][]byte) error

	// Update parity is use for change a few data shards and update it's parity.
	// Input 'newDatashards' containing data shards changed.
	// Input 'shards' containing old data shards (if data shard not changed, it can be nil) and old parity shards.
	// new parity shards will in shards[DataShards:]
	// Update is very useful if  DataShards much larger than ParityShards and changed data shards is few. It will
	// faster than Encode and not need read all data shards to encode.
	Update(shards [][]byte, newDatashards [][]byte) error

	// Split a data slice into the number of shards given to the encoder,
	// and create empty parity shards.
	//
	// The data will be split into equally sized shards.
	// If the data size isn't dividable by the number of shards,
	// the last shard will contain extra zeros.
	//
	// There must be at least 1 byte otherwise ErrShortData will be
	// returned.
	//
	// The data will not be copied, except for the last shard, so you
	// should not modify the data of the input slice afterwards.
	Split(data []byte) ([][]byte, error)

	// Join the shards and write the data segment to dst.
	//
	// Only the data shards are considered.
	// You must supply the exact output size you want.
	// If there are to few shards given, ErrTooFewShards will be returned.
	// If the total data size is less than outSize, ErrShortData will be returned.
	Join(dst io.Writer, shards [][]byte, outSize int) error
}

type lrc struct {
	DataShards   int // Number of data shards, should not be modified.
	LocalGroup   int //
	ParityShards int // Number of parity shards, should not be modified.
	Shards       int // Total number of shards. Calculated, and should not be modified.
	m            matrix
	tree         *inversionTree
	parity       [][]byte
	o            options
	mPool        sync.Pool
}

func (l *lrc) Encode(shards [][]byte) error {
	if len(shards) != l.Shards {
		return ErrTooFewShards
	}

	err := checkShards(shards, false)
	if err != nil {
		return err
	}

	// Get the slice of output buffers.
	output := shards[l.DataShards:]

	// Do the coding.
	l.codeSomeShards(l.parity, shards[0:l.DataShards], output, l.ParityShards+l.LocalGroup, len(shards[0]))
	return nil
}

// Multiplies a subset of rows from a coding matrix by a full set of
// input shards to produce some output shards.
// 'matrixRows' is The rows from the matrix to use.
// 'inputs' An array of byte arrays, each of which is one input shard.
// The number of inputs used is determined by the length of each matrix row.
// outputs Byte arrays where the computed shards are stored.
// The number of outputs computed, and the
// number of matrix rows used, is determined by
// outputCount, which is the number of outputs to compute.
func (l *lrc) codeSomeShards(matrixRows, inputs, outputs [][]byte, outputCount, byteCount int) {
	if len(outputs) == 0 {
		return
	}

	// Process using no goroutines
	start, end := 0, l.o.perRound
	if end > len(inputs[0]) {
		end = len(inputs[0])
	}

	for start < len(inputs[0]) {
		for c := 0; c < l.DataShards; c++ {
			in := inputs[c][start:end]
			for iRow := 0; iRow < outputCount; iRow++ {
				if c == 0 {
					galMulSlice(matrixRows[iRow][c], in, outputs[iRow][start:end], &l.o)
				} else {
					galMulSliceXor(matrixRows[iRow][c], in, outputs[iRow][start:end], &l.o)
				}
			}
		}
		start = end
		end += l.o.perRound
		if end > len(inputs[0]) {
			end = len(inputs[0])
		}
	}
}

func (l *lrc) Verify(shards [][]byte) (bool, error) {
	if len(shards) != l.Shards {
		return false, ErrTooFewShards
	}
	err := checkShards(shards, false)
	if err != nil {
		return false, err
	}

	// Slice of buffers being checked.
	toCheck := shards[l.DataShards:]

	// Do the checking.
	return l.checkSomeShards(l.parity, shards[0:l.DataShards], toCheck, l.ParityShards+l.LocalGroup, len(shards[0])), nil
}

func (l *lrc) checkSomeShards(matrixRows, inputs, toCheck [][]byte, outputCount, byteCount int) bool {
	outputs := make([][]byte, len(toCheck))
	for i := range outputs {
		outputs[i] = make([]byte, byteCount)
	}
	for c := 0; c < l.DataShards; c++ {
		in := inputs[c]
		for iRow := 0; iRow < outputCount; iRow++ {
			galMulSliceXor(matrixRows[iRow][c], in, outputs[iRow], &l.o)
		}
	}

	for i, calc := range outputs {
		if !bytes.Equal(calc, toCheck[i]) {
			return false
		}
	}
	return true
}

func (l *lrc) Reconstruct(shards [][]byte) error {
	panic("implement me")
}

func (l *lrc)reconstruct(shards [][]byte, dataOnly bool) error {
	if len(shards) != l.Shards {
		return ErrTooFewShards
	}
	// Check arguments.
	err := checkShards(shards, true)
	if err != nil {
		return err
	}

	shardSize := shardSize(shards)

	// Quick check: are all of the shards present?  If so, there's
	// nothing to do.
	numberPresent := 0
	dataPresent := 0
	group1:=make(map[int]struct{})
	group2:=make(map[int]struct{})
	group1[1]= struct{}{}
	group1[2]= struct{}{}
	group1[5]= struct{}{}
	group2[3]= struct{}{}
	group2[4]= struct{}{}
	group2[6]= struct{}{}
	group1Present:=make([]int,0)
	group2Present:=make([]int,0)
	for i := 0; i < l.Shards; i++ {
		if len(shards[i]) != 0 {
			numberPresent++
			if i < l.DataShards {
				dataPresent++
			}
		}
		if _,ok:=group1[i];ok {
			group1Present=append(group1Present, i)
		}
		if _,ok:=group2[i];ok {
			group2Present=append(group2Present, i)
		}
	}
	if numberPresent == l.Shards || dataOnly && dataPresent == l.DataShards {
		// Cool.  All of the shards data data.  We don't
		// need to do anything.
		return nil
	}

	groupShards:=make([][]byte,3)
	subMatrix,err:=newMatrix(2,2)
	if err != nil {
		return err
	}
	for k,v :=range group1Present{

	}
	if len(group1Present)==2 {
		reconstructGroup()
	}
}

func reconstructGroup(input,output [][]byte,m matrix)  {

}

func (l *lrc) ReconstructData(shards [][]byte) error {
	panic("implement me")
}

func (l *lrc) Update(shards [][]byte, newDatashards [][]byte) error {
	panic("implement me")
}

func (l *lrc) Split(data []byte) ([][]byte, error) {
	if len(data) == 0 {
		return nil, ErrShortData
	}
	// Calculate number of bytes per data shard.
	perShard := (len(data) + l.DataShards - 1) / l.DataShards

	if cap(data) > len(data) {
		data = data[:cap(data)]
	}

	// Only allocate memory if necessary
	var padding []byte
	if len(data) < (l.Shards * perShard) {
		// calculate maximum number of full shards in `data` slice
		fullShards := len(data) / perShard
		padding = make([]byte, l.Shards*perShard-perShard*fullShards)
		copy(padding, data[perShard*fullShards:])
		data = data[0 : perShard*fullShards]
	}

	// Split into equal-length shards.
	dst := make([][]byte, l.Shards)
	i := 0
	for ; i < len(dst) && len(data) >= perShard; i++ {
		dst[i] = data[:perShard:perShard]
		data = data[perShard:]
	}

	for j := 0; i+j < len(dst); j++ {
		dst[i+j] = padding[:perShard:perShard]
		padding = padding[perShard:]
	}

	return dst, nil
}

func (l *lrc) Join(dst io.Writer, shards [][]byte, outSize int) error {
	panic("implement me")
}

// ErrInvShardNum will be returned by New, if you attempt to create
// an Encoder where either data or parity shards is zero or less.
var ErrInvShardNum = errors.New("cannot create Encoder with zero or less data/parity shards")

// ErrMaxShardNum will be returned by New, if you attempt to create an
// Encoder where data and parity shards are bigger than the order of
// GF(2^8).
var ErrMaxShardNum = errors.New("cannot create Encoder with more than 256 data+parity shards")

func buildMatrix(dataShards, totalShards int) (matrix, error) {
	// Start with a Vandermonde matrix.  This matrix would work,
	// in theory, but doesn't have the property that the data
	// shards are unchanged after encoding.
	vm, err := vandermonde(totalShards, dataShards)
	if err != nil {
		return nil, err
	}

	// Multiply by the inverse of the top square of the matrix.
	// This will make the top square be the identity matrix, but
	// preserve the property that any square subset of rows is
	// invertible.
	top, err := vm.SubMatrix(0, 0, dataShards, dataShards)
	if err != nil {
		return nil, err
	}

	topInv, err := top.Invert()
	if err != nil {
		return nil, err
	}
	result,err:=vm.Multiply(topInv)
	if err != nil {
		return result,err
	}
	ma,err:=result.lrcaug(0,4,6)
	if err != nil {
		return result, err
	}
	return ma,nil;
}

func New(dataShards, localgroup, parityShards int, opts ...Option) (Encoder, error) {
	r := lrc{
		DataShards:   dataShards,
		LocalGroup:   localgroup,
		ParityShards: parityShards,
		Shards:       dataShards + parityShards + localgroup,
		o:            defaultOptions,
	}

	for _, opt := range opts {
		opt(&r.o)
	}
	if dataShards <= 0 || parityShards <= 0 {
		return nil, ErrInvShardNum
	}

	if dataShards+parityShards > 256 {
		return nil, ErrMaxShardNum
	}

	var err error
	r.m, err = buildMatrix(dataShards, r.Shards)
	if err != nil {
		return nil, err
	}

	// Calculate what we want per round
	r.o.perRound = cpuid.CPU.Cache.L2
	if r.o.perRound <= 0 {
		// Set to 128K if undetectable.
		r.o.perRound = 128 << 10
	}

	if cpuid.CPU.ThreadsPerCore > 1 && r.o.maxGoroutines > cpuid.CPU.PhysicalCores {
		// If multiple threads per core, make sure they don't contend for cache.
		r.o.perRound /= cpuid.CPU.ThreadsPerCore
	}
	// 1 input + parity must fit in cache, and we add one more to be safer.
	r.o.perRound = r.o.perRound / (1 + parityShards)
	// Align to 64 bytes.
	r.o.perRound = ((r.o.perRound + 63) / 64) * 64

	if r.o.minSplitSize <= 0 {
		// Set minsplit as high as we can, but still have parity in L1.
		cacheSize := cpuid.CPU.Cache.L1D
		if cacheSize <= 0 {
			cacheSize = 32 << 10
		}

		r.o.minSplitSize = cacheSize / (parityShards + 1)
		// Min 1K
		if r.o.minSplitSize < 1024 {
			r.o.minSplitSize = 1024
		}
	}

	if r.o.perRound < r.o.minSplitSize {
		r.o.perRound = r.o.minSplitSize
	}

	if r.o.shardSize > 0 {
		p := runtime.GOMAXPROCS(0)
		if p == 1 || r.o.shardSize <= r.o.minSplitSize*2 {
			// Not worth it.
			r.o.maxGoroutines = 1
		} else {
			g := r.o.shardSize / r.o.perRound

			// Overprovision by a factor of 2.
			if g < p*2 && r.o.perRound > r.o.minSplitSize*2 {
				g = p * 2
				r.o.perRound /= 2
			}

			// Have g be multiple of p
			g += p - 1
			g -= g % p

			r.o.maxGoroutines = g
		}
	}

	// Inverted matrices are cached in a tree keyed by the indices
	// of the invalid rows of the data to reconstruct.
	// The inversion root node will have the identity matrix as
	// its inversion matrix because it implies there are no errors
	// with the original data.
	if r.o.inversionCache {
		r.tree = newInversionTree(dataShards, parityShards)
	}

	r.parity = make([][]byte, parityShards+localgroup)
	for i := range r.parity {
		r.parity[i] = r.m[dataShards+i]
	}

	//if avx2CodeGen && r.o.useAVX2 {
	//	r.mPool.New = func() interface{} {
	//		return make([]byte, r.Shards*2*32)
	//	}
	//}
	return &r, err
}

// shardSize return the size of a single shard.
// The first non-zero size is returned,
// or 0 if all shards are size 0.
func shardSize(shards [][]byte) int {
	for _, shard := range shards {
		if len(shard) != 0 {
			return len(shard)
		}
	}
	return 0
}

// ErrShardNoData will be returned if there are no shards,
// or if the length of all shards is zero.
var ErrShardNoData = errors.New("no shard data")

// ErrShardSize is returned if shard length isn't the same for all
// shards.
var ErrShardSize = errors.New("shard sizes do not match")

func checkShards(shards [][]byte, nilok bool) error {
	size := shardSize(shards)
	if size == 0 {
		return ErrShardNoData
	}
	for _, shard := range shards {
		if len(shard) != size {
			if len(shard) != 0 || !nilok {
				return ErrShardSize
			}
		}
	}
	return nil
}