NVCC ?= nvcc
FLAGS ?= -O3 -std=c++17
SRCS := $(wildcard *.cu)
BINS := $(SRCS:.cu=)

all: $(BINS)

%: %.cu
	$(NVCC) $(FLAGS) $< -o $@

run: all
	for b in $(BINS); do echo "== $$b"; ./$$b; done

clean:
	rm -f $(BINS)
