# Makefile

EXE=d2q9-bgk

CC=icc
CFLAGS= -std=c99 -Wall -Ofast -march=broadwell
LIBS = -lm

PLATFORM = $(shell uname -s)
ifeq ($(PLATFORM), Darwin)
	LIBS += -framework OpenCL
else
	LIBS += -lOpenCL
endif

FINAL_STATE_FILE=./final_state.dat
AV_VELS_FILE=./av_vels.dat
REF_FINAL_STATE_FILE=check/128x128.final_state.dat
REF_AV_VELS_FILE=check/128x128.av_vels.dat

all: $(EXE)

$(EXE): $(EXE).c
	$(CC) $(CFLAGS) $^ $(LIBS) -o $@

check:
	python check/check.py --ref-av-vels-file=$(REF_AV_VELS_FILE) --ref-final-state-file=$(REF_FINAL_STATE_FILE) --av-vels-file=$(AV_VELS_FILE) --final-state-file=$(FINAL_STATE_FILE)

.PHONY: all check clean

clean:
	rm -f $(EXE)

cuda: 
	nvcc -o d2q9-bgk d2q9-bgk.cu