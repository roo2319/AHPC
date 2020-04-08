# Add any `module load` or `export` commands that your code needs to
# compile and run to this file.
module load languages/intel/2018-u3
module load libs/cuda/10.0-gcc-5.4.0-2.26
export OMP_PROC_BIND=close
