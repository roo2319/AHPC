/*
** Code to implement a d2q9-bgk lattice boltzmann scheme.
** 'd2' inidates a 2-dimensional grid, and
** 'q9' indicates 9 velocities per grid cell.
** 'bgk' refers to the Bhatnagar-Gross-Krook collision step.
**
** The 'speeds' in each cell are numbered as follows:
**
** 6 2 5
**  \|/
** 3-0-1
**  /|\
** 7 4 8
**
** A 2D grid:
**
**           cols
**       --- --- ---
**      | D | E | F |
** rows  --- --- ---
**      | A | B | C |
**       --- --- ---
**
** 'unwrapped' in row major order to give a 1D array:
**
**  --- --- --- --- --- ---
** | A | B | C | D | E | F |
**  --- --- --- --- --- ---
**
** Grid indicies are:
**
**          ny
**          ^       cols(ii)
**          |  ----- ----- -----
**          | | ... | ... | etc |
**          |  ----- ----- -----
** rows(jj) | | 1,0 | 1,1 | 1,2 |
**          |  ----- ----- -----
**          | | 0,0 | 0,1 | 0,2 |
**          |  ----- ----- -----
**          ----------------------> nx
**
** Note the names of the input parameter and obstacle files
** are passed on the command line, e.g.:
**
**   ./d2q9-bgk input.params obstacles.dat
**
** Be sure to adjust the grid dimensions in the parameter file
** if you choose a different obstacle file.
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include <sys/time.h>
#include <sys/resource.h>

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <cuda.h>
#endif

#define INDEX(ii,jj,sp,nx,ny) ((ii)+(jj)*(nx)+(sp)*(nx)*(ny))

#define NSPEEDS         9
#define FINALSTATEFILE  "final_state.dat"
#define AVVELSFILE      "av_vels.dat"

/* struct to hold the parameter values */
typedef struct
{
  int    nx;            /* no. of cells in x-direction */
  int    ny;            /* no. of cells in y-direction */
  int    maxIters;      /* no. of iterations */
  int    reynolds_dim;  /* dimension for Reynolds number */
  float density;       /* density per link */
  float accel;         /* density redistribution */
  float omega;         /* relaxation parameter */
  float w1;
  float w2;
  int free_cells;

  int reduction_count;
  int reduction_cap;

  unsigned int nworkgroupsX;
  unsigned int nworkgroupsY;
  unsigned int localnx;
  unsigned int localny;
} t_param;

/* struct to hold OpenCL objects */
typedef struct
{
  float* cells;
  float* tmp_cells;
  int* obstacles;
  float* partial_sums;
  float* averages;
} t_cuda;

__global__
void reduction(float* partial_sum,
  float* averages,
  int nGroups,
  int freeCells)
{
  int id = blockIdx.x *blockDim.x + threadIdx.x;
  float total = 0;
  for (int i = 0; i < nGroups; i++){
  total += partial_sum[i + (nGroups * id)];
  }
  averages[id] = total/(float) freeCells;
}


__global__
void accelerate_flow(float* cells,
  int* obstacles,
  int nx, int ny,
  float w1, float w2)
{
/* compute weighting factors */

/* modify the 2nd row of the grid */
int jj = ny - 2;

/* get column index */
int ii =  blockIdx.x *blockDim.x + threadIdx.x;

/* if the cell is not occupied and
** we don't send a negative density */
// 367 can be private
bool mask = (!obstacles[ii + jj* nx]
&& (cells[INDEX(ii,jj,3,nx,ny)] - w1) > 0.f
&& (cells[INDEX(ii,jj,6,nx,ny)] - w2) > 0.f
&& (cells[INDEX(ii,jj,7,nx,ny)] - w2) > 0.f);
/* increase 'east-side' densities */
cells[INDEX(ii,jj,1,nx,ny)] = mask * w1 + cells[INDEX(ii,jj,1,nx,ny)];
cells[INDEX(ii,jj,5,nx,ny)] = mask * w2 + cells[INDEX(ii,jj,5,nx,ny)];
cells[INDEX(ii,jj,8,nx,ny)] = mask * w2 + cells[INDEX(ii,jj,8,nx,ny)];
/* decrease 'west-side' densities */
cells[INDEX(ii,jj,3,nx,ny)] = mask * -w1 + cells[INDEX(ii,jj,3,nx,ny)];
cells[INDEX(ii,jj,6,nx,ny)] = mask * -w2 + cells[INDEX(ii,jj,6,nx,ny)];
cells[INDEX(ii,jj,7,nx,ny)] = mask * -w2 + cells[INDEX(ii,jj,7,nx,ny)];
}

__global__
void lbm(float* cells,
  float* tmp_cells,
  int* obstacles,
  float* partial_sum, //stores per workgroup
  int globalnx, int globalny, int localnx, int localny, 
  float omega, int iter)
  {
  extern __shared__ float local_sum[]; //stores per thread, maybe not needed
  float c_sq = 1.f / 3.f; /* square of speed of sound */
  float w0 = 4.f / 9.f;  /* weighting factor */
  float w1 = 1.f / 9.f;  /* weighting factor */
  float w2 = 1.f / 36.f; /* weighting factor */
  float tot_u = 0;          /* accumulated magnitudes of velocity for each cell */
  float speed0,speed1,speed2,speed3,speed4,speed5,speed6,speed7,speed8;
  /* get column and row indices */
  int ii = blockIdx.x * blockDim.x + threadIdx.x;
  int jj = blockIdx.y * blockDim.y + threadIdx.y;
  int lx = threadIdx.x;
  int ly = threadIdx.y;

  int idx = ii/localnx + (globalnx/localnx) * (jj/localny);
  int offset = iter * (globalnx/localnx)* (globalny/localny);

  //Array to lookup write direction
  int indexLookup[9][2] = {{0,0},{3,1},{4,2},{1,3},{2,4},{7,5},{8,6},{5,7},{6,8}};

  int y_n = (jj + 1) % globalny;
  int x_e = (ii + 1) % globalnx;
  int y_s = (jj == 0) ? (jj + globalny - 1) : (jj - 1);
  int x_w = (ii == 0) ? (ii + globalnx - 1) : (ii - 1);
  
  speed0 = cells[INDEX(ii,jj,0,globalnx,globalny)]; /* central cell, no movement */
  speed1 = cells[INDEX(x_w,jj,1,globalnx,globalny)]; /* west */
  speed2 = cells[INDEX(ii,y_s,2,globalnx,globalny)]; /* south */
  speed3 = cells[INDEX(x_e,jj,3,globalnx,globalny)]; /* east */
  speed4 = cells[INDEX(ii,y_n,4,globalnx,globalny)]; /* north */
  speed5 = cells[INDEX(x_w,y_s,5,globalnx,globalny)]; /* south-west */
  speed6 = cells[INDEX(x_e,y_s,6,globalnx,globalny)];  /* south-east */
  speed7 = cells[INDEX(x_e,y_n,7,globalnx,globalny)]; /* north-east */
  speed8 = cells[INDEX(x_w,y_n,8,globalnx,globalny)]; /* north-west */
  
  /* compute local density total */
  float local_density = 0.f;
  
  
  local_density = speed0 + speed1 + speed2 + speed3 + speed4 + speed5 + speed6 + speed7 + speed8;


  /* compute x velocity component */
  float u_x = (speed1
  + speed5
  + speed8
  - speed3
  - speed6
  - speed7)
  / local_density;
  /* compute y velocity component */
  float u_y = (speed2
  + speed5
  + speed6
  - speed4
  - speed7
  - speed8)
  / local_density;

  /* velocity squared */
  float u_sq = u_x * u_x + u_y * u_y;

  /* directional velocity components */
  float u[NSPEEDS];
  u[1] =   u_x;        /* east */
  u[2] =         u_y;  /* north */
  u[3] = - u_x;        /* west */
  u[4] =       - u_y;  /* south */
  u[5] =   u_x + u_y;  /* north-east */
  u[6] = - u_x + u_y;  /* north-west */
  u[7] = - u_x - u_y;  /* south-west */
  u[8] =   u_x - u_y;  /* south-east */

  /* equilibrium densities */
  float d_equ[NSPEEDS];
  /* zero velocity density: weight w0 */
  d_equ[0] = w0 * local_density
  * (1.f - u_sq / (2.f * c_sq));
  /* axis speeds: weight w1 */
  d_equ[1] = w1 * local_density * (1.f + u[1] / c_sq
                      + (u[1] * u[1]) / (2.f * c_sq * c_sq)
                      - u_sq / (2.f * c_sq));
  d_equ[2] = w1 * local_density * (1.f + u[2] / c_sq
                      + (u[2] * u[2]) / (2.f * c_sq * c_sq)
                      - u_sq / (2.f * c_sq));
  d_equ[3] = w1 * local_density * (1.f + u[3] / c_sq
                      + (u[3] * u[3]) / (2.f * c_sq * c_sq)
                      - u_sq / (2.f * c_sq));
  d_equ[4] = w1 * local_density * (1.f + u[4] / c_sq
                      + (u[4] * u[4]) / (2.f * c_sq * c_sq)
                      - u_sq / (2.f * c_sq));
  /* diagonal speeds: weight w2 */
  d_equ[5] = w2 * local_density * (1.f + u[5] / c_sq
                      + (u[5] * u[5]) / (2.f * c_sq * c_sq)
                      - u_sq / (2.f * c_sq));
  d_equ[6] = w2 * local_density * (1.f + u[6] / c_sq
                      + (u[6] * u[6]) / (2.f * c_sq * c_sq)
                      - u_sq / (2.f * c_sq));
  d_equ[7] = w2 * local_density * (1.f + u[7] / c_sq
                      + (u[7] * u[7]) / (2.f * c_sq * c_sq)
                      - u_sq / (2.f * c_sq));
  d_equ[8] = w2 * local_density * (1.f + u[8] / c_sq
                      + (u[8] * u[8]) / (2.f * c_sq * c_sq)
                      - u_sq / (2.f * c_sq));

  /* relaxation step */
  bool mask = !obstacles[ii+jj*globalnx];
  tmp_cells[INDEX(ii,jj,indexLookup[0][mask],globalnx,globalny)] = speed0 + mask * (omega * (d_equ[0] - speed0));
  tmp_cells[INDEX(ii,jj,indexLookup[1][mask],globalnx,globalny)] = speed1 + mask * (omega * (d_equ[1] - speed1));
  tmp_cells[INDEX(ii,jj,indexLookup[2][mask],globalnx,globalny)] = speed2 + mask * (omega * (d_equ[2] - speed2));
  tmp_cells[INDEX(ii,jj,indexLookup[3][mask],globalnx,globalny)] = speed3 + mask * (omega * (d_equ[3] - speed3));
  tmp_cells[INDEX(ii,jj,indexLookup[4][mask],globalnx,globalny)] = speed4 + mask * (omega * (d_equ[4] - speed4));
  tmp_cells[INDEX(ii,jj,indexLookup[5][mask],globalnx,globalny)] = speed5 + mask * (omega * (d_equ[5] - speed5));
  tmp_cells[INDEX(ii,jj,indexLookup[6][mask],globalnx,globalny)] = speed6 + mask * (omega * (d_equ[6] - speed6));
  tmp_cells[INDEX(ii,jj,indexLookup[7][mask],globalnx,globalny)] = speed7 + mask * (omega * (d_equ[7] - speed7));
  tmp_cells[INDEX(ii,jj,indexLookup[8][mask],globalnx,globalny)] = speed8 + mask * (omega * (d_equ[8] - speed8));


  tot_u += sqrt((u_x * u_x) + (u_y * u_y));

  // //take to outer loop
  local_sum[lx+ly*localnx] = mask?tot_u:0;

  // // Adapted from dournac.org
  for (int stride = (localnx*localny)/2; stride>0; stride /=2)
  {
  // Waiting for each 2x2 addition into given workgroup
  __syncthreads();

  // Add elements 2 by 2 between local_id and local_id + stride
  if ((lx+ly*localnx) < stride)
    local_sum[(lx+ly*localnx)] += local_sum[(lx+ly*localnx) + stride];
  }



  if (lx == 0 && ly == 0){
    partial_sum[ idx + offset ] = local_sum[0];
  }
}



/* struct to hold the 'speed' values */

/*
** function prototypes
*/

/* load params, allocate memory, load obstacles & initialise fluid particle densities */
int initialise(const char* paramfile, const char* obstaclefile,
               t_param* params, float** cells_ptr, float** tmp_cells_ptr, float** partial_sums_ptr,
               int** obstacles_ptr, float** av_vels_ptr, t_cuda* cuda);

/*
** The main calculation methods.
** timestep calls, in order, the functions:
** accelerate_flow(), propagate(), rebound() & collision()
*/

/* finalise, including freeing up allocated memory */
int finalise(const t_param* params, float** cells_ptr, float** tmp_cells_ptr,
             int** obstacles_ptr, float** av_vels_ptr, t_cuda cuda);

/* Sum all the densities in the grid.
** The total should remain constant from one timestep to the next. */
float total_density(const t_param params, float* cells);

/* compute average velocity */
float av_velocity(const t_param params, float* cells, int* obstacles, t_cuda cuda);

/* calculate Reynolds number */
float calc_reynolds(const t_param params, float average);
int write_values(const t_param params, float* cells, int* obstacles, float* av_vels);

/* utility functions */
void die(const char* message, const int line, const char* file);
void usage(const char* exe);

/*
** main program:
** initialise, timestep loop, finalise
*/

int main(int argc, char* argv[])
{
  char*    paramfile = NULL;    /* name of the input parameter file */
  char*    obstaclefile = NULL; /* name of a the input obstacle file */
  t_param  params;              /* struct to hold parameter values */
  t_cuda    cuda;                 /* struct to hold OpenCL objects */
  float* cells     = NULL;    /* grid containing fluid densities */
  float* tmp_cells = NULL;    /* scratch space */
  float* partial_sums = NULL;
  int*     obstacles = NULL;    /* grid indicating which cells are blocked */
  int free_cells = 0;
  float* av_vels   = NULL;     /* a record of the av. velocity computed for each timestep */
  struct timeval timstr;        /* structure to hold elapsed time */
  struct rusage ru;             /* structure to hold CPU time--system and user */
  double tic, toc;              /* floating point numbers to calculate elapsed wallclock time */
  double usrtim;                /* floating point number to record elapsed user CPU time */
  double systim;                /* floating point number to record elapsed system CPU time */

  /* parse the command line */
  if (argc != 3)
  {
    usage(argv[0]);
  }
  else
  {
    paramfile = argv[1];
    obstaclefile = argv[2];
  }

  /* initialise our data structures and load values from file */
  initialise(paramfile, obstaclefile, &params, &cells, &tmp_cells, &partial_sums, &obstacles, &av_vels, &cuda);

  /* iterate for maxIters timesteps */
  gettimeofday(&timstr, NULL);
  tic = timstr.tv_sec + (timstr.tv_usec / 1000000.0);

  for (int i = 0; i<params.nx * params.ny; i++){
    if (!obstacles[i]) free_cells++;
  }
  params.free_cells = free_cells;


  // Write cells and obstacles to device
  cudaMemcpy(cuda.cells,cells,params.nx * params.ny * NSPEEDS * sizeof(float),cudaMemcpyHostToDevice);
  cudaMemcpy(cuda.obstacles,obstacles,params.nx * params.ny * sizeof(int),cudaMemcpyHostToDevice);


  float* cellPointers[2] = {cuda.cells, cuda.tmp_cells};
  int read = 0;
  int write = 1;
  params.w1 = (params.density*params.accel)/9.0f;
  params.w2 = (params.density*params.accel)/36.0f;
  float* av_ptr = av_vels;
  dim3 gridsize(params.nworkgroupsX,params.nworkgroupsY);
  dim3 blocksize(params.localnx,params.localny);

  for (int tt = 0; tt < params.maxIters; tt++)
  {

    accelerate_flow<<<params.nx,1>>>(cellPointers[read], cuda.obstacles, params.nx,params.ny,params.w1,params.w2);
    
    lbm<<<gridsize,blocksize,sizeof(float) * params.localnx * params.localny>>>(cellPointers[read], cellPointers[write],cuda.obstacles,cuda.partial_sums,params.nx,params.ny,params.localnx,params.localny,params.omega,params.reduction_count);

    params.reduction_count++;
    if (params.reduction_count == params.reduction_cap){
      reduction<<<params.reduction_count,1>>>(cuda.partial_sums,cuda.averages,params.nworkgroupsX*params.nworkgroupsY,params.free_cells);
      cudaMemcpy(av_ptr,cuda.averages,params.reduction_count*sizeof(float),cudaMemcpyDeviceToHost);
      params.reduction_count = 0;
      av_ptr = &av_ptr[params.reduction_cap];
    }

    read ^=1; 
    write ^=1;

#ifdef DEBUG
    printf("==timestep: %d==\n", tt);
    printf("av velocity: %.12E\n", av_vels[tt]);
    printf("tot density: %.12E\n", total_density(params, cells));
#endif
  }

  // Final reduction
  if (params.reduction_count != 0){
    reduction<<<params.reduction_count,1>>>(cuda.partial_sums,cuda.averages,params.nworkgroupsX*params.nworkgroupsY,params.free_cells);
    cudaMemcpy(av_ptr,cuda.averages,params.reduction_count*sizeof(float),cudaMemcpyDeviceToHost);    
  }

  cudaMemcpy(cells,cuda.cells,params.nx * params.ny * NSPEEDS * sizeof(float),cudaMemcpyDeviceToHost);

  gettimeofday(&timstr, NULL);
  toc = timstr.tv_sec + (timstr.tv_usec / 1000000.0);
  getrusage(RUSAGE_SELF, &ru);
  timstr = ru.ru_utime;
  usrtim = timstr.tv_sec + (timstr.tv_usec / 1000000.0);
  timstr = ru.ru_stime;
  systim = timstr.tv_sec + (timstr.tv_usec / 1000000.0);

  /* write final values and free memory */
  printf("==done==\n");
  printf("Reynolds number:\t\t%.12E\n", calc_reynolds(params, av_vels[params.maxIters-1]));
  printf("Elapsed time:\t\t\t%.6lf (s)\n", toc - tic);
  printf("Elapsed user CPU time:\t\t%.6lf (s)\n", usrtim);
  printf("Elapsed system CPU time:\t%.6lf (s)\n", systim);
  write_values(params, cells, obstacles, av_vels);
  finalise(&params, &cells, &tmp_cells, &obstacles, &av_vels, cuda);

  return EXIT_SUCCESS;
}




float av_velocity(const t_param params, float* cells, int* obstacles, t_cuda cuda)
{
  int    tot_cells = 0;  /* no. of cells used in calculation */
  float tot_u;          /* accumulated magnitudes of velocity for each cell */

  /* initialise */
  tot_u = 0.f;

  /* loop over all non-blocked cells */
  for (int jj = 0; jj < params.ny; jj++)
  {
    for (int ii = 0; ii < params.nx; ii++)
    {
      /* ignore occupied cells */
      if (!obstacles[ii + jj*params.nx])
      {
        /* local density total */
        float local_density = 0.f;
        for (int kk=0; kk < NSPEEDS; kk++){
           local_density += cells[INDEX(ii,jj,kk,params.nx,params.ny)];
        }

        /* x-component of velocity */
        float u_x =  (cells[INDEX(ii,jj,1,params.nx,params.ny)]
                    + cells[INDEX(ii,jj,5,params.nx,params.ny)]
                    + cells[INDEX(ii,jj,8,params.nx,params.ny)]
                    - cells[INDEX(ii,jj,3,params.nx,params.ny)]
                    - cells[INDEX(ii,jj,6,params.nx,params.ny)]
                    - cells[INDEX(ii,jj,7,params.nx,params.ny)])
                    / local_density;
        /* compute y velocity component */
        float u_y =  (cells[INDEX(ii,jj,2,params.nx,params.ny)]
                    + cells[INDEX(ii,jj,5,params.nx,params.ny)]
                    + cells[INDEX(ii,jj,6,params.nx,params.ny)]
                    - cells[INDEX(ii,jj,4,params.nx,params.ny)]
                    - cells[INDEX(ii,jj,7,params.nx,params.ny)]
                    - cells[INDEX(ii,jj,8,params.nx,params.ny)])
                    / local_density;
        /* accumulate the norm of x- and y- velocity components */
        tot_u += sqrtf((u_x * u_x) + (u_y * u_y));
        /* increase counter of inspected cells */
        ++tot_cells;
      }
    }
  }

  return tot_u / (float)tot_cells;
}

int initialise(const char* paramfile, const char* obstaclefile,
               t_param* params, float** cells_ptr, float** tmp_cells_ptr, float** partial_sums_ptr,
               int** obstacles_ptr, float** av_vels_ptr, t_cuda *cuda)
{
  char   message[1024];  /* message buffer */
  FILE*   fp;            /* file pointer */
  int    xx, yy;         /* generic array indices */
  int    blocked;        /* indicates whether a cell is blocked by an obstacle */
  int    retval;         /* to hold return value for checking */

  /* open the parameter file */
  fp = fopen(paramfile, "r");

  if (fp == NULL)
  {
    sprintf(message, "could not open input parameter file: %s", paramfile);
    die(message, __LINE__, __FILE__);
  }

  /* read in the parameter values */
  retval = fscanf(fp, "%d\n", &(params->nx));

  if (retval != 1) die("could not read param file: nx", __LINE__, __FILE__);

  retval = fscanf(fp, "%d\n", &(params->ny));

  if (retval != 1) die("could not read param file: ny", __LINE__, __FILE__);

  retval = fscanf(fp, "%d\n", &(params->maxIters));

  if (retval != 1) die("could not read param file: maxIters", __LINE__, __FILE__);

  retval = fscanf(fp, "%d\n", &(params->reynolds_dim));

  if (retval != 1) die("could not read param file: reynolds_dim", __LINE__, __FILE__);

  retval = fscanf(fp, "%f\n", &(params->density));

  if (retval != 1) die("could not read param file: density", __LINE__, __FILE__);

  retval = fscanf(fp, "%f\n", &(params->accel));

  if (retval != 1) die("could not read param file: accel", __LINE__, __FILE__);

  retval = fscanf(fp, "%f\n", &(params->omega));

  if (retval != 1) die("could not read param file: omega", __LINE__, __FILE__);

  /* and close up the file */
  fclose(fp);

  // Local size
  params->localnx = 32;
  params->localny = 1;
  params->nworkgroupsX = params->nx/params->localnx;
  params->nworkgroupsY = params->ny/params->localny; 
  params->reduction_count = 0;
  params->reduction_cap = 10000; 

  /*
  ** Allocate memory.
  **
  ** Remember C is pass-by-value, so we need to
  ** pass pointers into the initialise function.
  **
  ** NB we are allocating a 1D array, so that the
  ** memory will be contiguous.  We still want to
  ** index this memory as if it were a (row major
  ** ordered) 2D array, however.  We will perform
  ** some arithmetic using the row and column
  ** coordinates, inside the square brackets, when
  ** we want to access elements of this array.
  **
  ** Note also that we are using a structure to
  ** hold an array of 'speeds'.  We will allocate
  ** a 1D array of these structs.
  */

  /* main grid */
 *cells_ptr = (float*) malloc(sizeof(float) * params->nx * params->ny * NSPEEDS);

  if (*cells_ptr == NULL) die("cannot allocate memory for cells", __LINE__, __FILE__);

  /* 'helper' grid, used as scratch space */
 *tmp_cells_ptr = (float*) malloc(sizeof(float) * params->nx * params->ny * NSPEEDS);
  
  if (*tmp_cells_ptr == NULL) die("cannot allocate memory for tmp_cells", __LINE__, __FILE__);

  *partial_sums_ptr = (float*) malloc(sizeof(float) * params->nworkgroupsX * params->nworkgroupsY * params->reduction_cap);

  if (*partial_sums_ptr == NULL) die("cannot allocate memory for partial_sums", __LINE__, __FILE__);

  /* the map of obstacles */
  *obstacles_ptr = (int*) malloc(sizeof(int) * (params->ny * params->nx));

  if (*obstacles_ptr == NULL) die("cannot allocate column memory for obstacles", __LINE__, __FILE__);

  /* initialise densities */
  float w0 = params->density * 4.f / 9.f;
  float w1 = params->density      / 9.f;
  float w2 = params->density      / 36.f;

  for (int jj = 0; jj < params->ny; jj++)
  {
    for (int ii = 0; ii < params->nx; ii++)
    {
      /* centre */
      (*cells_ptr)[INDEX(ii,jj,0,params->nx,params->ny)] = w0;
      /* axis directions */
      (*cells_ptr)[INDEX(ii,jj,1,params->nx,params->ny)] = w1;
      (*cells_ptr)[INDEX(ii,jj,2,params->nx,params->ny)] = w1;
      (*cells_ptr)[INDEX(ii,jj,3,params->nx,params->ny)] = w1;
      (*cells_ptr)[INDEX(ii,jj,4,params->nx,params->ny)] = w1;
      /* diagonals */
      (*cells_ptr)[INDEX(ii,jj,5,params->nx,params->ny)] = w2;
      (*cells_ptr)[INDEX(ii,jj,6,params->nx,params->ny)] = w2;
      (*cells_ptr)[INDEX(ii,jj,7,params->nx,params->ny)] = w2;
      (*cells_ptr)[INDEX(ii,jj,8,params->nx,params->ny)] = w2;
    }
  }

  /* first set all cells in obstacle array to zero */
  for (int jj = 0; jj < params->ny; jj++)
  {
    for (int ii = 0; ii < params->nx; ii++)
    {
      (*obstacles_ptr)[ii + jj*params->nx] = 0;
    }
  }

  /* open the obstacle data file */
  fp = fopen(obstaclefile, "r");

  if (fp == NULL)
  {
    sprintf(message, "could not open input obstacles file: %s", obstaclefile);
    die(message, __LINE__, __FILE__);
  }

  /* read-in the blocked cells list */
  while ((retval = fscanf(fp, "%d %d %d\n", &xx, &yy, &blocked)) != EOF)
  {
    /* some checks */
    if (retval != 3) die("expected 3 values per line in obstacle file", __LINE__, __FILE__);

    if (xx < 0 || xx > params->nx - 1) die("obstacle x-coord out of range", __LINE__, __FILE__);

    if (yy < 0 || yy > params->ny - 1) die("obstacle y-coord out of range", __LINE__, __FILE__);

    if (blocked != 1) die("obstacle blocked value should be 1", __LINE__, __FILE__);

    /* assign to array */
    (*obstacles_ptr)[xx + yy*params->nx] = blocked;
  }

  /* and close the file */
  fclose(fp);

  /*
  ** allocate space to hold a record of the avarage velocities computed
  ** at each timestep
  */
  *av_vels_ptr = (float*)malloc(sizeof(float) * params->maxIters);


  // Allocate memory on device
  cudaMalloc((void **)&cuda->cells, params->nx * params->ny * NSPEEDS * sizeof(float));
  cudaMalloc((void **)&cuda->tmp_cells, params->nx * params->ny * NSPEEDS * sizeof(float));
  cudaMalloc((void **)&cuda->obstacles, params->nx * params->ny * sizeof(int));
  cudaMalloc((void **)&cuda->partial_sums, params->nworkgroupsX * params->nworkgroupsY * params->reduction_cap * sizeof(float));
  cudaMalloc((void **)&cuda->averages, params->reduction_cap * sizeof(float));
  

  return EXIT_SUCCESS;
}

int finalise(const t_param* params, float** cells_ptr, float** tmp_cells_ptr,
             int** obstacles_ptr, float** av_vels_ptr, t_cuda cuda)
{
  /*
  ** free up allocated memory
  */
  free(*cells_ptr);
  *cells_ptr = NULL;

  free(*tmp_cells_ptr);
  *tmp_cells_ptr = NULL;

  free(*obstacles_ptr);
  *obstacles_ptr = NULL;

  free(*av_vels_ptr);
  *av_vels_ptr = NULL;


  cudaFree(cuda.cells);
  cudaFree(cuda.tmp_cells);
  cudaFree(cuda.obstacles);

  return EXIT_SUCCESS;
}


float calc_reynolds(const t_param params, float average)
{
  const float viscosity = 1.f / 6.f * (2.f / params.omega - 1.f);

  return average * params.reynolds_dim / viscosity;
}

float total_density(const t_param params, float* cells)
{
  float total = 0.f;  /* accumulator */

  for (int jj = 0; jj < params.ny; jj++)
  {
    for (int ii = 0; ii < params.nx; ii++)
    {
      total = 0;
      for (int kk=0; kk < NSPEEDS; kk++){
        total += cells[INDEX(ii,jj,kk,params.nx,params.ny)];
      }
    }
  }

  return total;
}

int write_values(const t_param params, float* cells, int* obstacles, float* av_vels)
{
  FILE* fp;                     /* file pointer */
  const float c_sq = 1.f / 3.f; /* sq. of speed of sound */
  float local_density;         /* per grid cell sum of densities */
  float pressure;              /* fluid pressure in grid cell */
  float u_x;                   /* x-component of velocity in grid cell */
  float u_y;                   /* y-component of velocity in grid cell */
  float u;                     /* norm--root of summed squares--of u_x and u_y */

  fp = fopen(FINALSTATEFILE, "w");

  if (fp == NULL)
  {
    die("could not open file output file", __LINE__, __FILE__);
  }

  for (int jj = 0; jj < params.ny; jj++)
  {
    for (int ii = 0; ii < params.nx; ii++)
    {
      /* an occupied cell */
      if (obstacles[ii + jj*params.nx])
      {
        u_x = u_y = u = 0.f;
        pressure = params.density * c_sq;
      }
      /* no obstacle */
      else
      {
        local_density = 0;
        for (int kk=0; kk < NSPEEDS; kk++){
          local_density += cells[INDEX(ii,jj,kk,params.nx,params.ny)];
        }

        /* compute x velocity component */
        float u_x =  (cells[INDEX(ii,jj,1,params.nx,params.ny)]
                    + cells[INDEX(ii,jj,5,params.nx,params.ny)]
                    + cells[INDEX(ii,jj,8,params.nx,params.ny)]
                    - cells[INDEX(ii,jj,3,params.nx,params.ny)]
                    - cells[INDEX(ii,jj,6,params.nx,params.ny)]
                    - cells[INDEX(ii,jj,7,params.nx,params.ny)])
                    / local_density;
        /* compute y velocity component */
        float u_y =  (cells[INDEX(ii,jj,2,params.nx,params.ny)]
                    + cells[INDEX(ii,jj,5,params.nx,params.ny)]
                    + cells[INDEX(ii,jj,6,params.nx,params.ny)]
                    - cells[INDEX(ii,jj,4,params.nx,params.ny)]
                    - cells[INDEX(ii,jj,7,params.nx,params.ny)]
                    - cells[INDEX(ii,jj,8,params.nx,params.ny)])
                    / local_density;
        /* compute norm of velocity */
        u = sqrtf((u_x * u_x) + (u_y * u_y));
        /* compute pressure */
        pressure = local_density * c_sq;
      }

      /* write to file */
      fprintf(fp, "%d %d %.12E %.12E %.12E %.12E %d\n", ii, jj, u_x, u_y, u, pressure, obstacles[ii * params.nx + jj]);
    }
  }

  fclose(fp);

  fp = fopen(AVVELSFILE, "w");

  if (fp == NULL)
  {
    die("could not open file output file", __LINE__, __FILE__);
  }

  for (int ii = 0; ii < params.maxIters; ii++)
  {
    fprintf(fp, "%d:\t%.12E\n", ii, av_vels[ii]);
  }

  fclose(fp);

  return EXIT_SUCCESS;
}


void die(const char* message, const int line, const char* file)
{
  fprintf(stderr, "Error at line %d of file %s:\n", line, file);
  fprintf(stderr, "%s\n", message);
  fflush(stderr);
  exit(EXIT_FAILURE);
}

void usage(const char* exe)
{
  fprintf(stderr, "Usage: %s <paramfile> <obstaclefile>\n", exe);
  exit(EXIT_FAILURE);
}

#define MAX_DEVICES 32
#define MAX_DEVICE_NAME 1024

