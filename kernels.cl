#pragma OPENCL EXTENSION cl_khr_fp64 : enable

#define NSPEEDS         9

#define INDEX(ii,jj,sp,nx,ny) ((ii)+(jj)*(nx)+(sp)*(nx)*(ny))

kernel void accelerate_flow(global float* cells,
                            global int* obstacles,
                            int nx, int ny,
                            float w1, float w2)
{
  /* compute weighting factors */

  /* modify the 2nd row of the grid */
  int jj = ny - 2;

  /* get column index */
  int ii = get_global_id(0);

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


kernel void lbm(global float* cells,
                global float* tmp_cells,
                global int* obstacles,
                local float* local_sum, //stores per thread, maybe not needed
                global float* partial_sum, //stores per workgroup
                int globalnx, int globalny, int localnx, int localny, 
                float omega, int iter)
{
  float c_sq = 1.f / 3.f; /* square of speed of sound */
  float w0 = 4.f / 9.f;  /* weighting factor */
  float w1 = 1.f / 9.f;  /* weighting factor */
  float w2 = 1.f / 36.f; /* weighting factor */
  float tot_u = 0;          /* accumulated magnitudes of velocity for each cell */
  float speed0,speed1,speed2,speed3,speed4,speed5,speed6,speed7,speed8;
  /* get column and row indices */
  int ii = get_global_id(0);
  int jj = get_global_id(1);
  int lx = get_local_id(0);
  int ly = get_local_id(1);

  int idx = ii/localnx + (globalnx/localnx) * jj/localny;
  int offset = iter * (globalnx/localnx)* (globalny/localny) ;

  int ngroupX = get_num_groups(0);
  int ngroupY = get_num_groups(1);  //Array to lookup write direction
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

tot_u += native_sqrt((u_x * u_x) + (u_y * u_y));
    /* increase counter of inspected cells */

//take to outer loop
local_sum[lx+ly*localnx] = mask?tot_u:0;

// Adapted from dournac.org
for (int stride = (localnx*localny)/2; stride>0; stride /=2)
    {
    // Waiting for each 2x2 addition into given workgroup
    barrier(CLK_LOCAL_MEM_FENCE);

    // Add elements 2 by 2 between local_id and local_id + stride
    if ((lx+ly*localnx) < stride)
      local_sum[(lx+ly*localnx)] += local_sum[(lx+ly*localnx) + stride];
    }



if (lx == 0 && ly == 0){
  partial_sum[ idx + offset ] = local_sum[0];
}
}

kernel void reduction(global float* partial_sum,
                global float* averages,
                int nGroups,
                int freeCells)
{
  int id = get_global_id(0);
  float total = 0;
  for (int i = 0; i < nGroups; i++){
    total += partial_sum[i + (nGroups * id)];
  }
  averages[id] = total/(float) freeCells;
}

