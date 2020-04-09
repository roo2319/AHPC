#pragma OPENCL EXTENSION cl_khr_fp64 : enable

#define NSPEEDS         9

#define INDEX(ii,jj,sp,nx,ny) ((ii)+(jj)*(nx)+(sp)*(nx)*(ny))

kernel void accelerate_flow(global float* cells,
                            global int* obstacles,
                            int nx, int ny,
                            float density, float accel)
{
  /* compute weighting factors */
  float w1 = native_divide(density * accel , 9.0f);
  float w2 = native_divide(density * accel , 36.0f);

  /* modify the 2nd row of the grid */
  int jj = ny - 2;

  /* get column index */
  int ii = get_global_id(0);

  /* if the cell is not occupied and
  ** we don't send a negative density */
  if (!obstacles[ii + jj* nx]
      && (cells[INDEX(ii,jj,3,nx,ny)] - w1) > 0.f
      && (cells[INDEX(ii,jj,6,nx,ny)] - w2) > 0.f
      && (cells[INDEX(ii,jj,7,nx,ny)] - w2) > 0.f)
    {
    /* increase 'east-side' densities */
    cells[INDEX(ii,jj,1,nx,ny)] += w1;
    cells[INDEX(ii,jj,5,nx,ny)] += w2;
    cells[INDEX(ii,jj,8,nx,ny)] += w2;
    /* decrease 'west-side' densities */
    cells[INDEX(ii,jj,3,nx,ny)] -= w1;
    cells[INDEX(ii,jj,6,nx,ny)] -= w2;
    cells[INDEX(ii,jj,7,nx,ny)] -= w2;
    }
}


kernel void lbm(global float* cells,
                global float* tmp_cells,
                global int* obstacles,
                local  float* cellblk,
                local float* local_sum, //stores per thread, maybe not needed
                global float* partial_sum, //stores per workgroup
                int globalnx, int globalny, int localnx, int localny, 
                float omega)
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


  //Loop over speeds
    //Loop over Y blocks
        // if (groupx == 1 && groupy == 1){
        // }      
        //can't actually write speed :((((((
          // Error with Lvalue (invalid left hand side)
          
  for (int kk = 0; kk < NSPEEDS; kk++){
    cellblk[INDEX(lx,ly,kk,localnx,localny)] = cells[INDEX(ii,jj,kk,globalnx,globalny)];
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  // if (ii == 0 && jj == 0) printf("iter"); 
  // if (get_group_id(0) == 1 && get_group_id(1) == 1){
  //         // printf("Local: %d, %d Global: %d,%d\n", lx,ly, globalx+lx,globaly+ly);
  //       printf("\n\n\n\n\n\n\n\niter\n");
  //       }

  //Array to lookup write direction
  int indexLookup[9][2] = {{0,0},{3,1},{4,2},{1,3},{2,4},{7,5},{8,6},{5,7},{6,8}};


  /* determine indices of axis-direction neighbours
  ** respecting periodic boundary conditions (wrap around) */
  if (ly+1 < localny && lx+1 < localnx && ly-1 >= 0 && lx-1 >= 0){
    int y_n = (ly + 1);
    int x_e = (lx + 1);
    int y_s = (ly - 1);
    int x_w = (lx - 1);
    speed0 = cellblk[INDEX(lx,ly,0,localnx,localny)]; /* central cell, no movement */
    speed1 = cellblk[INDEX(x_w,ly,1,localnx,localny)]; /* west */
    speed2 = cellblk[INDEX(lx,y_s,2,localnx,localny)]; /* south */
    speed3 = cellblk[INDEX(x_e,ly,3,localnx,localny)]; /* east */
    speed4 = cellblk[INDEX(lx,y_n,4,localnx,localny)]; /* north */
    speed5 = cellblk[INDEX(x_w,y_s,5,localnx,localny)]; /* south-west */
    speed6 = cellblk[INDEX(x_e,y_s,6,localnx,localny)];  /* south-east */
    speed7 = cellblk[INDEX(x_e,y_n,7,localnx,localny)]; /* north-east */
    speed8 = cellblk[INDEX(x_w,y_n,8,localnx,localny)]; /* north-west */
  }
  // remote access
  else{
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
  }

  if (obstacles[jj*globalnx + ii])
  {
      
    tmp_cells[INDEX(ii,jj,0,globalnx,globalny)] = speed0;
    tmp_cells[INDEX(ii,jj,3,globalnx,globalny)] = speed1;
    tmp_cells[INDEX(ii,jj,4,globalnx,globalny)] = speed2;
    tmp_cells[INDEX(ii,jj,1,globalnx,globalny)] = speed3;
    tmp_cells[INDEX(ii,jj,2,globalnx,globalny)] = speed4;
    tmp_cells[INDEX(ii,jj,7,globalnx,globalny)] = speed5;
    tmp_cells[INDEX(ii,jj,8,globalnx,globalny)] = speed6;
    tmp_cells[INDEX(ii,jj,5,globalnx,globalny)] = speed7;
    tmp_cells[INDEX(ii,jj,6,globalnx,globalny)] = speed8;
    local_sum[lx+ly*localnx] = 0;
  }
  else{
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
    tmp_cells[INDEX(ii,jj,0,globalnx,globalny)] = speed0 + (omega * (d_equ[0] - speed0));
    tmp_cells[INDEX(ii,jj,1,globalnx,globalny)] = speed1 + (omega * (d_equ[1] - speed1));
    tmp_cells[INDEX(ii,jj,2,globalnx,globalny)] = speed2 + (omega * (d_equ[2] - speed2));
    tmp_cells[INDEX(ii,jj,3,globalnx,globalny)] = speed3 + (omega * (d_equ[3] - speed3));
    tmp_cells[INDEX(ii,jj,4,globalnx,globalny)] = speed4 + (omega * (d_equ[4] - speed4));
    tmp_cells[INDEX(ii,jj,5,globalnx,globalny)] = speed5 + (omega * (d_equ[5] - speed5));
    tmp_cells[INDEX(ii,jj,6,globalnx,globalny)] = speed6 + (omega * (d_equ[6] - speed6));
    tmp_cells[INDEX(ii,jj,7,globalnx,globalny)] = speed7 + (omega * (d_equ[7] - speed7));
    tmp_cells[INDEX(ii,jj,8,globalnx,globalny)] = speed8 + (omega * (d_equ[8] - speed8));

    tot_u += native_sqrt((u_x * u_x) + (u_y * u_y));
        /* increase counter of inspected cells */

    //take to outer loop
    local_sum[lx+ly*localnx] = tot_u;
    }

  barrier(CLK_LOCAL_MEM_FENCE);

  if (lx == 0 && ly == 0){
    float sum = 0;
      for (int i = 0; i<localnx*localny; i++){
          sum += local_sum[i];
      }
  partial_sum[get_group_id(0) + get_group_id(1) * get_num_groups(0)] = sum;
    }
}

/* compute local density total */

