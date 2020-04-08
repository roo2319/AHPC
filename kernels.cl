#pragma OPENCL EXTENSION cl_khr_fp64 : enable

#define NSPEEDS         9

typedef struct
{
  float speeds[NSPEEDS];
} t_speed;

kernel void accelerate_flow(global t_speed* cells,
                            global int* obstacles,
                            int nx, int ny,
                            float density, float accel)
{
  /* compute weighting factors */
  float w1 = density * accel / 9.0;
  float w2 = density * accel / 36.0;

  /* modify the 2nd row of the grid */
  int jj = ny - 2;

  /* get column index */
  int ii = get_global_id(0);

  /* if the cell is not occupied and
  ** we don't send a negative density */
  if (!obstacles[ii + jj* nx]
      && (cells[ii + jj* nx].speeds[3] - w1) > 0.f
      && (cells[ii + jj* nx].speeds[6] - w2) > 0.f
      && (cells[ii + jj* nx].speeds[7] - w2) > 0.f)
  {
    /* increase 'east-side' densities */
    cells[ii + jj* nx].speeds[1] += w1;
    cells[ii + jj* nx].speeds[5] += w2;
    cells[ii + jj* nx].speeds[8] += w2;
    /* decrease 'west-side' densities */
    cells[ii + jj* nx].speeds[3] -= w1;
    cells[ii + jj* nx].speeds[6] -= w2;
    cells[ii + jj* nx].speeds[7] -= w2;
  }
}

kernel void lbm(global t_speed* cells,
                global t_speed* tmp_cells,
                global int* obstacles,
                int nx, int ny,float omega)
{
  float c_sq = 1.f / 3.f; /* square of speed of sound */
  float w0 = 4.f / 9.f;  /* weighting factor */
  float w1 = 1.f / 9.f;  /* weighting factor */
  float w2 = 1.f / 36.f; /* weighting factor */
  int    tot_cells = 0;  /* no. of cells used in calculation */
  float tot_u;          /* accumulated magnitudes of velocity for each cell */
  /* get column and row indices */
  int ii = get_global_id(0);
  int jj = get_global_id(1);

  /* determine indices of axis-direction neighbours
  ** respecting periodic boundary conditions (wrap around) */
  int y_n = (jj + 1) % ny;
  int x_e = (ii + 1) % nx;
  int y_s = (jj == 0) ? (jj + ny - 1) : (jj - 1);
  int x_w = (ii == 0) ? (ii + nx - 1) : (ii - 1);

  const float speed0 = cells[ii + jj*nx].speeds[0]; /* central cell, no movement */
  const float speed1 = cells[x_w + jj*nx].speeds[1]; /* east */
  const float speed2 = cells[ii + y_s*nx].speeds[2]; /* north */
  const float speed3 = cells[x_e + jj*nx].speeds[3]; /* west */
  const float speed4 = cells[ii + y_n*nx].speeds[4]; /* south */
  const float speed5 = cells[x_w + y_s*nx].speeds[5]; /* north-east */
  const float speed6 = cells[x_e + y_s*nx].speeds[6]; /* north-west */
  const float speed7 = cells[x_e + y_n*nx].speeds[7]; /* south-west */
  const float speed8 = cells[x_w + y_n*nx].speeds[8]; /* south-east */


  if (obstacles[jj*nx + ii])
  {

    tmp_cells[ii + jj*nx].speeds[0] = speed0;
    tmp_cells[ii + jj*nx].speeds[3] = speed1;
    tmp_cells[ii + jj*nx].speeds[4] = speed2;
    tmp_cells[ii + jj*nx].speeds[1] = speed3;
    tmp_cells[ii + jj*nx].speeds[2] = speed4;
    tmp_cells[ii + jj*nx].speeds[7] = speed5;
    tmp_cells[ii + jj*nx].speeds[8] = speed6;
    tmp_cells[ii + jj*nx].speeds[5] = speed7;
    tmp_cells[ii + jj*nx].speeds[6] = speed8;
  }
  else{
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
    tmp_cells[ii + jj*nx].speeds[0] = speed0 + omega * (d_equ[0] - speed0);
    tmp_cells[ii + jj*nx].speeds[1] = speed1 + omega * (d_equ[1] - speed1);
    tmp_cells[ii + jj*nx].speeds[2] = speed2 + omega * (d_equ[2] - speed2);
    tmp_cells[ii + jj*nx].speeds[3] = speed3 + omega * (d_equ[3] - speed3);
    tmp_cells[ii + jj*nx].speeds[4] = speed4 + omega * (d_equ[4] - speed4);
    tmp_cells[ii + jj*nx].speeds[5] = speed5 + omega * (d_equ[5] - speed5);
    tmp_cells[ii + jj*nx].speeds[6] = speed6 + omega * (d_equ[6] - speed6);
    tmp_cells[ii + jj*nx].speeds[7] = speed7 + omega * (d_equ[7] - speed7);
    tmp_cells[ii + jj*nx].speeds[8] = speed8 + omega * (d_equ[8] - speed8);

  }

}
