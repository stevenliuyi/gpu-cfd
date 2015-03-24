import numpy as np
import matplotlib.pyplot as plt
import pyopencl as cl
import time

# -----------------------------------------------------------------------------
# set computational mesh
def mesh(xmin, xmax, imax):
    x = np.linspace(xmin, xmax, imax)
    return x

# -----------------------------------------------------------------------------
# initial condition
def ic(imax):
    u = np.zeros((3, imax)).astype(np.float32)
    for i in range(0, (imax+1)/2):
        u[0, i] = 1.
        u[1, i] = 0.
        u[2, i] = 1 / (gamma-1)
    for i in range((imax+1)/2, imax):
        u[0, i] = 1 / r4r1
        u[1, i] = 0.
        u[2, i] = 1 / ((gamma-1) * p4p1)
    return u

# -----------------------------------------------------------------------------
# time step
def step():
    lam = max(u[1,:]/u[0,:] + np.sqrt(gamma*(gamma-1) * (u[2,:]/u[0,:] - \
            0.5 * u[1,:]**2 / u[0,:]**2)))
    dt  = cfl * dx / lam
    return dt

# -----------------------------------------------------------------------------
# compute Jacobi matrix
def jacobi(ui):
    rho = ui[0]
    vel = ui[1] / ui[0]
    et  = ui[2]

    j = np.zeros((3, 3))
    j[0, 0] = 0.
    j[0, 1] = 1.
    j[0, 2] = 0.
    j[1, 0] = .5 * (gamma-3) * vel**2
    j[1, 1] = (3-gamma) * vel
    j[1, 2] = gamma - 1
    j[2, 0] = (gamma-1) * vel**3 - gamma * vel * et / rho
    j[2, 1] = -1.5*(gamma-1) * vel**2 + gamma * et / rho
    j[2, 2] = gamma * vel

    return j

# -----------------------------------------------------------------------------
def update_opencl():
    global u
    u0 = u[0,:]
    u1 = u[1,:]
    u2 = u[2,:]
    # flux vector at inteface locations
    e0     = np.zeros(imax-1).astype(np.float32)
    e1     = np.zeros(imax-1).astype(np.float32)
    e2     = np.zeros(imax-1).astype(np.float32)
    # flux vector at cell centers
    ei0     = np.zeros(imax).astype(np.float32)
    ei1     = np.zeros(imax).astype(np.float32)
    ei2     = np.zeros(imax).astype(np.float32)
    # residual vector
    res0   = np.zeros(imax  ).astype(np.float32)
    res1   = np.zeros(imax  ).astype(np.float32)
    res2   = np.zeros(imax  ).astype(np.float32)

    u0_g = cl.Buffer(ctx, mf.COPY_HOST_PTR, hostbuf=u0)
    u1_g = cl.Buffer(ctx, mf.COPY_HOST_PTR, hostbuf=u1)
    u2_g = cl.Buffer(ctx, mf.COPY_HOST_PTR, hostbuf=u2)
    e0_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=e0)
    e1_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=e1)
    e2_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=e2)
    ei0_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=ei0)
    ei1_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=ei1)
    ei2_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=ei2)
    res0_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=res0)
    res1_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=res1)
    res2_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=res2)

    prg = cl.Program(ctx, """
    #define GAMMA 1.4f

    __kernel void update(
        __global float *u0_g,
        __global float *u1_g,
        __global float *u2_g,
        __global float *ei0_g,
        __global float *ei1_g,
        __global float *ei2_g
        ) {
        int i = get_global_id(0);
        
        // compute Jacobi matrix
        float rho = u0_g[i];
        float vel = u1_g[i] / u0_g[i];
        float et  = u2_g[i];

        float J00 = 0;
        float J01 = 1;
        float J02 = 0;
        float J10 = .5 * (GAMMA-3) * vel*vel;
        float J11 = (3-GAMMA) * vel;
        float J12 = GAMMA - 1;
        float J20 = (GAMMA-1) * vel*vel*vel - GAMMA * vel * et / rho;
        float J21 = -1.5*(GAMMA-1) * vel*vel + GAMMA * et / rho;
        float J22 = GAMMA * vel;

        // compute flux vector at cell center
        ei0_g[i] = J00 * u0_g[i] + J01 * u1_g[i] + J02 * u2_g[i];
        ei1_g[i] = J10 * u0_g[i] + J11 * u1_g[i] + J12 * u2_g[i];
        ei2_g[i] = J20 * u0_g[i] + J21 * u1_g[i] + J22 * u2_g[i];
        // barrier(CLK_GLOBAL_MEM_FENCE);
    }
    """).build()

    prg.update(queue, u0.shape, None, u0_g, u1_g, u2_g, \
            ei0_g, ei1_g, ei2_g)
    
    prg = cl.Program(ctx, """
    #define DX %(dx)f
    #define DT %(dt)f
    #define SIZE %(size)d
    #define GAMMA 1.4f

    __kernel void update(
        __global float *u0_g,
        __global float *u1_g,
        __global float *u2_g,
        __global float *e0_g,
        __global float *e1_g,
        __global float *e2_g,
        __global float *ei0_g,
        __global float *ei1_g,
        __global float *ei2_g
        ) {
        int i = get_global_id(0);
        
        // compute flux vector
        if (i != SIZE-1) {
            e0_g[i] = .5*(ei0_g[i+1] + ei0_g[i]) - \
                    .5*DX/DT * (u0_g[i+1] - u0_g[i]);
            e1_g[i] = .5*(ei1_g[i+1] + ei1_g[i]) - \
                    .5*DX/DT * (u1_g[i+1] - u1_g[i]);
            e2_g[i] = .5*(ei2_g[i+1] + ei2_g[i]) - \
                    .5*DX/DT * (u2_g[i+1] - u2_g[i]);
        }
    }
    """ % {"dx":dx, "dt":dt, "size":imax}).build()

    prg.update(queue, u0.shape, None, u0_g, u1_g, u2_g, e0_g, e1_g, e2_g, \
            ei0_g, ei1_g, ei2_g)

    prg = cl.Program(ctx, """
    #define DX %(dx)f
    #define DT %(dt)f
    #define SIZE %(size)d
    #define GAMMA 1.4f

    __kernel void update(
        __global float *u0_g,
        __global float *u1_g,
        __global float *u2_g,
        __global float *e0_g,
        __global float *e1_g,
        __global float *e2_g,
        __global float *res0_g,
        __global float *res1_g,
        __global float *res2_g
        ) {
        int i = get_global_id(0);
        
        // compute residual vector
        if (i != 0 && i != SIZE-1) {
            res0_g[i] = -(e0_g[i] - e0_g[i-1]) / DX;
            res1_g[i] = -(e1_g[i] - e1_g[i-1]) / DX;
            res2_g[i] = -(e2_g[i] - e2_g[i-1]) / DX;
        }

        // update
        u0_g[i] = u0_g[i] + DT * res0_g[i];
        u1_g[i] = u1_g[i] + DT * res1_g[i];
        u2_g[i] = u2_g[i] + DT * res2_g[i];
    }
    """ % {"dx":dx, "dt":dt, "size":imax}).build()

    prg.update(queue, u0.shape, None, u0_g, u1_g, u2_g, \
            e0_g, e1_g, e2_g, res0_g, res1_g, res2_g)

    cl.enqueue_copy(queue, u0, u0_g)
    cl.enqueue_copy(queue, u1, u1_g)
    cl.enqueue_copy(queue, u2, u2_g)
    u = np.vstack((u0, u1, u2))

    return

# -----------------------------------------------------------------------------
# update U
def update():
    global u

    # compute flux vector E   
    e = np.zeros((3, imax-1))

    for i in range(0, imax-1):
        j0 = jacobi(u[:,i+1])
        j1 = jacobi(u[:,  i])

        e[:,i] = .5*(np.dot(j0, u[:,i+1]) + np.dot(j1, u[:,i]))
        e[:,i] -= .5*dx/dt * (u[:,i+1] - u[:,i])

    # compute residual vector
    res = np.zeros((3, imax-1))
    for i in range(1, imax-1):
        res[:,i] = - (e[:,i] - e[:,i-1]) / dx

    # update
    for i in range(1, imax-1):
        u[:,i] += dt * res[:,i]
    return

# -----------------------------------------------------------------------------
# Lax method
def lax():
    global u

    if use_opencl:
        # parallel code
        update_opencl()
    else:
        # sequential code
        update()
    return

# -----------------------------------------------------------------------------
# parameters
use_opencl = True

imax = 3001
xmin = -0.5
xmax = 0.5
cfl  = .5
time_max = .214

p4p1  = 10.     # pressure ratio
r4r1  = 8.      # density ratio
gamma = 1.4     # ratio of sepcific heat


x  = mesh(xmin, xmax, imax)
dx = x[1] - x[0]

# initial condition
u = ic(imax)

t = 0.

platform = cl.get_platforms()[0]
device = platform.get_devices()[1]
ctx = cl.Context([device])
#ctx = cl.create_some_context()

queue = cl.CommandQueue(ctx)

mf = cl.mem_flags

start_time = time.time()
while (t < time_max):
    # time step
    dt = step()
    
    # solver
    lax()

    t += dt

print "execution time:", time.time() - start_time
plt.ylim(0,1)
plt.plot(x, u[0,:])
plt.show()
