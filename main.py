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
    # square wave
    u = np.zeros((2, imax)).astype(np.float32)
    for i in range(0, imax):
        if abs(x[i]) < 1: u[1, i] = 1.
    return u

# -----------------------------------------------------------------------------
# time step
def step():
    dt = cfl * (dx/max(abs(cp), abs(cm)))
    return dt

# -----------------------------------------------------------------------------
def update_opencl():
    global u
    u0 = u[0,:]
    u1 = u[1,:]
    e0     = np.zeros(imax-1).astype(np.float32)
    e1     = np.zeros(imax-1).astype(np.float32)
    res0   = np.zeros(imax  ).astype(np.float32)
    res1   = np.zeros(imax  ).astype(np.float32)

    a = .5 * (cp + cm)
    b = .5 * (cp - cm)
    u0_g = cl.Buffer(ctx, mf.COPY_HOST_PTR, hostbuf=u0)
    u1_g = cl.Buffer(ctx, mf.COPY_HOST_PTR, hostbuf=u1)
    e0_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=e0)
    e1_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=e1)
    res0_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=res0)
    res1_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=res1)

    prg = cl.Program(ctx, """
    #define A %(a)f
    #define B %(b)f
    #define DX %(dx)f
    #define DT %(dt)f
    #define SIZE %(size)d

    __kernel void update(
        __global float *u0_g,
        __global float *u1_g,
        __global float *e0_g,
        __global float *e1_g,
        __global float *res0_g,
        __global float *res1_g
        ) {
        int i = get_global_id(0);
        
        // compute flux vector
        float U0 = .5 * (u0_g[i] + u0_g[i+1]);
        float U1 = .5 * (u1_g[i] + u1_g[i+1]);
        barrier(CLK_GLOBAL_MEM_FENCE);

        if (i != SIZE-1) {
            e0_g[i] = A * U0 + B * U1 - .5*DX/DT * (u0_g[i+1] - u0_g[i]);
            e1_g[i] = B * U0 + A * U1 - .5*DX/DT * (u1_g[i+1] - u1_g[i]);
        }
        barrier(CLK_GLOBAL_MEM_FENCE);

        // compute residual vector
        if (i != 0 && i != SIZE-1) {
            res0_g[i] = -(e0_g[i] - e0_g[i-1]) / DX;
            res1_g[i] = -(e1_g[i] - e1_g[i-1]) / DX;
        }

        // update
        u0_g[i] = u0_g[i] + DT * res0_g[i];
        u1_g[i] = u1_g[i] + DT * res1_g[i];
    }
    """ % {"a":a, "b":b , "dx":dx, "dt":dt, "size":imax}).build()

    prg.update(queue, u0.shape, None, u0_g, u1_g, e0_g, e1_g, res0_g, res1_g)
    
    cl.enqueue_copy(queue, u0, u0_g)
    cl.enqueue_copy(queue, u1, u1_g)
    u = np.vstack((u0, u1))

    return

# -----------------------------------------------------------------------------
# update U
def update():
    global u

    # compute flux vector E   
    e = np.zeros((2, imax-1))

    a = .5 * (cp + cm)
    b = .5 * (cp - cm)
    
    for i in range(0, imax-1):
        u0 = .5 * (u[0,i] + u[0,i+1])
        u1 = .5 * (u[1,i] + u[1,i+1])

        e[0,i] = a * u0 + b * u1
        e[1,i] = b * u0 + a * u1

        e[:,i] -= .5*dx/dt * (u[:,i+1] - u[:,i])

    # compute residual vector
    res = np.zeros((2, imax-1))
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

imax = 10000
xmin = -10
xmax = 10
cfl  = .5
cp   = 1.
cm   = -.1
time_max = 7.


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
plt.ylim(-1,1)
plt.plot(x, u[1,:])
plt.show()
