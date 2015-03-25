import numpy as np
import matplotlib.pyplot as plt
import pyopencl as cl
import os
import time

# -----------------------------------------------------------------------------
def read_mesh(filename):
    f = open(filename)
    x = np.zeros((mx, my))
    y = np.zeros((mx, my))
    i = 0; j = 0
    for line in f:
        x[i,j] = line.split()[0]
        y[i,j] = line.split()[1]
        if (j == my-1):
            i += 1; j = 0
        else:
            j += 1
        if (i == mx): break

    f.close()
    return (x, y)

# -----------------------------------------------------------------------------
def plot_mesh():
    plt.plot(x, y, '.', color='b')
    plt.show()
    return 

# -----------------------------------------------------------------------------
def calc_normal():
    # initialize dn
    # dni/dnj_x : x-components
    # dnj/dnj_y : y-components
    # dni/dnj   : length
    # defined at boundaries
    dni_x = np.zeros((mx, my))
    dni_y = np.zeros((mx, my))
    dni   = np.zeros((mx, my))
    dnj_x = np.zeros((mx, my))
    dnj_y = np.zeros((mx, my))
    dnj   = np.zeros((mx, my))

    # compute dn_bc and |dn_bc| along constant-xi boundaries
    for i in range(0, mx):
        for j in range(1, my):
            dni_x[i,j] =  (y[i,j] - y[i,j-1])
            dni_y[i,j] = -(x[i,j] - x[i,j-1])
            dni[i,j]   = np.sqrt(dni_x[i,j]**2 + dni_y[i,j]**2)
            
    # compute dn_cd and |dn_cd| along constant-eta boundaries
    for i in range(1, mx):
        for j in range(0, my):
            dnj_x[i,j] =  (y[i-1,j] - y[i,j])
            dnj_y[i,j] = -(x[i-1,j] - x[i,j])
            dnj[i,j]   = np.sqrt(dnj_x[i,j]**2 + dnj_y[i,j]**2)

    return (dni_x, dni_y, dni, dnj_x, dnj_y, dnj)

# -----------------------------------------------------------------------------
# compute cell areas
def calc_area():
    # defined at cell-centers
    area = np.zeros((mx, my))
    for i in range(1, mx):
        for j in range(1, my):
            dxac = x[i,  j] - x[i-1,j-1]
            dyac = y[i,  j] - y[i-1,j-1]
            dxbd = x[i-1,j] - x[i,  j-1]
            dybd = y[i-1,j] - y[i,  j-1]
            area[i,j] = .5*abs(dxac*dybd - dxbd*dyac)
    return area

# -----------------------------------------------------------------------------
# compute pressure
def pressure(rho, u, v, et):
    return (gamma-1) * (et - .5 * rho * (u**2 + v**2))

# -----------------------------------------------------------------------------
# initialize conservative variables
def ic():
    # defined at cell-centers (including phatom cells)
    q = np.ones((lmax, mx+1, my+1)).astype(np.float32)
    if (import_ic):
        f = open(icfile)
        i = 1; j = 1
        for line in f:
            q[0,i,j] = float(line.split()[2])
            q[1,i,j] = float(line.split()[3])  
            q[2,i,j] = float(line.split()[4])  
            q[3,i,j] = float(line.split()[5])  
            if (j == my-1):
                i += 1; j = 1
            else:
                j += 1
            if (i == mx): break
        f.close()
        q[:,0, :] = q[:,1,   :]
        q[:,mx,:] = q[:,mx-1,:]
        q[:,:, 0] = q[:,:,   1]
        q[:,:,my] = q[:,:,my-1]
    else:
        if (ibcin == 1):
            if (pback == 0.85):
                q[0,:,:] = 1.
                q[1,:,:] = .5
                q[2,:,:] = 0.
                q[3,:,:] = 1.
            else:
                q[0,:,:] = .5
                q[1,:,:] = .25
                q[2,:,:] = 0.
                q[3,:,:] = 1.
        if (ibcin == 2):
            q[0,:,:] = 1.
            q[1,:,:] = 1.5
            q[2,:,:] = 0.
            q[3,:,:] = 2.
    return q

# -----------------------------------------------------------------------------
# evulation of the flux vectors
def flux():
    # initialize flux vectors
    ei = np.zeros((lmax, mx, my))
    ej = np.zeros((lmax, mx, my))
    fi = np.zeros((lmax, mx, my))
    fj = np.zeros((lmax, mx, my))

    # compute flux vectors along constant-xi boudnaries
    # defined at boundaries
    for i in range(0, mx):
        for j in range(1, my):
            rho = .5 * (q[0,i,  j] + q[0,i+1,j])
            u   = .5 * (q[1,i,  j] / q[0,i,  j] + \
                        q[1,i+1,j] / q[0,i+1,j])
            v   = .5 * (q[2,i,  j] / q[0,i,  j] + \
                        q[2,i+1,j] / q[0,i+1,j])
            et  = .5 * (q[3,i,  j] + q[3,i+1,j])
            p   = pressure(rho, u, v, et)
            ei[0,i,j] = rho * u
            ei[1,i,j] = rho * u**2 + p
            ei[2,i,j] = rho * u * v
            ei[3,i,j] = (et + p) * u
            fi[0,i,j] = rho * v
            fi[1,i,j] = rho * u * v
            fi[2,i,j] = rho * v**2 + p
            fi[3,i,j] = (et + p) * v

    # compute flux vectors along constant-eta boudnaries
    # defined at boundaries
    # interior cells
    for i in range(1, mx):
        for j in range(0, my):
            rho = .5 * (q[0,i,  j] + q[0,i,j+1])
            u   = .5 * (q[1,i,  j] / q[0,i,  j] + \
                        q[1,i,j+1] / q[0,i,j+1])
            v   = .5 * (q[2,i,  j] / q[0,i,  j] + \
                        q[2,i,j+1] / q[0,i,j+1])
            et  = .5 * (q[3,i,  j] + q[3,i,j+1])
            p   = pressure(rho, u, v, et)
            
            ej[0,i,j] = rho * u
            ej[1,i,j] = rho * u**2 + p
            ej[2,i,j] = rho * u * v
            ej[3,i,j] = (et + p) * u
            fj[0,i,j] = rho * v
            fj[1,i,j] = rho * u * v
            fj[2,i,j] = rho * v**2 + p
            fj[3,i,j] = (et + p) * v

    # upper boundary cells
    # zeroth-order extrapolation of pressure
    # j = my-1
    # for i in range(1, mx):
    #     rho = q[0,i,j]
    #     u   = q[1,i,j] / q[0,i,j]
    #     v   = q[2,i,j] / q[0,i,j]
    #     et  = q[3,i,j]
    #     p   = pressure(rho, u, v, et)

    #     ej[0,i,j] = 0.
    #     ej[1,i,j] = p
    #     ej[2,i,j] = 0.
    #     ej[3,i,j] = 0.
    #     fj[0,i,j] = 0.
    #     fj[1,i,j] = 0.
    #     fj[2,i,j] = p
    #     fj[3,i,j] = 0.

    return (ei, ej, fi, fj)

def flux_opencl():
    # initialize flux vectors
    ei = np.zeros((lmax, mx, my))
    ej = np.zeros((lmax, mx, my))
    fi = np.zeros((lmax, mx, my))
    fj = np.zeros((lmax, mx, my))
    
    ei0 = np.zeros(mx*my).astype(np.float32)
    ei1 = np.zeros(mx*my).astype(np.float32)
    ei2 = np.zeros(mx*my).astype(np.float32)
    ei3 = np.zeros(mx*my).astype(np.float32)
    ej0 = np.zeros(mx*my).astype(np.float32)
    ej1 = np.zeros(mx*my).astype(np.float32)
    ej2 = np.zeros(mx*my).astype(np.float32)
    ej3 = np.zeros(mx*my).astype(np.float32)
    fi0 = np.zeros(mx*my).astype(np.float32)
    fi1 = np.zeros(mx*my).astype(np.float32)
    fi2 = np.zeros(mx*my).astype(np.float32)
    fi3 = np.zeros(mx*my).astype(np.float32)
    fj0 = np.zeros(mx*my).astype(np.float32)
    fj1 = np.zeros(mx*my).astype(np.float32)
    fj2 = np.zeros(mx*my).astype(np.float32)
    fj3 = np.zeros(mx*my).astype(np.float32)

    q0_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=q0)
    q1_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=q1)
    q2_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=q2)
    q3_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=q3)

    # compute flux vectors along constant-xi boudnaries
    # defined at boundaries
    ei0_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=ei0)
    ei1_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=ei1)
    ei2_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=ei2)
    ei3_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=ei3)
    fi0_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=fi0)
    fi1_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=fi1)
    fi2_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=fi2)
    fi3_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=fi3)

    prg_flux_xi.update(queue, ((mx+1),(my+1)), None, q0_g, q1_g, q2_g, q3_g, \
            ei0_g, ei1_g, ei2_g, ei3_g, fi0_g, fi1_g, fi2_g, fi3_g)

    cl.enqueue_copy(queue, ei0, ei0_g)
    cl.enqueue_copy(queue, ei1, ei1_g)
    cl.enqueue_copy(queue, ei2, ei2_g)
    cl.enqueue_copy(queue, ei3, ei3_g)
    cl.enqueue_copy(queue, fi0, fi0_g)
    cl.enqueue_copy(queue, fi1, fi1_g)
    cl.enqueue_copy(queue, fi2, fi2_g)
    cl.enqueue_copy(queue, fi3, fi3_g)

    ei[0,:,:] = np.reshape(ei0, (mx, my))
    ei[1,:,:] = np.reshape(ei1, (mx, my))
    ei[2,:,:] = np.reshape(ei2, (mx, my))
    ei[3,:,:] = np.reshape(ei3, (mx, my))
    fi[0,:,:] = np.reshape(fi0, (mx, my))
    fi[1,:,:] = np.reshape(fi1, (mx, my))
    fi[2,:,:] = np.reshape(fi2, (mx, my))
    fi[3,:,:] = np.reshape(fi3, (mx, my))

    # compute flux vectors along constant-eta boudnaries
    # defined at boundaries
    ej0_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=ej0)
    ej1_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=ej1)
    ej2_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=ej2)
    ej3_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=ej3)
    fj0_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=fj0)
    fj1_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=fj1)
    fj2_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=fj2)
    fj3_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=fj3)

    prg_flux_eta.update(queue, ((mx+1),(my+1)), None, q0_g, q1_g, q2_g, q3_g, \
            ej0_g, ej1_g, ej2_g, ej3_g, fj0_g, fj1_g, fj2_g, fj3_g)

    cl.enqueue_copy(queue, ej0, ej0_g)
    cl.enqueue_copy(queue, ej1, ej1_g)
    cl.enqueue_copy(queue, ej2, ej2_g)
    cl.enqueue_copy(queue, ej3, ej3_g)
    cl.enqueue_copy(queue, fj0, fj0_g)
    cl.enqueue_copy(queue, fj1, fj1_g)
    cl.enqueue_copy(queue, fj2, fj2_g)
    cl.enqueue_copy(queue, fj3, fj3_g)

    ej[0,:,:] = np.reshape(ej0, (mx, my))
    ej[1,:,:] = np.reshape(ej1, (mx, my))
    ej[2,:,:] = np.reshape(ej2, (mx, my))
    ej[3,:,:] = np.reshape(ej3, (mx, my))
    fj[0,:,:] = np.reshape(fj0, (mx, my))
    fj[1,:,:] = np.reshape(fj1, (mx, my))
    fj[2,:,:] = np.reshape(fj2, (mx, my))
    fj[3,:,:] = np.reshape(fj3, (mx, my))

    return (ei, ej, fi, fj)

# -----------------------------------------------------------------------------
# evulation of source term
def source():
    # defined at cell-centers
    h = np.zeros((lmax, mx, my))
    for i in range(1, mx):
        for j in range(1, my):
            yloc = .25 * (y[i,  j] + y[i-1,  j] \
                        + y[i,j-1] + y[i-1,j-1])
            rho  = q[0,i,j]
            u    = q[1,i,j] / q[0,i,j]
            v    = q[2,i,j] / q[0,i,j]
            et   = q[3,i,j]
            p    = pressure(rho, u, v, et)

            h[0,i,j] = - rho*v / yloc
            h[1,i,j] = - rho*u*v / yloc
            h[2,i,j] = - rho*v*v / yloc
            h[3,i,j] = - (et+p)*v / yloc
    return h

def source_opencl():
    # defined at cell-centers
    h = np.zeros((lmax, mx, my))
    h0 = np.zeros(mx*my).astype(np.float32)
    h1 = np.zeros(mx*my).astype(np.float32)
    h2 = np.zeros(mx*my).astype(np.float32)
    h3 = np.zeros(mx*my).astype(np.float32)

    q0_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=q0)
    q1_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=q1)
    q2_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=q2)
    q3_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=q3)
    y_g  = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=y0)
    h0_g  = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=h0)
    h1_g  = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=h1)
    h2_g  = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=h2)
    h3_g  = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=h3)

    prg_source.update(queue, ((mx+1),(my+1)), None, q0_g, q1_g, q2_g, q3_g, \
            y_g, h0_g, h1_g, h2_g, h3_g)

    cl.enqueue_copy(queue, h0, h0_g)
    cl.enqueue_copy(queue, h1, h1_g)
    cl.enqueue_copy(queue, h2, h2_g)
    cl.enqueue_copy(queue, h3, h3_g)
    h[0,:,:] = h0.reshape((mx, my))
    h[1,:,:] = h1.reshape((mx, my))
    h[2,:,:] = h2.reshape((mx, my))
    h[3,:,:] = h3.reshape((mx, my))

    return h

# -----------------------------------------------------------------------------
# evulation of the dissipation vectors
def dissp():
    kappa2 = visc/4.
    kappa4 = visc/256.

    # compute flow solutions at cell-centers
    pr = np.zeros((mx+1, my+1))
    for i in range(0, mx+1):
        for j in range(0, my):
            rho = q[0,i,j]
            u   = q[1,i,j] / q[0,i,j]
            v   = q[2,i,j] / q[0,i,j]
            et  = q[3,i,j]
            pr[i,j] = pressure(rho, u, v, et)
    
    # compute dxi at constant-xi boundaries
    dwx = np.zeros((lmax, mx, my))
    for j in range(1, my):
        # compute switch function nu_xi at cell-centers
        dp = np.zeros(mx)
        for i in range(1, mx):
            dp[i] = abs(pr[i+1,j] - 2*pr[i,j] + pr[i-1,j]) / \
                    abs(pr[i+1,j] + 2*pr[i,j] + pr[i-1,j])

        # compute dxi*dU/dxi at boundaries
        d1q = np.zeros((lmax, mx))
        for i in range(1, mx-1):
            d1q[:,i] = q[:,i+1,j] - q[:,i,j]

        # compute dxi3*d3U/dxi3 at boundaries
        d3q = np.zeros((lmax, mx))
        for i in range(1, mx-1):
            d3q[:,i] = d1q[:,i+1] - 2*d1q[:,i] + d1q[:,i-1]

        # compute dxi
        for i in range(1, mx-1):
            dnx  = dni_x[i,j]
            dny  = dni_y[i,j]
            rho  = .5 * (q[0,i,  j] + q[0,i+1,j])
            u    = .5 * (q[1,i,  j] / q[0,i,  j] + \
                         q[1,i+1,j] / q[0,i+1,j])
            v    = .5 * (q[2,i,  j] / q[0,i,  j] + \
                         q[2,i+1,j] / q[0,i+1,j])
            p    = .5 * (pr[i,j] + pr[i+1,j])
            c    = np.sqrt(max(0.01,gamma * p / rho))
            lam  = abs((u*dnx + v*dny) / np.sqrt(dnx**2 + dny**2)) + c
            eps2 = kappa2 * max(dp[i], dp[i+1])
            eps4 = max(0., kappa4 - eps2)

            dwx[:,i,j] = lam*(eps2 * d1q[:,i] - eps4 * d3q[:,i])
        dwx[:,0,   j] = 0.
        dwx[:,mx-1,j] = 0.
    
    # compute deta at constant-xi boundaries
    dwy = np.zeros((lmax, mx, my))
    for i in range(1, mx):
        # compute switch function nu_eta at cell-centers
        dp = np.zeros(my)
        for j in range(1, my-1):
            dp[j] = abs(pr[i,j+1] - 2*pr[i,j] + pr[i,j-1]) / \
                    abs(pr[i,j+1] + 2*pr[i,j] + pr[i,j-1])
        dp[0]    = -dp[1]
        dp[my-1] = 2*dp[my-2] - dp[my-3]
        # HERE!

        # compute deta*dU/deta at boundaries
        d1q = np.zeros((lmax, my))
        for j in range(1, my-1):
            d1q[:,j] = q[:,i,j+1] - q[:,i,j]

        # compute deta3*d3U/deta3 at boundaries
        d3q = np.zeros((lmax, my))
        for j in range(1, my-1):
            d3q[:,j] = d1q[:,j+1] - 2*d1q[:,j] + d1q[:,j-1]

        # compute deta
        for j in range(1, my-1):
            dnx  = dnj_x[i,j]
            dny  = dnj_y[i,j]
            rho  = .5 * (q[0,i,  j] + q[0,i,j+1])
            u    = .5 * (q[1,i,  j] / q[0,i,  j] + \
                         q[1,i,j+1] / q[0,i,j+1])
            v    = .5 * (q[2,i,  j] / q[0,i,  j] + \
                         q[2,i,j+1] / q[0,i,j+1])
            p    = .5 * (pr[i,j] + pr[i,j+1])
            c    = np.sqrt(max(0.01,gamma * p / rho))
            lam  = abs((u*dnx + v*dny) / np.sqrt(dnx**2 + dny**2)) + c
            eps2 = kappa2 * max(dp[j], dp[j+1])
            eps4 = max(0., kappa4 - eps2)

            dwy[:,i,j] = lam*(eps2 * d1q[:,j] - eps4 * d3q[:,j])
        dwy[:,i,my-1] = 0.


    return (dwx, dwy, pr)


def dissp_opencl():
    # compute flow solutions at cell-centers
    q0_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=q0)
    q1_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=q1)
    q2_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=q2)
    q3_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=q3)

    pr0  = np.zeros((mx+1)*(my+1)).astype(np.float32)
    pr_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=pr0)

    prg_pressure.update(queue, ((mx+1),(my+1)), None, q0_g, q1_g, q2_g, q3_g, pr_g)

    cl.enqueue_copy(queue, pr0, pr_g)
    pr = np.reshape(pr0, (mx+1, my+1))
    
    # compute dxi at constant-xi boundaries
    dwx = np.zeros((lmax, mx, my)).astype(np.float32)

    dwx0 = np.zeros(mx*my).astype(np.float32)
    dwx1 = np.zeros(mx*my).astype(np.float32)
    dwx2 = np.zeros(mx*my).astype(np.float32)
    dwx3 = np.zeros(mx*my).astype(np.float32)
    dp   = np.zeros(mx*my).astype(np.float32)
    
    dwx0_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=dwx0)
    dwx1_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=dwx1)
    dwx2_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=dwx2)
    dwx3_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=dwx3)
    dp_g   = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=dp)

    dnix_g   = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=dni_x0)
    dniy_g   = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=dni_y0)


    prg_dissp_xi.update(queue, ((mx+1),(my+1)), None, q0_g, q1_g, q2_g, q3_g, \
            pr_g, dwx0_g, dwx1_g, dwx2_g, dwx3_g, dp_g, dnix_g, dniy_g)

    cl.enqueue_copy(queue, dwx0, dwx0_g)
    cl.enqueue_copy(queue, dwx1, dwx1_g)
    cl.enqueue_copy(queue, dwx2, dwx2_g)
    cl.enqueue_copy(queue, dwx3, dwx3_g)
    dwx[0,:,:] = dwx0.reshape((mx, my))
    dwx[1,:,:] = dwx1.reshape((mx, my))
    dwx[2,:,:] = dwx2.reshape((mx, my))
    dwx[3,:,:] = dwx3.reshape((mx, my))
    
    # compute deta at constant-xi boundaries
    dwy = np.zeros((lmax, mx, my)).astype(np.float32)

    dwy0 = np.zeros(mx*my).astype(np.float32)
    dwy1 = np.zeros(mx*my).astype(np.float32)
    dwy2 = np.zeros(mx*my).astype(np.float32)
    dwy3 = np.zeros(mx*my).astype(np.float32)
    dp   = np.zeros(mx*my).astype(np.float32)

    dnjx_g   = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=dnj_x0)
    dnjy_g   = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=dnj_y0)
    
    dwy0_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=dwy0)
    dwy1_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=dwy1)
    dwy2_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=dwy2)
    dwy3_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=dwy3)
    dp_g   = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=dp)

    prg_dissp_eta.update(queue, ((mx+1),(my+1)), None, q0_g, q1_g, q2_g, q3_g, \
            pr_g, dwy0_g, dwy1_g, dwy2_g, dwy3_g, dp_g, dnjx_g, dnjy_g)

    cl.enqueue_copy(queue, dwy0, dwy0_g)
    cl.enqueue_copy(queue, dwy1, dwy1_g)
    cl.enqueue_copy(queue, dwy2, dwy2_g)
    cl.enqueue_copy(queue, dwy3, dwy3_g)
    dwy[0,:,:] = dwy0.reshape((mx, my))
    dwy[1,:,:] = dwy1.reshape((mx, my))
    dwy[2,:,:] = dwy2.reshape((mx, my))
    dwy[3,:,:] = dwy3.reshape((mx, my))

    return (dwx, dwy, pr)

# -----------------------------------------------------------------------------
# evulation of residual vectors
def residual():
    res = np.zeros((lmax, mx, my))
    for i in range(1, mx):
        for j in range(1, my):
            # physical flux
            flux_ab = ej[:,i,j-1] * dnj_x[i,j-1] \
                    + fj[:,i,j-1] * dnj_y[i,j-1]
            flux_bc = ei[:,i,  j] * dni_x[i,  j] \
                    + fi[:,i,  j] * dni_y[i,  j]
            flux_cd = ej[:,i,  j] * dnj_x[i,  j] \
                    + fj[:,i,  j] * dnj_y[i,  j]
            flux_da = ei[:,i-1,j] * dni_x[i-1,j] \
                    + fi[:,i-1,j] * dni_y[i-1,j]
            flux_phys = - flux_ab + flux_bc + flux_cd - flux_da

            # AV flux
            flux_ab = dwy[:,i,j-1] * dnj[i,j-1]
            flux_bc = dwx[:,i,  j] * dni[i,  j]
            flux_cd = dwy[:,i,  j] * dnj[i,  j]
            flux_da = dwx[:,i-1,j] * dni[i-1,j]
            flux_av = - flux_ab + flux_bc + flux_cd - flux_da

            # compute residual vector
            res[:,i,j] = - (flux_phys - flux_av)/area[i,j] \
                         + alpha*h[:,i,j]
    return res

def residual_opencl():
    res = np.zeros((lmax, mx, my)).astype(np.float32)

    res0 = np.zeros(mx*my).astype(np.float32)
    res1 = np.zeros(mx*my).astype(np.float32)
    res2 = np.zeros(mx*my).astype(np.float32)
    res3 = np.zeros(mx*my).astype(np.float32)

    ei0  = np.reshape(ei[0,:,:], mx*my).astype(np.float32)
    ei1  = np.reshape(ei[1,:,:], mx*my).astype(np.float32)
    ei2  = np.reshape(ei[2,:,:], mx*my).astype(np.float32)
    ei3  = np.reshape(ei[3,:,:], mx*my).astype(np.float32)
    ej0  = np.reshape(ej[0,:,:], mx*my).astype(np.float32)
    ej1  = np.reshape(ej[1,:,:], mx*my).astype(np.float32)
    ej2  = np.reshape(ej[2,:,:], mx*my).astype(np.float32)
    ej3  = np.reshape(ej[3,:,:], mx*my).astype(np.float32)
    fi0  = np.reshape(fi[0,:,:], mx*my).astype(np.float32)
    fi1  = np.reshape(fi[1,:,:], mx*my).astype(np.float32)
    fi2  = np.reshape(fi[2,:,:], mx*my).astype(np.float32)
    fi3  = np.reshape(fi[3,:,:], mx*my).astype(np.float32)
    fj0  = np.reshape(fj[0,:,:], mx*my).astype(np.float32)
    fj1  = np.reshape(fj[1,:,:], mx*my).astype(np.float32)
    fj2  = np.reshape(fj[2,:,:], mx*my).astype(np.float32)
    fj3  = np.reshape(fj[3,:,:], mx*my).astype(np.float32)
    h0   = np.reshape(h[0,:,:], mx*my).astype(np.float32)
    h1   = np.reshape(h[1,:,:], mx*my).astype(np.float32)
    h2   = np.reshape(h[2,:,:], mx*my).astype(np.float32)
    h3   = np.reshape(h[3,:,:], mx*my).astype(np.float32)
    dwx0  = np.reshape(dwx[0,:,:], mx*my).astype(np.float32)
    dwx1  = np.reshape(dwx[1,:,:], mx*my).astype(np.float32)
    dwx2  = np.reshape(dwx[2,:,:], mx*my).astype(np.float32)
    dwx3  = np.reshape(dwx[3,:,:], mx*my).astype(np.float32)
    dwy0  = np.reshape(dwy[0,:,:], mx*my).astype(np.float32)
    dwy1  = np.reshape(dwy[1,:,:], mx*my).astype(np.float32)
    dwy2  = np.reshape(dwy[2,:,:], mx*my).astype(np.float32)
    dwy3  = np.reshape(dwy[3,:,:], mx*my).astype(np.float32)

    res0_g   = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=res0)
    res1_g   = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=res1)
    res2_g   = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=res2)
    res3_g   = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=res3)

    ei0_g   = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=ei0)
    ei1_g   = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=ei1)
    ei2_g   = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=ei2)
    ei3_g   = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=ei3)
    ej0_g   = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=ej0)
    ej1_g   = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=ej1)
    ej2_g   = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=ej2)
    ej3_g   = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=ej3)
    fi0_g   = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=fi0)
    fi1_g   = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=fi1)
    fi2_g   = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=fi2)
    fi3_g   = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=fi3)
    fj0_g   = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=fj0)
    fj1_g   = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=fj1)
    fj2_g   = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=fj2)
    fj3_g   = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=fj3)
    h0_g    = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=h0)
    h1_g    = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=h1)
    h2_g    = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=h2)
    h3_g    = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=h3)
    dwx0_g    = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=dwx0)
    dwx1_g    = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=dwx1)
    dwx2_g    = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=dwx2)
    dwx3_g    = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=dwx3)
    dwy0_g    = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=dwy0)
    dwy1_g    = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=dwy1)
    dwy2_g    = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=dwy2)
    dwy3_g    = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=dwy3)

    dnix_g   = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=dni_x0)
    dniy_g   = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=dni_y0)
    dnjx_g   = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=dnj_x0)
    dnjy_g   = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=dnj_y0)
    dni_g   = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=dni0)
    dnj_g   = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=dnj0)
    area_g   = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=area0)


    prg_res.update(queue, (mx,my), None, res0_g, res1_g, res2_g, res3_g, \
        ei0_g, ei1_g, ei2_g, ei3_g, ej0_g, ej1_g, ej2_g, ej3_g, fi0_g, fi1_g, \
        fi2_g, fi3_g, fj0_g, fj1_g, fj2_g, fj3_g, h0_g, h1_g, h2_g, h3_g, \
        dwx0_g, dwx1_g, dwx2_g, dwx3_g, dwy0_g, dwy1_g, dwy2_g, dwy3_g, \
        dnix_g, dniy_g, dnjx_g, dnjy_g, dni_g, dnj_g, area_g)

    cl.enqueue_copy(queue, res0, res0_g)
    cl.enqueue_copy(queue, res1, res1_g)
    cl.enqueue_copy(queue, res2, res2_g)
    cl.enqueue_copy(queue, res3, res3_g)

    res[0,:,:] = res0.reshape((mx, my))
    res[1,:,:] = res1.reshape((mx, my))
    res[2,:,:] = res2.reshape((mx, my))
    res[3,:,:] = res3.reshape((mx, my))

    return res

# -----------------------------------------------------------------------------
# update q in phantom cells along solid boundary
def bc_wall():
    global q
    j = my
    for i in range(1, mx):
        u         = q[1,i,j-1] / q[0,i,j-1]
        v         = q[2,i,j-1] / q[0,i,j-1]
        vn        = (u*dnj_x[i,j-1] + v*dnj_y[i,j-1]) / dnj[i,j-1]
        u_ghost   = u - 2*vn*dnj_x[i,j-1] / dnj[i,j-1]
        v_ghost   = v - 2*vn*dnj_y[i,j-1] / dnj[i,j-1]
        rho_ghost = q[0,i,j-1]
        p_ghost   = pr[i,j-1]
        et_ghost  = p_ghost/(gamma-1) + .5*rho_ghost* \
                    (u_ghost**2 + v_ghost**2)

        q[0,i,j]  = rho_ghost
        q[1,i,j]  = rho_ghost * u_ghost
        q[2,i,j]  = rho_ghost * v_ghost
        q[3,i,j]  = et_ghost
    return

# -----------------------------------------------------------------------------
# update q in phantom cells along symmetric boundary
def bc_symmetric():
    global q
    j = 0
    for i in range(1, mx):
        q[0,i,j] =  q[0,i,j+1]
        q[1,i,j] =  q[1,i,j+1]
        q[2,i,j] = -q[2,i,j+1]
        q[3,i,j] =  q[3,i,j+1]
    return

# -----------------------------------------------------------------------------
# update q in phantom cells along inlet boundary
def bc_inflow():
    global q
    i = 0
    
    # subsonic inflow (ibcin = 1)
    # specify following conditions:
    # inlet flow angle = 0
    # inlet stagnation pressure and temperature
    # extrapolate: static pressure
    if (ibcin == 1):
        for j in range(1, my):
            # extrapolate static pressure
            rho  = q[0,i+1,j]
            u    = q[1,i+1,j] / q[0,i+1,j]
            v    = q[2,i+1,j] / q[0,i+1,j]
            et   = q[3,i+1,j]
            p1   = pressure(rho, u, v, et)
            rho  = q[0,i+2,j]
            u    = q[1,i+2,j] / q[0,i+2,j]
            v    = q[2,i+2,j] / q[0,i+2,j]
            et   = q[3,i+2,j]
            p2   = pressure(rho, u, v, et)
            p    = 2*p1 - p2

            m2   = 2/(gamma-1) * (p**((1-gamma)/gamma) - 1)
            t    = 1/(1 + .5*(gamma-1)*m2)
            rho  = p / t
            u    = np.sqrt(max(0.01, m2*gamma*t))
            v    = 0.
            et   = p/(gamma-1) + .5*rho*(u**2 + v**2)

            q[0,i,j] = rho
            q[1,i,j] = rho*u
            q[2,i,j] = rho*v
            q[3,i,j] = et


    # supersonic inflow (ibcin = 2)
    # specify following conditions:
    # inlet Mach number
    # inlet flow angle = 0
    # inlet stagnation pressue = 1
    # inlet stagnation temperature = 1
    if (ibcin == 2):
        m = 1.5
        for j in range(1, my):
            term = 1 / (1 + .5*(gamma-1) * m**2)
            p    = term**(gamma/(gamma-1))
            rho  = p/term
            c    = np.sqrt(gamma*p/rho)
            u    = m * c
            v    = 0.
            et   = p/(gamma-1) + .5*rho*(u**2 + v**2)

            q[0,i,j] = rho
            q[1,i,j] = rho*u
            q[2,i,j] = rho*v
            q[3,i,j] = et
    return

# ----------------------------------------------------------------------------- # update q in phantom cells along inlet boundary
def bc_outflow():
    global q
    i = mx

    # subsonic outflow (ibcout = 1)
    # specify following condition:
    # static pressure pback
    # extrapolate other variables
    if (ibcout == 1):

        for j in range(1,my):
            p     = pback
            # extraploate rho, u, v
            rho1  = q[0,i-1,j]
            u1    = q[1,i-1,j] / q[0,i-1,j]
            v1    = q[2,i-1,j] / q[0,i-1,j]
            rho2  = q[0,i-2,j]
            u2    = q[1,i-2,j] / q[0,i-2,j]
            v2    = q[2,i-2,j] / q[0,i-2,j]
            rho   = 2*rho1 - rho2
            u     = 2*u1 - u2
            v     = 2*v1 - v2

            et = p/(gamma-1) + .5*rho*(u**2 + v**2)
            q[0,i,j] = rho
            q[1,i,j] = rho*u
            q[2,i,j] = rho*v
            q[3,i,j] = et

    # supersonic outflow (ibcout = 2)
    if (ibcout == 2):
        for j in range(1, my):
            q[:,i,j] = 2*q[:,i-1,j] - q[:,i-2,j]
            #q[:,i,j] = q[:,i-1,j]
    return

# -----------------------------------------------------------------------------
# compute time step
def step():
    # defined at cell-centers
    dt = np.zeros((mx,my))
    for i in range(1, mx):
        for j in range(1, my):
            xi = .5*(x[i,j] - x[i-1,j] + x[i,j-1] - x[i-1,j-1])
            xj = .5*(x[i,j] - x[i,j-1] + x[i-1,j] - x[i-1,j-1])
            yi = .5*(y[i,j] - y[i-1,j] + y[i,j-1] - y[i-1,j-1])
            yj = .5*(y[i,j] - y[i,j-1] + y[i-1,j] - y[i-1,j-1])
            
            rho = q[0,i,j]
            u   = q[1,i,j] / q[0,i,j]
            v   = q[2,i,j] / q[0,i,j]
            et  = q[3,i,j]
            p   = pressure(rho, u, v, et)
            c   = np.sqrt(max(0.01,gamma*p/rho))
            
            qi  =  yj*u - xj*v
            qj  = -yi*u + xi*v
            ci  = c*np.sqrt(xj**2 + yj**2)
            cj  = c*np.sqrt(xi**2 + yi**2)
            dti = area[i,j]/(abs(qi)+abs(ci))
            dtj = area[i,j]/(abs(qj)+abs(cj))
            
            dt[i,j]  = cfl * (dti*dtj) / (dti+dtj)
    return dt

def step_opencl():
    # defined at cell-centers
    dt = np.zeros((mx,my))

    dt0 = np.zeros(mx*my).astype(np.float32)

    dt_g   = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=dt0)

    q0_g   = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=q0)
    q1_g   = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=q1)
    q2_g   = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=q2)
    q3_g   = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=q3)
    x_g    = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=x0)
    y_g    = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=y0)
    area_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=area0)


    prg_step.update(queue, (mx+1,my+1), None, q0_g, q1_g, q2_g, q3_g, \
            x_g, y_g, area_g, dt_g)

    cl.enqueue_copy(queue, dt0, dt_g)

    dt = dt0.reshape((mx, my))

    return dt

# -----------------------------------------------------------------------------
# convergence moniter
def monitor():
    global reslist, stop, line1, line2, line3, line4, cb
    ijmax        = abs(res[0,:,:]).argmax()
    (imax, jmax) = np.unravel_index(ijmax, res[0,:,:].shape)
    resmax       = abs(res[0,imax,jmax])
    resavg       = np.mean(abs(res[0,:,:]))

    if (t % 10 == 0):
        print ' step', '%10s' % 'avg.res', '%10s' % 'max.res', \
                       '%11s' % 'x',       '%11s' % 'y'

    print '%5d' % t, '%.4e' % resavg,        '%.4e' % resmax, \
                     '%+.4e' % x[imax,jmax], '%+.4e' % y[imax,jmax]
    if (resmax > zero): reslist.append(resmax)

    if (t % plot_interval == 0):
        if (t == 1):
            # mid-height
            j = int(mach.shape[-1]/2)
            line1, = ax1.plot(xc[1:,j],mach[1:,j],'-o')
            # symmetric line
            j = 1
            line2, = ax1.plot(xc[1:,j],mach[1:,j],'-o')
            # solid boundary
            j = -1
            line3, = ax1.plot(xc[1:,j],mach[1:,j],'-o')
            # supersonic
            if (ibcin == 2): ax1.set_ylim(1,3)
            # transonic
            if (ibcin == 1): ax1.set_ylim(0.,1.8)

            # contour plot of mach number
            con = ax2.contourf(xc[1:,1:],yc[1:,1:],mach[1:,1:], \
                    10,cmap=plt.cm.rainbow)
            #plt.clabel(con, fontsize=12,colors='k')
            ax2.set_xlim(-1,2)
            ax2.set_ylim(0,1)

            # residual plot
            line4, = ax3.plot(range(0, len(reslist)), reslist)
            ax3.set_yscale('log')
            ax3.set_xlim(0,100)
            ax3.set_ylim(10**(int(np.log10(np.amin(reslist)))-1), \
                         10**(int(np.log10(np.amax(reslist)))+1))
            fig.show()
        elif (t != 0):
            j = int(mach.shape[-1]/2)
            line1.set_ydata(mach[1:,j])
            j = 1
            line2.set_ydata(mach[1:,j])
            j = -1
            line3.set_ydata(mach[1:,j])

            ax2.cla()
            con = ax2.contourf(xc[1:,1:],yc[1:,1:],mach[1:,1:], \
                    10,cmap=plt.cm.rainbow)
            #plt.clabel(con, fontsize=12,colors='k')
            ax2.set_xlim(-1,2)
            ax2.set_ylim(0,1)

            line4.set_xdata(range(0, len(reslist)))
            line4.set_ydata(reslist)
            ax3.set_xlim(0, (t/100+1)*100)
            ax3.set_ylim(10**(int(np.log10(np.amin(reslist)))-1), \
                         10**(int(np.log10(np.amax(reslist)))+1))
            fig.canvas.draw()

    if (resmax < eps): stop = True
    return

# -----------------------------------------------------------------------------
# four-stage Runge-Kutta scheme
def rk4():
    global q, ei, ej, fi, fj, dwx, dwy, pr, h, res
    global q0, q1, q2, q3, dni_x0, dni_y0, dnj_x0, dnj_y0, \
            dni0, dnj0, area0, x0, y0

    rk   = [1./4, 1./3, 1./2, 1.]
    
    dt = np.ones((mx, my))*.01
    
    q_old = np.copy(q)

    if use_opencl:
        dni_x0 = np.reshape(dni_x, mx*my).astype(np.float32)
        dni_y0 = np.reshape(dni_y, mx*my).astype(np.float32)
        dnj_x0 = np.reshape(dnj_x, mx*my).astype(np.float32)
        dnj_y0 = np.reshape(dnj_y, mx*my).astype(np.float32)
        dni0   = np.reshape(dni,   mx*my).astype(np.float32)
        dnj0   = np.reshape(dnj,   mx*my).astype(np.float32)
        area0  = np.reshape(area,  mx*my).astype(np.float32)
        x0     = np.reshape(x,     mx*my).astype(np.float32)
        y0     = np.reshape(y,     mx*my).astype(np.float32)

    # time stepping
    for m in range(0,4):
        # compute flux vectors
        if use_opencl:
            q0  = np.reshape(q[0,:,:], (mx+1)*(my+1))
            q1  = np.reshape(q[1,:,:], (mx+1)*(my+1))
            q2  = np.reshape(q[2,:,:], (mx+1)*(my+1))
            q3  = np.reshape(q[3,:,:], (mx+1)*(my+1))

            (ei, ej, fi, fj) = flux_opencl()
        else:
            (ei, ej, fi, fj) = flux()

        # compute artificial viscosity dissipation vectors
        if use_opencl:
            (dwx, dwy, pr) = dissp_opencl()
        else:
            (dwx, dwy, pr) = dissp()

        # compute source vector
        if (alpha == 1):
            if use_opencl:
                h = source_opencl()
            else:
                h = source()

        # compute residual vector
        if use_opencl:
            res = residual_opencl()
        else:
            res = residual()

        # time step
        if use_opencl:
            dt = step_opencl()
        else:
            dt = step()

        # update solution
        if use_opencl:
            qold0  = np.reshape(q_old[0,:,:], (mx+1)*(my+1)).astype(np.float32)
            qold1  = np.reshape(q_old[1,:,:], (mx+1)*(my+1)).astype(np.float32)
            qold2  = np.reshape(q_old[2,:,:], (mx+1)*(my+1)).astype(np.float32)
            qold3  = np.reshape(q_old[3,:,:], (mx+1)*(my+1)).astype(np.float32)
            res0   = np.reshape(res[0,:,:], mx*my).astype(np.float32)
            res1   = np.reshape(res[1,:,:], mx*my).astype(np.float32)
            res2   = np.reshape(res[2,:,:], mx*my).astype(np.float32)
            res3   = np.reshape(res[3,:,:], mx*my).astype(np.float32)
            dt0    = np.reshape(dt, mx*my).astype(np.float32)

            q0_g   = cl.Buffer(ctx, mf.COPY_HOST_PTR, hostbuf=qold0)
            q1_g   = cl.Buffer(ctx, mf.COPY_HOST_PTR, hostbuf=qold1)
            q2_g   = cl.Buffer(ctx, mf.COPY_HOST_PTR, hostbuf=qold2)
            q3_g   = cl.Buffer(ctx, mf.COPY_HOST_PTR, hostbuf=qold3)

            dt_g   = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=dt0)
            res0_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=res0)
            res1_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=res1)
            res2_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=res2)
            res3_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=res3)

            prg_update[m].update(queue, (mx+1,my+1), None, q0_g, q1_g, q2_g, q3_g, \
            res0_g, res1_g, res2_g, res3_g, dt_g)

            cl.enqueue_copy(queue, qold0, q0_g)
            cl.enqueue_copy(queue, qold1, q1_g)
            cl.enqueue_copy(queue, qold2, q2_g)
            cl.enqueue_copy(queue, qold3, q3_g)
            q[0,:,:] = np.reshape(qold0, (mx+1, my+1))
            q[1,:,:] = np.reshape(qold1, (mx+1, my+1))
            q[2,:,:] = np.reshape(qold2, (mx+1, my+1))
            q[3,:,:] = np.reshape(qold3, (mx+1, my+1))
        else:
            for i in range(1, mx):
                for j in range(1, my):
                    q[:,i,j] = q_old[:,i,j] + rk[m]*dt[i,j]*res[:,i,j]

        # boundary conditions
        bc_wall()
        bc_symmetric()
        bc_inflow()
        bc_outflow()

    # convergence monitor
    calc_mach()

    monitor()
    return

# -----------------------------------------------------------------------------
def calc_mach():
    global xc, yc, mach
    mach = np.zeros((mx,my)).astype(np.float32)
    xc   = np.zeros((mx,my)).astype(np.float32)
    yc   = np.zeros((mx,my)).astype(np.float32)
    for i in range(1,mx):
        for j in range(1,my):
            rho = q[0,i,j]
            u   = q[1,i,j] / q[0,i,j]
            v   = q[2,i,j] / q[0,i,j]
            et  = q[3,i,j]
            p   = pressure(rho, u, v, et)
            c   = np.sqrt(max(0.01,gamma*p/rho))
            mach[i,j] = u/c

            xloc = .25 * (x[i,  j] + x[i-1,  j] \
                        + x[i,j-1] + x[i-1,j-1])
            yloc = .25 * (y[i,  j] + y[i-1,  j] \
                        + y[i,j-1] + y[i-1,j-1])
            xc[i,j] = xloc
            yc[i,j] = yloc

    xc   = xc[1:,1:]
    yc   = yc[1:,1:]
    mach = mach[1:,1:]
    return

# -----------------------------------------------------------------------------
def write_data():
    filename = 'data.txt'
    f = open(filename, 'w')
    for i in range(1, mx):
        for j in range(1, my):
            f.write(str(xc[i,j]) + ' ' + \
                    str(yc[i,j]) + ' ' + \
                    str(q[0,i,j]) + ' ' + \
                    str(q[1,i,j]) + ' ' + \
                    str(q[2,i,j]) + ' ' + \
                    str(q[3,i,j]) + ' ' + \
                    str(mach[i,j]) + '\n')
    f.close()

    # write residual data
    filename = 'residual.txt'
    f = open(filename, 'w')
    for i in range(0, len(reslist)):
        f.write(str(reslist[i]) + '\n')
    f.close()
    return

# -----------------------------------------------------------------------------
# build programs for OpenCL
def opencl_build_programs():
    prg_flux_xi = cl.Program(ctx, """
    #define MX %(mx)d
    #define MY %(my)d
    #define GAMMA 1.4f

    __kernel void update(
        __global float *q0_g,
        __global float *q1_g,
        __global float *q2_g,
        __global float *q3_g,
        __global float *ei0_g,
        __global float *ei1_g,
        __global float *ei2_g,
        __global float *ei3_g,
        __global float *fi0_g,
        __global float *fi1_g,
        __global float *fi2_g,
        __global float *fi3_g
        ) {
        int i = get_global_id(0);
        int j = get_global_id(1);
        
        if (i != MX && j != 0 && j != MY) {
            float rho = .5 * (q0_g[i    *(MY+1)+j] + q0_g[(i+1)*(MY+1)+j]);
            float u   = .5 * (q1_g[i    *(MY+1)+j] / q0_g[i    *(MY+1)+j] \ 
                            + q1_g[(i+1)*(MY+1)+j] / q0_g[(i+1)*(MY+1)+j]);
            float v   = .5 * (q2_g[i    *(MY+1)+j] / q0_g[i    *(MY+1)+j] \ 
                            + q2_g[(i+1)*(MY+1)+j] / q0_g[(i+1)*(MY+1)+j]);
            float et  = .5 * (q3_g[i    *(MY+1)+j] + q3_g[(i+1)*(MY+1)+j]);
            float p   = (GAMMA-1) * (et - .5*rho*(u*u + v*v));

            ei0_g[i*MY+j] = rho * u;
            ei1_g[i*MY+j] = rho * u * u + p;
            ei2_g[i*MY+j] = rho * u * v;
            ei3_g[i*MY+j] = (et + p) * u;
            fi0_g[i*MY+j] = rho * v;
            fi1_g[i*MY+j] = rho * u * v;
            fi2_g[i*MY+j] = rho * v * v + p;
            fi3_g[i*MY+j] = (et + p) * v;
        }
    }
    """ % {"mx": mx, "my": my, "mod":"%"}).build()

    prg_flux_eta = cl.Program(ctx, """
    #define MX %(mx)d
    #define MY %(my)d
    #define GAMMA 1.4f

    __kernel void update(
        __global float *q0_g,
        __global float *q1_g,
        __global float *q2_g,
        __global float *q3_g,
        __global float *ej0_g,
        __global float *ej1_g,
        __global float *ej2_g,
        __global float *ej3_g,
        __global float *fj0_g,
        __global float *fj1_g,
        __global float *fj2_g,
        __global float *fj3_g
        ) {
        int i = get_global_id(0);
        int j = get_global_id(1);
        
        if (i !=0 && i != MX && j != MY) {
            float rho = .5 * (q0_g[i*(MY+1)+j    ] + q0_g[i*(MY+1)+(j+1)]);
            float u   = .5 * (q1_g[i*(MY+1)+j    ] / q0_g[i*(MY+1)+j    ] \ 
                            + q1_g[i*(MY+1)+(j+1)] / q0_g[i*(MY+1)+(j+1)]);
            float v   = .5 * (q2_g[i*(MY+1)+j    ] / q0_g[i*(MY+1)+j    ] \ 
                            + q2_g[i*(MY+1)+(j+1)] / q0_g[i*(MY+1)+(j+1)]);
            float et  = .5 * (q3_g[i*(MY+1)+j    ] + q3_g[i*(MY+1)+(j+1)]);
            float p   = (GAMMA-1) * (et - .5*rho*(u*u + v*v));

            ej0_g[i*MY+j] = rho * u;
            ej1_g[i*MY+j] = rho * u * u + p;
            ej2_g[i*MY+j] = rho * u * v;
            ej3_g[i*MY+j] = (et + p) * u;
            fj0_g[i*MY+j] = rho * v;
            fj1_g[i*MY+j] = rho * u * v;
            fj2_g[i*MY+j] = rho * v * v + p;
            fj3_g[i*MY+j] = (et + p) * v;
        }
    }
    """ % {"mx": mx, "my": my, "mod":"%"}).build()

    prg_source = cl.Program(ctx, """
    #define MX %(mx)d
    #define MY %(my)d
    #define GAMMA 1.4f

    __kernel void update(
        __global float *q0_g,
        __global float *q1_g,
        __global float *q2_g,
        __global float *q3_g,
        __global float *y_g,
        __global float *h0_g,
        __global float *h1_g,
        __global float *h2_g,
        __global float *h3_g
        ) {
        int i = get_global_id(0);
        int j = get_global_id(1);
        
        if (i != 0 && i != MX && j != 0 && j != MY) {
            float yloc = .25 * (y_g[i*MY+j] + y_g[(i-1)*MY+j] \
                    + y_g[i*MY+(j-1)] + y_g[(i-1)*MY+(j-1)]);
            float rho = q0_g[i*(MY+1)+j];
            float u   = q1_g[i*(MY+1)+j] / q0_g[i*(MY+1)+j];
            float v   = q2_g[i*(MY+1)+j] / q0_g[i*(MY+1)+j];
            float et  = q3_g[i*(MY+1)+j];
            float p   = (GAMMA-1) * (et - .5*rho*(u*u + v*v));
            
            h0_g[i*MY+j] = - rho*v / yloc;
            h1_g[i*MY+j] = - rho*u*v / yloc;
            h2_g[i*MY+j] = - rho*v*v / yloc;
            h3_g[i*MY+j] = - (et+p)*v / yloc;
        }
    }
    """ % {"mx": mx, "my": my, "mod":"%"}).build()

    prg_pressure = cl.Program(ctx, """
    #define MX %(mx)d
    #define MY %(my)d
    #define GAMMA 1.4f

    __kernel void update(
        __global float *q0_g,
        __global float *q1_g,
        __global float *q2_g,
        __global float *q3_g,
        __global float *pr_g
        ) {
        int i = get_global_id(0);
        int j = get_global_id(1);
        
        if (j != MY) {
            float rho = q0_g[i*(MY+1)+j];
            float u   = q1_g[i*(MY+1)+j] / q0_g[i*(MY+1)+j];
            float v   = q2_g[i*(MY+1)+j] / q0_g[i*(MY+1)+j];
            float et  = q3_g[i*(MY+1)+j];

            pr_g[i*(MY+1)+j] = (GAMMA-1) * (et - .5*rho*(u*u + v*v));
        }
    }
    """ % {"mx": mx, "my": my, "mod":"%"}).build()

    prg_dissp_xi = cl.Program(ctx, """
    #define MX %(mx)d
    #define MY %(my)d
    #define KAPPA2 %(k2)f
    #define KAPPA4 %(k4)f
    #define GAMMA 1.4f

    __kernel void update(
        __global float *q0_g,
        __global float *q1_g,
        __global float *q2_g,
        __global float *q3_g,
        __global float *pr_g,
        __global float *dwx0_g,
        __global float *dwx1_g,
        __global float *dwx2_g,
        __global float *dwx3_g,
        __global float *dp_g,
        __global float *dnix_g,
        __global float *dniy_g
        ) {
        int i = get_global_id(0);
        int j = get_global_id(1);
        
        if (j != 0 && j != MY) {
            if (i != 0 && i != MX) {
                // compute switch function nu_xi at cell-centers
                dp_g[i*MY+j] = (pr_g[(i+1)*(MY+1)+j] - 2*pr_g[i*(MY+1)+j] + \
                        pr_g[(i-1)*(MY+1)+j]) / (pr_g[(i+1)*(MY+1)+j] + \
                        2*pr_g[i*(MY+1)+j] + pr_g[(i-1)*(MY+1)+j]);
                if (dp_g[i*MY+j] < 0) dp_g[i*MY+j] = -dp_g[i*MY+j];
            }
        }
        barrier(CLK_GLOBAL_MEM_FENCE);

        if (j != 0 && j != MY) {
            if (i != 0 && i < MX-1) {
                // compute dxi*dU/dxi at boundaries
                float d1q0 = q0_g[(i+1)*(MY+1)+j] - q0_g[i*(MY+1)+j];
                float d1q1 = q1_g[(i+1)*(MY+1)+j] - q1_g[i*(MY+1)+j];
                float d1q2 = q2_g[(i+1)*(MY+1)+j] - q2_g[i*(MY+1)+j];
                float d1q3 = q3_g[(i+1)*(MY+1)+j] - q3_g[i*(MY+1)+j];

                // compute dxi3*d3U/dxi3 at boundaries
                float d3q0 = q0_g[(i+2)*(MY+1)+j] - 3*q0_g[(i+1)*(MY+1)+j] \
                        + 3*q0_g[i*(MY+1)+j] - q0_g[(i-1)*(MY+1)+j];
                float d3q1 = q1_g[(i+2)*(MY+1)+j] - 3*q1_g[(i+1)*(MY+1)+j] \
                        + 3*q1_g[i*(MY+1)+j] - q1_g[(i-1)*(MY+1)+j];
                float d3q2 = q2_g[(i+2)*(MY+1)+j] - 3*q2_g[(i+1)*(MY+1)+j] \
                        + 3*q2_g[i*(MY+1)+j] - q2_g[(i-1)*(MY+1)+j];
                float d3q3 = q3_g[(i+2)*(MY+1)+j] - 3*q3_g[(i+1)*(MY+1)+j] \
                        + 3*q3_g[i*(MY+1)+j] - q3_g[(i-1)*(MY+1)+j];

                // compute dxi
                float dnx  = dnix_g[i*MY+j];
                float dny  = dniy_g[i*MY+j];
                float rho = .5 * (q0_g[i    *(MY+1)+j] + q0_g[(i+1)*(MY+1)+j]);
                float u   = .5 * (q1_g[i    *(MY+1)+j] / q0_g[i    *(MY+1)+j] \ 
                                + q1_g[(i+1)*(MY+1)+j] / q0_g[(i+1)*(MY+1)+j]);
                float v   = .5 * (q2_g[i    *(MY+1)+j] / q0_g[i    *(MY+1)+j] \ 
                                + q2_g[(i+1)*(MY+1)+j] / q0_g[(i+1)*(MY+1)+j]);
                float p   = .5 * (pr_g[i*(MY+1)+j] + pr_g[(i+1)*(MY+1)+j]);
                float c   = sqrt(GAMMA * p / rho);
                float lam = (u*dnx + v*dny) / sqrt(dnx*dnx + dny*dny);
                if (lam < 0) lam = -lam;
                lam += c;

                float eps2 = (dp_g[i*MY+j] > dp_g[(i+1)*MY+j]) ? \
                        (KAPPA2 * dp_g[i*MY+j]) : (KAPPA2 * dp_g[(i+1)*MY+j]);
                float eps4 = KAPPA4 - eps2;
                if (eps4 < 0) eps4 = 0;

                dwx0_g[i*MY+j] = lam*(eps2 * d1q0 - eps4 * d3q0);
                dwx1_g[i*MY+j] = lam*(eps2 * d1q1 - eps4 * d3q1);
                dwx2_g[i*MY+j] = lam*(eps2 * d1q2 - eps4 * d3q2);
                dwx3_g[i*MY+j] = lam*(eps2 * d1q3 - eps4 * d3q3);

            }
        }
    }
    """ % {"mx": mx, "my": my, "k2": kappa2, "k4": kappa4, "mod":"%"}).build()

    prg_dissp_eta = cl.Program(ctx, """
    #define MX %(mx)d
    #define MY %(my)d
    #define KAPPA2 %(k2)f
    #define KAPPA4 %(k4)f
    #define GAMMA 1.4f

    __kernel void update(
        __global float *q0_g,
        __global float *q1_g,
        __global float *q2_g,
        __global float *q3_g,
        __global float *pr_g,
        __global float *dwy0_g,
        __global float *dwy1_g,
        __global float *dwy2_g,
        __global float *dwy3_g,
        __global float *dp_g,
        __global float *dnjx_g,
        __global float *dnjy_g
        ) {
        int i = get_global_id(0);
        int j = get_global_id(1);
        
        if (i != 0 && i != MX) {
            if (j != 0 && j < MY-1) {
                // compute switch function nu_eta at cell-centers
                dp_g[i*MY+j] = (pr_g[i*(MY+1)+(j+1)] - 2*pr_g[i*(MY+1)+j] + \
                        pr_g[i*(MY+1)+(j-1)]) / (pr_g[i*(MY+1)+(j+1)] + \
                        2*pr_g[i*(MY+1)+j] + pr_g[i*(MY+1)+(j-1)]);
                if (dp_g[i*MY+j] < 0) dp_g[i*MY+j] = -dp_g[i*MY+j];
            }
        }
        barrier(CLK_GLOBAL_MEM_FENCE);
        if (i != 0 && i != MX) {
            if (j == 0) {
                dp_g[i*MY] = -dp_g[i*MY+1];
            }
            else if (j == MY-1) {
                dp_g[i*MY+j] = 2*dp_g[i*MY+j-1] - dp_g[i*MY+j-2];
            }
        }
        barrier(CLK_GLOBAL_MEM_FENCE);

        if (i != 0 && i != MX) {
            if (j != 0 && j < MY-1) {
                // compute deta*dU/deta at boundaries
                float d1q0 = q0_g[i*(MY+1)+(j+1)] - q0_g[i*(MY+1)+j];
                float d1q1 = q1_g[i*(MY+1)+(j+1)] - q1_g[i*(MY+1)+j];
                float d1q2 = q2_g[i*(MY+1)+(j+1)] - q2_g[i*(MY+1)+j];
                float d1q3 = q3_g[i*(MY+1)+(j+1)] - q3_g[i*(MY+1)+j];

                // compute deta3*d3U/deta3 at boundaries
                float d3q0 = q0_g[i*(MY+1)+(j+2)] - 3*q0_g[i*(MY+1)+(j+1)] \
                        + 3*q0_g[i*(MY+1)+j] - q0_g[i*(MY+1)+(j-1)];
                float d3q1 = q1_g[i*(MY+1)+(j+2)] - 3*q1_g[i*(MY+1)+(j+1)] \
                        + 3*q1_g[i*(MY+1)+j] - q1_g[i*(MY+1)+(j-1)];
                float d3q2 = q2_g[i*(MY+1)+(j+2)] - 3*q2_g[i*(MY+1)+(j+1)] \
                        + 3*q2_g[i*(MY+1)+j] - q2_g[i*(MY+1)+(j-1)];
                float d3q3 = q3_g[i*(MY+1)+(j+2)] - 3*q3_g[i*(MY+1)+(j+1)] \
                        + 3*q3_g[i*(MY+1)+j] - q3_g[i*(MY+1)+(j-1)];

                // compute deta
                float dnx  = dnjx_g[i*MY+j];
                float dny  = dnjy_g[i*MY+j];
                float rho = .5 * (q0_g[i*(MY+1)+j    ] + q0_g[i*(MY+1)+(j+1)]);
                float u   = .5 * (q1_g[i*(MY+1)+j    ] / q0_g[i*(MY+1)+j    ] \ 
                                + q1_g[i*(MY+1)+(j+1)] / q0_g[i*(MY+1)+(j+1)]);
                float v   = .5 * (q2_g[i*(MY+1)+j    ] / q0_g[i*(MY+1)+j    ] \ 
                                + q2_g[i*(MY+1)+(j+1)] / q0_g[i*(MY+1)+(j+1)]);
                float p   = .5 * (pr_g[i*(MY+1)+j    ] + pr_g[i*(MY+1)+(j+1)]);
                float c   = sqrt(GAMMA * p / rho);
                float lam = (u*dnx + v*dny) / sqrt(dnx*dnx + dny*dny);
                if (lam < 0) lam = -lam;
                lam += c;

                float eps2 = (dp_g[i*MY+j] > dp_g[i*MY+j+1]) ? \
                        (KAPPA2 * dp_g[i*MY+j]) : (KAPPA2 * dp_g[i*MY+j+1]);
                float eps4 = KAPPA4 - eps2;
                if (eps4 < 0) eps4 = 0;

                dwy0_g[i*MY+j] = lam*(eps2 * d1q0 - eps4 * d3q0);
                dwy1_g[i*MY+j] = lam*(eps2 * d1q1 - eps4 * d3q1);
                dwy2_g[i*MY+j] = lam*(eps2 * d1q2 - eps4 * d3q2);
                dwy3_g[i*MY+j] = lam*(eps2 * d1q3 - eps4 * d3q3);

            }
        }
    }
    """ % {"mx": mx, "my": my, "k2": kappa2, "k4": kappa4, "mod":"%"}).build()

    prg_res = cl.Program(ctx, """
    #define MX %(mx)d
    #define MY %(my)d
    #define GAMMA 1.4f
    #define ALPHA %(alpha)d

    __kernel void update(
        __global float *res0_g,
        __global float *res1_g,
        __global float *res2_g,
        __global float *res3_g,
        __global float *ei0_g,
        __global float *ei1_g,
        __global float *ei2_g,
        __global float *ei3_g,
        __global float *ej0_g,
        __global float *ej1_g,
        __global float *ej2_g,
        __global float *ej3_g,
        __global float *fi0_g,
        __global float *fi1_g,
        __global float *fi2_g,
        __global float *fi3_g,
        __global float *fj0_g,
        __global float *fj1_g,
        __global float *fj2_g,
        __global float *fj3_g,
        __global float *h0_g,
        __global float *h1_g,
        __global float *h2_g,
        __global float *h3_g,
        __global float *dwx0_g,
        __global float *dwx1_g,
        __global float *dwx2_g,
        __global float *dwx3_g,
        __global float *dwy0_g,
        __global float *dwy1_g,
        __global float *dwy2_g,
        __global float *dwy3_g,
        __global float *dnix_g,
        __global float *dniy_g,
        __global float *dnjx_g,
        __global float *dnjy_g,
        __global float *dni_g,
        __global float *dnj_g,
        __global float *area_g
        ) {
        int i = get_global_id(0);
        int j = get_global_id(1);
        
        if (i != 0 && j != 0) {
            // physical flux
            float flux0_ab = ej0_g[i*MY+(j-1)] * dnjx_g[i*MY+(j-1)] \
                           + fj0_g[i*MY+(j-1)] * dnjy_g[i*MY+(j-1)];
            float flux1_ab = ej1_g[i*MY+(j-1)] * dnjx_g[i*MY+(j-1)] \
                           + fj1_g[i*MY+(j-1)] * dnjy_g[i*MY+(j-1)];
            float flux2_ab = ej2_g[i*MY+(j-1)] * dnjx_g[i*MY+(j-1)] \
                           + fj2_g[i*MY+(j-1)] * dnjy_g[i*MY+(j-1)];
            float flux3_ab = ej3_g[i*MY+(j-1)] * dnjx_g[i*MY+(j-1)] \
                           + fj3_g[i*MY+(j-1)] * dnjy_g[i*MY+(j-1)];
            
            float flux0_bc = ei0_g[i*MY+j] * dnix_g[i*MY+j] \
                           + fi0_g[i*MY+j] * dniy_g[i*MY+j];
            float flux1_bc = ei1_g[i*MY+j] * dnix_g[i*MY+j] \
                           + fi1_g[i*MY+j] * dniy_g[i*MY+j];
            float flux2_bc = ei2_g[i*MY+j] * dnix_g[i*MY+j] \
                           + fi2_g[i*MY+j] * dniy_g[i*MY+j];
            float flux3_bc = ei3_g[i*MY+j] * dnix_g[i*MY+j] \
                           + fi3_g[i*MY+j] * dniy_g[i*MY+j];
            
            float flux0_cd = ej0_g[i*MY+j] * dnjx_g[i*MY+j] \
                           + fj0_g[i*MY+j] * dnjy_g[i*MY+j];
            float flux1_cd = ej1_g[i*MY+j] * dnjx_g[i*MY+j] \
                           + fj1_g[i*MY+j] * dnjy_g[i*MY+j];
            float flux2_cd = ej2_g[i*MY+j] * dnjx_g[i*MY+j] \
                           + fj2_g[i*MY+j] * dnjy_g[i*MY+j];
            float flux3_cd = ej3_g[i*MY+j] * dnjx_g[i*MY+j] \
                           + fj3_g[i*MY+j] * dnjy_g[i*MY+j];
            
            float flux0_da = ei0_g[(i-1)*MY+j] * dnix_g[(i-1)*MY+j] \
                           + fi0_g[(i-1)*MY+j] * dniy_g[(i-1)*MY+j];
            float flux1_da = ei1_g[(i-1)*MY+j] * dnix_g[(i-1)*MY+j] \
                           + fi1_g[(i-1)*MY+j] * dniy_g[(i-1)*MY+j];
            float flux2_da = ei2_g[(i-1)*MY+j] * dnix_g[(i-1)*MY+j] \
                           + fi2_g[(i-1)*MY+j] * dniy_g[(i-1)*MY+j];
            float flux3_da = ei3_g[(i-1)*MY+j] * dnix_g[(i-1)*MY+j] \
                           + fi3_g[(i-1)*MY+j] * dniy_g[(i-1)*MY+j];

            float flux_phys0 = - flux0_ab + flux0_bc + flux0_cd - flux0_da;
            float flux_phys1 = - flux1_ab + flux1_bc + flux1_cd - flux1_da;
            float flux_phys2 = - flux2_ab + flux2_bc + flux2_cd - flux2_da;
            float flux_phys3 = - flux3_ab + flux3_bc + flux3_cd - flux3_da;

            // AV flux
            flux0_ab = dwy0_g[i*MY+(j-1)] * dnj_g[i*MY+(j-1)];
            flux1_ab = dwy1_g[i*MY+(j-1)] * dnj_g[i*MY+(j-1)];
            flux2_ab = dwy2_g[i*MY+(j-1)] * dnj_g[i*MY+(j-1)];
            flux3_ab = dwy3_g[i*MY+(j-1)] * dnj_g[i*MY+(j-1)];

            flux0_bc = dwx0_g[i*MY+j] * dni_g[i*MY+j];
            flux1_bc = dwx1_g[i*MY+j] * dni_g[i*MY+j];
            flux2_bc = dwx2_g[i*MY+j] * dni_g[i*MY+j];
            flux3_bc = dwx3_g[i*MY+j] * dni_g[i*MY+j];

            flux0_cd = dwy0_g[i*MY+j] * dnj_g[i*MY+j];
            flux1_cd = dwy1_g[i*MY+j] * dnj_g[i*MY+j];
            flux2_cd = dwy2_g[i*MY+j] * dnj_g[i*MY+j];
            flux3_cd = dwy3_g[i*MY+j] * dnj_g[i*MY+j];

            flux0_da = dwx0_g[(i-1)*MY+j] * dni_g[(i-1)*MY+j];
            flux1_da = dwx1_g[(i-1)*MY+j] * dni_g[(i-1)*MY+j];
            flux2_da = dwx2_g[(i-1)*MY+j] * dni_g[(i-1)*MY+j];
            flux3_da = dwx3_g[(i-1)*MY+j] * dni_g[(i-1)*MY+j];

            float flux_av0 = - flux0_ab + flux0_bc + flux0_cd - flux0_da;
            float flux_av1 = - flux1_ab + flux1_bc + flux1_cd - flux1_da;
            float flux_av2 = - flux2_ab + flux2_bc + flux2_cd - flux2_da;
            float flux_av3 = - flux3_ab + flux3_bc + flux3_cd - flux3_da;

            // compute residual vector
            res0_g[i*MY+j] = - (flux_phys0 - flux_av0) / area_g[i*MY+j] \
                    + ALPHA * h0_g[i*MY+j];
            res1_g[i*MY+j] = - (flux_phys1 - flux_av1) / area_g[i*MY+j] \
                    + ALPHA * h1_g[i*MY+j];
            res2_g[i*MY+j] = - (flux_phys2 - flux_av2) / area_g[i*MY+j] \
                    + ALPHA * h2_g[i*MY+j];
            res3_g[i*MY+j] = - (flux_phys3 - flux_av3) / area_g[i*MY+j] \
                    + ALPHA * h3_g[i*MY+j];
        }
    }
    """ % {"mx": mx, "my": my, "mod":"%", "alpha": alpha}).build()

    prg_step = cl.Program(ctx, """
    #define MX %(mx)d
    #define MY %(my)d
    #define CFL %(cfl)f
    #define GAMMA 1.4f

    __kernel void update(
        __global float *q0_g,
        __global float *q1_g,
        __global float *q2_g,
        __global float *q3_g,
        __global float *x_g,
        __global float *y_g,
        __global float *area_g,
        __global float *dt_g
        ) {
        int i = get_global_id(0);
        int j = get_global_id(1);
        
        if (i != 0 && i != MX && j != 0 && j != MY) {
            float xi = .5 * (x_g[i*MY+j] - x_g[(i-1)*MY+j] \
                    + x_g[i*MY+(j-1)] - x_g[(i-1)*MY+(j-1)]);
            float xj = .5 * (x_g[i*MY+j] - x_g[i*MY+(j-1)] \
                    + x_g[(i-1)*MY+j] - x_g[(i-1)*MY+(j-1)]);
            float yi = .5 * (y_g[i*MY+j] - y_g[(i-1)*MY+j] \
                    + y_g[i*MY+(j-1)] - y_g[(i-1)*MY+(j-1)]);
            float yj = .5 * (y_g[i*MY+j] - y_g[i*MY+(j-1)] \
                    + y_g[(i-1)*MY+j] - y_g[(i-1)*MY+(j-1)]);

            float rho = q0_g[i*(MY+1)+j];
            float u   = q1_g[i*(MY+1)+j] / q0_g[i*(MY+1)+j];
            float v   = q2_g[i*(MY+1)+j] / q0_g[i*(MY+1)+j];
            float et  = q3_g[i*(MY+1)+j];
            float p   = (GAMMA-1) * (et - .5*rho*(u*u + v*v));
            float c   = sqrt(GAMMA * p / rho);
            
            float qi  =  yj*u - xj*v;
            float qj  = -yi*u + xi*v;
            float ci  = c * sqrt(xj*xj + yj*yj);
            float cj  = c * sqrt(xi*xi + yi*yi);

            if (qi < 0) qi = -qi;
            if (qj < 0) qj = -qj;
            float dti = area_g[i*MY+j] / (qi + ci);
            float dtj = area_g[i*MY+j] / (qj + cj);

            dt_g[i*MY+j] = CFL * (dti*dtj) / (dti+dtj);
        }
    }
    """ % {"mx": mx, "my": my, "cfl": cfl}).build()

    rk   = [1./4, 1./3, 1./2, 1.]
    prg_update = []
    for m in range(0, 4):
            prg_update.append(cl.Program(ctx, """
            #define MX %(mx)d
            #define MY %(my)d
            #define GAMMA 1.4f
            #define RK %(rk)f

            __kernel void update(
                __global float *q0_g,
                __global float *q1_g,
                __global float *q2_g,
                __global float *q3_g,
                __global float *res0_g,
                __global float *res1_g,
                __global float *res2_g,
                __global float *res3_g,
                __global float *dt_g
                ) {
                int i = get_global_id(0);
                int j = get_global_id(1);
                
                if (i != 0 && i != MX && j != 0 && j != MY) {
                    q0_g[i*(MY+1)+j] += RK * dt_g[i*MY+j] * res0_g[i*MY+j];
                    q1_g[i*(MY+1)+j] += RK * dt_g[i*MY+j] * res1_g[i*MY+j];
                    q2_g[i*(MY+1)+j] += RK * dt_g[i*MY+j] * res2_g[i*MY+j];
                    q3_g[i*(MY+1)+j] += RK * dt_g[i*MY+j] * res3_g[i*MY+j];
                }
            }
            """ % {"mx": mx, "my": my, "rk": rk[m]}).build())

    return prg_flux_xi, prg_flux_eta, prg_source, prg_pressure, prg_dissp_xi, \
            prg_dissp_eta, prg_res, prg_step, prg_update

# -----------------------------------------------------------------------------
# main program

use_opencl = True

# grid points
mx = 65; my = 17 
#mx = 129; my = 33

# parameters
lmax = 4
gamma = 1.4
alpha = 1       # axisymmetric flow
plot_interval = 1

cfl  = 0.5
visc = 3
eps  = 1e-3
zero = 1e-12    # define as zero
tmax = 2000

kappa2 = visc/4.
kappa4 = visc/256.

# initial conditions
import_ic = False
icfile    = 'data.txt'

# boundary conditions
pback  = .9
ibcin  = 2
ibcout = 2

# read mesh file
filename = 'ft03.128x32_trans.dat'
filename = 'ft03.dat'
(x, y) = read_mesh(filename)
# plot_mesh()

(dni_x, dni_y, dni, dnj_x, dnj_y, dnj) = calc_normal()
area = calc_area()

# intial condtions
q = ic()

fig = plt.figure(figsize=(6,9))
ax1 = fig.add_subplot(311)
ax2 = fig.add_subplot(312, aspect='equal')
ax3 = fig.add_subplot(313)

# OpenCL setting
platform = cl.get_platforms()[0]
device = platform.get_devices()[1]
ctx = cl.Context([device])
queue = cl.CommandQueue(ctx)

mf = cl.mem_flags
#os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'

# build programs for OpenCL
(prg_flux_xi, prg_flux_eta, prg_source, prg_pressure, prg_dissp_xi, \
        prg_dissp_eta, prg_res, prg_step, prg_update) = opencl_build_programs()

reslist = []
start_time = time.time()
for t in range(0,tmax):
    stop = False
    rk4()

    if (stop and t>10): break

print "execution time:", time.time() - start_time
# write data to file
# write_data()
