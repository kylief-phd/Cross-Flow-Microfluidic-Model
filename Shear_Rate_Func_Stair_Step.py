# Functions used to calculate components of shear rate
# h is the step size (same for both x and y direction)

# ==================================================
#                      dvy_dx
# ==================================================
# dvy_dx forward finite difference
def dvy_dx_forward(vy_ip1_j, vy_i_j, h):
    return (vy_ip1_j - vy_i_j) / h


# dvy_dx center finite difference
def dvy_dx_center(vy_ip1_j, vy_im1_j, h):
    return (vy_ip1_j - vy_im1_j) / (2 * h)


# ==================================================
#                      dvx_dy
# ==================================================
# dvx_dy backward finite difference
def dvx_dy_backward(vx_i_j, vx_i_jm1, h):
    return (vx_i_j - vx_i_jm1) / h


# dvx_dy center finite difference
def dvx_dy_center(vx_i_jp1, vx_i_jm1, h):
    return (vx_i_jp1 - vx_i_jm1) / (2 * h)


# ==================================================
#                      dvy_dy
# ==================================================
def dvy_dy_backward(vy_i_j, vy_i_jm1, h):
    return (vy_i_j - vy_i_jm1) / h


def dvy_dy_center(vy_i_jp1, vy_i_jm1, h):
    return (vy_i_jp1 - vy_i_jm1) / (2 * h)


# ==================================================
#                      dvx_dx
# ==================================================
def dvx_dx_forward(vx_ip1_j, vx_i_j, h):
    return (vx_ip1_j - vx_i_j) / h


def dvx_dx_center(vx_ip1_j, vx_im1_j, h):
    return (vx_ip1_j - vx_im1_j) / (2 * h)
