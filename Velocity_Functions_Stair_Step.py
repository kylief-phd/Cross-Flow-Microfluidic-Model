# Functions used to obtain values for velocity components (x and y) from the solved Psi matrix
# Used in Solve_Velocity_Profiles.py
# h is the step size (same for both x and y direction)
def v_x_center(Psi_i_jp1, Psi_i_jm1, h):
    return -(Psi_i_jp1 - Psi_i_jm1) / (2 * h)


def v_x_backward(Psi_i_j, Psi_i_jm1, h):
    return -(Psi_i_j - Psi_i_jm1) / h


def v_x_forward(Psi_i_jp1, Psi_i_j, h):
    return -(Psi_i_jp1 - Psi_i_j) / h


def v_y_center(Psi_ip1_j, Psi_im1_j, h):
    return (Psi_ip1_j - Psi_im1_j) / (2 * h)


def v_y_backward(Psi_i_j, Psi_im1_j, h):
    return (Psi_i_j - Psi_im1_j) / h


def v_y_forward(Psi_ip1_j, Psi_i_j, h):
    return (Psi_ip1_j - Psi_i_j) / h
