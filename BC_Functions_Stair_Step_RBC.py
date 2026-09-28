# Boundary condition functions to solve for the points outside the flow field
# used in Solve_Psi_stair_step_RBC.py

# BC1 based on vx = 0
def BC1_Psi_i_neg1(Psi_i_1):
    return Psi_i_1

# BC2 based on vy = 0
def BC2_Psi_ntopp1_j(Psi_ntopm1_j):
    return Psi_ntopm1_j

# BC3 based on vx=0
def BC3_Psi_i_ntopm1(Psi_i_ntopp1):
    return Psi_i_ntopp1

# BC4 based on vy = 0 condition
def BC4_Psi_nbotp1_j(Psi_nbotm1_j):
    return Psi_nbotm1_j

# BC5 based on dvx/dy = 0
def BC5_Psi_i_nbotp1(Psi_i_nbot, Psi_i_nbotm1):
    return (2 * Psi_i_nbot) - Psi_i_nbotm1

# BC6 based on vx = 0: two versions
# BC6_horizontal, generalized such that i, j references point being solved, ghost point is (i, j+2)
def BC6_horizontal_i_jp2(Psi_i_j):
    return Psi_i_j
# BC6_vertical, generalized such that i, j references point being solved, ghost point is (i-2, j)
def BC6_vertical_im2_j(Psi_i_j):
    return Psi_i_j

# BC7 based on dvy/dx = 0
def BC7_Psi_neg1_j(Psi_0_j, Psi_1_j):
    return (2 * Psi_0_j) - Psi_1_j