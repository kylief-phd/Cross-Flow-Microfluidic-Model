# equations derived from the finite differences of the vorticity equation of the stream function
# used in Gauss_Seidel_Solve_Psi.py

# center x, center y
def Cx_Cy(Psi_ip1_j, Psi_im1_j, Psi_i_jp1, Psi_i_jm1, Psi_ip1_jp1, Psi_im1_jp1, Psi_ip1_jm1, Psi_im1_jm1, Psi_ip2_j, Psi_im2_j, Psi_i_jp2, Psi_i_jm2):
    return (8 * (Psi_ip1_j + Psi_im1_j + Psi_i_jp1 + Psi_i_jm1) + 2 * (-Psi_ip1_jp1 - Psi_im1_jp1 - Psi_ip1_jm1 - Psi_im1_jm1) - Psi_ip2_j - Psi_im2_j - Psi_i_jp2 - Psi_i_jm2) / 20
