# ==================================================
# region     Import Libraries/Scripts
# ==================================================
# Python Libraries
import numpy as np
import sys
from decimal import Decimal, getcontext
import math

# Other Scripts
import Shear_Rate_Func_Stair_Step as SR

# endregion

# ==================================================
# region        Shear Rate Function
# ==================================================


def Shear_Rate(row_expanded, column_expanded, n_row_expanded, n_column_expanded,
               n_rbc_x_expd_approx, n_rbc_y_expd_approx, h_exp_decimal,
               vx_zoom_expd, vy_zoom_expd, In_or_Out_zoom_expd_fill, BC6_zoom_expd_fill, precision, a, b):
    # Function OUTPUTS: dvx_dx_expd, dvy_dy_expd, dvy_dx_expd, dvx_dy_expd, gamma_xx, gamma_yy, gamma_yx, gamma_xy,
    #                   gamma_xx_func_of_x, gamma_yy_func_of_x, gamma_yx_func_of_x, gamma_xy_func_of_x

    # NOTE: Outputs are all in inverse seconds, NO outputs multiplied by -1 in tau equation yet
    # Multiplication by -1 and viscosity will happen in Tension and Force Calculations

    # ----- Set Decimal Precision -----
    getcontext().prec = precision

    # ----- Initialize Matrices to Hold Velocity Gradients -----
    dvy_dx_expd = np.full((row_expanded, column_expanded), Decimal('0'), dtype=object)
    dvx_dy_expd = np.full((row_expanded, column_expanded), Decimal('0'), dtype=object)
    dvy_dy_expd = np.full((row_expanded, column_expanded), Decimal('0'), dtype=object)
    dvx_dx_expd = np.full((row_expanded, column_expanded), Decimal('0'), dtype=object)

    # ----- Calculate Velocity Gradients -----
    # UNITS: all velocity gradients in 1/s
    for j in range(0, n_row_expanded + 1):
        for i in range(0, n_column_expanded + 1):
            if BC6_zoom_expd_fill[j, i] != 0:
                # ON the RBC (BC6) boarder
                # Check conditions at (i-1, j) to determine finite diff for derivatives wrt x
                if i == 0 and j == n_rbc_y_expd_approx:
                    # (i-1, j) references outside the grid, use forward difference for derivatives wrt x
                    dvy_dx_expd[j, i] = SR.dvy_dx_forward(vy_zoom_expd[j, i + 1], vy_zoom_expd[j, i], h_exp_decimal)
                    dvx_dx_expd[j, i] = SR.dvx_dx_forward(vx_zoom_expd[j, i + 1], vx_zoom_expd[j, i], h_exp_decimal)

                elif BC6_zoom_expd_fill[j, i - 1] == 0 and In_or_Out_zoom_expd_fill[j, i - 1] == 0:
                    # (i-1, j) references inside the RBC, use forward difference for derivatives wrt x
                    dvy_dx_expd[j, i] = SR.dvy_dx_forward(vy_zoom_expd[j, i + 1], vy_zoom_expd[j, i], h_exp_decimal)
                    dvx_dx_expd[j, i] = SR.dvx_dx_forward(vx_zoom_expd[j, i + 1], vx_zoom_expd[j, i], h_exp_decimal)

                elif BC6_zoom_expd_fill[j, i - 1] != 0:
                    # (i-1, j) still on RBC boarder (BC6), use centered difference for derivatives wrt x
                    dvy_dx_expd[j, i] = SR.dvy_dx_center(vy_zoom_expd[j, i + 1], vy_zoom_expd[j, i - 1], h_exp_decimal)
                    dvx_dx_expd[j, i] = SR.dvx_dx_center(vx_zoom_expd[j, i + 1], vx_zoom_expd[j, i - 1], h_exp_decimal)

                else:
                    print("dvy_dx or dvx_dx finite difference equation not selected, condition not coded for")
                    sys.exit("Force Stop, Shear Rate, line 56")

                # Check conditions at (i, j+1) to determine finite diff for derivatives wrt y
                if i == n_rbc_x_expd_approx and j == n_row_expanded:
                    # (i, j+1) references outside grid, use backward difference for derivatives wrt y
                    dvx_dy_expd[j, i] = SR.dvx_dy_backward(vx_zoom_expd[j, i], vx_zoom_expd[j - 1, i], h_exp_decimal)
                    dvy_dy_expd[j, i] = SR.dvy_dy_backward(vy_zoom_expd[j, i], vy_zoom_expd[j - 1, i], h_exp_decimal)

                elif BC6_zoom_expd_fill[j + 1, i] == 0 and In_or_Out_zoom_expd_fill[j + 1, i] == 0:
                    # (i, j+1) references inside RBC, use backward difference for derivatives wrt y
                    dvx_dy_expd[j, i] = SR.dvx_dy_backward(vx_zoom_expd[j, i], vx_zoom_expd[j - 1, i], h_exp_decimal)
                    dvy_dy_expd[j, i] = SR.dvy_dy_backward(vy_zoom_expd[j, i], vy_zoom_expd[j - 1, i], h_exp_decimal)

                elif BC6_zoom_expd_fill[j + 1, i] != 0:
                    # (i, j+1) still on RBC boarder (BC6), use centered difference for derivatives wrt y
                    dvx_dy_expd[j, i] = SR.dvx_dy_center(vx_zoom_expd[j + 1, i], vx_zoom_expd[j - 1, i], h_exp_decimal)
                    dvy_dy_expd[j, i] = SR.dvy_dy_center(vy_zoom_expd[j + 1, i], vy_zoom_expd[j - 1, i], h_exp_decimal)

                else:
                    print("dvx_dy or dvy_dy finite difference equation not selected, condition not coded for")
                    sys.exit("Force Stop, Shear Rate, line 82")
            else:
                # We only care about RBC boarder (BC6)
                pass

    # ----- Combine velocity gradients into shear rate components -----
    # Shear rate NOT multiplied by -1 yet
    # UNITS: 1/s
    gamma_xx = 2 * dvx_dx_expd
    gamma_yy = 2 * dvy_dy_expd
    gamma_yx = dvy_dx_expd + dvx_dy_expd
    gamma_xy = dvy_dx_expd + dvx_dy_expd

    # ----- Map back to exact geometry -----
    # Initialize Matrices
    # row 0 = x coordinate (micron), row 1 = corresponding gamma value (1/s)
    gamma_xx_func_of_x = np.full((2, 1), Decimal('0'), dtype=object)
    gamma_yy_func_of_x = np.full((2, 1), Decimal('0'), dtype=object)
    gamma_yx_func_of_x = np.full((2, 1), Decimal('0'), dtype=object)
    gamma_xy_func_of_x = np.full((2, 1), Decimal('0'), dtype=object)

    # Height of zoom/expanded grid
    H_expd = n_row_expanded * h_exp_decimal

    # Fill in matrices
    for j in range(0, n_row_expanded + 1):
        for i in range(0, n_column_expanded + 1):
            if BC6_zoom_expd_fill[j, i] != 0 and i == 0:
                # ON boarder and ON first node of RBC (ie radius from line of symmetry = b)
                # Check for horizontal, vertical, or diagonal (should be horizontal always)
                # should just reset first column values, after which we will append the func_of_x arrays
                if BC6_zoom_expd_fill[j, i + 1] != 0:
                    # On horizontal segment
                    x_pos = i * h_exp_decimal
                    gamma_xx_func_of_x[0, i] = x_pos
                    gamma_xx_func_of_x[1, i] = gamma_xx[j, i]
                    gamma_yy_func_of_x[0, i] = x_pos
                    gamma_yy_func_of_x[1, i] = gamma_yy[j, i]
                    gamma_yx_func_of_x[0, i] = x_pos
                    gamma_yx_func_of_x[1, i] = gamma_yx[j, i]
                    gamma_xy_func_of_x[0, i] = x_pos
                    gamma_xy_func_of_x[1, i] = gamma_xy[j, i]
                elif BC6_zoom_expd_fill[j+1, i] != 0:
                    # On a vertical segment
                    y_pos = j * h_exp_decimal
                    # Find corresponding x on exact geometry
                    x_exact = math.sqrt(a ** 2 * (1 - ((y_pos - H_expd) ** 2 / b ** 2)))
                    gamma_xx_func_of_x[0, i] = x_exact
                    gamma_xx_func_of_x[1, i] = gamma_xx[j, i]
                    gamma_yy_func_of_x[0, i] = x_exact
                    gamma_yy_func_of_x[1, i] = gamma_yy[j, i]
                    gamma_yx_func_of_x[0, i] = x_exact
                    gamma_yx_func_of_x[1, i] = gamma_yx[j, i]
                    gamma_xy_func_of_x[0, i] = x_exact
                    gamma_xy_func_of_x[1, i] = gamma_xy[j, i]
                elif BC6_zoom_expd_fill[j+1, i+1] != 0:
                    # On diagonal segment
                    x_pos = i * h_exp_decimal
                    gamma_xx_func_of_x[0, i] = x_pos
                    gamma_xx_func_of_x[1, i] = gamma_xx[j, i]
                    gamma_yy_func_of_x[0, i] = x_pos
                    gamma_yy_func_of_x[1, i] = gamma_yy[j, i]
                    gamma_yx_func_of_x[0, i] = x_pos
                    gamma_yx_func_of_x[1, i] = gamma_yx[j, i]
                    gamma_xy_func_of_x[0, i] = x_pos
                    gamma_xy_func_of_x[1, i] = gamma_xy[j, i]
                else:
                    print("Condition not coded for")
                    sys.exit("Force Stop, Shear Rate, line 144")
            elif BC6_zoom_expd_fill[j, i] != 0 and i != 0 and j != n_row_expanded:
                # ON boarder and NOT on any endpoint
                # Determine horizontal, vertical, or diagonal segment
                if BC6_zoom_expd_fill[j, i + 1] != 0:
                    # On horizontal
                    x_pos = i * h_exp_decimal
                    gamma_xx_func_of_x = np.append(gamma_xx_func_of_x, np.array([[x_pos], [gamma_xx[j, i]]]), axis=1)
                    gamma_yy_func_of_x = np.append(gamma_yy_func_of_x, np.array([[x_pos], [gamma_yy[j, i]]]), axis=1)
                    gamma_yx_func_of_x = np.append(gamma_yx_func_of_x, np.array([[x_pos], [gamma_yx[j, i]]]), axis=1)
                    gamma_xy_func_of_x = np.append(gamma_xy_func_of_x, np.array([[x_pos], [gamma_xy[j, i]]]), axis=1)
                elif BC6_zoom_expd_fill[j + 1, i] != 0:
                    # On vertical
                    y_pos = j * h_exp_decimal
                    # Find corresponding x on exact geometry
                    x_exact = math.sqrt(a ** 2 * (1 - ((y_pos - H_expd) ** 2 / b ** 2)))
                    gamma_xx_func_of_x = np.append(gamma_xx_func_of_x, np.array([[x_exact], [gamma_xx[j, i]]]), axis=1)
                    gamma_yy_func_of_x = np.append(gamma_yy_func_of_x, np.array([[x_exact], [gamma_yy[j, i]]]), axis=1)
                    gamma_yx_func_of_x = np.append(gamma_yx_func_of_x, np.array([[x_exact], [gamma_yx[j, i]]]), axis=1)
                    gamma_xy_func_of_x = np.append(gamma_xy_func_of_x, np.array([[x_exact], [gamma_xy[j, i]]]), axis=1)
                elif BC6_zoom_expd_fill[j+1, i+1] != 0:
                    # On diagonal
                    x_pos = i * h_exp_decimal
                    gamma_xx_func_of_x = np.append(gamma_xx_func_of_x, np.array([[x_pos], [gamma_xx[j, i]]]), axis=1)
                    gamma_yy_func_of_x = np.append(gamma_yy_func_of_x, np.array([[x_pos], [gamma_yy[j, i]]]), axis=1)
                    gamma_yx_func_of_x = np.append(gamma_yx_func_of_x, np.array([[x_pos], [gamma_yx[j, i]]]), axis=1)
                    gamma_xy_func_of_x = np.append(gamma_xy_func_of_x, np.array([[x_pos], [gamma_xy[j, i]]]), axis=1)
                else:
                    print("Condition not coded for")
                    sys.exit("Force Stop, Shear Rate, line 173")
            elif BC6_zoom_expd_fill[j, i] != 0 and j == n_row_expanded:
                # ON boarder and ON last node of RBC (ie radius from line of symmetry = 0)
                # we have to check in reverse direction for horizontal, vertical, or diagonal
                if BC6_zoom_expd_fill[j, i-1] != 0:
                    # ON horizontal
                    x_pos = i * h_exp_decimal
                    gamma_xx_func_of_x = np.append(gamma_xx_func_of_x, np.array([[x_pos], [gamma_xx[j, i]]]), axis=1)
                    gamma_yy_func_of_x = np.append(gamma_yy_func_of_x, np.array([[x_pos], [gamma_yy[j, i]]]), axis=1)
                    gamma_yx_func_of_x = np.append(gamma_yx_func_of_x, np.array([[x_pos], [gamma_yx[j, i]]]), axis=1)
                    gamma_xy_func_of_x = np.append(gamma_xy_func_of_x, np.array([[x_pos], [gamma_xy[j, i]]]), axis=1)
                elif BC6_zoom_expd_fill[j-1, i] != 0:
                    # ON vertical
                    # Check for backwards check, check vertical prior to diagonal
                    # Sometime last node on j = n_bot will have two neighbors (one vertical and one diagonal)
                    # Checking vertical first ensures we "draw" the shape correctly
                    y_pos = j * h_exp_decimal
                    # Find corresponding x on exact geometry
                    x_exact = math.sqrt(a ** 2 * (1 - ((y_pos - H_expd) ** 2 / b ** 2)))
                    gamma_xx_func_of_x = np.append(gamma_xx_func_of_x, np.array([[x_exact], [gamma_xx[j, i]]]), axis=1)
                    gamma_yy_func_of_x = np.append(gamma_yy_func_of_x, np.array([[x_exact], [gamma_yy[j, i]]]), axis=1)
                    gamma_yx_func_of_x = np.append(gamma_yx_func_of_x, np.array([[x_exact], [gamma_yx[j, i]]]), axis=1)
                    gamma_xy_func_of_x = np.append(gamma_xy_func_of_x, np.array([[x_exact], [gamma_xy[j, i]]]), axis=1)
                elif BC6_zoom_expd_fill[j-1, i-1] != 0:
                    # ON diagonal
                    x_pos = i * h_exp_decimal
                    gamma_xx_func_of_x = np.append(gamma_xx_func_of_x, np.array([[x_pos], [gamma_xx[j, i]]]), axis=1)
                    gamma_yy_func_of_x = np.append(gamma_yy_func_of_x, np.array([[x_pos], [gamma_yy[j, i]]]), axis=1)
                    gamma_yx_func_of_x = np.append(gamma_yx_func_of_x, np.array([[x_pos], [gamma_yx[j, i]]]), axis=1)
                    gamma_xy_func_of_x = np.append(gamma_xy_func_of_x, np.array([[x_pos], [gamma_xy[j, i]]]), axis=1)
                else:
                    print("Condition not coded for")
                    sys.exit("Force Stop, Shear Rate, line 205")

    # ----- Function Outputs -----
    return (dvx_dx_expd, dvy_dy_expd, dvy_dx_expd, dvx_dy_expd,
            gamma_xx, gamma_yy, gamma_yx, gamma_xy,
            gamma_xx_func_of_x, gamma_yy_func_of_x, gamma_yx_func_of_x, gamma_xy_func_of_x)

# endregion
