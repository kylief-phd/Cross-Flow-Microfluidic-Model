# ==================================================
# region     Import Libraries/Scripts
# ==================================================
# Python Libraries
import numpy as np
import sys
from decimal import Decimal, getcontext

# Other Scripts
import Velocity_Functions_Stair_Step as VF

# endregion

# ==================================================
# region    Zoom and Resolve Velocity Function
# ==================================================


def Zoom_Resolve_Vel(row_expanded, column_expanded, n_row_expanded, n_column_expanded,
                     n_rbc_x_expd_approx, n_rbc_y_expd_approx, V_max, L, h_expd_decimal,
                     Psi_Expd_converged, In_or_Out_zoom_expd_fill, BC6_zoom_expd_fill, precision):
    # Function OUTPUTS: Psi_dimensions_zoom_expd, vx_zoom_expd, vy_zoom_expd, v_y_neg_expd,
    # v_mag_zoom_expd, v_x_unit_expd, v_y_neg_unit_expd)
    # ----- Set Decimal Precision -----
    getcontext().prec = precision

    # ----- Dimensionalize Psi -----
    # UNITS: micron^2/s
    Psi_dimensions_zoom_expd = Psi_Expd_converged * V_max * L

    # ----- Initialize v_x & v_y arrays -----
    vx_zoom_expd = np.full((row_expanded, column_expanded), Decimal('0'), dtype=object)
    vy_zoom_expd = np.full((row_expanded, column_expanded), Decimal('0'), dtype=object)

    # ----- Calculate Velocity Components using Zoomed, Expanded, and Re-Solved Psi Matrix -----\
    # UNITS: All velocity components in micron/s
    for j in range(0, n_row_expanded + 1):
        for i in range(0, n_column_expanded + 1):
            # Case 1
            if i == 0 and j == 0:
                # v_x_forward and v_y_forward
                vx_zoom_expd[j, i] = VF.v_x_forward(Psi_dimensions_zoom_expd[j + 1, i],
                                                    Psi_dimensions_zoom_expd[j, i], h_expd_decimal)
                vy_zoom_expd[j, i] = VF.v_y_forward(Psi_dimensions_zoom_expd[j, i + 1],
                                                    Psi_dimensions_zoom_expd[j, i], h_expd_decimal)

            # Case 2
            if i == 0 and 1 <= j <= (n_rbc_y_expd_approx - 1):
                # use v_x_center and v_y_forward
                vx_zoom_expd[j, i] = VF.v_x_center(Psi_dimensions_zoom_expd[j + 1, i],
                                                   Psi_dimensions_zoom_expd[j - 1, i], h_expd_decimal)
                vy_zoom_expd[j, i] = VF.v_y_forward(Psi_dimensions_zoom_expd[j, i + 1],
                                                    Psi_dimensions_zoom_expd[j, i], h_expd_decimal)

            # Case 3
            elif 1 <= i <= (n_column_expanded - 1) and j == 0:
                # use v_x_forward and v_y_center
                vx_zoom_expd[j, i] = VF.v_x_forward(Psi_dimensions_zoom_expd[j + 1, i],
                                                    Psi_dimensions_zoom_expd[j, i], h_expd_decimal)
                vy_zoom_expd[j, i] = VF.v_y_center(Psi_dimensions_zoom_expd[j, i + 1],
                                                   Psi_dimensions_zoom_expd[j, i - 1], h_expd_decimal)

            # Case 4
            elif i == n_column_expanded and j == 0:
                # use v_x_forward and v_y_backward
                vx_zoom_expd[j, i] = VF.v_x_forward(Psi_dimensions_zoom_expd[j + 1, i],
                                                    Psi_dimensions_zoom_expd[j, i], h_expd_decimal)
                vy_zoom_expd[j, i] = VF.v_y_backward(Psi_dimensions_zoom_expd[j, i],
                                                     Psi_dimensions_zoom_expd[j, i - 1], h_expd_decimal)

            # Case 5
            elif i == n_column_expanded and 1 <= j <= (n_row_expanded - 1):
                # use v_x_center and v_y_backward
                vx_zoom_expd[j, i] = VF.v_x_center(Psi_dimensions_zoom_expd[j + 1, i],
                                                   Psi_dimensions_zoom_expd[j - 1, i], h_expd_decimal)
                vy_zoom_expd[j, i] = VF.v_y_backward(Psi_dimensions_zoom_expd[j, i],
                                                     Psi_dimensions_zoom_expd[j, i - 1], h_expd_decimal)

            # Case 6
            elif i == n_column_expanded and j == n_row_expanded:
                # use v_x_backward and v_y_backward
                vx_zoom_expd[j, i] = VF.v_x_backward(Psi_dimensions_zoom_expd[j, i],
                                                     Psi_dimensions_zoom_expd[j - 1, i], h_expd_decimal)
                vy_zoom_expd[j, i] = VF.v_y_backward(Psi_dimensions_zoom_expd[j, i],
                                                     Psi_dimensions_zoom_expd[j, i - 1], h_expd_decimal)

            # Case 7
            elif (n_rbc_x_expd_approx + 1) <= i <= (n_column_expanded - 1) and j == n_row_expanded:
                # use v_x_backward and v_y_center
                vx_zoom_expd[j, i] = VF.v_x_backward(Psi_dimensions_zoom_expd[j, i],
                                                     Psi_dimensions_zoom_expd[j - 1, i], h_expd_decimal)
                vy_zoom_expd[j, i] = VF.v_y_center(Psi_dimensions_zoom_expd[j, i + 1],
                                                   Psi_dimensions_zoom_expd[j, i - 1], h_expd_decimal)

            # Case 8
            elif i == 0 and n_rbc_y_expd_approx <= j <= n_row_expanded:
                if BC6_zoom_expd_fill[j, i] == 0 and In_or_Out_zoom_expd_fill[j, i] != 0:
                    print("This node should have been taken care of in Case 2")
                    sys.exit("Force Stop, Zoom & Resolve Vel, line 88")
                elif BC6_zoom_expd_fill[j, i] != 0:
                    # ON RBC boundary
                    # check values at (i, j+1) to determine v_x finite difference equation
                    if BC6_zoom_expd_fill[j + 1, i] == 0 and In_or_Out_zoom_expd_fill[j + 1, i] == 0:
                        vx_zoom_expd[j, i] = VF.v_x_backward(Psi_dimensions_zoom_expd[j, i],
                                                             Psi_dimensions_zoom_expd[j - 1, i], h_expd_decimal)
                        vy_zoom_expd[j, i] = VF.v_y_forward(Psi_dimensions_zoom_expd[j, i + 1],
                                                            Psi_dimensions_zoom_expd[j, i], h_expd_decimal)
                    elif BC6_zoom_expd_fill[j + 1, i] != 0:
                        vx_zoom_expd[j, i] = VF.v_x_center(Psi_dimensions_zoom_expd[j + 1, i],
                                                           Psi_dimensions_zoom_expd[j - 1, i], h_expd_decimal)
                        vy_zoom_expd[j, i] = VF.v_y_forward(Psi_dimensions_zoom_expd[j, i + 1],
                                                            Psi_dimensions_zoom_expd[j, i], h_expd_decimal)
                    else:
                        print("Velocity not calculated when it should have been, bypassed all conditions coded for")
                        sys.exit("Force Stop, Zoom & Resolve Vel, line 104")
                elif BC6_zoom_expd_fill[j, i] == 0 and In_or_Out_zoom_expd_fill[j, i] == 0:
                    # INSIDE RBC
                    pass
                else:
                    print("Condition not coded for")
                    sys.exit("Force Stop, Zoom & Resolve Vel, line 110")

            # Case 9
            elif 1 <= i <= n_rbc_x_expd_approx and j == n_row_expanded:
                if BC6_zoom_expd_fill[j, i] == 0 and In_or_Out_zoom_expd_fill[j, i] != 0:
                    print("This node should have been taken care of in Case 7")
                    sys.exit("Force Stop, Zoom & Resolve Vel, line 116")
                elif BC6_zoom_expd_fill[j, i] != 0:
                    # ON RBC boundary
                    # check values at (i-1, j) to determine v_y finite difference equation
                    if BC6_zoom_expd_fill[j, i - 1] == 0 and In_or_Out_zoom_expd_fill[j, i - 1] == 0:
                        vx_zoom_expd[j, i] = VF.v_x_backward(Psi_dimensions_zoom_expd[j, i],
                                                             Psi_dimensions_zoom_expd[j - 1, i], h_expd_decimal)
                        vy_zoom_expd[j, i] = VF.v_y_forward(Psi_dimensions_zoom_expd[j, i + 1],
                                                            Psi_dimensions_zoom_expd[j, i], h_expd_decimal)
                    elif BC6_zoom_expd_fill[j, i - 1] != 0:
                        vx_zoom_expd[j, i] = VF.v_x_backward(Psi_dimensions_zoom_expd[j, i],
                                                             Psi_dimensions_zoom_expd[j - 1, i], h_expd_decimal)
                        vy_zoom_expd[j, i] = VF.v_y_center(Psi_dimensions_zoom_expd[j, i + 1],
                                                           Psi_dimensions_zoom_expd[j, i - 1], h_expd_decimal)
                    else:
                        print("Velocity not calculated when it should have been, bypassed all conditions coded for")
                        sys.exit("Force Stop, Zoom & Resolve Vel, line 132")
                elif BC6_zoom_expd_fill[j, i] == 0 and In_or_Out_zoom_expd_fill[j, i] == 0:
                    # INSIDE RBC
                    pass
                else:
                    print("Condition not coded for")
                    sys.exit("Force Stop, Zoom & Resolve Vel, line 138")

            # Case 10
            elif 1 <= i <= (n_column_expanded - 1) and 1 <= j <= (n_row_expanded - 1):
                if BC6_zoom_expd_fill[j, i] == 0 and In_or_Out_zoom_expd_fill[j, i] != 0:
                    vx_zoom_expd[j, i] = VF.v_x_center(Psi_dimensions_zoom_expd[j + 1, i],
                                                       Psi_dimensions_zoom_expd[j - 1, i], h_expd_decimal)
                    vy_zoom_expd[j, i] = VF.v_y_center(Psi_dimensions_zoom_expd[j, i + 1],
                                                       Psi_dimensions_zoom_expd[j, i - 1], h_expd_decimal)
                elif BC6_zoom_expd_fill[j, i] != 0:
                    # ON RBC boundary
                    # Check values at (i, j+1) to determine v_x finite difference equation
                    if BC6_zoom_expd_fill[j + 1, i] == 0 and In_or_Out_zoom_expd_fill[j + 1, i] == 0:
                        vx_zoom_expd[j, i] = VF.v_x_backward(Psi_dimensions_zoom_expd[j, i],
                                                             Psi_dimensions_zoom_expd[j - 1, i], h_expd_decimal)
                    elif BC6_zoom_expd_fill[j + 1, i] != 0:
                        vx_zoom_expd[j, i] = VF.v_x_center(Psi_dimensions_zoom_expd[j + 1, i],
                                                           Psi_dimensions_zoom_expd[j - 1, i], h_expd_decimal)
                    else:
                        print("v_x finite difference equation not selected, condition not coded for")
                        sys.exit("Force Stop, Zoom & Resolve Vel, line 158")

                    # Check values at (i-1, j) to determine v_y finite difference equation
                    if BC6_zoom_expd_fill[j, i - 1] == 0 and In_or_Out_zoom_expd_fill[j, i - 1] == 0:
                        vy_zoom_expd[j, i] = VF.v_y_forward(Psi_dimensions_zoom_expd[j, i + 1],
                                                            Psi_dimensions_zoom_expd[j, i], h_expd_decimal)
                    elif BC6_zoom_expd_fill[j, i - 1] != 0:
                        vy_zoom_expd[j, i] = VF.v_y_center(Psi_dimensions_zoom_expd[j, i + 1],
                                                           Psi_dimensions_zoom_expd[j, i - 1], h_expd_decimal)
                    else:
                        print("v_y finite difference equation not selected, condition not coded for")
                        sys.exit("Force Stop, Zoom & Resolve Vel, line 169")
                elif BC6_zoom_expd_fill[j, i] == 0 and In_or_Out_zoom_expd_fill[j, i] == 0:
                    # INSIDE RBC
                    pass
                else:
                    print("Condition not coded for")
                    sys.exit("Force Stop, Zoom & Resolve Vel, line 175")

    # ----- Check for any negative velocities -----
    for j in range(0, n_row_expanded + 1):
        for i in range(0, n_column_expanded + 1):
            if vx_zoom_expd[j, i] < 0:
                vx_zoom_expd[j, i] = 0
            if vy_zoom_expd[j, i] < 0:
                vy_zoom_expd[j, i] = 0

    # ----- v_y_Expd_neg -----
    # Multiply v_y by -1 since our +y is in the opposite direction of normal.
    # Python assumes normal +y, so for plotting and determining unit vector direction we need to multiply by -1
    # UNITS: micron/s
    v_y_neg_expd = vy_zoom_expd * -1

    # ----- Velocity Magnitude: Zoom/Expd Grid -----
    # UNITS: micron/s
    # set up matrix to hold results
    v_mag_zoom_expd = np.full((row_expanded, column_expanded), Decimal('0'), dtype=object)
    # fill in magnitude at each node
    for j in range(0, n_row_expanded + 1):
        for i in range(0, n_column_expanded + 1):
            v_mag_zoom_expd[j, i] = (vx_zoom_expd[j, i]**Decimal('2') +
                                     vy_zoom_expd[j, i]**Decimal('2'))**Decimal('0.5')

    # ----- Velocity Direction as Unit Vectors: Zoom/Expd Grid -----
    # UNITS: unit vectors are dimensionless
    # set up matrix to hold results
    v_x_unit_expd = np.full((row_expanded, column_expanded), Decimal('0'), dtype=object)
    v_y_neg_unit_expd = np.full((row_expanded, column_expanded), Decimal('0'), dtype=object)
    # fill in with components of unit vector
    for j in range(0, n_row_expanded + 1):
        for i in range(0, n_column_expanded + 1):
            if v_mag_zoom_expd[j, i] == 0:
                v_x_unit_expd[j, i] = 0
                v_y_neg_unit_expd[j, i] = 0
            else:
                v_x_unit_expd[j, i] = vx_zoom_expd[j, i]/v_mag_zoom_expd[j, i]
                v_y_neg_unit_expd[j, i] = v_y_neg_expd[j, i]/v_mag_zoom_expd[j, i]

    # ----- Function Outputs -----
    return (Psi_dimensions_zoom_expd, vx_zoom_expd, vy_zoom_expd, v_y_neg_expd,
            v_mag_zoom_expd, v_x_unit_expd, v_y_neg_unit_expd)

# endregion
