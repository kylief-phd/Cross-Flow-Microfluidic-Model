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
# region      Velocity_Solve Function
# ==================================================


def Vel_Solve(n_top, n_bot, n_rbc_x_approx, n_rbc_y_approx, V_max, L, h_decimal,
              Psi_Converged, In_or_Out, BC6_location, precision):
    # Function OUTPUTS: v_x, v_y, v_y_neg, v_mag, v_x_unit, v_y_neg_unit
    # ----- Set Decimal Precision -----
    getcontext().prec = precision

    # ----- Dimensionalize Psi -----
    # UNITS: micron^2/s
    Psi_dimensions = Psi_Converged * V_max * L

    # ----- Initialize v_x & v_y arrays -----
    v_x = np.full((n_bot + 1, n_bot + 1), Decimal('0'), dtype=object)
    v_y = np.full((n_bot + 1, n_bot + 1), Decimal('0'), dtype=object)

    # ----- Calculate Velocity Components, done in 2 steps -----
    # UNITS: All velocity components in micron/s
    # STEP 1
    for j in range(0, n_top):
        for i in range(0, n_top + 1):
            # Step 1, Case 1
            if i == 0 and j == 0:
                # use v_x_forward and v_y_forward
                v_x[j, i] = VF.v_x_forward(Psi_dimensions[j + 1, i], Psi_dimensions[j, i], h_decimal)
                v_y[j, i] = VF.v_y_forward(Psi_dimensions[j, i + 1], Psi_dimensions[j, i], h_decimal)

            # Step 1, Case 2
            elif 1 <= i <= (n_top - 1) and j == 0:
                # use v_y_center and v_x_forward
                v_x[j, i] = VF.v_x_forward(Psi_dimensions[j + 1, i], Psi_dimensions[j, i], h_decimal)
                v_y[j, i] = VF.v_y_center(Psi_dimensions[j, i + 1], Psi_dimensions[j, i - 1], h_decimal)

            # Step 1, Case 3
            elif i == n_top and j == 0:
                # use v_x_forward and v_y_backward
                v_x[j, i] = VF.v_x_forward(Psi_dimensions[j + 1, i], Psi_dimensions[j, i], h_decimal)
                v_y[j, i] = VF.v_y_backward(Psi_dimensions[j, i], Psi_dimensions[j, i - 1], h_decimal)

            # Step 1, Case 4
            elif i == 0 and 1 <= j <= (n_top - 1):
                # use v_x_center and v_y_forward
                v_x[j, i] = VF.v_x_center(Psi_dimensions[j + 1, i], Psi_dimensions[j - 1, i], h_decimal)
                v_y[j, i] = VF.v_y_forward(Psi_dimensions[j, i + 1], Psi_dimensions[j, i], h_decimal)

            # Step 1, Case 5
            elif 1 <= i <= (n_top - 1) and 1 <= j <= (n_top - 1):
                # use v_x_center and v_y_center
                v_x[j, i] = VF.v_x_center(Psi_dimensions[j + 1, i], Psi_dimensions[j - 1, i], h_decimal)
                v_y[j, i] = VF.v_y_center(Psi_dimensions[j, i + 1], Psi_dimensions[j, i - 1], h_decimal)

            # Step 1, Case 6
            elif i == n_top and 1 <= j <= (n_top - 1):
                # use v_x_center and v_y_backward
                v_x[j, i] = VF.v_x_center(Psi_dimensions[j + 1, i], Psi_dimensions[j - 1, i], h_decimal)
                v_y[j, i] = VF.v_y_backward(Psi_dimensions[j, i], Psi_dimensions[j, i - 1], h_decimal)

    # STEP 2
    for j in range(n_top, n_bot + 1):
        for i in range(0, n_bot + 1):
            # Step 2, Case 1
            if i == 0 and n_top <= j <= (n_rbc_y_approx - 1):
                # use v_x_center and v_y_forward
                v_x[j, i] = VF.v_x_center(Psi_dimensions[j + 1, i], Psi_dimensions[j - 1, i], h_decimal)
                v_y[j, i] = VF.v_y_forward(Psi_dimensions[j, i + 1], Psi_dimensions[j, i], h_decimal)

            # Step 2, Case 2
            elif 1 <= i <= n_top and j == n_top:
                # use v_x_center adn v_y_center
                v_x[j, i] = VF.v_x_center(Psi_dimensions[j + 1, i], Psi_dimensions[j - 1, i], h_decimal)
                v_y[j, i] = VF.v_y_center(Psi_dimensions[j, i + 1], Psi_dimensions[j, i - 1], h_decimal)

            # Step 2, Case 3
            elif (n_top + 1) <= i <= (n_bot - 1) and j == n_top:
                # use v_x_forward and v_y_center
                v_x[j, i] = VF.v_x_forward(Psi_dimensions[j + 1, i], Psi_dimensions[j, i], h_decimal)
                v_y[j, i] = VF.v_y_center(Psi_dimensions[j, i + 1], Psi_dimensions[j, i - 1], h_decimal)

            # Step 2, Case 4
            elif i == n_bot and j == n_top:
                # use v_x_forward and v_y_backward
                v_x[j, i] = VF.v_x_forward(Psi_dimensions[j + 1, i], Psi_dimensions[j, i], h_decimal)
                v_y[j, i] = VF.v_y_backward(Psi_dimensions[j, i], Psi_dimensions[j, i - 1], h_decimal)

            # Step 2, Case 5
            elif i == n_bot and (n_top + 1) <= j <= (n_bot - 1):
                # use v_x_center and v_y_backward
                v_x[j, i] = VF.v_x_center(Psi_dimensions[j + 1, i], Psi_dimensions[j - 1, i], h_decimal)
                v_y[j, i] = VF.v_y_backward(Psi_dimensions[j, i], Psi_dimensions[j, i - 1], h_decimal)

            # Step 2, Case 6
            elif i == n_bot and j == n_bot:
                # use v_x_backward and v_y_backward
                v_x[j, i] = VF.v_x_backward(Psi_dimensions[j, i], Psi_dimensions[j - 1, i], h_decimal)
                v_y[j, i] = VF.v_y_backward(Psi_dimensions[j, i], Psi_dimensions[j, i - 1], h_decimal)

            # Step 2, Case 7
            elif (n_rbc_x_approx + 1) <= i <= (n_bot - 1) and j == n_bot:
                # use v_x_backward and v_y_center
                v_x[j, i] = VF.v_x_backward(Psi_dimensions[j, i], Psi_dimensions[j - 1, i], h_decimal)
                v_y[j, i] = VF.v_y_center(Psi_dimensions[j, i + 1], Psi_dimensions[j, i - 1], h_decimal)

            # Step 2, Case 8
            elif 1 <= i <= (n_bot - 1) and (n_top + 1) <= j <= (n_rbc_y_approx - 1):
                # use v_x_center and v_y_center
                v_x[j, i] = VF.v_x_center(Psi_dimensions[j + 1, i], Psi_dimensions[j - 1, i], h_decimal)
                v_y[j, i] = VF.v_y_center(Psi_dimensions[j, i + 1], Psi_dimensions[j, i - 1], h_decimal)

            # Step 2, Case 9
            elif (n_rbc_x_approx + 1) <= i <= (n_bot - 1) and n_rbc_y_approx <= j <= (n_bot - 1):
                # use v_x_center and v_y_center
                v_x[j, i] = VF.v_x_center(Psi_dimensions[j + 1, i], Psi_dimensions[j - 1, i], h_decimal)
                v_y[j, i] = VF.v_y_center(Psi_dimensions[j, i + 1], Psi_dimensions[j, i - 1], h_decimal)

            # Step 2, Case 10
            elif i == 0 and n_rbc_y_approx <= j <= n_bot:
                if BC6_location[j, i] == 0 and In_or_Out[j, i] != 0:
                    print("This node should have been taken care of in Step 2, Case 1")
                    sys.exit("Force Stop, line 124")
                elif BC6_location[j, i] != 0:
                    # ON RBC boundary
                    # check values at (i, j+1) to determine v_x finite difference equation
                    if BC6_location[j + 1, i] == 0 and In_or_Out[j + 1, i] == 0:
                        v_x[j, i] = VF.v_x_backward(Psi_dimensions[j, i], Psi_dimensions[j - 1, i], h_decimal)
                        v_y[j, i] = VF.v_y_forward(Psi_dimensions[j, i + 1], Psi_dimensions[j, i], h_decimal)
                    elif BC6_location[j + 1, i] != 0:
                        v_x[j, i] = VF.v_x_center(Psi_dimensions[j + 1, i], Psi_dimensions[j - 1, i], h_decimal)
                        v_y[j, i] = VF.v_y_forward(Psi_dimensions[j, i + 1], Psi_dimensions[j, i], h_decimal)
                    else:
                        print("Velocity not calculated when it should have been, bypassed all conditions coded for")
                        sys.exit("Force Stop, Velocity_Solver, line 140")
                elif BC6_location[j, i] == 0 and In_or_Out[j, i] == 0:
                    # INSIDE RBC
                    pass
                else:
                    print("Condition not coded for")
                    sys.exit("Force Stop, Velocity_Solver, line 146")

            # Step 2, Case 11
            elif 1 <= i <= n_rbc_x_approx and j == n_bot:
                if BC6_location[j, i] == 0 and In_or_Out[j, i] != 0:
                    print("This node should have been taken care of in Step 2, Case 7")
                    sys.exit("Force Stop, Velocity_Solver, line 152")
                elif BC6_location[j, i] != 0:
                    # ON RBC boundary
                    # check values at (i-1, j) to determine v_y finite difference equation
                    if BC6_location[j, i - 1] == 0 and In_or_Out[j, i - 1] == 0:
                        v_x[j, i] = VF.v_x_backward(Psi_dimensions[j, i], Psi_dimensions[j - 1, i], h_decimal)
                        v_y[j, i] = VF.v_y_forward(Psi_dimensions[j, i + 1], Psi_dimensions[j, i], h_decimal)
                    elif BC6_location[j, i - 1] != 0:
                        v_x[j, i] = VF.v_x_backward(Psi_dimensions[j, i], Psi_dimensions[j - 1, i], h_decimal)
                        v_y[j, i] = VF.v_y_center(Psi_dimensions[j, i + 1], Psi_dimensions[j, i - 1], h_decimal)
                    else:
                        print("Velocity not calculated when it should have been, bypassed all conditions coded for")
                        sys.exit("Force Stop, Velocity_Solver, line 164")
                elif BC6_location[j, i] == 0 and In_or_Out[j, i] == 0:
                    # INSIDE RBC
                    pass
                else:
                    print("Condition not coded for")
                    sys.exit("Force Stop, Velocity_Solver, line 170")

            # Step 2, Case 12
            elif 1 <= i <= n_rbc_x_approx and n_rbc_y_approx <= j <= (n_bot - 1):
                if BC6_location[j, i] == 0 and In_or_Out[j, i] != 0:
                    v_x[j, i] = VF.v_x_center(Psi_dimensions[j + 1, i], Psi_dimensions[j - 1, i], h_decimal)
                    v_y[j, i] = VF.v_y_center(Psi_dimensions[j, i + 1], Psi_dimensions[j, i - 1], h_decimal)
                elif BC6_location[j, i] != 0:
                    # ON RBC boundary
                    # Check values at (i, j+1) to determine v_x finite difference equation
                    if BC6_location[j + 1, i] == 0 and In_or_Out[j + 1, i] == 0:
                        v_x[j, i] = VF.v_x_backward(Psi_dimensions[j, i], Psi_dimensions[j - 1, i], h_decimal)
                    elif BC6_location[j + 1, i] != 0:
                        v_x[j, i] = VF.v_x_center(Psi_dimensions[j + 1, i], Psi_dimensions[j - 1, i], h_decimal)
                    else:
                        print("v_x finite difference equation not selected, condition not coded for")
                        sys.exit("Force Stop, Velocity_Solver, line 186")

                    # Check values at (i-1, j) to determine v_y finite difference equation
                    if BC6_location[j, i - 1] == 0 and In_or_Out[j, i - 1] == 0:
                        v_y[j, i] = VF.v_y_forward(Psi_dimensions[j, i + 1], Psi_dimensions[j, i], h_decimal)
                    elif BC6_location[j, i - 1] != 0:
                        v_y[j, i] = VF.v_y_center(Psi_dimensions[j, i + 1], Psi_dimensions[j, i - 1], h_decimal)
                    else:
                        print("v_y finite difference equation not selected, condition not coded for")
                        sys.exit("Force Stop, Velocity_Solver, line 195")
                elif BC6_location[j, i] == 0 and In_or_Out[j, i] == 0:
                    # INSIDE RBC
                    pass
                else:
                    print("Condition not coded for")
                    sys.exit("Force Stop, Velocity_Solver, line 201")

    # ----- Check for any negative velocities -----
    # step 1 area
    for j in range(0, n_top):
        for i in range(0, n_top + 1):
            if v_x[j, i] < 0:
                v_x[j, i] = 0
            if v_y[j, i] < 0:
                v_y[j, i] = 0
    # step 2 area
    for j in range(n_top, n_bot + 1):
        for i in range(0, n_bot + 1):
            if v_x[j, i] < 0:
                v_x[j, i] = 0
            if v_y[j, i] < 0:
                v_y[j, i] = 0

    # ----- v_y_neg -----
    # multiply v_y by -1 to use with plotting the vector field
    # BECAUSE when plotting the positive y-axis is in the opposite direction of what it was when Psi was solved
    # UNITS: micron/s
    v_y_neg = v_y * -1

    # ----- Velocity Magnitude -----
    # UNITS: micron/s
    # set up matrix to hold results
    v_mag = np.full((n_bot + 1, n_bot + 1), Decimal('0'), dtype=object)
    # fill in magnitude at each node
    for j in range(0, n_bot + 1):
        for i in range(0, n_bot + 1):
            v_mag[j, i] = (v_x[j, i] ** Decimal('2') + v_y[j, i] ** Decimal('2')) ** Decimal('0.5')

    # ----- Velocity Direction as Unit Vectors -----
    # UNITS: unit vectors are dimensionless
    # set up matrices to hold results
    v_x_unit = np.full((n_bot + 1, n_bot + 1), Decimal('0'), dtype=object)
    v_y_neg_unit = np.full((n_bot + 1, n_bot + 1), Decimal('0'), dtype=object)
    # fill in with components of unit vector
    for j in range(0, n_bot + 1):
        for i in range(0, n_bot + 1):
            if v_mag[j, i] == 0:
                v_x_unit[j, i] = 0
                v_y_neg_unit[j, i] = 0
            else:
                v_x_unit[j, i] = v_x[j, i] / v_mag[j, i]
                v_y_neg_unit[j, i] = v_y_neg[j, i] / v_mag[j, i]

    # ----- Function Outputs -----
    return Psi_dimensions, v_x, v_y, v_y_neg, v_mag, v_x_unit, v_y_neg_unit,
