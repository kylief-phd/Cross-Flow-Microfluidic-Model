# ==================================================
# region     Import Libraries/Scripts
# ==================================================
# Python Libraries
import numpy as np
import sys
import math
from decimal import Decimal, getcontext

# Other Scripts
import BC_Functions_Stair_Step_RBC as BCF
import Psi_Finite_Diff_Func_Stair_Step as PsiFunc

# endregion

# ==================================================
# region      Psi_Solve Function
# ==================================================


def Psi_Solve(n_top, n_bot, n_rbc_x_approx, n_rbc_y_approx, epsilon_stop, iter_max,
              Psi_initial, In_or_Out, BC6_location, precision):
    # Function OUTPUT: Psi (numpy array), err (numpy array), k (integer)
    # ----- Set decimal precision -----
    getcontext().prec = precision

    # ----- Initialize Psi Matrix used for solving -----
    # This will start as a copy of the Psi_Initial, which is a function input
    Psi = np.copy(Psi_initial)

    # ----- Initialize Psi_0 -----
    # Psi_0 is used to test convergence of Psi, it holds previous values (before next iteration occurs)
    # Values will be updated after each iteration
    # At this point it holds the initial values as no iterations have occurred
    Psi_0 = np.copy(Psi)

    # ----- Convergence criteria -----
    # Global relative error was set by user
    epsilon_global = epsilon_stop
    # Max absolute error two orders of magnitude tighter
    epsilon_max = epsilon_stop / (Decimal('1e2'))

    # ----- Begin Iterations, done in 2 steps -----
    Max_AE = 1
    Global_RE = 1
    RMS = 1
    converged = False  # set to false, so we don't start "converged"
    k = 0  # iteration counter
    print('Begin Iterations, k = ' + str(k))

    # iterate so long as we don't meet convergence criteria
    while converged is False:
        # add one to iteration counter at start of each iteration
        k += 1

        # visual indicator that code hasn't frozen and convergence criteria status
        if k % 1000 == 0:
            print("k = " + str(k) + '; Global_RE = ' + str(Global_RE) + '; Max_AE = ' + str(Max_AE) + '; RMS = ' + str(RMS))

        # STEP 1: (i = [1, n_top-1] and j = [1,  n_top], this excludes boundaries)
        for j in range(1, n_top + 1):
            for i in range(1, n_top):
                # Step 1, Case 1
                if i == 1 and j == 1:
                    # The top left corner of step 1 area (BC1 and BC7 ghost points needed), Psi(i, -1) & Psi(-1, j)
                    Psi_i_minus1_BC1 = BCF.BC1_Psi_i_neg1(Psi[1, i])
                    Psi_minus1_j_BC7 = BCF.BC7_Psi_neg1_j(Psi[j, 0], Psi[j, 1])
                    # Apply finite diff (Cx_Cy)
                    Psi[j, i] = PsiFunc.Cx_Cy(Psi[j, i + 1], Psi[j, i - 1], Psi[j + 1, i],
                                              Psi[j - 1, i], Psi[j + 1, i + 1], Psi[j + 1, i - 1],
                                              Psi[j - 1, i + 1], Psi[j - 1, i - 1], Psi[j, i + 2],
                                              Psi_minus1_j_BC7, Psi[j + 2, i], Psi_i_minus1_BC1)
                # Step 1, Case 2
                elif 2 <= i <= (n_top - 2) and j == 1:
                    # Top edge of step 1 area (BC1 ghost point needed), Psi(i, -1)
                    Psi_i_minus1_BC1 = BCF.BC1_Psi_i_neg1(Psi[1, i])
                    # Apply finite diff (Cx_Cy)
                    Psi[j, i] = PsiFunc.Cx_Cy(Psi[j, i + 1], Psi[j, i - 1], Psi[j + 1, i],
                                              Psi[j - 1, i], Psi[j + 1, i + 1], Psi[j + 1, i - 1],
                                              Psi[j - 1, i + 1], Psi[j - 1, i - 1], Psi[j, i + 2],
                                              Psi[j, i - 2], Psi[j + 2, i], Psi_i_minus1_BC1)
                # Step 1, Case 3
                elif i == (n_top - 1) and j == 1:
                    # Top right corner of step 1 area (BC1 and BC2 ghost points needed), Psi(i, -1) & Psi(n_top+1, j)
                    Psi_i_minus1_BC1 = BCF.BC1_Psi_i_neg1(Psi[1, i])
                    Psi_ntopplus1_j_BC2 = BCF.BC2_Psi_ntopp1_j(Psi[j, n_top - 1])
                    # Apply finite diff (Cx_Cy)
                    Psi[j, i] = PsiFunc.Cx_Cy(Psi[j, i + 1], Psi[j, i - 1], Psi[j + 1, i],
                                              Psi[j - 1, i], Psi[j + 1, i + 1], Psi[j + 1, i - 1],
                                              Psi[j - 1, i + 1], Psi[j - 1, i - 1], Psi_ntopplus1_j_BC2,
                                              Psi[j, i - 2], Psi[j + 2, i], Psi_i_minus1_BC1)
                # Step 1, Case 4
                elif i == 1 and j >= 2:
                    # The left edge of step 1 area (BC7 ghost point needed), Psi(-1, j)
                    Psi_minus1_j_BC7 = BCF.BC7_Psi_neg1_j(Psi[j, 0], Psi[j, 1])
                    # Apply finite diff (Cx_Cy)
                    Psi[j, i] = PsiFunc.Cx_Cy(Psi[j, i + 1], Psi[j, i - 1], Psi[j + 1, i],
                                              Psi[j - 1, i], Psi[j + 1, i + 1], Psi[j + 1, i - 1],
                                              Psi[j - 1, i + 1], Psi[j - 1, i - 1], Psi[j, i + 2],
                                              Psi_minus1_j_BC7, Psi[j + 2, i], Psi[j - 2, i])
                # Step 1, Case 5
                elif 2 <= i <= (n_top - 2) and j >= 2:
                    # Center of step 1 area, No ghost points needed
                    # Apply finite diff (Cx_Cy)
                    Psi[j, i] = PsiFunc.Cx_Cy(Psi[j, i + 1], Psi[j, i - 1], Psi[j + 1, i],
                                              Psi[j - 1, i], Psi[j + 1, i + 1], Psi[j + 1, i - 1],
                                              Psi[j - 1, i + 1], Psi[j - 1, i - 1], Psi[j, i + 2],
                                              Psi[j, i - 2], Psi[j + 2, i], Psi[j - 2, i])
                # Step 1, Case 6
                elif i == n_top - 1 and 2 <= j <= (n_top - 1):
                    # The right edge of step 1 area (BC2 ghost point needed), Psi(n_top+1, j)
                    Psi_ntopplus1_j_BC2 = BCF.BC2_Psi_ntopp1_j(Psi[j, n_top - 1])
                    # Apply finite diff (Cx_Cy)
                    Psi[j, i] = PsiFunc.Cx_Cy(Psi[j, i + 1], Psi[j, i - 1], Psi[j + 1, i],
                                              Psi[j - 1, i], Psi[j + 1, i + 1], Psi[j + 1, i - 1],
                                              Psi[j - 1, i + 1], Psi[j - 1, i - 1], Psi_ntopplus1_j_BC2,
                                              Psi[j, i - 2], Psi[j + 2, i], Psi[j - 2, i])
                # Step 1, Case 7
                elif i == n_top - 1 and j == n_top:
                    # Bottom right corner of step 1 area, No ghost points needed due to geometry of step 2
                    # Apply finite diff (Cx_Cy)
                    Psi[j, i] = PsiFunc.Cx_Cy(Psi[j, i + 1], Psi[j, i - 1], Psi[j + 1, i],
                                              Psi[j - 1, i], Psi[j + 1, i + 1], Psi[j + 1, i - 1],
                                              Psi[j - 1, i + 1], Psi[j - 1, i - 1], Psi[j, i + 2],
                                              Psi[j, i - 2], Psi[j + 2, i], Psi[j - 2, i])
                # Step 2, "Case 8"
                else:
                    pass

        # STEP 2 (i = [1, n_bot-1] and j = [n_top+1, n_bot-1], this excludes the boundaries)
        for j in range(n_top + 1, n_bot):
            for i in range(1, n_bot):
                # Step 2, Case 1
                if i == 1 and (n_top + 1) <= j <= (n_rbc_y_approx - 3):
                    # NOTE: For i=0, n_rbc_y_approx is a BC6 location (most likely horizontal)
                    # So, n_rbc_y_approx-1 would reference a ghost point and BC6
                    # n_rbc_y_approx-2 would reference BC6
                    # n_rbc_y_approx-3 is the last point where we KNOW for certain that we don't need BC6 information
                    # Need BC7 ghost point
                    # Fail Safe Checks
                    if In_or_Out[j, i] != 1:
                        print("In_or_Out < 1 where it is expected to be = 1.  Code force stop.")
                        print("In_or_Out = " + str(In_or_Out[j, i]))
                        sys.exit("Force Stop, Psi_Solver, line 126")
                    elif BC6_location[j, i] != 0:
                        print("BC6_location value was not 0, meaning we are on the boundary when not expected to be")
                        print("BC6_location = " + str(BC6_location[j, i]))
                        sys.exit("Force Stop, Psi_Solver, line 130")
                    else:
                        pass
                    # Actual Code assuming fail safe's pass
                    Psi_minus1_j_BC7 = BCF.BC7_Psi_neg1_j(Psi[j, 0], Psi[j, 1])
                    Psi[j, i] = PsiFunc.Cx_Cy(Psi[j, i + 1], Psi[j, i - 1], Psi[j + 1, i],
                                              Psi[j - 1, i], Psi[j + 1, i + 1], Psi[j + 1, i - 1],
                                              Psi[j - 1, i + 1], Psi[j - 1, i - 1], Psi[j, i + 2],
                                              Psi_minus1_j_BC7, Psi[j + 2, i], Psi[j - 2, i])
                # Step 2, Case 2
                elif i == 1 and (n_rbc_y_approx - 2) <= j <= (n_bot - 1):
                    # In the area of the RBC, proceed with further checks
                    # We will need BC7 for some areas, can call before conditional statements with no adverse effects
                    Psi_minus1_j_BC7 = BCF.BC7_Psi_neg1_j(Psi[j, 0], Psi[j, 1])

                    if BC6_location[j, i] == 0 and In_or_Out[j, i] != 0:
                        # NOT inside RBC, NOT on border: will need to perform iterations
                        # Perform further conditional checks
                        # For the general case (12), we would check BC6_location at (i, j+1) and (i-1, j)
                        # At i = 1, we KNOW there should NOT be any BC6 != 0 at (i-1, j)
                        if BC6_location[j, i - 1] != 0:
                            # checking the point to the left of solving node (i, j)
                            print("BC6_location at (i-1, j) should = 0 and it is NOT. Code force stop.")
                            print("It should be 0 because (i-1, j) should be BC7")
                            print("BC6_location(i-1, j) = " + str(BC6_location[j, i - 1]))
                            sys.exit("Force Stop, Psi_Solver, line 155")
                        elif BC6_location[j + 1, i] == 0:
                            # checking the point under the solving node (i, j)
                            # means NOT adjacent to RBC boundary, only need BC7 ghost
                            Psi[j, i] = PsiFunc.Cx_Cy(Psi[j, i + 1], Psi[j, i - 1], Psi[j + 1, i],
                                                      Psi[j - 1, i], Psi[j + 1, i + 1], Psi[j + 1, i - 1],
                                                      Psi[j - 1, i + 1], Psi[j - 1, i - 1], Psi[j, i + 2],
                                                      Psi_minus1_j_BC7, Psi[j + 2, i], Psi[j - 2, i])
                        elif BC6_location[j + 1, i] != 0:
                            # ARE adjacent to RBC boundary
                            # should only ever need ghost point (GP) at (i, j+2), defined by BC6
                            # GP at (i-2, j) will ALWAYS be defined by BC7
                            # Will check GPs as (i, j+2) and (i-1, j+1) just in case
                            # Set GP conditional variables to FALSE initially
                            GP_i_jp2_needed = False
                            GP_im1_jp1_needed = False

                            # Check GP at (i, j+2)
                            if In_or_Out[j + 2, i] == 0:
                                # GP at (i, j+2) is needed
                                GP_i_jp2_needed = True
                                if BC6_location[j + 1, i] == 1:
                                    # BC6 horizontal check first, horizontal is default
                                    GP_i_jp2 = Psi[j, i]
                                elif BC6_location[j + 2, i + 1] == 2:
                                    # BC6 vertical check
                                    GP_i_jp2 = Psi[j + 2, i + 2]
                                else:
                                    print("Ghost Point needed but not defined")
                                    sys.exit("Force Stop, Psi_Solver, line 184")
                            else:
                                # GP at (i, j+2) NOT needed
                                GP_i_jp2_needed = False

                            # Check GP at (i-1, j+1)
                            if In_or_Out[j + 1, i - 1] == 0:
                                # GP at (i-1, j+1) is needed
                                GP_im1_jp1_needed = True
                                if BC6_location[j, i - 1] == 1:
                                    # BC6 horizontal check first, horizontal is default
                                    GP_im1_jp1 = Psi[j - 1, i - 1]
                                elif BC6_location[j + 1, i] == 2:
                                    # BC6 vertical check
                                    GP_im1_jp1 = Psi[j + 1, i + 1]
                                else:
                                    print("Ghost Point needed but not defined")
                                    sys.exit("Force Stop, Psi_Solver, line 201")
                            else:
                                # GP at (i-1, j+1) NOT needed
                                GP_im1_jp1_needed = False

                            # All possible combinations of GPs, all use BC7
                            if GP_i_jp2_needed is True and GP_im1_jp1_needed is False:
                                Psi[j, i] = PsiFunc.Cx_Cy(Psi[j, i + 1], Psi[j, i - 1], Psi[j + 1, i],
                                                          Psi[j - 1, i], Psi[j + 1, i + 1], Psi[j + 1, i - 1],
                                                          Psi[j - 1, i + 1], Psi[j - 1, i - 1], Psi[j, i + 2],
                                                          Psi_minus1_j_BC7, GP_i_jp2, Psi[j - 2, i])
                            elif GP_i_jp2_needed is False and GP_im1_jp1_needed is True:
                                Psi[j, i] = PsiFunc.Cx_Cy(Psi[j, i + 1], Psi[j, i - 1], Psi[j + 1, i],
                                                          Psi[j - 1, i], Psi[j + 1, i + 1], GP_im1_jp1,
                                                          Psi[j - 1, i + 1], Psi[j - 1, i - 1], Psi[j, i + 2],
                                                          Psi_minus1_j_BC7, Psi[j + 2, i], Psi[j - 2, i])
                            elif GP_i_jp2_needed is True and GP_im1_jp1_needed is True:
                                Psi[j, i] = PsiFunc.Cx_Cy(Psi[j, i + 1], Psi[j, i - 1], Psi[j + 1, i],
                                                          Psi[j - 1, i], Psi[j + 1, i + 1], GP_im1_jp1,
                                                          Psi[j - 1, i + 1], Psi[j - 1, i - 1], Psi[j, i + 2],
                                                          Psi_minus1_j_BC7, GP_i_jp2, Psi[j - 2, i])
                            elif GP_i_jp2_needed is False and GP_im1_jp1_needed is False:
                                # should be accounted for in another area of the code, but just in case
                                Psi[j, i] = PsiFunc.Cx_Cy(Psi[j, i + 1], Psi[j, i - 1], Psi[j + 1, i],
                                                          Psi[j - 1, i], Psi[j + 1, i + 1], Psi[j + 1, i - 1],
                                                          Psi[j - 1, i + 1], Psi[j - 1, i - 1], Psi[j, i + 2],
                                                          Psi_minus1_j_BC7, Psi[j + 2, i], Psi[j - 2, i])
                            else:
                                pass
                    elif BC6_location[j, i] != 0:
                        # ON the BOUNDARY
                        pass
                    elif BC6_location[j, i] == 0 and In_or_Out[j, i] == 0:
                        # INSIDE RBC
                        pass
                    else:
                        pass

                # Step 2, Case 3
                elif 2 <= i <= n_top and j == (n_top + 1):
                    # no BC's needed
                    # Fail Safe Checks
                    if In_or_Out[j, i] != 1:
                        print("In_or_Out < 1 where it is expected to be = 1.  Code force stop.")
                        print("In_or_Out = " + str(In_or_Out[j, i]))
                        sys.exit("Force Stop, Psi_Solver, line 246")
                    elif BC6_location[j, i] != 0:
                        print("BC6_location value was not 0, meaning we are on the boundary when not expected to be")
                        print("BC6_location = " + str(BC6_location[j, i]))
                        sys.exit("Force Stop, Psi_Solver, line 250")
                    else:
                        pass
                    # Actual Code assuming fail safe's pass
                    Psi[j, i] = PsiFunc.Cx_Cy(Psi[j, i + 1], Psi[j, i - 1], Psi[j + 1, i],
                                              Psi[j - 1, i], Psi[j + 1, i + 1], Psi[j + 1, i - 1],
                                              Psi[j - 1, i + 1], Psi[j - 1, i - 1], Psi[j, i + 2],
                                              Psi[j, i - 2], Psi[j + 2, i], Psi[j - 2, i])
                # Step 2, Case 4
                elif (n_top + 1) <= i <= (n_bot - 2) and j == (n_top + 1):
                    # Need BC3 ghost point
                    # Fail Safe Checks
                    if In_or_Out[j, i] != 1:
                        print("In_or_Out < 1 where it is expected to be = 1.  Code force stop.")
                        print("In_or_Out = " + str(In_or_Out[j, i]))
                        sys.exit("Force Stop, Psi_Solver, line 265")
                    elif BC6_location[j, i] != 0:
                        print("BC6_location value was not 0, meaning we are on the boundary when not expected to be")
                        print("BC6_location = " + str(BC6_location[j, i]))
                        sys.exit("Force Stop, Psi_Solver, line 269")
                    else:
                        pass
                    # Actual Code assuming fail safe's pass
                    Psi_i_ntopminus1_BC3 = BCF.BC3_Psi_i_ntopm1(Psi[n_top + 1, i])
                    Psi[j, i] = PsiFunc.Cx_Cy(Psi[j, i + 1], Psi[j, i - 1], Psi[j + 1, i],
                                              Psi[j - 1, i], Psi[j + 1, i + 1], Psi[j + 1, i - 1],
                                              Psi[j - 1, i + 1], Psi[j - 1, i - 1], Psi[j, i + 2],
                                              Psi[j, i - 2], Psi[j + 2, i], Psi_i_ntopminus1_BC3)
                # Step 2, Case 5
                elif i == (n_bot - 1) and j == (n_top + 1):
                    # Need BC3 and BC4 ghost points
                    # Fail Safe Checks
                    if In_or_Out[j, i] != 1:
                        print("In_or_Out < 1 where it is expected to be = 1.  Code force stop.")
                        print("In_or_Out = " + str(In_or_Out[j, i]))
                        sys.exit("Force Stop, Psi_Solver, line 285")
                    elif BC6_location[j, i] != 0:
                        print("BC6_location value was not 0, meaning we are on the boundary when not expected to be")
                        print("BC6_location = " + str(BC6_location[j, i]))
                        sys.exit("Force Stop, Psi_Solver, line 289")
                    else:
                        pass
                    # Actual Code assuming fail safe's pass
                    Psi_i_ntopminus1_BC3 = BCF.BC3_Psi_i_ntopm1(Psi[n_top + 1, i])
                    Psi_nbotplus1_j_BC4 = BCF.BC4_Psi_nbotp1_j(Psi[j, n_bot - 1])
                    Psi[j, i] = PsiFunc.Cx_Cy(Psi[j, i + 1], Psi[j, i - 1], Psi[j + 1, i],
                                              Psi[j - 1, i], Psi[j + 1, i + 1], Psi[j + 1, i - 1],
                                              Psi[j - 1, i + 1], Psi[j - 1, i - 1], Psi_nbotplus1_j_BC4,
                                              Psi[j, i - 2], Psi[j + 2, i], Psi_i_ntopminus1_BC3)
                # Step 2, Case 6
                elif i == (n_bot - 1) and (n_top + 2) <= j <= (n_bot - 2):
                    # Need BC4 ghost points
                    # Fail Safe Checks
                    if In_or_Out[j, i] != 1:
                        print("In_or_Out < 1 where it is expected to be = 1.  Code force stop.")
                        print("In_or_Out = " + str(In_or_Out[j, i]))
                        sys.exit("Force Stop, Psi_Solver, line 306")
                    elif BC6_location[j, i] != 0:
                        print("BC6_location value was not 0, meaning we are on the boundary when not expected to be")
                        print("BC6_location = " + str(BC6_location[j, i]))
                        sys.exit("Force Stop, Psi_Solver, line 310")
                    else:
                        pass
                    # Actual Code assuming fail safe's pass
                    Psi_nbotplus1_j_BC4 = BCF.BC4_Psi_nbotp1_j(Psi[j, n_bot - 1])
                    Psi[j, i] = PsiFunc.Cx_Cy(Psi[j, i + 1], Psi[j, i - 1], Psi[j + 1, i],
                                              Psi[j - 1, i], Psi[j + 1, i + 1], Psi[j + 1, i - 1],
                                              Psi[j - 1, i + 1], Psi[j - 1, i - 1], Psi_nbotplus1_j_BC4,
                                              Psi[j, i - 2], Psi[j + 2, i], Psi[j - 2, i])
                # Step 2, Case 7
                elif i == (n_bot - 1) and j == (n_bot - 1):
                    # Need BC4 and BC5 ghost points
                    # Fail Safe Checks
                    if In_or_Out[j, i] != 1:
                        print("In_or_Out < 1 where it is expected to be = 1.  Code force stop.")
                        print("In_or_Out = " + str(In_or_Out[j, i]))
                        sys.exit("Force Stop, Psi_Solver, line 326")
                    elif BC6_location[j, i] != 0:
                        print("BC6_location value was not 0, meaning we are on the boundary when not expected to be")
                        print("BC6_location = " + str(BC6_location[j, i]))
                        sys.exit("Force Stop, Psi_Solver, line 330")
                    else:
                        pass
                    # Actual Code assuming fail safe's pass
                    Psi_nbotplus1_j_BC4 = BCF.BC4_Psi_nbotp1_j(Psi[j, n_bot - 1])
                    Psi_i_nbotplus1_BC5 = BCF.BC5_Psi_i_nbotp1(Psi[n_bot, i], Psi[n_bot - 1, i])
                    Psi[j, i] = PsiFunc.Cx_Cy(Psi[j, i + 1], Psi[j, i - 1], Psi[j + 1, i],
                                              Psi[j - 1, i], Psi[j + 1, i + 1], Psi[j + 1, i - 1],
                                              Psi[j - 1, i + 1], Psi[j - 1, i - 1], Psi_nbotplus1_j_BC4,
                                              Psi[j, i - 2], Psi_i_nbotplus1_BC5, Psi[j - 2, i])
                # Step 2, Case 8
                elif (n_rbc_x_approx + 3) <= i <= (n_bot - 2) and j == (n_bot - 1):
                    # i = n_rbc_x_approx + 3 is the first node where we won't reference RBC BC6
                    # Need BC5 ghost point
                    # Fail Safe Checks
                    if In_or_Out[j, i] != 1:
                        print("In_or_Out < 1 where it is expected to be = 1.  Code force stop.")
                        print("In_or_Out = " + str(In_or_Out[j, i]))
                        sys.exit("Force Stop, Psi_Solver, line 348")
                    elif BC6_location[j, i] != 0:
                        print("BC6_location value was not 0, meaning we are on the boundary when not expected to be")
                        print("BC6_location = " + str(BC6_location[j, i]))
                        sys.exit("Force Stop, Psi_Solver, line 352")
                    else:
                        pass
                    # Actual Code assuming fail safe's pass
                    Psi_i_nbotplus1_BC5 = BCF.BC5_Psi_i_nbotp1(Psi[n_bot, i], Psi[n_bot - 1, i])
                    Psi[j, i] = PsiFunc.Cx_Cy(Psi[j, i + 1], Psi[j, i - 1], Psi[j + 1, i],
                                              Psi[j - 1, i], Psi[j + 1, i + 1], Psi[j + 1, i - 1],
                                              Psi[j - 1, i + 1], Psi[j - 1, i - 1], Psi[j, i + 2],
                                              Psi[j, i - 2], Psi_i_nbotplus1_BC5, Psi[j - 2, i])
                # Step 2, Case 9
                elif 2 <= i <= (n_rbc_x_approx + 2) and j == (n_bot - 1):
                    # in region where we MUST check for RBC
                    # BC5 will likely be needed
                    # EXCEPT when solving node is vertically adjacent to RBC border (on j=n_bot)
                    # So we will only call BC5 when needed in each conditional statement instead of before as in Case 2
                    if BC6_location[j, i] == 0 and In_or_Out[j, i] != 0:
                        # NOT inside RBC, NOT on border. Perform iterations
                        # Check BC6_location at (i, j+1), ends up being a bit more complex for this Case
                        # Normally I would also check (i-1, j) BUT I lumped that code into nested conditional statements
                        # there should be a max of three nodes checked in this if then loop, so it shouldn't
                        # slow things down too much
                        if BC6_location[j + 1, i] == 0:
                            # need BC5 ghost: GP at (i, j+2) defined BC5
                            Psi_i_nbotplus1_BC5 = BCF.BC5_Psi_i_nbotp1(Psi[n_bot, i], Psi[n_bot - 1, i])
                            # Check GPs at (i-1, j+1) and (i-2, j)
                            # Set GP conditional variables to FALSE initially
                            GP_im1_jp1_needed = False
                            GP_im2_j_needed = False

                            # Check GP at (i-1, j+1)
                            if In_or_Out[j + 1, i - 1] == 0:
                                # GP at (i-1, j+1) is needed
                                GP_im1_jp1_needed = True
                                if BC6_location[j, i - 1] == 1:
                                    # BC6 horizontal check first, horizontal is default
                                    GP_im1_jp1 = Psi[j - 1, i - 1]
                                elif BC6_location[j + 1, i] == 2:
                                    # BC6 vertical check
                                    GP_im1_jp1 = Psi[j + 1, i + 1]
                                else:
                                    print("Ghost Point needed but not defined")
                                    sys.exit("Force Stop, Psi_Solver, line 393")
                            else:
                                # GP at (i-1, j+1) NOT needed
                                GP_im1_jp1_needed = False

                            # Check GP at (i-2, j)
                            if In_or_Out[j, i - 2] == 0:
                                # GP at (i-2, j) is needed
                                GP_im2_j_needed = True
                                if BC6_location[j - 1, i - 2] == 1:
                                    # BC6 horizontal check first, horizontal is default
                                    GP_im2_j = Psi[j - 2, i - 2]
                                elif BC6_location[j, i - 1] == 2:
                                    # BC6 vertical check
                                    GP_im2_j = Psi[j, i]
                                else:
                                    print("Ghost Point needed but not defined")
                                    sys.exit("Force Stop, Psi_Solver, line 410")
                            else:
                                # GP at (i-2, j) NOT needed
                                GP_im2_j_needed = False

                            # All possible combinations of GPs, all use BC5
                            if GP_im1_jp1_needed is True and GP_im2_j_needed is False:
                                Psi[j, i] = PsiFunc.Cx_Cy(Psi[j, i + 1], Psi[j, i - 1], Psi[j + 1, i],
                                                          Psi[j - 1, i], Psi[j + 1, i + 1], GP_im1_jp1,
                                                          Psi[j - 1, i + 1], Psi[j - 1, i - 1], Psi[j, i + 2],
                                                          Psi[j, i - 2], Psi_i_nbotplus1_BC5, Psi[j - 2, i])
                            elif GP_im1_jp1_needed is False and GP_im2_j_needed is True:
                                Psi[j, i] = PsiFunc.Cx_Cy(Psi[j, i + 1], Psi[j, i - 1], Psi[j + 1, i],
                                                          Psi[j - 1, i], Psi[j + 1, i + 1], Psi[j + 1, i - 1],
                                                          Psi[j - 1, i + 1], Psi[j - 1, i - 1], Psi[j, i + 2],
                                                          GP_im2_j, Psi_i_nbotplus1_BC5, Psi[j - 2, i])
                            elif GP_im1_jp1_needed is True and GP_im2_j_needed is True:
                                Psi[j, i] = PsiFunc.Cx_Cy(Psi[j, i + 1], Psi[j, i - 1], Psi[j + 1, i],
                                                          Psi[j - 1, i], Psi[j + 1, i + 1], GP_im1_jp1,
                                                          Psi[j - 1, i + 1], Psi[j - 1, i - 1], Psi[j, i + 2],
                                                          GP_im2_j, Psi_i_nbotplus1_BC5, Psi[j - 2, i])
                            elif GP_im1_jp1_needed is False and GP_im2_j_needed is False:
                                Psi[j, i] = PsiFunc.Cx_Cy(Psi[j, i + 1], Psi[j, i - 1], Psi[j + 1, i],
                                                          Psi[j - 1, i], Psi[j + 1, i + 1], Psi[j + 1, i - 1],
                                                          Psi[j - 1, i + 1], Psi[j - 1, i - 1], Psi[j, i + 2],
                                                          Psi[j, i - 2], Psi_i_nbotplus1_BC5, Psi[j - 2, i])
                            else:
                                pass

                        elif BC6_location[j + 1, i] == 2 and i == n_rbc_x_approx:
                            # DO NOT call BC5, instead take advantage of symmetry to define GP at (i, j+2)
                            # doesn't need a conditional variable, we will always need it for this if then loop
                            GP_i_jp2 = Psi[j + 1, i]
                            if GP_i_jp2 != 0:
                                # Basically this should be on the RBC boundary due to the symmetry of the channel
                                # ie it's Psi value should = 0
                                print("Ghost Point lands on RBC boundary (based on symmetry) but does not = 0")
                                sys.exit("Force Stop, Psi_Solver, line 447")

                            # Check GPs at (i-1, j+1) and (i-2, j)
                            # Set GP conditional variables to FALSE initially
                            GP_im1_jp1_needed = False
                            GP_im2_j_needed = False

                            # Check GP at (i-1, j+1)
                            if In_or_Out[j + 1, i - 1] == 0:
                                # GP at (i-1, j+1) is needed
                                GP_im1_jp1_needed = True
                                if BC6_location[j, i - 1] == 1:
                                    # BC6 horizontal check first, horizontal is default
                                    GP_im1_jp1 = Psi[j - 1, i - 1]
                                elif BC6_location[j + 1, i] == 2:
                                    # BC6 vertical check
                                    GP_im1_jp1 = Psi[j + 1, i + 1]
                                else:
                                    print("Ghost Point needed but not defined")
                                    sys.exit("Force Stop, Psi_Solver, line 466")
                            else:
                                # GP at (i-1, j+1) NOT needed
                                GP_im1_jp1_needed = False

                            # Check GP at (i-2, j)
                            if In_or_Out[j, i - 2] == 0:
                                # GP at (i-2, j) is needed
                                GP_im2_j_needed = True
                                if BC6_location[j - 1, i - 2] == 1:
                                    # BC6 horizontal check first, horizontal is default
                                    GP_im2_j = Psi[j - 2, i - 2]
                                elif BC6_location[j, i - 1] == 2:
                                    # BC6 vertical check
                                    GP_im2_j = Psi[j, i]
                                else:
                                    print("Ghost Point needed but not defined")
                                    sys.exit("Force Stop, Psi_Solver, line 483")
                            else:
                                # GP at (i-2, j) NOT needed
                                GP_im2_j_needed = False

                            # All possible combinations of GPs, all use GP_i_jp2
                            if GP_im1_jp1_needed is True and GP_im2_j_needed is False:
                                Psi[j, i] = PsiFunc.Cx_Cy(Psi[j, i + 1], Psi[j, i - 1], Psi[j + 1, i],
                                                          Psi[j - 1, i], Psi[j + 1, i + 1], GP_im1_jp1,
                                                          Psi[j - 1, i + 1], Psi[j - 1, i - 1], Psi[j, i + 2],
                                                          Psi[j, i - 2], GP_i_jp2, Psi[j - 2, i])
                            elif GP_im1_jp1_needed is False and GP_im2_j_needed is True:
                                Psi[j, i] = PsiFunc.Cx_Cy(Psi[j, i + 1], Psi[j, i - 1], Psi[j + 1, i],
                                                          Psi[j - 1, i], Psi[j + 1, i + 1], Psi[j + 1, i - 1],
                                                          Psi[j - 1, i + 1], Psi[j - 1, i - 1], Psi[j, i + 2],
                                                          GP_im2_j, GP_i_jp2, Psi[j - 2, i])
                            elif GP_im1_jp1_needed is True and GP_im2_j_needed is True:
                                Psi[j, i] = PsiFunc.Cx_Cy(Psi[j, i + 1], Psi[j, i - 1], Psi[j + 1, i],
                                                          Psi[j - 1, i], Psi[j + 1, i + 1], GP_im1_jp1,
                                                          Psi[j - 1, i + 1], Psi[j - 1, i - 1], Psi[j, i + 2],
                                                          GP_im2_j, GP_i_jp2, Psi[j - 2, i])
                            elif GP_im1_jp1_needed is False and GP_im2_j_needed is False:
                                Psi[j, i] = PsiFunc.Cx_Cy(Psi[j, i + 1], Psi[j, i - 1], Psi[j + 1, i],
                                                          Psi[j - 1, i], Psi[j + 1, i + 1], Psi[j + 1, i - 1],
                                                          Psi[j - 1, i + 1], Psi[j - 1, i - 1], Psi[j, i + 2],
                                                          Psi[j, i - 2], GP_i_jp2, Psi[j - 2, i])
                            else:
                                pass

                        elif BC6_location[j + 1, i] == 1 and i == n_rbc_x_approx:
                            print("RBC boundary at j = n_bot is horizontal, when it should be vertical")
                            print("BC6_location at (i, j+1) = " + str(BC6_location[j + 1, i]))
                            sys.exit("Force Stop, Psi_Solver, line 515")
                        else:
                            print("A condition occurred that is not coded for")
                            sys.exit("Force Stop, Psi_Solver, line 518")

                    elif BC6_location[j, i] != 0:
                        # ON the BOUNDARY
                        pass
                    elif BC6_location[j, i] == 0 and In_or_Out[j, i] == 0:
                        # INSIDE RBC
                        pass
                    else:
                        pass

                # Step 2, Case 10
                elif 2 <= i <= (n_bot - 2) and (n_top + 2) <= j <= (n_rbc_y_approx - 3):
                    # In an area where we should always need Cx_Cy because we are above the max y value of the RBC
                    # Do NOT need any BC's
                    # Fail Safe Checks
                    if In_or_Out[j, i] != 1:
                        print("In_or_Out < 1 where it is expected to be = 1.  Code force stop.")
                        print("In_or_Out = " + str(In_or_Out[j, i]))
                        sys.exit("Force Stop, Psi_Solver, line 537")
                    elif BC6_location[j, i] != 0:
                        print("BC6_location value was not 0, meaning we are on the boundary when not expected to be")
                        print("BC6_location = " + str(BC6_location[j, i]))
                        sys.exit("Force Stop, Psi_Solver, line 541")
                    else:
                        pass
                    # Actual Code assuming fail safe's pass
                    Psi[j, i] = PsiFunc.Cx_Cy(Psi[j, i + 1], Psi[j, i - 1], Psi[j + 1, i],
                                              Psi[j - 1, i], Psi[j + 1, i + 1], Psi[j + 1, i - 1],
                                              Psi[j - 1, i + 1], Psi[j - 1, i - 1], Psi[j, i + 2],
                                              Psi[j, i - 2], Psi[j + 2, i], Psi[j - 2, i])
                # Step 2, Case 11
                elif (n_rbc_x_approx + 3) <= i <= (n_bot - 2) and (n_rbc_y_approx - 2) <= j <= (n_bot - 2):
                    # In an area where we should always need Cx_Cy
                    # because we are to the right of the max x value of the RBC
                    # Do NOT need any BC's
                    # Fail Safe Checks
                    if In_or_Out[j, i] != 1:
                        print("In_or_Out < 1 where it is expected to be = 1.  Code force stop.")
                        print("In_or_Out = " + str(In_or_Out[j, i]))
                        sys.exit("Force Stop, Psi_Solver, line 558")
                    elif BC6_location[j, i] != 0:
                        print("BC6_location value was not 0, meaning we are on the boundary when not expected to be")
                        print("BC6_location = " + str(BC6_location[j, i]))
                        sys.exit("Force Stop, Psi_Solver, line 562")
                    else:
                        pass
                    # Actual Code assuming fail safe's pass
                    Psi[j, i] = PsiFunc.Cx_Cy(Psi[j, i + 1], Psi[j, i - 1], Psi[j + 1, i],
                                              Psi[j - 1, i], Psi[j + 1, i + 1], Psi[j + 1, i - 1],
                                              Psi[j - 1, i + 1], Psi[j - 1, i - 1], Psi[j, i + 2],
                                              Psi[j, i - 2], Psi[j + 2, i], Psi[j - 2, i])
                # Step 2, Case 12
                elif 2 <= i <= (n_rbc_x_approx + 2) and (n_rbc_y_approx - 2) <= j <= (n_bot - 2):
                    # in region where we MUST check for RBC
                    if BC6_location[j, i] == 0 and In_or_Out[j, i] != 0:
                        # NOT inside RBC, NOT on boarder. Perform iterations
                        # Further Check BC6_location at (i, j+1) and (i-1, j)
                        if BC6_location[j + 1, i] == 0 and BC6_location[j, i - 1] == 0:
                            # NOT adjacent to ANY boundary, no GPs needed
                            Psi[j, i] = PsiFunc.Cx_Cy(Psi[j, i + 1], Psi[j, i - 1], Psi[j + 1, i],
                                                      Psi[j - 1, i], Psi[j + 1, i + 1], Psi[j + 1, i - 1],
                                                      Psi[j - 1, i + 1], Psi[j - 1, i - 1], Psi[j, i + 2],
                                                      Psi[j, i - 2], Psi[j + 2, i], Psi[j - 2, i])
                        elif BC6_location[j + 1, i] != 0 or BC6_location[j, i - 1] != 0:
                            # ARE adjacent to a boundary, further checks needed
                            # Check for GPs at (i, j+2), (i-1, j+1), and (i-2, j)
                            # Set GP conditional variables to FALSE initially
                            GP_i_jp2_needed = False
                            GP_im1_jp1_needed = False
                            GP_im2_j_needed = False

                            # Check GP at (i, j+2)
                            if In_or_Out[j + 2, i] == 0:
                                # GP at (i, j+2) is needed
                                GP_i_jp2_needed = True
                                if BC6_location[j + 1, i] == 1:
                                    # BC6 horizontal check first, horizontal is default
                                    GP_i_jp2 = Psi[j, i]
                                elif BC6_location[j + 2, i + 1] == 2:
                                    # BC6 vertical check
                                    GP_i_jp2 = Psi[j + 2, i + 2]
                                else:
                                    print("Ghost Point needed but not defined")
                                    sys.exit("Force Stop, Psi_Solver, line 602")
                            else:
                                # GP at (i, j+2) NOT needed
                                GP_i_jp2_needed = False

                            # Check GP at (i-1, j+1)
                            if In_or_Out[j + 1, i - 1] == 0:
                                # GP at (i-1, j+1) is needed
                                GP_im1_jp1_needed = True
                                if BC6_location[j, i - 1] == 1:
                                    # BC6 horizontal check first, horizontal is default
                                    GP_im1_jp1 = Psi[j - 1, i - 1]
                                elif BC6_location[j + 1, i] == 2:
                                    # BC6 vertical check
                                    GP_im1_jp1 = Psi[j + 1, i + 1]
                                else:
                                    print("Ghost Point needed but not defined")
                                    sys.exit("Force Stop, Psi_Solver, line 619")
                            else:
                                # GP at (i-1, j+1) is NOT needed
                                GP_im1_jp1_needed = False

                            # Check GP at (i-2, j)
                            if In_or_Out[j, i - 2] == 0:
                                # GP at (i-2, j) is needed
                                GP_im2_j_needed = True
                                if BC6_location[j - 1, i - 2] == 1:
                                    # BC6 horizontal check first, horizontal is default
                                    GP_im2_j = Psi[j - 2, i - 2]
                                elif BC6_location[j, i - 1] == 2:
                                    # BC6 vertical check
                                    GP_im2_j = Psi[j, i]
                                else:
                                    print("Ghost Point needed but not defined")
                                    sys.exit("Force Stop, Psi_Solver, line 636")
                            else:
                                # GP at (i-2, j) NOT needed
                                GP_im2_j_needed = False

                            # All possible combinations of GPs
                            if GP_i_jp2_needed is True and GP_im1_jp1_needed is False and GP_im2_j_needed is False:
                                Psi[j, i] = PsiFunc.Cx_Cy(Psi[j, i + 1], Psi[j, i - 1], Psi[j + 1, i],
                                                          Psi[j - 1, i], Psi[j + 1, i + 1], Psi[j + 1, i - 1],
                                                          Psi[j - 1, i + 1], Psi[j - 1, i - 1], Psi[j, i + 2],
                                                          Psi[j, i - 2], GP_i_jp2, Psi[j - 2, i])
                            elif GP_i_jp2_needed is False and GP_im1_jp1_needed is True and GP_im2_j_needed is False:
                                Psi[j, i] = PsiFunc.Cx_Cy(Psi[j, i + 1], Psi[j, i - 1], Psi[j + 1, i],
                                                          Psi[j - 1, i], Psi[j + 1, i + 1], GP_im1_jp1,
                                                          Psi[j - 1, i + 1], Psi[j - 1, i - 1], Psi[j, i + 2],
                                                          Psi[j, i - 2], Psi[j + 2, i], Psi[j - 2, i])
                            elif GP_i_jp2_needed is False and GP_im1_jp1_needed is False and GP_im2_j_needed is True:
                                Psi[j, i] = PsiFunc.Cx_Cy(Psi[j, i + 1], Psi[j, i - 1], Psi[j + 1, i],
                                                          Psi[j - 1, i], Psi[j + 1, i + 1], Psi[j + 1, i - 1],
                                                          Psi[j - 1, i + 1], Psi[j - 1, i - 1], Psi[j, i + 2],
                                                          GP_im2_j, Psi[j + 2, i], Psi[j - 2, i])
                            elif GP_i_jp2_needed is True and GP_im1_jp1_needed is True and GP_im2_j_needed is False:
                                Psi[j, i] = PsiFunc.Cx_Cy(Psi[j, i + 1], Psi[j, i - 1], Psi[j + 1, i],
                                                          Psi[j - 1, i], Psi[j + 1, i + 1], GP_im1_jp1,
                                                          Psi[j - 1, i + 1], Psi[j - 1, i - 1], Psi[j, i + 2],
                                                          Psi[j, i - 2], GP_i_jp2, Psi[j - 2, i])
                            elif GP_i_jp2_needed is True and GP_im1_jp1_needed is False and GP_im2_j_needed is True:
                                Psi[j, i] = PsiFunc.Cx_Cy(Psi[j, i + 1], Psi[j, i - 1], Psi[j + 1, i],
                                                          Psi[j - 1, i], Psi[j + 1, i + 1], Psi[j + 1, i - 1],
                                                          Psi[j - 1, i + 1], Psi[j - 1, i - 1], Psi[j, i + 2],
                                                          GP_im2_j, GP_i_jp2, Psi[j - 2, i])
                            elif GP_i_jp2_needed is False and GP_im1_jp1_needed is True and GP_im2_j_needed is True:
                                Psi[j, i] = PsiFunc.Cx_Cy(Psi[j, i + 1], Psi[j, i - 1], Psi[j + 1, i],
                                                          Psi[j - 1, i], Psi[j + 1, i + 1], GP_im1_jp1,
                                                          Psi[j - 1, i + 1], Psi[j - 1, i - 1], Psi[j, i + 2],
                                                          GP_im2_j, Psi[j + 2, i], Psi[j - 2, i])
                            elif GP_i_jp2_needed is True and GP_im1_jp1_needed is True and GP_im2_j_needed is True:
                                Psi[j, i] = PsiFunc.Cx_Cy(Psi[j, i + 1], Psi[j, i - 1], Psi[j + 1, i],
                                                          Psi[j - 1, i], Psi[j + 1, i + 1], GP_im1_jp1,
                                                          Psi[j - 1, i + 1], Psi[j - 1, i - 1], Psi[j, i + 2],
                                                          GP_im2_j, GP_i_jp2, Psi[j - 2, i])
                            elif GP_i_jp2_needed is False and GP_im1_jp1_needed is False and GP_im2_j_needed is False:
                                # shouldn't be needed here, coded elsewhere. Including for completeness
                                Psi[j, i] = PsiFunc.Cx_Cy(Psi[j, i + 1], Psi[j, i - 1], Psi[j + 1, i],
                                                          Psi[j - 1, i], Psi[j + 1, i + 1], Psi[j + 1, i - 1],
                                                          Psi[j - 1, i + 1], Psi[j - 1, i - 1], Psi[j, i + 2],
                                                          Psi[j, i - 2], Psi[j + 2, i], Psi[j - 2, i])
                            else:
                                pass

                    elif BC6_location[j, i] != 0:
                        # ON the BOUNDARY
                        pass
                    elif BC6_location[j, i] == 0 and In_or_Out[j, i] == 0:
                        # INSIDE RBC
                        pass
                    else:
                        pass

                # Step 2, "Case 13"
                else:
                    pass

        # ----- Check convergence criteria, again done in 2 steps -----
        # Find absolute value of absolute errors at each node
        # We can perform the subtraction on the entire grid as out of bound values are always zero
        err_absolute_placeholder_arr = np.subtract(Psi, Psi_0)
        err_absolute_arr = np.abs(err_absolute_placeholder_arr)

        # Global absolute error (frobenius norm of err_absolute_arr, which requires squaring all elements)
        # Global_AE = math.sqrt(np.sum(err_absolute_arr**2))

        # Max absolute error
        Max_AE = np.max(err_absolute_arr)

        # Global relative error (frobenius norm of err_absolute_arr (ie global_abs_err)/ frobenius of current Psi)
        # should be able to perform on entire grid as zeros won't contribute to overall value
        # numerator is frobenius of err_absolute_arr and all out of bounds and boundaries will have a zero value to contribute (ie no effect)
        # denominator is frobenius of current Psi matrix (ie size of solution). Out of bounds are zeros and won't contribute.
        # will inlcude boundaries for now, some boundaries don't contribute to overall size of current Psi
        Global_RE = math.sqrt(np.sum(err_absolute_arr**2)) / math.sqrt(np.sum(Psi**2))

        # Root Mean Square (RMS)- average size of update per point
        # Perform node by node since we don't use entire grid.
        # include boundaries since Global_AE and Global_RE include boundaries
        # set node counter to zero
        N = 0
        for j in range(0, n_top + 1):
            for i in range(0, n_top + 1):
                # in step 1 area
                N += 1
        for j in range(n_top + 1, n_bot + 1):
            for i in range(0, n_bot + 1):
                # need conditional statements in step 2. The raster area includes areas where Psi is always 0
                # For example on RBC boarder and inside RBC, if we check convergence here we divide by 0
                if In_or_Out[j, i] == 0 and BC6_location[j, i] == 0:
                    # This only happens inside the RBC, and we do not need to check convergence
                    pass
                else:
                    # all other areas should be where we want to check convergence
                    N += 1
        # Calculate RMS
        RMS = math.sqrt( np.sum(err_absolute_arr**2) / N)

        if Global_RE <= epsilon_global and Max_AE <= epsilon_max:
            # solution has converged
            converged = True
        else:
            # solution has NOT converged
            converged = False
            # reset Psi_0
            Psi_0 = np.copy(Psi)

        # check out iteration number against max iteration (failsafe to prevent an infinite loop)
        if k >= iter_max:
            break  # break the while loop

    if k >= iter_max:
        print("Program reached max number of iterations allowed to prevent an infinite loop. Number of iterations = " + str(k))
    else:
        print("Number of iterations = " + str(k))
        print('Global_RE = ' + str(Global_RE) + '; Max_AE = ' + str(Max_AE) + '; RMS = ' + str(RMS))

    # ----- Function Outputs -----
    return Psi, err_absolute_arr, k, Global_RE, Max_AE, RMS, epsilon_global, epsilon_max
# endregion
