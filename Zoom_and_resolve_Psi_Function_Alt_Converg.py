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


def Zoom_Resolve_Psi(row_delete, column_delete, row_expanded, column_expanded,
                     n_row_expanded, n_column_expanded, n_rbc_x_expd_approx, epsilon_stop, iter_max,
                     Psi_Converged, In_or_Out_zoom_expd_fill, BC6_zoom_expd_fill, precision):
    # Function OUTPUT: Psi_zoom_expd_original, Psi_initial_zoom_expd, Psi_zoom_expd_fill, err_zoom_expd, k
    # ----- Set Decimal Precision -----
    getcontext().prec = precision

    # ==================================================
    # region     Set-Up for Resolving Psi
    # ==================================================
    # ----- Zoom into Converged Psi -----
    # Copy of full grid that will get zoomed into
    Psi_zoom = np.copy(Psi_Converged)

    # zoom into grids by deleting rows/columns of the zoomed grids currently set up
    Psi_zoom = np.delete(Psi_zoom, row_delete, axis=0)
    Psi_zoom = np.delete(Psi_zoom, column_delete, axis=1)

    # ----- Initialize Needed Matrices -----
    # Create empty matrices to hold zoomed/expanded Psi
    Psi_zoom_expd_original = np.full((row_expanded, column_expanded), Decimal('0'), dtype=object)

    # ----- Expand Psi Matrix -----
    # Fill in expanded matrix with original values (remember original always goes on even nodes)
    for j in range(0, n_row_expanded + 1):
        for i in range(0, n_column_expanded + 1):
            if i % 2 == 0 and j % 2 == 0:
                # both i and j are even, meaning they hold values from original zoomed matrix prior to expansion
                # when referencing zoomed matrix prior to expansion i and j should be divided by 2 (works because even)
                Psi_zoom_expd_original[j, i] = Psi_zoom[int(j / 2), int(i / 2)]

    # Create copies of zoom_expd_original matrices used to fill in missing values
    # Do this so that we can still reference expanded matrices with original values only if needed
    Psi_zoom_expd_fill = np.copy(Psi_zoom_expd_original)

    # NOW we can start filling in the expanded Psi matrix
    # Fill in expanded Psi matrix boarders (including a pseudo boarder on the top and right side), done in 2 steps
    # above solving area we will need 2 additional points because no boundary conditions exist here (ie pseudo boarder)
    # Top pseudo boarder is j = [0, 1] (ie row 0 and row 1)
    # Right of solving area is similar, so right pseudo boarder is i =[n_column_expanded - 1, n_column_expanded]
    # Solving area is then i = [0, n_column_expanded - 2] and j = [2, n_row_expanded]
    # STEP 1
    for j in range(0, n_row_expanded + 1):
        for i in range(0, n_column_expanded + 1):
            if i % 2 == 0 and j % 2 == 0:
                # original data
                pass
            elif i % 2 != 0 and j % 2 != 0:
                # will calculate in a second step, not all interpolation points will be known during this raster
                pass
            elif i == 0:
                # ON BC7, all Psi should be 0 already
                pass
            elif j == n_row_expanded:
                # ON BC5, all Psi should be 0 already
                pass
            elif BC6_zoom_expd_fill[j, i] != 0:
                # ON BC6, all Psi should already be 0
                pass
            elif j == 0 or j == 2:
                if i % 2 != 0:
                    # on top pseudo boarder, for every ODD column, average Psi from left and right points
                    # for j = 2, this technically the solving area, so this will be our initial guess
                    # j = 2 values are needed for more accurate estimate of odd i and odd j nodes in pseudo boarder
                    Psi_zoom_expd_fill[j, i] = ((Psi_zoom_expd_fill[j, i - 1] + Psi_zoom_expd_fill[j, i + 1]) /
                                                Decimal('2'))
            elif i == n_column_expanded or i == (n_column_expanded - 2):
                if j % 2 != 0:
                    # on right pseudo boarder, for every ODD row, average Psi from above and below points
                    # for i = (n_column_expanded - 2), this is technically the solving area, this is our initial guess
                    # i = (n_column_expanded - 2) values are needed for more accurate estimate of odd i and odd j nodes
                    # in pseudo boarder
                    Psi_zoom_expd_fill[j, i] = ((Psi_zoom_expd_fill[j - 1, i] + Psi_zoom_expd_fill[j + 1, i]) /
                                                Decimal('2'))
            elif j == 1 and i % 2 == 0:
                # on second row of top "boarder," for every even column, average Psi from above and below points
                Psi_zoom_expd_fill[j, i] = (Psi_zoom_expd_fill[j - 1, i] + Psi_zoom_expd_fill[j + 1, i]) / Decimal('2')
            elif i == (n_column_expanded - 1) and j % 2 == 0:
                # on second column of right "boarder," for every even row, average Psi from left and right points
                Psi_zoom_expd_fill[j, i] = (Psi_zoom_expd_fill[j, i - 1] + Psi_zoom_expd_fill[j, i + 1]) / Decimal('2')
            else:
                pass

    # STEP 2
    # Fill in remaining boarder nodes (where i & j are both odd)
    for j in range(0, n_row_expanded + 1):
        for i in range(0, n_column_expanded + 1):
            if j == 1 and i % 2 != 0:
                # take average from its 4 neighbors (not diagonals)
                Psi_zoom_expd_fill[j, i] = (Psi_zoom_expd_fill[j, i - 1] + Psi_zoom_expd_fill[j, i + 1] +
                                            Psi_zoom_expd_fill[j - 1, i] + Psi_zoom_expd_fill[j + 1, i]) / Decimal('4')
            elif i == (n_column_expanded - 1) and j % 2 != 0:
                # take average from its three known neighbors (not diagonals)
                Psi_zoom_expd_fill[j, i] = (Psi_zoom_expd_fill[j - 1, i] + Psi_zoom_expd_fill[j + 1, i] +
                                            Psi_zoom_expd_fill[j, i + 1] + Psi_zoom_expd_fill[j, i - 1]) / Decimal('4')

    # Finally, in Psi_zoom_expd_fill add initial guesses for all zeros that are in the solving area
    # (ie not on boarder real or pseudo)
    for j in range(3, n_row_expanded):
        for i in range(1, n_column_expanded - 2):
            if BC6_zoom_expd_fill[j, i] != 0:
                # ON RBC boarder
                pass
            elif i % 2 == 0 and j % 2 == 0:
                # original node, don't change
                pass
            elif In_or_Out_zoom_expd_fill[j, i] != 0 and BC6_zoom_expd_fill[j, i] == 0:
                # in the RBC, fill with initial guess
                Psi_zoom_expd_fill[j, i] = Decimal('0.125')

    # ----- Create Psi_Initial and Psi_0 for Solving -----
    # Save initial expanded Psi matrix in a new array
    Psi_initial_zoom_expd = np.copy(Psi_zoom_expd_fill)

    # Create Psi_0_zoom_expd, which will be used to test for convergence of Psi
    # Psi_0_zoom_expd holds the previous values of Psi_zoom_expd_fill (before the next iteration occurs)
    # Values will be updated after each iteration
    # At this point it holds the initial values as no iterations have occurred
    Psi_0_zoom_expd = np.copy(Psi_zoom_expd_fill)

    # endregion

    # ==================================================
    # region     Begin Iterations: Resolve Psi
    # ==================================================
    # Don't perform iterations in pseudo boarder (top 2 rows and 2 rightmost columns)
    # use the same convergence criteria and max iterations as in original solving script
    # ----- Convergence criteria -----
    # Global relative error was set by user
    epsilon_global = epsilon_stop
    # Max absolute error two orders of magnitude tighter
    epsilon_max = epsilon_stop / (Decimal('1e2'))

    # ----- Begin Iterations -----
    Max_AE = 1
    Global_RE = 1
    RMS = 1
    converged = False  # set to false, so we don't start "converged"
    k = 0  # iteration counter

    # iterate so long as we don't meet convergence criteria
    while converged is False:
        # add one to iteration counter at start of each iteration
        k += 1

        # visual indicator that code hasn't frozen and convergence criteria status
        if k % 1000 == 0:
            print("k_expd = " + str(k)+ '; Global_RE = ' + str(Global_RE) + '; Max_AE = ' + str(Max_AE) + '; RMS = ' + str(RMS))

        # raster though solving area (skipping all boarders including pseudo boarder)
        for j in range(2, n_row_expanded):
            for i in range(1, n_column_expanded - 1):
                # Case 1
                if i == 1 and 2 <= j <= (n_row_expanded - 1):
                    # We will need BC7 for sure, may also need BC6
                    Psi_minus1_j_BC7_expd = BCF.BC7_Psi_neg1_j(Psi_zoom_expd_fill[j, 0], Psi_zoom_expd_fill[j, 1])

                    if BC6_zoom_expd_fill[j, i] == 0 and In_or_Out_zoom_expd_fill[j, i] != 0:
                        # NOT inside RBC, NOT on border: will need to perform iterations
                        # Perform further conditional checks
                        # For the general case (12), we would check BC6_location at (i, j+1) and (i-1, j)
                        # At i = 1, we KNOW there should NOT be any BC6 != 0 at (i-1, j)
                        if BC6_zoom_expd_fill[j, i - 1] != 0:
                            # checking the point to the left of solving node (i, j)
                            print("BC6_zoom_expd_fill at (i-1, j) should = 0 and it is NOT. Code force stop.")
                            print("It should be 0 because (i-1, j) should be BC7")
                            print("BC6_zoom_expd_fill (i-1, j) = " + str(BC6_zoom_expd_fill[j, i - 1]))
                            sys.exit("Force Stop, Zoom & Resolve Psi, line 169")
                        elif BC6_zoom_expd_fill[j + 1, i] == 0:
                            # checking the point under the solving node (i, j)
                            # means NOT adjacent to RBC boundary, only need BC7 ghost
                            Psi_zoom_expd_fill[j, i] = PsiFunc.Cx_Cy(Psi_zoom_expd_fill[j, i + 1],
                                                                     Psi_zoom_expd_fill[j, i - 1],
                                                                     Psi_zoom_expd_fill[j + 1, i],
                                                                     Psi_zoom_expd_fill[j - 1, i],
                                                                     Psi_zoom_expd_fill[j + 1, i + 1],
                                                                     Psi_zoom_expd_fill[j + 1, i - 1],
                                                                     Psi_zoom_expd_fill[j - 1, i + 1],
                                                                     Psi_zoom_expd_fill[j - 1, i - 1],
                                                                     Psi_zoom_expd_fill[j, i + 2],
                                                                     Psi_minus1_j_BC7_expd,
                                                                     Psi_zoom_expd_fill[j + 2, i],
                                                                     Psi_zoom_expd_fill[j - 2, i])
                        elif BC6_zoom_expd_fill[j + 1, i] != 0:
                            # ARE adjacent to RBC boundary
                            # should only ever need ghost point (GP) at (i, j+2), defined by BC6
                            # GP at (i-2, j) will ALWAYS be defined by BC7
                            # Will check GPs as (i, j+2) and (i-1, j+1) just in case
                            # Set GP conditional variables to FALSE initially
                            GP_i_jp2_needed = False
                            GP_im1_jp1_needed = False

                            # Check GP at (i, j+2)
                            if In_or_Out_zoom_expd_fill[j+2, i] == 0:
                                # GP at (i, j+2) is needed
                                GP_i_jp2_needed = True
                                if BC6_zoom_expd_fill[j+1, i] == 1:
                                    # BC6 horizontal check first, horizontal is default
                                    GP_i_jp2 = Psi_zoom_expd_fill[j, i]
                                elif BC6_zoom_expd_fill[j+2, i+1] == 2:
                                    # BC6 vertical check
                                    GP_i_jp2 = Psi_zoom_expd_fill[j+2, i+2]
                                else:
                                    print("Ghost Point needed but not defined")
                                    sys.exit("Force Stop, Zoom & Resolve Psi, line 206")
                            else:
                                # GP at (i, j+2) NOT needed
                                GP_i_jp2_needed = False

                            # Check GP at (i-1, j+1)
                            if In_or_Out_zoom_expd_fill[j+1, i-1] == 0:
                                # GP at (i-1, j+1) is needed
                                GP_im1_jp1_needed = True
                                if BC6_zoom_expd_fill[j, i-1] == 1:
                                    # BC6 horizontal check first, horizontal is default
                                    GP_im1_jp1 = Psi_zoom_expd_fill[j-1, i-1]
                                elif BC6_zoom_expd_fill[j+1, i] == 2:
                                    # BC6 vertical check
                                    GP_im1_jp1 = Psi_zoom_expd_fill[j+1, i+1]
                                else:
                                    print("Ghost Point needed but not defined")
                                    sys.exit("Force Stop, Zoom & Resolve Psi, line 223")
                            else:
                                # GP at (i-1, j+1) NOT needed
                                GP_im1_jp1_needed = False

                            # All possible combinations of GPs, all use BC7
                            if GP_i_jp2_needed is True and GP_im1_jp1_needed is False:
                                Psi_zoom_expd_fill[j, i] = PsiFunc.Cx_Cy(Psi_zoom_expd_fill[j, i + 1],
                                                                         Psi_zoom_expd_fill[j, i - 1],
                                                                         Psi_zoom_expd_fill[j + 1, i],
                                                                         Psi_zoom_expd_fill[j - 1, i],
                                                                         Psi_zoom_expd_fill[j + 1, i + 1],
                                                                         Psi_zoom_expd_fill[j + 1, i - 1],
                                                                         Psi_zoom_expd_fill[j - 1, i + 1],
                                                                         Psi_zoom_expd_fill[j - 1, i - 1],
                                                                         Psi_zoom_expd_fill[j, i + 2],
                                                                         Psi_minus1_j_BC7_expd,
                                                                         GP_i_jp2,
                                                                         Psi_zoom_expd_fill[j - 2, i])
                            elif GP_i_jp2_needed is False and GP_im1_jp1_needed is True:
                                Psi_zoom_expd_fill[j, i] = PsiFunc.Cx_Cy(Psi_zoom_expd_fill[j, i + 1],
                                                                         Psi_zoom_expd_fill[j, i - 1],
                                                                         Psi_zoom_expd_fill[j + 1, i],
                                                                         Psi_zoom_expd_fill[j - 1, i],
                                                                         Psi_zoom_expd_fill[j + 1, i + 1],
                                                                         GP_im1_jp1,
                                                                         Psi_zoom_expd_fill[j - 1, i + 1],
                                                                         Psi_zoom_expd_fill[j - 1, i - 1],
                                                                         Psi_zoom_expd_fill[j, i + 2],
                                                                         Psi_minus1_j_BC7_expd,
                                                                         Psi_zoom_expd_fill[j + 2, i],
                                                                         Psi_zoom_expd_fill[j - 2, i])
                            elif GP_i_jp2_needed is True and GP_im1_jp1_needed is True:
                                Psi_zoom_expd_fill[j, i] = PsiFunc.Cx_Cy(Psi_zoom_expd_fill[j, i + 1],
                                                                         Psi_zoom_expd_fill[j, i - 1],
                                                                         Psi_zoom_expd_fill[j + 1, i],
                                                                         Psi_zoom_expd_fill[j - 1, i],
                                                                         Psi_zoom_expd_fill[j + 1, i + 1],
                                                                         GP_im1_jp1,
                                                                         Psi_zoom_expd_fill[j - 1, i + 1],
                                                                         Psi_zoom_expd_fill[j - 1, i - 1],
                                                                         Psi_zoom_expd_fill[j, i + 2],
                                                                         Psi_minus1_j_BC7_expd,
                                                                         GP_i_jp2,
                                                                         Psi_zoom_expd_fill[j - 2, i])
                            elif GP_i_jp2_needed is False and GP_im1_jp1_needed is False:
                                # should be accounted for in another area of the code, but just in case
                                Psi_zoom_expd_fill[j, i] = PsiFunc.Cx_Cy(Psi_zoom_expd_fill[j, i + 1],
                                                                         Psi_zoom_expd_fill[j, i - 1],
                                                                         Psi_zoom_expd_fill[j + 1, i],
                                                                         Psi_zoom_expd_fill[j - 1, i],
                                                                         Psi_zoom_expd_fill[j + 1, i + 1],
                                                                         Psi_zoom_expd_fill[j + 1, i - 1],
                                                                         Psi_zoom_expd_fill[j - 1, i + 1],
                                                                         Psi_zoom_expd_fill[j - 1, i - 1],
                                                                         Psi_zoom_expd_fill[j, i + 2],
                                                                         Psi_minus1_j_BC7_expd,
                                                                         Psi_zoom_expd_fill[j + 2, i],
                                                                         Psi_zoom_expd_fill[j - 2, i])
                            else:
                                pass
                    elif BC6_zoom_expd_fill[j, i] != 0:
                        # ON the BOUNDARY
                        pass
                    elif BC6_zoom_expd_fill[j, i] == 0 and In_or_Out_zoom_expd_fill[j, i] == 0:
                        # INSIDE RBC
                        pass
                    else:
                        pass

                # Case 2
                elif 2 <= i <= (n_column_expanded - 2) and j == (n_row_expanded - 1):
                    # in region where we MUST check for RBC
                    # BC5 will likely be needed, EXCEPT when solving node is vertically adjacent to RBC border
                    # So we will only call BC5 when needed in each conditional statement
                    if BC6_zoom_expd_fill[j, i] == 0 and In_or_Out_zoom_expd_fill[j, i] != 0:
                        # NOT inside RBC, NOT on border. Perform iterations
                        # Check BC6_location at (i, j+1), ends up being a bit more complex for this Case
                        # Normally I would also check (i-1, j) BUT I lumped that code into nested conditional statements
                        # there should be a max of three nodes checked in this if then loop, so it shouldn't
                        # slow things down too much
                        if BC6_zoom_expd_fill[j+1, i] == 0:
                            # need BC5 ghost: GP at (i, j+2) defined BC5
                            Psi_i_nbotplus1_BC5_expd = BCF.BC5_Psi_i_nbotp1(Psi_zoom_expd_fill[n_row_expanded, i],
                                                                            Psi_zoom_expd_fill[n_row_expanded - 1, i])
                            # Check GPs at (i-1, j+1) and (i-2, j)
                            # Set GP conditional variables to FALSE initially
                            GP_im1_jp1_needed = False
                            GP_im2_j_needed = False

                            # Check GP at (i-1, j+1)
                            if In_or_Out_zoom_expd_fill[j+1, i-1] == 0:
                                # GP at (i-1, j+1) is needed
                                GP_im1_jp1_needed = True
                                if BC6_zoom_expd_fill[j, i-1] == 1:
                                    # BC6 horizontal check first, horizontal is default
                                    GP_im1_jp1 = Psi_zoom_expd_fill[j-1, i-1]
                                elif BC6_zoom_expd_fill[j+1, i] == 2:
                                    # BC6 vertical check
                                    GP_im1_jp1 = Psi_zoom_expd_fill[j+1, i+1]
                                else:
                                    print("Ghost Point needed but not defined")
                                    sys.exit("Force Stop, Zoom & Resolve Psi, line 325")
                            else:
                                # GP at (i-1, j+1) NOT needed
                                GP_im1_jp1_needed = False

                            # Check GP at (i-2, j)
                            if In_or_Out_zoom_expd_fill[j, i-2] == 0:
                                # GP at (i-2, j) is needed
                                GP_im2_j_needed = True
                                if BC6_zoom_expd_fill[j-1, i-2] == 1:
                                    # BC6 horizontal check first, horizontal is default
                                    GP_im2_j = Psi_zoom_expd_fill[j-2, i-2]
                                elif BC6_zoom_expd_fill[j, i-1] == 2:
                                    # BC6 vertical check
                                    GP_im2_j = Psi_zoom_expd_fill[j, i]
                                else:
                                    print("Ghost Point needed but not defined")
                                    sys.exit("Force Stop, Zoom & Resolve Psi, line 342")
                            else:
                                # GP at (i-2, j) NOT needed
                                GP_im2_j_needed = False

                            # All possible combinations of GPs, all use BC5
                            if GP_im1_jp1_needed is True and GP_im2_j_needed is False:
                                Psi_zoom_expd_fill[j, i] = PsiFunc.Cx_Cy(Psi_zoom_expd_fill[j, i + 1],
                                                                         Psi_zoom_expd_fill[j, i - 1],
                                                                         Psi_zoom_expd_fill[j + 1, i],
                                                                         Psi_zoom_expd_fill[j - 1, i],
                                                                         Psi_zoom_expd_fill[j + 1, i + 1],
                                                                         GP_im1_jp1,
                                                                         Psi_zoom_expd_fill[j - 1, i + 1],
                                                                         Psi_zoom_expd_fill[j - 1, i - 1],
                                                                         Psi_zoom_expd_fill[j, i + 2],
                                                                         Psi_zoom_expd_fill[j, i - 2],
                                                                         Psi_i_nbotplus1_BC5_expd,
                                                                         Psi_zoom_expd_fill[j - 2, i])
                            elif GP_im1_jp1_needed is False and GP_im2_j_needed is True:
                                Psi_zoom_expd_fill[j, i] = PsiFunc.Cx_Cy(Psi_zoom_expd_fill[j, i + 1],
                                                                         Psi_zoom_expd_fill[j, i - 1],
                                                                         Psi_zoom_expd_fill[j + 1, i],
                                                                         Psi_zoom_expd_fill[j - 1, i],
                                                                         Psi_zoom_expd_fill[j + 1, i + 1],
                                                                         Psi_zoom_expd_fill[j + 1, i - 1],
                                                                         Psi_zoom_expd_fill[j - 1, i + 1],
                                                                         Psi_zoom_expd_fill[j - 1, i - 1],
                                                                         Psi_zoom_expd_fill[j, i + 2],
                                                                         GP_im2_j,
                                                                         Psi_i_nbotplus1_BC5_expd,
                                                                         Psi_zoom_expd_fill[j - 2, i])
                            elif GP_im1_jp1_needed is True and GP_im2_j_needed is True:
                                Psi_zoom_expd_fill[j, i] = PsiFunc.Cx_Cy(Psi_zoom_expd_fill[j, i + 1],
                                                                         Psi_zoom_expd_fill[j, i - 1],
                                                                         Psi_zoom_expd_fill[j + 1, i],
                                                                         Psi_zoom_expd_fill[j - 1, i],
                                                                         Psi_zoom_expd_fill[j + 1, i + 1],
                                                                         GP_im1_jp1,
                                                                         Psi_zoom_expd_fill[j - 1, i + 1],
                                                                         Psi_zoom_expd_fill[j - 1, i - 1],
                                                                         Psi_zoom_expd_fill[j, i + 2],
                                                                         GP_im2_j,
                                                                         Psi_i_nbotplus1_BC5_expd,
                                                                         Psi_zoom_expd_fill[j - 2, i])
                            elif GP_im1_jp1_needed is False and GP_im2_j_needed is False:
                                Psi_zoom_expd_fill[j, i] = PsiFunc.Cx_Cy(Psi_zoom_expd_fill[j, i + 1],
                                                                         Psi_zoom_expd_fill[j, i - 1],
                                                                         Psi_zoom_expd_fill[j + 1, i],
                                                                         Psi_zoom_expd_fill[j - 1, i],
                                                                         Psi_zoom_expd_fill[j + 1, i + 1],
                                                                         Psi_zoom_expd_fill[j + 1, i - 1],
                                                                         Psi_zoom_expd_fill[j - 1, i + 1],
                                                                         Psi_zoom_expd_fill[j - 1, i - 1],
                                                                         Psi_zoom_expd_fill[j, i + 2],
                                                                         Psi_zoom_expd_fill[j, i - 2],
                                                                         Psi_i_nbotplus1_BC5_expd,
                                                                         Psi_zoom_expd_fill[j - 2, i])
                            else:
                                pass

                        elif BC6_zoom_expd_fill[j+1, i] == 2 and i == n_rbc_x_expd_approx:
                            # DO NOT call BC5, instead take advantage of symmetry to define GP at (i, j+2)
                            # doesn't need a conditional variable, we will always need it for this if then loop
                            GP_i_jp2 = Psi_zoom_expd_fill[j+1, i]
                            if GP_i_jp2 != 0:
                                # Basically this should be on the RBC boundary due to the symmetry of the channel
                                # ie it's Psi value should = 0
                                print("Ghost Point lands on RBC boundary (based on symmetry) but does not = 0")
                                sys.exit("Force Stop, Zoom & Resolve Psi, line 411")

                            # Check GPs at (i-1, j+1) and (i-2, j)
                            # Set GP conditional variables to FALSE initially
                            GP_im1_jp1_needed = False
                            GP_im2_j_needed = False

                            # Check GP at (i-1, j+1)
                            if In_or_Out_zoom_expd_fill[j + 1, i - 1] == 0:
                                # GP at (i-1, j+1) is needed
                                GP_im1_jp1_needed = True
                                if BC6_zoom_expd_fill[j, i - 1] == 1:
                                    # BC6 horizontal check first, horizontal is default
                                    GP_im1_jp1 = Psi_zoom_expd_fill[j - 1, i - 1]
                                elif BC6_zoom_expd_fill[j + 1, i] == 2:
                                    # BC6 vertical check
                                    GP_im1_jp1 = Psi_zoom_expd_fill[j + 1, i + 1]
                                else:
                                    print("Ghost Point needed but not defined")
                                    sys.exit("Force Stop, Zoom & Resolve Psi, line 430")
                            else:
                                # GP at (i-1, j+1) NOT needed
                                GP_im1_jp1_needed = False

                            # Check GP at (i-2, j)
                            if In_or_Out_zoom_expd_fill[j, i - 2] == 0:
                                # GP at (i-2, j) is needed
                                GP_im2_j_needed = True
                                if BC6_zoom_expd_fill[j - 1, i - 2] == 1:
                                    # BC6 horizontal check first, horizontal is default
                                    GP_im2_j = Psi_zoom_expd_fill[j - 2, i - 2]
                                elif BC6_zoom_expd_fill[j, i - 1] == 2:
                                    # BC6 vertical check
                                    GP_im2_j = Psi_zoom_expd_fill[j, i]
                                else:
                                    print("Ghost Point needed but not defined")
                                    sys.exit("Force Stop, Zoom & Resolve Psi, line 447")
                            else:
                                # GP at (i-2, j) NOT needed
                                GP_im2_j_needed = False

                            # All possible combinations of GPs, all use GP_i_jp2
                            if GP_im1_jp1_needed is True and GP_im2_j_needed is False:
                                Psi_zoom_expd_fill[j, i] = PsiFunc.Cx_Cy(Psi_zoom_expd_fill[j, i + 1],
                                                                         Psi_zoom_expd_fill[j, i - 1],
                                                                         Psi_zoom_expd_fill[j + 1, i],
                                                                         Psi_zoom_expd_fill[j - 1, i],
                                                                         Psi_zoom_expd_fill[j + 1, i + 1],
                                                                         GP_im1_jp1,
                                                                         Psi_zoom_expd_fill[j - 1, i + 1],
                                                                         Psi_zoom_expd_fill[j - 1, i - 1],
                                                                         Psi_zoom_expd_fill[j, i + 2],
                                                                         Psi_zoom_expd_fill[j, i - 2],
                                                                         GP_i_jp2,
                                                                         Psi_zoom_expd_fill[j - 2, i])
                            elif GP_im1_jp1_needed is False and GP_im2_j_needed is True:
                                Psi_zoom_expd_fill[j, i] = PsiFunc.Cx_Cy(Psi_zoom_expd_fill[j, i + 1],
                                                                         Psi_zoom_expd_fill[j, i - 1],
                                                                         Psi_zoom_expd_fill[j + 1, i],
                                                                         Psi_zoom_expd_fill[j - 1, i],
                                                                         Psi_zoom_expd_fill[j + 1, i + 1],
                                                                         Psi_zoom_expd_fill[j + 1, i - 1],
                                                                         Psi_zoom_expd_fill[j - 1, i + 1],
                                                                         Psi_zoom_expd_fill[j - 1, i - 1],
                                                                         Psi_zoom_expd_fill[j, i + 2],
                                                                         GP_im2_j,
                                                                         GP_i_jp2,
                                                                         Psi_zoom_expd_fill[j - 2, i])
                            elif GP_im1_jp1_needed is True and GP_im2_j_needed is True:
                                Psi_zoom_expd_fill[j, i] = PsiFunc.Cx_Cy(Psi_zoom_expd_fill[j, i + 1],
                                                                         Psi_zoom_expd_fill[j, i - 1],
                                                                         Psi_zoom_expd_fill[j + 1, i],
                                                                         Psi_zoom_expd_fill[j - 1, i],
                                                                         Psi_zoom_expd_fill[j + 1, i + 1],
                                                                         GP_im1_jp1,
                                                                         Psi_zoom_expd_fill[j - 1, i + 1],
                                                                         Psi_zoom_expd_fill[j - 1, i - 1],
                                                                         Psi_zoom_expd_fill[j, i + 2],
                                                                         GP_im2_j,
                                                                         GP_i_jp2,
                                                                         Psi_zoom_expd_fill[j - 2, i])
                            elif GP_im1_jp1_needed is False and GP_im2_j_needed is False:
                                Psi_zoom_expd_fill[j, i] = PsiFunc.Cx_Cy(Psi_zoom_expd_fill[j, i + 1],
                                                                         Psi_zoom_expd_fill[j, i - 1],
                                                                         Psi_zoom_expd_fill[j + 1, i],
                                                                         Psi_zoom_expd_fill[j - 1, i],
                                                                         Psi_zoom_expd_fill[j + 1, i + 1],
                                                                         Psi_zoom_expd_fill[j + 1, i - 1],
                                                                         Psi_zoom_expd_fill[j - 1, i + 1],
                                                                         Psi_zoom_expd_fill[j - 1, i - 1],
                                                                         Psi_zoom_expd_fill[j, i + 2],
                                                                         Psi_zoom_expd_fill[j, i - 2],
                                                                         GP_i_jp2,
                                                                         Psi_zoom_expd_fill[j - 2, i])
                            else:
                                pass

                        elif BC6_zoom_expd_fill[j+1, i] == 1 and i == n_rbc_x_expd_approx:
                            print("RBC boundary at j = n_row_expanded is horizontal, when it should be vertical")
                            print("BC6_zoom_expd_fill at (i, j+1) = " + str(BC6_zoom_expd_fill[j+1, i]))
                            sys.exit("Force Stop, Zoom & Resolve Psi, line 511")
                        else:
                            print("A condition occurred that is not coded for")
                            sys.exit("Force Stop, Zoom & Resolve Psi, line 514")

                    elif BC6_zoom_expd_fill[j, i] != 0:
                        # ON the BOUNDARY
                        pass
                    elif BC6_zoom_expd_fill[j, i] == 0 and In_or_Out_zoom_expd_fill[j, i] == 0:
                        # INSIDE RBC
                        pass
                    else:
                        pass

                # Case 3
                elif 2 <= i <= (n_column_expanded - 2) and 2 <= j <= (n_row_expanded - 2):
                    # in region where we MUST check for RBC
                    if BC6_zoom_expd_fill[j, i] == 0 and In_or_Out_zoom_expd_fill[j, i] != 0:
                        # NOT inside RBC, NOT on boarder. Perform iterations
                        # Further Check BC6_location at (i, j+1) and (i-1, j)
                        if BC6_zoom_expd_fill[j+1, i] == 0 and BC6_zoom_expd_fill[j, i-1] == 0:
                            # NOT adjacent to ANY boundary, no GPs needed
                            Psi_zoom_expd_fill[j, i] = PsiFunc.Cx_Cy(Psi_zoom_expd_fill[j, i + 1],
                                                                     Psi_zoom_expd_fill[j, i - 1],
                                                                     Psi_zoom_expd_fill[j + 1, i],
                                                                     Psi_zoom_expd_fill[j - 1, i],
                                                                     Psi_zoom_expd_fill[j + 1, i + 1],
                                                                     Psi_zoom_expd_fill[j + 1, i - 1],
                                                                     Psi_zoom_expd_fill[j - 1, i + 1],
                                                                     Psi_zoom_expd_fill[j - 1, i - 1],
                                                                     Psi_zoom_expd_fill[j, i + 2],
                                                                     Psi_zoom_expd_fill[j, i - 2],
                                                                     Psi_zoom_expd_fill[j + 2, i],
                                                                     Psi_zoom_expd_fill[j - 2, i])
                        elif BC6_zoom_expd_fill[j+1, i] != 0 or BC6_zoom_expd_fill[j, i-1] != 0:
                            # ARE adjacent to a boundary, further checks needed
                            # Check for GPs at (i, j+2), (i-1, j+1), and (i-2, j)
                            # Set GP conditional variables to FALSE initially
                            GP_i_jp2_needed = False
                            GP_im1_jp1_needed = False
                            GP_im2_j_needed = False

                            # Check GP at (i, j+2)
                            if In_or_Out_zoom_expd_fill[j+2, i] == 0:
                                # GP at (i, j+2) is needed
                                GP_i_jp2_needed = True
                                if BC6_zoom_expd_fill[j+1, i] == 1:
                                    # BC6 horizontal check first, horizontal is default
                                    GP_i_jp2 = Psi_zoom_expd_fill[j, i]
                                elif BC6_zoom_expd_fill[j+2, i+1] == 2:
                                    # BC6 vertical check
                                    GP_i_jp2 = Psi_zoom_expd_fill[j+2, i+2]
                                else:
                                    print("Ghost Point needed but not defined")
                                    print('i = ' + str(i))
                                    print('j = ' + str(j))
                                    sys.exit("Force Stop, Zoom & Resolve Psi, line 567")
                            else:
                                # GP at (i, j+2) NOT needed
                                GP_i_jp2_needed = False

                            # Check GP at (i-1, j+1)
                            if In_or_Out_zoom_expd_fill[j+1, i-1] == 0:
                                # GP at (i-1, j+1) is needed
                                GP_im1_jp1_needed = True
                                if BC6_zoom_expd_fill[j, i-1] == 1:
                                    # BC6 horizontal check first, horizontal is default
                                    GP_im1_jp1 = Psi_zoom_expd_fill[j-1, i-1]
                                elif BC6_zoom_expd_fill[j+1, i] == 2:
                                    # BC6 vertical check
                                    GP_im1_jp1 = Psi_zoom_expd_fill[j+1, i+1]
                                else:
                                    print("Ghost Point needed but not defined")
                                    sys.exit("Force Stop, Zoom & Resolve Psi, line 584")
                            else:
                                # GP at (i-1, j+1) is NOT needed
                                GP_im1_jp1_needed = False

                            # Check GP at (i-2, j)
                            if In_or_Out_zoom_expd_fill[j, i-2] == 0:
                                # GP at (i-2, j) is needed
                                GP_im2_j_needed = True
                                if BC6_zoom_expd_fill[j-1, i-2] == 1:
                                    # BC6 horizontal check first, horizontal is default
                                    GP_im2_j = Psi_zoom_expd_fill[j-2, i-2]
                                elif BC6_zoom_expd_fill[j, i-1] == 2:
                                    # BC6 vertical check
                                    GP_im2_j = Psi_zoom_expd_fill[j, i]
                                else:
                                    print("Ghost Point needed but not defined")
                                    sys.exit("Force Stop, Zoom & Resolve Psi, line 601")
                            else:
                                # GP at (i-2, j) NOT needed
                                GP_im2_j_needed = False

                            # All possible combinations of GPs
                            if GP_i_jp2_needed is True and GP_im1_jp1_needed is False and GP_im2_j_needed is False:
                                Psi_zoom_expd_fill[j, i] = PsiFunc.Cx_Cy(Psi_zoom_expd_fill[j, i + 1],
                                                                         Psi_zoom_expd_fill[j, i - 1],
                                                                         Psi_zoom_expd_fill[j + 1, i],
                                                                         Psi_zoom_expd_fill[j - 1, i],
                                                                         Psi_zoom_expd_fill[j + 1, i + 1],
                                                                         Psi_zoom_expd_fill[j + 1, i - 1],
                                                                         Psi_zoom_expd_fill[j - 1, i + 1],
                                                                         Psi_zoom_expd_fill[j - 1, i - 1],
                                                                         Psi_zoom_expd_fill[j, i + 2],
                                                                         Psi_zoom_expd_fill[j, i - 2],
                                                                         GP_i_jp2,
                                                                         Psi_zoom_expd_fill[j - 2, i])
                            elif GP_i_jp2_needed is False and GP_im1_jp1_needed is True and GP_im2_j_needed is False:
                                Psi_zoom_expd_fill[j, i] = PsiFunc.Cx_Cy(Psi_zoom_expd_fill[j, i + 1],
                                                                         Psi_zoom_expd_fill[j, i - 1],
                                                                         Psi_zoom_expd_fill[j + 1, i],
                                                                         Psi_zoom_expd_fill[j - 1, i],
                                                                         Psi_zoom_expd_fill[j + 1, i + 1],
                                                                         GP_im1_jp1,
                                                                         Psi_zoom_expd_fill[j - 1, i + 1],
                                                                         Psi_zoom_expd_fill[j - 1, i - 1],
                                                                         Psi_zoom_expd_fill[j, i + 2],
                                                                         Psi_zoom_expd_fill[j, i - 2],
                                                                         Psi_zoom_expd_fill[j + 2, i],
                                                                         Psi_zoom_expd_fill[j - 2, i])
                            elif GP_i_jp2_needed is False and GP_im1_jp1_needed is False and GP_im2_j_needed is True:
                                Psi_zoom_expd_fill[j, i] = PsiFunc.Cx_Cy(Psi_zoom_expd_fill[j, i + 1],
                                                                         Psi_zoom_expd_fill[j, i - 1],
                                                                         Psi_zoom_expd_fill[j + 1, i],
                                                                         Psi_zoom_expd_fill[j - 1, i],
                                                                         Psi_zoom_expd_fill[j + 1, i + 1],
                                                                         Psi_zoom_expd_fill[j + 1, i - 1],
                                                                         Psi_zoom_expd_fill[j - 1, i + 1],
                                                                         Psi_zoom_expd_fill[j - 1, i - 1],
                                                                         Psi_zoom_expd_fill[j, i + 2],
                                                                         GP_im2_j,
                                                                         Psi_zoom_expd_fill[j + 2, i],
                                                                         Psi_zoom_expd_fill[j - 2, i])
                            elif GP_i_jp2_needed is True and GP_im1_jp1_needed is True and GP_im2_j_needed is False:
                                Psi_zoom_expd_fill[j, i] = PsiFunc.Cx_Cy(Psi_zoom_expd_fill[j, i + 1],
                                                                         Psi_zoom_expd_fill[j, i - 1],
                                                                         Psi_zoom_expd_fill[j + 1, i],
                                                                         Psi_zoom_expd_fill[j - 1, i],
                                                                         Psi_zoom_expd_fill[j + 1, i + 1],
                                                                         GP_im1_jp1,
                                                                         Psi_zoom_expd_fill[j - 1, i + 1],
                                                                         Psi_zoom_expd_fill[j - 1, i - 1],
                                                                         Psi_zoom_expd_fill[j, i + 2],
                                                                         Psi_zoom_expd_fill[j, i - 2],
                                                                         GP_i_jp2,
                                                                         Psi_zoom_expd_fill[j - 2, i])
                            elif GP_i_jp2_needed is True and GP_im1_jp1_needed is False and GP_im2_j_needed is True:
                                Psi_zoom_expd_fill[j, i] = PsiFunc.Cx_Cy(Psi_zoom_expd_fill[j, i + 1],
                                                                         Psi_zoom_expd_fill[j, i - 1],
                                                                         Psi_zoom_expd_fill[j + 1, i],
                                                                         Psi_zoom_expd_fill[j - 1, i],
                                                                         Psi_zoom_expd_fill[j + 1, i + 1],
                                                                         Psi_zoom_expd_fill[j + 1, i - 1],
                                                                         Psi_zoom_expd_fill[j - 1, i + 1],
                                                                         Psi_zoom_expd_fill[j - 1, i - 1],
                                                                         Psi_zoom_expd_fill[j, i + 2],
                                                                         GP_im2_j,
                                                                         GP_i_jp2,
                                                                         Psi_zoom_expd_fill[j - 2, i])
                            elif GP_i_jp2_needed is False and GP_im1_jp1_needed is True and GP_im2_j_needed is True:
                                Psi_zoom_expd_fill[j, i] = PsiFunc.Cx_Cy(Psi_zoom_expd_fill[j, i + 1],
                                                                         Psi_zoom_expd_fill[j, i - 1],
                                                                         Psi_zoom_expd_fill[j + 1, i],
                                                                         Psi_zoom_expd_fill[j - 1, i],
                                                                         Psi_zoom_expd_fill[j + 1, i + 1],
                                                                         GP_im1_jp1,
                                                                         Psi_zoom_expd_fill[j - 1, i + 1],
                                                                         Psi_zoom_expd_fill[j - 1, i - 1],
                                                                         Psi_zoom_expd_fill[j, i + 2],
                                                                         GP_im2_j,
                                                                         Psi_zoom_expd_fill[j + 2, i],
                                                                         Psi_zoom_expd_fill[j - 2, i])
                            elif GP_i_jp2_needed is True and GP_im1_jp1_needed is True and GP_im2_j_needed is True:
                                Psi_zoom_expd_fill[j, i] = PsiFunc.Cx_Cy(Psi_zoom_expd_fill[j, i + 1],
                                                                         Psi_zoom_expd_fill[j, i - 1],
                                                                         Psi_zoom_expd_fill[j + 1, i],
                                                                         Psi_zoom_expd_fill[j - 1, i],
                                                                         Psi_zoom_expd_fill[j + 1, i + 1],
                                                                         GP_im1_jp1,
                                                                         Psi_zoom_expd_fill[j - 1, i + 1],
                                                                         Psi_zoom_expd_fill[j - 1, i - 1],
                                                                         Psi_zoom_expd_fill[j, i + 2],
                                                                         GP_im2_j,
                                                                         GP_i_jp2,
                                                                         Psi_zoom_expd_fill[j - 2, i])
                            elif GP_i_jp2_needed is False and GP_im1_jp1_needed is False and GP_im2_j_needed is False:
                                # shouldn't be needed here, coded elsewhere. Including for completeness
                                Psi_zoom_expd_fill[j, i] = PsiFunc.Cx_Cy(Psi_zoom_expd_fill[j, i + 1],
                                                                         Psi_zoom_expd_fill[j, i - 1],
                                                                         Psi_zoom_expd_fill[j + 1, i],
                                                                         Psi_zoom_expd_fill[j - 1, i],
                                                                         Psi_zoom_expd_fill[j + 1, i + 1],
                                                                         Psi_zoom_expd_fill[j + 1, i - 1],
                                                                         Psi_zoom_expd_fill[j - 1, i + 1],
                                                                         Psi_zoom_expd_fill[j - 1, i - 1],
                                                                         Psi_zoom_expd_fill[j, i + 2],
                                                                         Psi_zoom_expd_fill[j, i - 2],
                                                                         Psi_zoom_expd_fill[j + 2, i],
                                                                         Psi_zoom_expd_fill[j - 2, i])
                            else:
                                pass

                    elif BC6_zoom_expd_fill[j, i] != 0:
                        # ON the BOUNDARY
                        pass
                    elif BC6_zoom_expd_fill[j, i] == 0 and In_or_Out_zoom_expd_fill[j, i] == 0:
                        # INSIDE RBC
                        pass
                    else:
                        pass

                # "Case 4"
                else:
                    pass

        # ----- Check convergence criteria -----
        # Find absolute value of absolute error at each node
        # # We can perform the subtraction on the entire grid as changes in boarder and out of bound areas are always zero
        err_Expd_absolute_placeholder_arr = np.subtract(Psi_zoom_expd_fill, Psi_0_zoom_expd)
        err_Expd_absolute_arr = np.abs(err_Expd_absolute_placeholder_arr)

        # Global absolute error (frobenius norm of err_Expd_absolute_arr, which requires squaring all elements)
        # Global_AE = math.sqrt(np.sum(err_Expd_absolute_arr**2))

        # Max absolute error
        Max_AE = np.max(err_Expd_absolute_arr)

        # Global relative error (frobenius norm of err_Expd_absolute_arr (ie global_abs_err)/ frobenius of current Psi_expd)
        # should be able to perform on entire grid as zeros won't contribute to overall value
        # numerator is frobenius of err_Expd_absolute_arr and all out of bounds and boundaries will have a zero value to contribute (ie no effect)
        # denominator is frobenius of current Psi matrix (ie size of solution). Out of bounds are zeros and won't contribute.
        # will inlcude boundaries for now, some boundaries don't contribute to overall size of current Psi
        Global_RE = math.sqrt(np.sum(err_Expd_absolute_arr ** 2)) / math.sqrt(np.sum(Psi_zoom_expd_fill ** 2))

        # Root Mean Square (RMS)- average size of update per point
        # Perform node by node since we don't use entire grid.
        # include boundaries since Global_AE and Global_RE include boundaries
        # set node counter to zero
        N_Expd = 0
        for j in range(0, n_row_expanded + 1):
            for i in range(0, n_column_expanded + 1):
                # need conditional statements in step 2. The raster area includes areas where Psi is always 0
                # For example on RBC boarder and inside RBC, if we check convergence here we divide by 0
                if In_or_Out_zoom_expd_fill[j, i] == 0 and BC6_zoom_expd_fill[j, i] == 0:
                    # This only happens inside the RBC, and we do not need to check convergence
                    pass
                else:
                    # all other areas should be where we want to check convergence
                    N_Expd += 1
        # Calculate RMS
        RMS = math.sqrt(np.sum(err_Expd_absolute_arr ** 2) / N_Expd)

        if Global_RE <= epsilon_global and Max_AE <= epsilon_max:
            # solution has converged
            converged = True
        else:
            # solution has NOT converged
            converged = False
            # reset Psi_0
            Psi_0_zoom_expd = np.copy(Psi_zoom_expd_fill)

        # check out iteration number against max iteration (failsafe to prevent an infinite loop)
        if k >= iter_max:
            break  # break the while loop

    if k >= iter_max:
        print("Program reached max number of iterations allowed to prevent an infinite loop. Number of iterations = " + str(k))
    else:
        print("Number of iterations = " + str(k))
        print('Global_RE = ' + str(Global_RE) + '; Max_AE = ' + str(Max_AE) + '; RMS = ' + str(RMS))

    # ==================================================
    # region     Function Outputs
    # ==================================================
    return (Psi_zoom_expd_original, Psi_initial_zoom_expd, Psi_zoom_expd_fill, err_Expd_absolute_arr,
            k, Global_RE, Max_AE, RMS)
    # endregion
