# ==================================================
# region      Script Description
# ==================================================
# Script is used to solve dimensionless Psi matrix (full and zoomed grid) given known channel and mesh dimensions.
# Use this script to visually check convergence for the provided convergence criteria
# Uses an arbitrary max velocity to draw the velocity vector field for visual evaluation
# "Solved" Psi and arbitrary velocities are exported for use in Psi_Convergence_Checker.py
# (needs two solutions with different convergence criteria to compare)
# Psi*, arbitrary velocities, and other needed matrices are exported as Excel spreadsheets

# endregion

# ==================================================
# region      Import Libraries/Scripts
# ==================================================
# Python Libraries
import numpy as np
import pandas as pd
import math
import sys
import time
# import winsound
import matplotlib.pyplot as plt
from decimal import Decimal, getcontext

# Other scripts
import Psi_Solver_Function_Alt_Converg as PsiSol
import Velocity_Solver_Function as VelSol
import Zoom_and_resolve_Psi_Function_Alt_Converg as ZoomPsi
import Zoom_and_resolve_Vel_Function as ZoomVel

# Set decimal precision
# Incorporating decimal precision in case we need accuracy past the 15-17 digits of
# accuracy python innately has with floats
precision = 15  # int(input('Decimal.decimal precision (ie # of digits) = '))
getcontext().prec = precision

# Set Print Options
np.set_printoptions(linewidth=200)

# User defined base file naming convention
filename = input('Base filename: ')
filename_expd = filename + "_EXPD"

# Start Timer
start_time = time.time()
# endregion

# ==================================================
# region    Initial Guess Info from user
# ==================================================
Initial_Guess_Type = input('Import previous solution (identical channel and mesh size) for initial Psi guess (Y/N)? ')
filepath_initial = 'na'
if Initial_Guess_Type == 'Y':
    filepath_initial = "C:\\Users\kylie\\OneDrive - University of Oklahoma\\Documents\\Graduate School\\Crossflow Elongational Stresses Model\\CrossFlow Python Excel File Outputs\\17_L=25, 200x200, err=6e-8(NEW), initial V3.xlsx"
    # input('Filepath for Excel File containing previous Psi solution to use as initial guess\n '
    #                  'Locate Excel file to upload in File Explorer, copy filepath and paste here\n'
    #                  'DO NOT forget to include filename and extension as well: ')
# endregion

# ==================================================
# region    Channel Parameters: Full Grid
# ==================================================
# ----- Pre-set Parameters -----
L = 25  # Half channel width (from centerline to wall) in microns
D = L  # Half height of channel set such that square cross-section
aspect_ratio = D / L

# ----- User Input Parameters -----
n_top = 100  # int(input('Number of steps across the top edge, equivalent to L (NOT the number of nodes) = '))
epsilon_stop = Decimal('5e-8')  # Decimal(input('Convergence criteria for global relative error as a decimal (ie if 1% input 0.01) = '))
iter_max = int(10e6)  # int(input('Max number of iterations to perform (fail-safe to prevent infinite loop) = '))

# ----- Calculated from User Input -----
# Full grid parameters
h = L / n_top  # step size in microns
h_decimal = Decimal(h)  # as a decimal number for pecision calculations
n_bot = n_top * 2  # steps across bottom (2L in length)

# endregion

# ==================================================
# region      RBC Geometry: Elliptic
# ==================================================

# ----- Elliptic Parameters -----
# a & b are found by fitting known SA & V of an RBC to the SA and V equations for an ellipse
# O'Rear solved on MathCad, essentially this is the max deformation without changing SA and V of the RBC
a = 8.03926  # major radius in microns
b = 1.67075  # minor radius in microns

# ----- Exact Index Where Ellipse Crosses x/y Axes -----
n_rbc_x_true = a / h  # final node index where the RBC ends on x-axis, also the number of steps (likely a fraction)
n_rbc_y_true = ((2 * L) - b) / h  # index where the RBC begins on y-axis (likely a fraction)

# ----- Elliptic Position Arrays as f(x) and f(y) -----
# Create two 1D arrays that will hold the RBC x and y positions
X_RBC_arr = np.zeros(n_bot + 1)  # this is a function of the y position (j)
Y_RBC_arr = np.zeros((n_bot + 1))  # this is a function of the x position (i)

# Fill in the X_RBC_arr
for j in range(0, n_bot + 1):
    y_pos_i_j = h * j  # at i, j
    if y_pos_i_j >= ((2 * L) - b):  # Checking if we are even in an area with a value for the ellipse
        X_RBC_arr[j] = math.sqrt(a ** 2 * (1 - ((y_pos_i_j - (2 * L)) ** 2) / b ** 2))

# Fill in the Y_RBC_arr
for i in range(0, n_bot + 1):
    x_pos_i_j = h * i  # at i, j
    if x_pos_i_j <= a:  # Checking we are even in an area with a value for the ellipse
        Y_RBC_arr[i] = (2 * L) - math.sqrt(b ** 2 * (1 - (x_pos_i_j / a) ** 2))

# endregion

# ==================================================
# region      RBC Geometry: Stair Step
# ==================================================
# ----- Scaling Factors Alpha and Beta -----
# Create a square array for Alpha_1 and Beta_1 values that all start with a value of 1
Alpha_1 = np.ones((n_bot+1, n_bot+1))
Beta_1 = np.ones((n_bot+1, n_bot+1))

# Fill in the Alpha_1 and Beta_1 Arrays if they deviate from 1
for j in range(0, n_bot + 1):  # plus 1 because range is inclusive of start and exclusive of end
    for i in range(0, n_bot + 1):
        x_pos_i_j = h * i  # at i, j
        y_pos_i_j = h * j  # at i, j
        x_pos_im1_j = h * (i - 1)  # at i-1,j
        y_pos_i_jp1 = h * (j + 1)
        # Check if we need to calculate alphas
        if X_RBC_arr[j] == 0:  # we are NOT in the area of the ellipse
            pass
        elif x_pos_im1_j < X_RBC_arr[j] < x_pos_i_j:  # Calculate alpha_1
            Alpha_1[j, i] = (x_pos_i_j - X_RBC_arr[j]) / h
        elif X_RBC_arr[j] != 0 and x_pos_i_j < X_RBC_arr[j]:
            Alpha_1[j, i] = 0
        else:
            pass
        # Check if we need to calculate betas
        if Y_RBC_arr[i] == 0:  # we are NOT in the area of the ellipse
            pass
        elif y_pos_i_j < Y_RBC_arr[i] < y_pos_i_jp1:  # Calculate beta_1
            Beta_1[j, i] = (Y_RBC_arr[i]-y_pos_i_j)/h
        elif Y_RBC_arr[i] != 0 and y_pos_i_j > Y_RBC_arr[i]:
            Beta_1[j, i] = 0
        else:
            pass

# ----- Stair-Step RBC Boundary (BC6) -----
# Use Alpha and Betas to determine plane locations of RBC (for stair step approximation)
# Create a matrix to hold plane locations and use to determine where to apply BC6
BC6_location = np.zeros((n_bot + 1, n_bot + 1))

# NEW METHOD FOR BC6/ RBC ENDPOINT NODES
# raster though and find where alpha or beta is non-zero, and use value to determine BC6 location
for j in range(0, n_bot + 1):
    for i in range(0, n_bot + 1):
        # we are including the boundaries here
        if 0 < Alpha_1[j, i] < 0.5:
            BC6_location[j, i] = 1
        elif 0.5 <= Alpha_1[j, i] < 1:
            BC6_location[j, i - 1] = 1
for j in range(0, n_bot + 1):
    for i in range(0, n_bot + 1):
        if 0 < Beta_1[j, i] < 0.5:
            BC6_location[j, i] = 1
        elif 0.5 <= Beta_1[j, i] < 1:
            BC6_location[j + 1, i] = 1

# Go back though and set horizontal or vertical "plane"
for j in range(0, n_bot + 1):
    for i in range(0, n_bot + 1):
        if BC6_location[j, i] != 0:
            # need to check orientation
            if j == n_bot:
                # RBC endpoint on x-axis, always vertical
                # BC6 = 2
                BC6_location[j, i] = 2
            elif BC6_location[j - 1, i] != 0 or BC6_location[j + 1, i] != 0:
                # vertical, BC6 = 2
                BC6_location[j, i] = 2
            else:
                pass

# ----- Indices Where Approximated RBC Crosses x/y Axes -----
# initially set to zero so can't be undefined
n_rbc_y_approx = 0
n_rbc_x_approx = 0
# reset to correct values
for j in range(0, n_bot + 1):
    # i = 0
    if BC6_location[j, 0] != 0:
        n_rbc_y_approx = j
for i in range(0, n_bot + 1):
    # j = n_bot
    if BC6_location[n_bot, i] != 0:
        n_rbc_x_approx = i

# ----- In or Outside RBC Binary Matrix -----
# Use Alpha or Beta to set up a matrix that tell you if you are inside or outside of RBC
# Where Alpha or Beta is non-zero = outside RBC
# Where Alpha or Beta is 0 (if one is 0 the other is as well) = inside RBC
# so only need to use Alpha or Beta, not both
# First, create a copy of Alpha_1 or Beta_1 to base In_or_Out off of
In_or_Out = np.copy(Alpha_1)
# convert to binary (1's/0's)
# 1 = outside RBC or on boarder
# 0 = inside RBC
for j in range(0, n_bot + 1):
    for i in range(0, n_bot + 1):
        if 0 < In_or_Out[j, i] < 1:
            In_or_Out[j, i] = 1

# Update In_or_Out matrix based on finalized BC6
# Sometimes BC6 is drawn where fractional Alpha/Beta existed (just outside RBC) which has already been converted to 1
# Sometimes BC6 is drawn just inside RBC (dependent on fractional Alpha/Beta value) and currently reads 0
# even though it is actually outside the approximated RBC
for j in range(0, n_bot + 1):
    for i in range(0, n_bot + 1):
        if BC6_location[j, i] != 0 and In_or_Out[j, i] == 0:
            In_or_Out[j, i] = 1

# ----- Approximated RBC Volume -----
# Volume of ENTIRE stair-step RBC (trapezoidal method, exact for straight lines)
V_RBC_arr = np.full((n_bot + 1, n_bot + 1), Decimal('0'), dtype=object)
for j in range(0, n_bot + 1):
    for i in range(0, n_bot + 1):
        if BC6_location[j, i] != 0 and j != n_bot:
            # We know we are on boundary BUT unknown if horizontal, vertical, or diagonal portion
            if BC6_location[j, i + 1] != 0:
                # On horizontal segment, volume in cubic micron
                Vh_ij_ip1j = Decimal(math.pi) * h_decimal * (((2 * L) - (h_decimal * j)) ** 2)
                V_RBC_arr[j, i] = Vh_ij_ip1j
            elif BC6_location[j - 1, i] != 0 or BC6_location[j + 1, i] != 0:
                # On a vertical segment, does not contribute to volume
                # Check for vertical prior to diagonal because sometime last node on j = n_bot
                # will have two neighbors (one vertical and one diagonal) and checking vertical ensures
                # it's "drawing" the shape correctly
                pass
            elif BC6_location[j + 1, i + 1] != 0:
                # On a diagonal segment, volume in cubic micron
                Vd_ij_ip1jp1 = Decimal(math.pi/2) * h_decimal * ((((2 * L) - (h_decimal * j)) ** 2) +
                                                                 (((2 * L) - (h_decimal * (j + 1))) ** 2))
                V_RBC_arr[j, i] = Vd_ij_ip1jp1
V_RBC = 2 * np.sum(V_RBC_arr)

# ----- Create a mask for plotting full grid -----
# array holds boolean values
# False means the data should be plotted (ie our flow area)
# True means the data should not be plotted (ie outside grid or inside RBC)
mask_full = np.full((n_bot+1, n_bot+1), False)
for j in range(0, n_bot + 1):
    for i in range(0, n_bot + 1):
        if n_top < i <= n_bot and 0 <= j < n_top:
            # quarter of the grid that isn't used
            mask_full[j, i] = True
        elif In_or_Out[j, i] == 0:
            # inside the cell
            mask_full[j, i] = True

# endregion

# ==================================================
# region  Channel Parameters: Zoomed/Expanded Grid
# ==================================================
# ----- Zoom into BC6 & In_or_Out Matrices -----
# copies of full grid that will get zoomed into
BC6_zoom = np.copy(BC6_location)
In_or_Out_zoom = np.copy(In_or_Out)

# set up which rows and columns are deleted from full grid to "zoom in", based on approximated RBC position
row_delete = list(range(0, n_rbc_y_approx - 3))  # zooms in such that there are 3 rows above max height of RBC
column_delete = list(range(n_rbc_x_approx + 4, n_bot + 1))  # leaves 3 columns to the right of max RBC length

# zoom into grids by deleting rows/columns of the zoomed grids currently set up
BC6_zoom = np.delete(BC6_zoom, row_delete, axis=0)
BC6_zoom = np.delete(BC6_zoom, column_delete, axis=1)
In_or_Out_zoom = np.delete(In_or_Out_zoom, row_delete, axis=0)
In_or_Out_zoom = np.delete(In_or_Out_zoom, column_delete, axis=1)

# ----- Dimensions of Zoomed/Expd Grid -----
# Find dimensions of zoomed grids NOT expanded yet (total number of nodes in x and y)
dimensions_zoom = BC6_zoom.shape
row_zoom, column_zoom = dimensions_zoom

# NOW calculate dimensions in the zoomed AND expanded grids
# zoomed matrices are expanded by adding new rows and columns between original rows and columns
# total number j nodes
row_expanded = row_zoom + (row_zoom - 1)
# total number i nodes
column_expanded = column_zoom + (column_zoom - 1)

# Convert expanded dimensions to number of steps which is equal to last node index since index starts at 0
# j index
n_row_expanded = row_expanded - 1
# i index
n_column_expanded = column_expanded - 1

# Step size for zoomed/expanded grid
h_expd = h / 2  # step size in microns
h_expd_decimal = h_decimal / 2  # as a decimal for precision calculations

# ----- Expand Zoomed BC6 & In_or_Out Matrices -----
# Create empty matrices to hold zoomed/expanded BC6/IN_or_Out
BC6_zoom_expd_original = np.zeros((row_expanded, column_expanded))
In_or_Out_zoom_expd_original = np.zeros((row_expanded, column_expanded))

# Fill in expanded matrices with original values (remember original always goes on even nodes)
for j in range(0, n_row_expanded + 1):
    for i in range(0, n_column_expanded + 1):
        if i % 2 == 0 and j % 2 == 0:
            # both i and j are even, meaning they hold values from original zoomed matrix prior to expansion
            # when referencing zoomed matrix prior to expansion i and j should be divided by 2 (works because even)
            BC6_zoom_expd_original[j, i] = BC6_zoom[int(j / 2), int(i / 2)]
            In_or_Out_zoom_expd_original[j, i] = In_or_Out_zoom[int(j / 2), int(i / 2)]

# Create copies of zoom_expd_original matrices used to fill in missing values
# Do this so that we can still reference expanded matrices with original values only if needed
BC6_zoom_expd_fill = np.copy(BC6_zoom_expd_original)
In_or_Out_zoom_expd_fill = np.copy(In_or_Out_zoom_expd_original)

# NOW we can start filling in the expanded matrices
# Adjust BC6_zoom_expd_fill (fill in missing boundary locations from expansion)
for j in range(1, n_row_expanded):
    for i in range(1, n_column_expanded):
        if BC6_zoom_expd_fill[j, i] == 0:
            # we may need to add a BC6 location
            if BC6_zoom_expd_fill[j, i - 1] != 0 and BC6_zoom_expd_fill[j, i + 1] != 0:
                # connect horizontal planes
                BC6_zoom_expd_fill[j, i] = 1
            elif BC6_zoom_expd_fill[j - 1, i - 1] != 0 and BC6_zoom_expd_fill[j + 1, i + 1] != 0:
                # connect diagonals
                # for fine mesh, nose can get tricky and fill in a BC6 where not needed
                if BC6_zoom_expd_fill[j - 1, i + 1] == 0:
                    # draw diagonal as normal
                    if BC6_zoom_expd_fill[j - 1, i - 1] == 2 and BC6_zoom_expd_fill[j + 1, i + 1] == 2:
                        # connect with 'vertical'
                        BC6_zoom_expd_fill[j, i] = 2
                    else:
                        # connect with 'horizontal'
                        BC6_zoom_expd_fill[j, i] = 1
                elif BC6_zoom_expd_fill[j - 1, i + 1] != 0:
                    # will want to draw diagonal but that would cut off part of cell
                    pass
            elif BC6_zoom_expd_fill[j - 1, i] != 0 and BC6_zoom_expd_fill[j + 1, i] != 0:
                # connect vertical planes
                BC6_zoom_expd_fill[j, i] = 2
            else:
                # not in a BC6 location
                pass

# Adjust In_or_Out_zoom_expd_fill (fill in missing 1's from expansion)
for i in range(0, n_column_expanded + 1):
    for j in range(0, n_row_expanded + 1):
        # raster column by column
        if BC6_zoom_expd_fill[j, i] != 0:
            if BC6_zoom_expd_fill[j, i] == 1:
                # we've reached a horizontal plane on the cell, no longer need to update In_or_Out in this column (i)
                # break out of j loop, move to next i
                # this logic takes advantage of where the cell is (lower left corner) and it's geometry
                # by rastering column by column once we hit the RBC, we know everything else will be inside the cell
                In_or_Out_zoom_expd_fill[j, i] = 1
                # adding 1 to In_or_Out isn't necessary because on boarder,
                # BUT doing it so the matrix doesn't look so wierd
                break
            elif BC6_zoom_expd_fill[j, i] == 2:
                In_or_Out_zoom_expd_fill[j, i] = 1
                # check to see if we've reached the end of this vertical plane
                if j == n_row_expanded:
                    pass
                elif BC6_zoom_expd_fill[j + 1, i] == 0:
                    # we've reached the end of vertical plane, break out of j loop
                    break
        else:
            # this should only fill before we hit the RBC
            # as soon as BC6 != 0 we should break out of j loop and not fill In_or_Out in that are IN the RBC
            In_or_Out_zoom_expd_fill[j, i] = 1

# ----- Create a mask for plotting zoom/expd grid -----
# array holds boolean values
# False means the data should be plotted (ie our flow area)
# True means the data should not be plotted (ie outside grid or inside RBC)
mask_expd = np.full((row_expanded, column_expanded), False)
for j in range(0, n_row_expanded + 1):
    for i in range(0, n_column_expanded + 1):
        if In_or_Out_zoom_expd_fill[j, i] == 0:
            # inside the cell
            mask_expd[j, i] = True

# ----- RBC Indices: Exact (Elliptic) and Approximate (Stair Step) -----
# Exact rbc indices on x and y-axis in zoomed/expanded grid (likely fractional)
n_rbc_x_expd_true = a / h_expd
n_rbc_y_expd_true = ((h_expd * n_row_expanded) - b) / h_expd

# Approximated RBC Indices on x and y-axis in zoomed/expanded grid
# initially set to zero so can't be undefined
n_rbc_x_expd_approx = 0
n_rbc_y_expd_approx = 0
# reset to correct values
for i in range(0, n_column_expanded + 1):
    if BC6_zoom_expd_fill[n_row_expanded, i] != 0:
        n_rbc_x_expd_approx = i
for j in range(0, n_row_expanded + 1):
    if BC6_zoom_expd_fill[j, 0] != 0:
        n_rbc_y_expd_approx = j

# endregion

# ==================================================
# region   Set up Initial Guesses-Full Grid
# ==================================================
if Initial_Guess_Type == 'Y':
    # Importing an initial guess (ie a previous Psi solution)
    # Imports a numpy array full of strings
    Psi_initial = (pd.read_excel(filepath_initial, sheet_name='Psi(Full)', header=None, dtype=object)).to_numpy()
    # Convert to decimal.Decimal
    for j in range(0, n_bot + 1):
        for i in range(0, n_bot + 1):
            Psi_initial[j, i] = Decimal(Psi_initial[j, i])
elif Initial_Guess_Type == 'N':
    # Create Psi_Initial from scratch

    # ----- Initialize Psi matrix to hold boundary conditions and initial guesses -----
    # Create an empty square matrix (full of 0's) for Psi initial guess values and boundary conditions
    # they are input as strings, so we can use Decimal, for more precise calculations
    # we won't use the entire space
    Psi_initial = np.full((n_bot + 1, n_bot + 1), Decimal('0'), dtype=object)

    # ----- Fill in Psi_initial with boundary conditions -----
    # i will be a counter in the x direction (ie i = column)
    # j will be a counter in the y direction (ie j = row)

    # Boundary Condition 1
    # i = [0, n_top], j = 0
    for i in range(n_top + 1):  # its n_top + 1 because range is inclusive of the start, and exclusive of the end
        x_star = (i * h) / L  # x_star is the dimensionless x position & the x-position is related to i and the step size
        Psi_initial[0, i] = round((Decimal(str(x_star)) - ((Decimal(str(x_star)) ** 3) / 3)), precision-2)

    # Boundary Condition 2
    # i = n_top, j = [0, n_top]
    for j in range(n_top + 1):
        Psi_initial[j, n_top] = round(Decimal(str(2 / 3)), precision-2)

    # Boundary Condition 3
    # i = [n_top, n_bot], j = n_top
    for i in range(n_top, n_bot + 1):
        Psi_initial[n_top, i] = round(Decimal(str(2 / 3)), precision-2)

    # Boundary Condition 4
    # i = n_bot, j = [n_top, n_bot]
    for j in range(n_top, n_bot + 1):
        y_star = (j * h) / L  # y_star is the dimensionless y-position & the y-position is related to j and the stepsize
        Psi_initial[j, n_bot] = round((((Decimal(str(y_star)) ** 3) / 3) - (2 * Decimal(str(y_star)) ** 2) +
                                       (3 * Decimal(str(y_star))) - Decimal(str(2/3))), precision-2)

    # Boundary Conditions 5-7:
    # These will all have a value of zero
    # Since the Psi matrix already starts as a matrix of zeros, it is arbitrary to raster through and set each BC to 0
    # Will keep the code and just comment out the lines in case these BC's are ever not 0 in the future

    # # Boundary Condition 5
    # # i = [n_rbc_x_approx + 1, n_bot], j = n_bot, where n_rbc_x_approx + 1 is the node to the right of the RBC
    # for i in range(n_rbc_x_approx + 1, n_bot+1):
    #    Psi_initial[n_bot, i] = 0

    # Boundary Condition 6
    # set up code if ever needed (ie BC6 not 0), use BC6_location to determine BC6 positions

    # Boundary Condition 7
    # i = 0, j = [0, n_rbc_y_approx - 1], where n_rbc_y_approx - 1 is the node just above the RBC
    # for j in range(n_rbc_y_approx):
    #    Psi_initial[j, 0] = 0

    # ----- Fill in remaining Psi_initial V1 -----
    # # This has to be done in 2 steps
    # # Step 1
    # for j in range(1, n_top + 1):
    #     for i in range(1, n_top):
    #         Psi_initial[j, i] = Decimal('0.25')
    # # Step 2
    # for j in range(n_top + 1, n_bot):
    #     for i in range(1, n_bot):
    #         if In_or_Out[j, i] > 0:
    #             # If In_or_Out > 0 we are in the main part of the channel (outside of RBC) and possibly on the RBC boundary
    #             # Also check BC6_location to determine if we actually need to set initial guess
    #             if BC6_location[j, i] == 0:
    #                 # We are NOT on the boundary and should set the initial guess
    #                 Psi_initial[j, i] = Decimal('0.25')
    #             else:
    #                 # meaning BC6_location is 1 or 2, and therefore we are on the boundary and the boundary conditions
    #                 # have already been applied
    #                 pass
    #         elif In_or_Out[j, i] == 0:
    #             # this should mean we are in RBC, and shouldn't need to update with initial guess.
    #             # Double check BC6_location is as expected (ie it should be 0)
    #             if BC6_location[j, i] != 0:
    #                 print("In_or_Out = 0 and BC6_location is non-zero which shouldn't happen")
    #                 print("In_or_Out = " + str(In_or_Out[j, i]))
    #                 print("BC6_location = " + str(BC6_location[j, i]))
    #                 sys.exit("Force Stop, line 583")
    #             else:
    #                 pass
    #         else:
    #             pass

    # ----- Fill in remaining Psi_initial with initial guesses: V2 -----
    # Before the rest of the initial guesses were all one value (0.25)
    # Now creating an initial guess that varies with position
    # Doing this to try and start as close to the solution as possible, so that less iterations are required

    # Determine how much Psi* guess will change from node to node
    # use similar to how we would determine location in grid (h*i or h*j)
    # h_Psi = Decimal(2/3) / n_top

    # Fill in guesses in three steps
    # Step 1
    # for j in range(1, n_rbc_y_approx - 5):
    #     for i in range(1, n_top):
    #         if In_or_Out[j, i] > 0 and BC6_location[j, i] == 0:
    #             # Outside RBC, not on the Boarder
    #             x_star = (i * h) / L
    #             Psi_initial[j, i] = round((Decimal(str(x_star)) - ((Decimal(str(x_star)) ** 3) / 3)), 6)
    #             # Psi_initial[j, i] = Decimal('0.25')
    #         elif In_or_Out[j, i] == 0 and BC6_location[j, i] != 0:
    #             # Inside RBC, Double check BC6_location is as expected (ie it should be 0)
    #             print("In_or_Out = 0 and BC6_location is non-zero which shouldn't happen")
    #             print("In_or_Out = " + str(In_or_Out[j, i]))
    #             print("BC6_location = " + str(BC6_location[j, i]))
    #             sys.exit("Force Stop, line 504")
    # # Step 2
    # for j in range(n_top + 1, n_bot):
    #     for i in range(n_rbc_x_approx + 6, n_bot):
    #         if In_or_Out[j, i] > 0 and BC6_location[j, i] == 0:
    #             # Outside RBC, not on boarder
    #             y_star = (j * h) / L
    #             Psi_initial[j, i] = round((((Decimal(str(y_star)) ** 3) / 3) - (2 * Decimal(str(y_star)) ** 2) +
    #                                            (3 * Decimal(str(y_star))) - Decimal(str(2 / 3))), 6)
    #             # Psi_initial[j, i] = Decimal('0.25')
    #         elif In_or_Out[j, i] == 0 and BC6_location[j, i] != 0:
    #             # Inside RBC, Double check BC6_location is as expected (ie it should be 0)
    #             print("In_or_Out = 0 and BC6_location is non-zero which shouldn't happen")
    #             print("In_or_Out = " + str(In_or_Out[j, i]))
    #             print("BC6_location = " + str(BC6_location[j, i]))
    #             sys.exit("Force Stop, line 519")
    #
    # # Step 3
    # for j in range(n_rbc_y_approx - 5, n_bot):
    #     for i in range(1, n_rbc_x_approx + 6):
    #         if In_or_Out[j, i] > 0 and BC6_location[j, i] == 0:
    #             # Outside RBC, not on boarder
    #             Psi_initial[j, i] = Decimal('0.125')
    #         elif In_or_Out[j, i] == 0 and BC6_location[j, i] != 0:
    #             # Inside RBC, Double check BC6_location is as expected (ie it should be 0)
    #             print("In_or_Out = 0 and BC6_location is non-zero which shouldn't happen")
    #             print("In_or_Out = " + str(In_or_Out[j, i]))
    #             print("BC6_location = " + str(BC6_location[j, i]))
    #             sys.exit("Force Stop, line 536")

    # ----- Fill in remaining Psi_initial with initial guesses: V3 -----
    # Fill in guesses in three steps
    # Step 1
    for j in range(1, n_top):
        for i in range(1, n_top):
            x_star = (i * h) / L
            Psi_initial[j, i] = round((Decimal(str(x_star)) - ((Decimal(str(x_star)) ** 3) / 3)), precision-2)
    # Step 2
    for j in range(n_top + 1, n_bot):
        for i in range(n_top + 1, n_bot):
            y_star = (j * h) / L
            Psi_initial[j, i] = round((((Decimal(str(y_star)) ** 3) / 3) - (2 * Decimal(str(y_star)) ** 2) +
                                       (3 * Decimal(str(y_star))) - Decimal(str(2 / 3))), precision-2)

    # Step 3a
    for j in range(n_top, n_bot):
        modifier = j - n_top
        for i in range(1, (n_top - modifier) + 1):
            if In_or_Out[j, i] > 0 and BC6_location[j, i] == 0:
                # Outside RBC, not on boarder
                x_star = (i * h) / L
                Psi_initial[j, i] = round((Decimal(str(x_star)) - ((Decimal(str(x_star)) ** 3) / 3)), precision-2)
            elif In_or_Out[j, i] == 0 and BC6_location[j, i] != 0:
                # Inside RBC, Double check BC6_location is as expected (ie it should be 0)
                print("In_or_Out = 0 and BC6_location is non-zero which shouldn't happen")
                print("In_or_Out = " + str(In_or_Out[j, i]))
                print("BC6_location = " + str(BC6_location[j, i]))
                sys.exit("Force Stop, line 536")

    # Step 3b
    for j in range(n_top, n_bot):
        modifier = j - n_top
        for i in range((n_top - modifier) + 1, n_top + 1):
            if In_or_Out[j, i] > 0 and BC6_location[j, i] == 0:
                # Outside RBC, not on boarder
                y_star = (j * h) / L
                Psi_initial[j, i] = round((((Decimal(str(y_star)) ** 3) / 3) - (2 * Decimal(str(y_star)) ** 2) +
                                           (3 * Decimal(str(y_star))) - Decimal(str(2 / 3))), precision-2)
            elif In_or_Out[j, i] == 0 and BC6_location[j, i] != 0:
                # Inside RBC, Double check BC6_location is as expected (ie it should be 0)
                print("In_or_Out = 0 and BC6_location is non-zero which shouldn't happen")
                print("In_or_Out = " + str(In_or_Out[j, i]))
                print("BC6_location = " + str(BC6_location[j, i]))
                sys.exit("Force Stop, line 536")

# ----- Uncomment if you want to plot initial matrix -----
#
# Psi_initial_flt = Psi_initial.astype(float)
#
# # ----- Mask data to 'hide' parts outside grid or inside RBC -----
# mask_Psi_initial = np.ma.masked_array(Psi_initial_flt, mask=mask_full)
#
# t = np.linspace(0, (np.pi/2), 50)
#
# # ----- Full Grid -----
# x_ellipse = n_rbc_x_true * np.cos(t)
# y_ellipse = (2 * n_top) - ((n_bot - n_rbc_y_true) * np.sin(t))
#
# filename_Psi_initial_cmap_svg = filename + "_Psi_initial cmap.svg"
# filepath_Psi_initial_cmap_svg = "C:\\Users\kylie\\OneDrive - University of Oklahoma\\Documents\\Graduate School\\Crossflow Elongational Stresses Model\\CrossFlow Python Excel File Outputs\\" + filename_Psi_initial_cmap_svg
#
# # ----- Full Grid Plot -----
# fig_Psi_i, ax_Psi_i = plt.subplots()
# map_Psi_i = ax_Psi_i.imshow(mask_Psi_initial, cmap='turbo', vmin=0, vmax=0.7)
# bar_Psi_i = plt.colorbar(map_Psi_i, ax=ax_Psi_i)
# ax_Psi_i.set_aspect('equal')
# ax_Psi_i.plot(x_ellipse, y_ellipse, color='red')
# fig_Psi_i.suptitle('Dimensionless Psi Values: Initial Guess')
# fig_Psi_i.savefig(filepath_Psi_initial_cmap_svg, dpi=400)
#
# sys.exit()


# endregion

# ==================================================
# region             Solve Psi
# ==================================================
# Should only need to solve Psi once as it is dimensionless and does NOT depend on Q
(Psi_Converged, Psi_Err_Abs, Psi_k, Global_RE,
 Max_AE, RMS, epsilon_global, epsilon_max) = PsiSol.Psi_Solve(n_top, n_bot, n_rbc_x_approx, n_rbc_y_approx, epsilon_stop,
                                                              iter_max, Psi_initial, In_or_Out, BC6_location, precision)

# endregion

# ==================================================
# region   Find Velocities (Arbitrary Values)
# ==================================================
# ----- Arbitrary V_max Value -----
# Used to set up arbitrary vector field to visually evaluate convergence
V_max_arb = 100  # V_max = micron/sec

(Psi_Dimen_arb, v_x_arb, v_y_arb, v_y_neg_arb,
 v_mag_arb, v_x_unit_arb, v_y_neg_unit_arb) = VelSol.Vel_Solve(n_top, n_bot, n_rbc_x_approx, n_rbc_y_approx,
                                                               V_max_arb, L, h_decimal, Psi_Converged,
                                                               In_or_Out, BC6_location, precision)

# endregion

# ==================================================
# region   Visualize Arbitrary Velocities
# ==================================================
# filename_vector = filename + "_conv check vector.svg"
# filename_mag = filename + '_conv check mag.svg'
# filename_comp = filename + '_conv check comp.svg'
# filepath_vector = "C:\\Users\kylie\\OneDrive - University of Oklahoma\\Documents\\Graduate School\\Crossflow Elongational Stresses Model\\CrossFlow Python Excel File Outputs\\" + filename_vector
# filepath_mag = "C:\\Users\kylie\\OneDrive - University of Oklahoma\\Documents\\Graduate School\\Crossflow Elongational Stresses Model\\CrossFlow Python Excel File Outputs\\" + filename_mag
# filepath_comp = "C:\\Users\kylie\\OneDrive - University of Oklahoma\\Documents\\Graduate School\\Crossflow Elongational Stresses Model\\CrossFlow Python Excel File Outputs\\" + filename_comp
#
# # ----- Specify Spacing in x/y dimensions -----
# n_nodes = n_bot + 1
# x1 = np.linspace(0, n_bot, n_nodes)
# y1 = np.linspace(0, n_bot, n_nodes)
# x, y = np.meshgrid(x1, y1)
#
# # ----- Convert Decimal.decimal np arrays to floats -----
# v_x_arb_flt = v_x_arb.astype(float)
# v_y_arb_flt = v_y_arb.astype(float)
# v_y_neg_arb_flt = v_y_neg_arb.astype(float)
# v_mag_arb_flt = v_mag_arb.astype(float)
# v_x_unit_arb_flt = v_x_unit_arb.astype(float)
# v_y_neg_unit_arb_flt = v_y_neg_unit_arb.astype(float)
#
# # ----- Mask data to 'hide' parts outside grid or inside RBC -----
# mask_vx = np.ma.masked_array(v_x_arb_flt, mask=mask_full)
# mask_vy = np.ma.masked_array(v_y_arb_flt, mask=mask_full)
# mask_vy_neg = np.ma.masked_array(v_y_neg_arb_flt, mask=mask_full)
# mask_vmag = np.ma.masked_array(v_mag_arb_flt, mask=mask_full)
# mask_vx_unit = np.ma.masked_array(v_x_unit_arb_flt, mask=mask_full)
# mask_vy_neg_unit = np.ma.masked_array(v_y_neg_unit_arb_flt, mask=mask_full)
#
# # ----- Plot V_Mag and unit vectors -----
# # cmap_velocity = plt.cm.turbo
# # custom_cmap_velocity = cmap_velocity.copy()
# # custom_cmap_velocity.set_under(color='white')
#
# fig_arb_cmap, ax_arb_cmap = plt.subplots()
# fig_arb_cmap.set_size_inches(6, 6)
# map_velocity = ax_arb_cmap.imshow(mask_vmag, cmap='turbo', vmin=0, vmax=V_max_arb)
# bar_velocity = plt.colorbar(map_velocity, ax=ax_arb_cmap)
# ax_arb_cmap.quiver(x, y, mask_vx_unit, mask_vy_neg_unit, zorder=10)
# ax_arb_cmap.set_aspect('equal')
# # fig_arb_cmap.suptitle('Visually Check for Convergence\n Close Window to Continue Script')
# fig_arb_cmap.suptitle('Velocity Color Map with Unit Vectors (micron/s)\n Arbitrary Vmax = ' + str(V_max_arb))
# fig_arb_cmap.savefig(filepath_mag, dpi=400)
#
# # ----- Plot Vector Field -----
# fig_arb_vector, ax_arb_vector = plt.subplots()
# fig_arb_vector.set_size_inches(6, 6)
# ax_arb_vector.quiver(x, y, mask_vx, mask_vy_neg, mask_vmag, cmap='turbo')
# bar_velocity = plt.colorbar(map_velocity, ax=ax_arb_vector)
# ax_arb_vector.set_aspect('equal')
# ax_arb_vector.invert_yaxis()
# ax_arb_vector.set_facecolor('black')
# # fig_arb_vector.suptitle('Visually Check for Convergence\n Close Window to Continue Script')
# fig_arb_vector.suptitle('Velocity Vector Field (micron/s)\n Arbitrary Vmax = ' + str(V_max_arb))
# fig_arb_vector.savefig(filepath_vector, dpi=400)
#
# # ----- Plot Individual Velocity Components -----
# fig_arb_v_components, ax_arb_v_components = plt.subplots(1, 2)
# fig_arb_v_components.set_size_inches(12, 6)
# map_velocity_comp = ax_arb_v_components[0].imshow(mask_vx, cmap='turbo', vmin=0, vmax=V_max_arb)
# bar_velocity_comp = plt.colorbar(map_velocity_comp, ax=ax_arb_v_components)
# ax_arb_v_components[1].imshow(mask_vy, cmap='turbo', vmin=0, vmax=V_max_arb)
# ax_arb_v_components[0].set_title('v_x')
# ax_arb_v_components[1].set_title('v_y')
# ax_arb_v_components[0].set_aspect('equal')
# ax_arb_v_components[1].set_aspect('equal')
# # fig_arb_v_components.suptitle('Visually Check for Convergence\n Close Window to Continue Script')
# fig_arb_v_components.suptitle('Velocity Components (micron/s)\n Arbitrary Vmax = ' + str(V_max_arb))
# fig_arb_v_components.savefig(filepath_comp, dpi=400)

# winsound.PlaySound('SystemQuestion', winsound.SND_ALIAS)
# plt.show()
# endregion

# ==================================================
# region     User Check for Convergence
# ==================================================
# User will evaluate if Psi converged < iter_max and visually evaluate arbitrary velocity plots
# These parameters will indicate to the user if Psi converged
# Confirmation of convergence, code will continue onto solving zoomed matrix
# If NOT converged, code force stops so user can reset iter_max and/or epsilon_stop
# User_Converge_Check = input('Did Psi adequately converge (less than max_iteration, '
#                             'visually on arbitrary velocity plots)? (Y/N): ')

# if User_Converge_Check == 'N':
#     print("User indicated Psi did NOT adequately converge. Update convergence criteria/max number of iterations.")
#     sys.exit("Force Stop")
# endregion

# ==================================================
# region     Zoom, Expand and Resolve Psi
# ==================================================
# ----- Zoom & Resolve Psi -----
(Psi_Expd_orig_values, Psi_Expd_initial,
 Psi_Expd_converged, Psi_Expd_Err_Abs,
 Psi_k_Expd, Global_RE_Expd, Max_AE_Expd, RMS_Expd) = ZoomPsi.Zoom_Resolve_Psi(row_delete, column_delete,
                                                                               row_expanded, column_expanded,
                                                                               n_row_expanded, n_column_expanded,
                                                                               n_rbc_x_expd_approx, epsilon_stop,
                                                                               iter_max, Psi_Converged,
                                                                               In_or_Out_zoom_expd_fill,
                                                                               BC6_zoom_expd_fill, precision)
# endregion

# ==================================================
# region  Zoom, Expand Velocity (Arbitrary Values)
# ==================================================
(Psi_Dimen_Expd_arb, v_x_Expd_arb, v_y_Expd_arb,
 v_y_neg_Expd_arb, v_mag_Expd_arb, v_x_unit_Expd_arb,
 v_y_neg_unit_Expd_arb) = ZoomVel.Zoom_Resolve_Vel(row_expanded, column_expanded, n_row_expanded, n_column_expanded,
                                                   n_rbc_x_expd_approx, n_rbc_y_expd_approx, V_max_arb, L,
                                                   h_expd_decimal, Psi_Expd_converged,
                                                   In_or_Out_zoom_expd_fill, BC6_zoom_expd_fill, precision)
# endregion

# ==================================================
# region Visualize Arbitrary Velocities (Zoom/Expd)
# ==================================================
# filename_expd_vector = filename_expd + "_conv check vector.svg"
# filename_expd_mag = filename_expd + '_conv check mag.svg'
# filename_expd_comp = filename_expd + '_conv check comp.svg'
# filepath_expd_vector = "C:\\Users\kylie\\OneDrive - University of Oklahoma\\Documents\\Graduate School\\Crossflow Elongational Stresses Model\\CrossFlow Python Excel File Outputs\\" + filename_expd_vector
# filepath_expd_mag = "C:\\Users\kylie\\OneDrive - University of Oklahoma\\Documents\\Graduate School\\Crossflow Elongational Stresses Model\\CrossFlow Python Excel File Outputs\\" + filename_expd_mag
# filepath_expd_comp = "C:\\Users\kylie\\OneDrive - University of Oklahoma\\Documents\\Graduate School\\Crossflow Elongational Stresses Model\\CrossFlow Python Excel File Outputs\\" + filename_expd_comp
#
# # ----- Specify Spacing in x/y dimensions (EXPD GRID) -----
# x1_expd = np.linspace(0, n_column_expanded, n_column_expanded + 1)
# y1_expd = np.linspace(0, n_row_expanded, n_row_expanded + 1)
# x_expd, y_expd = np.meshgrid(x1_expd, y1_expd)
#
# # ----- Convert Decimal.decimal np arrays to floats -----
# v_x_Expd_arb_flt = v_x_Expd_arb.astype(float)
# v_y_Expd_arb_flt = v_y_Expd_arb.astype(float)
# v_y_neg_Expd_arb_flt = v_y_neg_Expd_arb.astype(float)
# v_mag_Expd_arb_flt = v_mag_Expd_arb.astype(float)
# v_x_unit_Expd_arb_flt = v_x_unit_Expd_arb.astype(float)
# v_y_neg_unit_Expd_arb_flt = v_y_neg_unit_Expd_arb.astype(float)
#
# # ----- Mask data to 'hide' parts outside grid or inside RBC -----
# mask_vx_expd = np.ma.masked_array(v_x_Expd_arb_flt, mask=mask_expd)
# mask_vy_expd = np.ma.masked_array(v_y_Expd_arb_flt, mask=mask_expd)
# mask_vy_neg_expd = np.ma.masked_array(v_y_neg_Expd_arb_flt, mask=mask_expd)
# mask_vmag_expd = np.ma.masked_array(v_mag_Expd_arb_flt, mask=mask_expd)
# mask_vx_unit_expd = np.ma.masked_array(v_x_unit_Expd_arb_flt, mask=mask_expd)
# mask_vy_neg_unit_expd = np.ma.masked_array(v_y_neg_unit_Expd_arb_flt, mask=mask_expd)
#
# # ----- Plot V_Mag and unit vectors (EXPD GRID) -----
# fig_expd_arb_cmap, ax_expd_arb_cmap = plt.subplots()
# # fig_expd_arb_cmap.set_size_inches(6, 4)
# map_velocity_expd = ax_expd_arb_cmap.imshow(mask_vmag_expd, cmap='turbo', vmin=0, vmax=V_max_arb)
# bar_velocity_expd = plt.colorbar(map_velocity_expd, ax=ax_expd_arb_cmap)
# ax_expd_arb_cmap.quiver(x_expd, y_expd, mask_vx_unit_expd, mask_vy_neg_unit_expd, zorder=10)
# ax_expd_arb_cmap.set_aspect('equal')
# # fig_expd_arb_cmap.suptitle('Visually Check for Convergence\n Close Window to Continue Script')
# fig_expd_arb_cmap.suptitle('Velocity Color Map with Unit Vectors (micron/s)\n Arbitrary Vmax = ' + str(V_max_arb))
# fig_expd_arb_cmap.savefig(filepath_expd_mag, dpi=400)
#
# # ----- Plot Vector Field (EXPD GRID) -----
# fig_expd_arb_vector, ax_expd_arb_vector = plt.subplots()
# # fig_expd_arb_vector.set_size_inches(6, 4)
# ax_expd_arb_vector.quiver(x_expd, y_expd, mask_vx_expd, mask_vy_neg_expd, mask_vmag_expd, cmap='turbo')
# bar_velocity_expd = plt.colorbar(map_velocity_expd, ax=ax_expd_arb_vector)
# ax_expd_arb_vector.set_aspect('equal')
# ax_expd_arb_vector.invert_yaxis()
# ax_expd_arb_vector.set_facecolor('black')
# # fig_expd_arb_vector.suptitle('Visually Check for Convergence\n Close Window to Continue Script')
# fig_expd_arb_vector.suptitle('Velocity Vector Field (micron/s)\n Arbitrary Vmax = ' + str(V_max_arb))
# fig_expd_arb_vector.savefig(filepath_expd_vector, dpi=400)
#
# # ----- Plot Individual Velocity Components -----
# fig_expd_arb_v_components, ax_expd_arb_v_components = plt.subplots(2, 1)
# # fig_expd_arb_v_components.set_size_inches(12, 4)
# map_expd_velocity_comp = ax_expd_arb_v_components[0].imshow(mask_vx_expd, cmap='turbo', vmin=0, vmax=V_max_arb)
# bar_expd_velocity_comp = plt.colorbar(map_velocity_comp, ax=ax_expd_arb_v_components)
# ax_expd_arb_v_components[1].imshow(mask_vy_expd, cmap='turbo', vmin=0, vmax=V_max_arb)
# ax_expd_arb_v_components[0].set_title('v_x')
# ax_expd_arb_v_components[1].set_title('v_y')
# ax_expd_arb_v_components[0].set_aspect('equal')
# ax_expd_arb_v_components[1].set_aspect('equal')
# # fig_arb_v_components.suptitle('Visually Check for Convergence\n Close Window to Continue Script')
# fig_arb_v_components.suptitle('Velocity Components (micron/s)\n Arbitrary Vmax = ' + str(V_max_arb))
# fig_expd_arb_v_components.savefig(filepath_expd_comp, dpi=400)

# winsound.PlaySound('SystemQuestion', winsound.SND_ALIAS)
# plt.show()
# endregion

# ==================================================
# region   User Check for Convergence (EXPD GRID)
# ==================================================
# User will evaluate if Psi converged < iter_max and visually evaluate arbitrary velocity plots
# These parameters will indicate to the user if Psi converged
# Confirmation of convergence, code will continue onto solving zoomed matrix
# If NOT converged, code force stops so user can reset iter_max and/or epsilon_stop
# User_Converge_Check = input('Did Zoomed/Expanded Psi adequately converge'
#                             ' (less than max_iteration, visually on arbitrary velocity plots)? (Y/N): ')
# if User_Converge_Check == 'N':
#     print("User indicated Psi did NOT adequately converge. Update convergence criteria/max number of iterations.")
#     sys.exit("Force Stop")
# endregion

# ==================================================
# region Convergence Time (FUll and EXPD Combined)
# ==================================================
end_time = time.time()
run_time = end_time - start_time
run_time_min = run_time/60
run_time_hr = run_time_min/60

# endregion

# ==================================================
# region    Filenames and Filepaths
# ==================================================
# ----- Excel Output -----
filename_xlsx = filename + ".xlsx"
filepath_xlsx = "C:\\Users\kylie\\OneDrive - University of Oklahoma\\Documents\\Graduate School\\Crossflow Elongational Stresses Model\\CrossFlow Python Excel File Outputs\\" + filename_xlsx
filename_other_xlsx = filename + ' other arrays.xlsx'
filepath_other_xlsx = "C:\\Users\kylie\\OneDrive - University of Oklahoma\\Documents\\Graduate School\\Crossflow Elongational Stresses Model\\CrossFlow Python Excel File Outputs\\" + filename_other_xlsx

# ----- Psi Plots -----
filename_Psi_cmap_svg = filename + "_Psi cmap.svg"
filepath_Psi_cmap_svg = "C:\\Users\kylie\\OneDrive - University of Oklahoma\\Documents\\Graduate School\\Crossflow Elongational Stresses Model\\CrossFlow Python Excel File Outputs\\" + filename_Psi_cmap_svg
filename_expd_Psi_cmap_svg = filename_expd + "_Psi cmap.svg"
filepath_expd_Psi_cmap_svg = "C:\\Users\kylie\\OneDrive - University of Oklahoma\\Documents\\Graduate School\\Crossflow Elongational Stresses Model\\CrossFlow Python Excel File Outputs\\" + filename_expd_Psi_cmap_svg
# endregion

# ==================================================
# region           Excel Output
# ==================================================
# ----- Parameter Arrays -----
general_parameters = np.array([["Variable Name", "Variable Description", 'Variable Value'],
                               ["2L", "Width of Channel (micron)", 2*L],
                               ['2D', 'Depth of Channel (micron)', 2*D],
                               ['aspect ratio', '', aspect_ratio],
                               ['a', 'RBC major radius (micron)', a],
                               ['b', 'RBC minor radius (micron)', b],
                               ['V_RBC', 'Approximate RBC Volume (cubic micron)', V_RBC],
                               ['epsilon_global_Psi', 'convergence criteria for GRE (% as decimal)', epsilon_global],
                               ['epsilon_max_Psi', 'convergence criteria for MAE (dimensionless Psi)', epsilon_max],
                               ['iter_max_Psi', 'max iterations allowed Psi convergence', iter_max],
                               ['precision', 'precision for Decimal.decimal (ie # digits)', precision],
                               ['run_time', 'time to converge (full and expd combined) (s)', run_time],
                               ['run_time_min', 'time to converge (full and expd combined) (min)', run_time_min],
                               ['run_time_hr', 'time to converge (full and expd combined) (hr)', run_time_hr],
                               ['Initial_Guess_Type', 'Previous solution for initial guess (Y/N)', Initial_Guess_Type],
                               ['Initial_Guess_Location', 'filepath for previous solution used', filepath_initial]])
Full_Grid_Parameters = np.array([["Variable Name", "Variable Description", 'Variable Value'],
                                 ['n_top', 'steps across top (center to wall)', n_top],
                                 ['n_bot', 'steps across bottom (center point to exit)', n_bot],
                                 ['h', 'step size (micron)', h],
                                 ['h_decimal', 'step size as a string (micron)', h_decimal],
                                 ['n_rbc_x_true', 'node where elliptic rbc ends on x-axis', n_rbc_x_true],
                                 ['n_rbc_y_true', 'node where elliptic rbc starts on y-axis', n_rbc_y_true],
                                 ['n_rbc_x_approx', 'node where stair step rbc ends on x-axis', n_rbc_x_approx],
                                 ['n_rbc_y_approx', 'node where stair step rbc starts on y-axis', n_rbc_y_approx],
                                 ['GRE', 'Global Relative Error (% as decimal)', Global_RE],
                                 ['MAE', 'Max Absolute Error (dimensionless Psi change)', Max_AE],
                                 ['RMS', 'Root Mean Square, avg absolute change per active node', RMS],
                                 ['k_full_Psi', 'number of iterations for full grid (Psi Convergence)', Psi_k]])
Zoom_EXPD_Parameters = np.array([["Variable Name", "Variable Description", 'Variable Value'],
                                 ['n_column_expanded', 'steps across length of expanded grid', n_column_expanded],
                                 ['n_row_expanded', 'steps across height of expanded grid', n_row_expanded],
                                 ['h_expd', 'step size for expanded grid (micron)', h_expd],
                                 ['h_expd_decimal', 'step size for expanded grid as a string (micron)', h_expd_decimal],
                                 ['n_rbc_x_expd_true', 'node where elliptic rbc ends on EXPD x-axis', n_rbc_x_expd_true],
                                 ['n_rbc_y_expd_true', 'node where elliptic rbc starts on EXPD y-axis', n_rbc_y_expd_true],
                                 ['n_rbc_x_expd_approx', 'node where stair step rbc ends on x-axis', n_rbc_x_expd_approx],
                                 ['n_rbc_y_expd_approx', 'node where stair step rbc starts on y-axis', n_rbc_y_expd_approx],
                                 ['L_expd', 'Length of expd grid (micron)', (n_column_expanded * h_expd)],
                                 ['H_expd', 'Height of expd grid (micron)', (n_row_expanded * h_expd)],
                                 ['GRE', 'Global Relative Error (% as decimal)', Global_RE_Expd],
                                 ['MAE', 'Max Absolute Error (dimensionless Psi change)', Max_AE_Expd],
                                 ['RMS', 'Root Mean Square, avg absolute change per active node', RMS_Expd],
                                 ['k_expd_Psi', 'number of iterations for expd grid (Psi Convergence)', Psi_k_Expd]])

# ----- Convert numpy arrays to dataframes -----
# Important Parameters
general_parameters_df = pd.DataFrame(general_parameters)
Full_Grid_Parameters_df = pd.DataFrame(Full_Grid_Parameters)
Zoom_EXPD_Parameters_df = pd.DataFrame(Zoom_EXPD_Parameters)

# Other arrays used for calculations
X_RBC_df = pd.DataFrame(X_RBC_arr)
Y_RBC_df = pd.DataFrame(Y_RBC_arr)
Alpha_1_df = pd.DataFrame(Alpha_1)
Beta_1_df = pd.DataFrame(Beta_1)
BC6_location_df = pd.DataFrame(BC6_location)
In_or_Out_df = pd.DataFrame(In_or_Out)
BC6_zoom_expd_original_df = pd.DataFrame(BC6_zoom_expd_original)
In_or_Out_zoom_expd_original_df = pd.DataFrame(In_or_Out_zoom_expd_original)
BC6_zoom_expd_fill_df = pd.DataFrame(BC6_zoom_expd_fill)
In_or_Out_zoom_expd_fill_df = pd.DataFrame(In_or_Out_zoom_expd_fill)
mask_full_df = pd.DataFrame(mask_full)
mask_expd_df = pd.DataFrame(mask_expd)

# Psi Arrays
Psi_initial_df = pd.DataFrame(Psi_initial)
Psi_Converged_df = pd.DataFrame(Psi_Converged)
Psi_Abs_Err_df = pd.DataFrame(Psi_Err_Abs)
Psi_Expd_orig_values_df = pd.DataFrame(Psi_Expd_orig_values)
Psi_Expd_initial_df = pd.DataFrame(Psi_Expd_initial)
Psi_Expd_converged_df = pd.DataFrame(Psi_Expd_converged)
Psi_Expd_Err_Abs_df = pd.DataFrame(Psi_Expd_Err_Abs)

# Arbitrary Flow Fields
v_x_arb_df = pd.DataFrame(v_x_arb)
v_y_arb_df = pd.DataFrame(v_y_arb)
v_x_Expd_arb_df = pd.DataFrame(v_x_Expd_arb)
v_y_Expd_arb_df = pd.DataFrame(v_y_Expd_arb)

# ----- Write to Excel -----
with pd.ExcelWriter(filepath_xlsx) as writer:
    general_parameters_df.to_excel(writer, sheet_name="GenParameter", header=False, index=False)
    Full_Grid_Parameters_df.to_excel(writer, sheet_name="FullParameter", header=False, index=False)
    Zoom_EXPD_Parameters_df.to_excel(writer, sheet_name="EXPDParameter", header=False, index=False)
    Psi_initial_df.to_excel(writer, sheet_name="Psi Initial(Full)", header=False, index=False)
    Psi_Converged_df.to_excel(writer, sheet_name="Psi(Full)", header=False, index=False)
    Psi_Abs_Err_df.to_excel(writer, sheet_name="Abs_Error(Full)", header=False, index=False)
    Psi_Expd_orig_values_df.to_excel(writer, sheet_name="Psi Expd_Original", header=False, index=False)
    Psi_Expd_initial_df.to_excel(writer, sheet_name="Psi Expd_Initial", header=False, index=False)
    Psi_Expd_converged_df.to_excel(writer, sheet_name="Psi Expd", header=False, index=False)
    Psi_Expd_Err_Abs_df.to_excel(writer, sheet_name="Abs_Error(Full)", header=False, index=False)
    v_x_arb_df.to_excel(writer, sheet_name="Arbitrary vx(Full)", header=False, index=False)
    v_y_arb_df.to_excel(writer, sheet_name="Arbitrary vy(Full)", header=False, index=False)
    v_x_Expd_arb_df.to_excel(writer, sheet_name="Arbitrary vx(Expd)", header=False, index=False)
    v_y_Expd_arb_df.to_excel(writer, sheet_name="Arbitrary vy(Expd)", header=False, index=False)

with pd.ExcelWriter(filepath_other_xlsx) as writer:
    X_RBC_df.to_excel(writer, sheet_name="RBC x positions", header=False, index=False)
    Y_RBC_df.to_excel(writer, sheet_name="RBC y positions", header=False, index=False)
    Alpha_1_df.to_excel(writer, sheet_name="Alpha(Full)", header=False, index=False)
    Beta_1_df.to_excel(writer, sheet_name="Beta(Full)", header=False, index=False)
    BC6_location_df.to_excel(writer, sheet_name="BC6(Full)", header=False, index=False)
    In_or_Out_df.to_excel(writer, sheet_name="In_Out(Full)", header=False, index=False)
    BC6_zoom_expd_original_df.to_excel(writer, sheet_name="BC6 Expd_Original", header=False, index=False)
    In_or_Out_zoom_expd_original_df.to_excel(writer, sheet_name="In_Out Expd_Original", header=False, index=False)
    BC6_zoom_expd_fill_df.to_excel(writer, sheet_name="BC6 Expd", header=False, index=False)
    In_or_Out_zoom_expd_fill_df.to_excel(writer, sheet_name="In_Out Expd", header=False, index=False)
    mask_full_df.to_excel(writer, sheet_name="Plot Mask(Full)", header=False, index=False)
    mask_expd_df.to_excel(writer, sheet_name="Plot Mask(Expd)", header=False, index=False)

# endregion

# ==================================================
# region     Plotting Setup - Elliptic RBC
# ==================================================
# ----- Parametric Range -----
# Create vectors, x_ellipse and y_ellipse, that hold values of parametric ellipse
# First, create a vector of t values to plug into parametric functions
# Use np.linspace(start, stop(inclusive), total number of values in array)
# Ex np.linspace(0, 2pi, 5) would give us [0, pi/2, pi, 3pi/2, and 2pi)
# tracing from 0 to pi/2 gives us the region of the ellipse we're interested in for the model
# From t, create x_ellipse and y_ellipse using numpy sin and cos functions in terms of nodes NOT L, a, and b
t = np.linspace(0, (np.pi/2), 50)

# ----- Full Grid -----
x_ellipse = n_rbc_x_true * np.cos(t)
y_ellipse = (2 * n_top) - ((n_bot - n_rbc_y_true) * np.sin(t))
# ----- Zoom/Expd Grid -----
x_ellipse_expd = n_rbc_x_expd_true * np.cos(t)
y_ellipse_expd = n_row_expanded - ((n_row_expanded - n_rbc_y_expd_true) * np.sin(t))

# endregion

# ==================================================
# region              Plot Psi
# ==================================================
# ----- Convert Decimal.decimal to floats -----
Psi_Converged_flt = Psi_Converged.astype(float)
Psi_Expd_converged_flt = Psi_Expd_converged.astype(float)

# ----- Mask data to 'hide' parts outside grid or inside RBC -----
mask_Psi_Converged = np.ma.masked_array(Psi_Converged_flt, mask=mask_full)
mask_Psi_Expd_converged = np.ma.masked_array(Psi_Expd_converged_flt, mask=mask_expd)

# ----- Define Color Map -----
# cmap_Psi = plt.cm.turbo
# custom_cmap_Psi = cmap_Psi.copy()
# custom_cmap_Psi.set_under(color='white')

# ----- Full Grid Plot -----
fig_Psi, ax_Psi = plt.subplots()
map_Psi = ax_Psi.imshow(mask_Psi_Converged, cmap='turbo', vmin=0, vmax=0.7)
bar_Psi = plt.colorbar(map_Psi, ax=ax_Psi)
ax_Psi.set_aspect('equal')
ax_Psi.plot(x_ellipse, y_ellipse, color='red')
fig_Psi.suptitle('Dimensionless Psi Values')
fig_Psi.savefig(filepath_Psi_cmap_svg, dpi=400)


# ----- Expd Grid -----
fig_Psi_EXPD, ax_Psi_EXPD = plt.subplots()
map_Psi_EXPD = ax_Psi_EXPD.imshow(mask_Psi_Expd_converged, cmap='turbo', vmin=0, vmax=0.7)
bar_Psi_EXPD = plt.colorbar(map_Psi_EXPD)
ax_Psi_EXPD.set_aspect('equal')
ax_Psi_EXPD.plot(x_ellipse_expd, y_ellipse_expd, color='red')
fig_Psi_EXPD.suptitle('Dimensionless Psi Values')
fig_Psi_EXPD.savefig(filepath_expd_Psi_cmap_svg, dpi=400)

# winsound.PlaySound('SystemQuestion', winsound.SND_ALIAS)

# plt.show()

# endregion
