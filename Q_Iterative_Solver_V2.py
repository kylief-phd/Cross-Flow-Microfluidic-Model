# ==================================================
# region      Script Description
# ==================================================
# Script is used to solve for flow rate given a given known EV diameter.
# Import solved Psi matrix (found using Initial_Psi_Convergence.py)
# V2 is the new version that always integrates along the exact elliptic RBC

# endregion

# ==================================================
# region      Import Needed Libraries
# ==================================================
# Python Libraries
import numpy as np
import pandas as pd
import math
import sympy as sp
import sys
import matplotlib.pyplot as plt
from decimal import Decimal, getcontext

# Other scripts
import Zoom_and_resolve_Vel_Function as ZoomVel
import Zoomed_Shear_Rate_Solver_Function as ZoomShear
import T_and_F_Solver_Function_V2 as TF
import Velocity_Solver_Function as VelSol

# endregion

# ==================================================
# region    Import Data From Psi Convergence
# ==================================================
# ----- User Defined Filepath for Excel Location -----
filepath_Psi = "C:\\Users\kylie\\OneDrive - University of Oklahoma\\Documents\\Graduate School\\Crossflow Elongational Stresses Model\\CrossFlow Python Excel File Outputs\\18_L=25, 200x200, err=5e-8(NEW), initial V3.xlsx"
# input('Filepath for Excel Workbook containing converged Psi data:\n '
#                  'Locate Excel file to upload in File Explorer, copy filepath and paste here\n'
#                  'DO NOT forget to include filename as well: ')
filepath_other_arr = "C:\\Users\kylie\\OneDrive - University of Oklahoma\\Documents\\Graduate School\\Crossflow Elongational Stresses Model\\CrossFlow Python Excel File Outputs\\18_L=25, 200x200, err=5e-8(NEW), initial V3 other arrays.xlsx"
# input('Filepath for Excel Workbook containing converged Psi data:\n '
#                        'Locate Excel file to upload in File Explorer, copy filepath and paste here\n'
#                        'DO NOT forget to include filename as well: ')

# ----- Parameters for BOTH Full and EXPD Grids -----
GenParameters = (pd.read_excel(filepath_Psi, sheet_name='GenParameter', header=None, dtype=object)).to_numpy()
L = int(GenParameters[1, 2] / 2)  # microns
D = int(GenParameters[2, 2] / 2)  # microns
a = Decimal(GenParameters[4, 2])  # microns
b = Decimal(GenParameters[5, 2])  # microns
precision = GenParameters[10, 2]  # IMPORTANT: index will change depending on conv criteria method used

# ----- Channel Dimensions Full Grid -----
FullParameters = (pd.read_excel(filepath_Psi, sheet_name='FullParameter', header=None, dtype=object)).to_numpy()
n_top = FullParameters[1, 2]
n_bot = FullParameters[2, 2]
h = FullParameters[3, 2]  # microns
h_decimal = Decimal(FullParameters[4, 2])  # microns
n_rbc_x_true = FullParameters[5, 2]
n_rbc_y_true = FullParameters[6, 2]
n_rbc_x_approx = FullParameters[7, 2]
n_rbc_y_approx = FullParameters[8, 2]

# ----- Channel Dimensions Zoom/Expd Grid -----
ExpdParameters = (pd.read_excel(filepath_Psi, sheet_name='EXPDParameter', header=None, dtype=object)).to_numpy()
n_column_expanded = ExpdParameters[1, 2]
n_row_expanded = ExpdParameters[2, 2]
h_expd = ExpdParameters[3, 2]  # microns
h_expd_decimal = Decimal(ExpdParameters[4, 2])  # microns
n_rbc_x_expd_true = ExpdParameters[5, 2]
n_rbc_y_expd_true = ExpdParameters[6, 2]
n_rbc_x_expd_approx = ExpdParameters[7, 2]
n_rbc_y_expd_approx = ExpdParameters[8, 2]

row_expanded = n_row_expanded + 1
column_expanded = n_column_expanded + 1
H_expd = n_row_expanded * h_expd_decimal

# ----- Converged Psi -----
# UNITS: dimensionless
# Full Grid as a np array full of strings
Psi_Converged_Full = (pd.read_excel(filepath_Psi, sheet_name='Psi(Full)', header=None, dtype=object)).to_numpy()
# Convert to decimal.Decimal
for j in range(0, n_bot + 1):
    for i in range(0, n_bot + 1):
        Psi_Converged_Full[j, i] = Decimal(Psi_Converged_Full[j, i])

# Zoom/Expanded Grid as a np array full of strings
# UNITS: dimensionless
Psi_Converged_Expd = (pd.read_excel(filepath_Psi, sheet_name='Psi Expd', header=None, dtype=object)).to_numpy()
# Convert to decimal.Decimal
for j in range(0, n_row_expanded + 1):
    for i in range(0, n_column_expanded + 1):
        Psi_Converged_Expd[j, i] = Decimal(Psi_Converged_Expd[j, i])

# ----- In_or_Out & BC6 Arrays -----
# Full Grid
In_or_Out_Full = (pd.read_excel(filepath_other_arr, sheet_name='In_Out(Full)', header=None)).to_numpy()
BC6_Full = (pd.read_excel(filepath_other_arr, sheet_name='BC6(Full)', header=None)).to_numpy()

# Zoom/Expanded Grid
In_or_Out_Expd = (pd.read_excel(filepath_other_arr, sheet_name='In_Out Expd', header=None)).to_numpy()
BC6_Expd = (pd.read_excel(filepath_other_arr, sheet_name='BC6 Expd', header=None)).to_numpy()

# ----- Plotting Masks -----
mask_full = (pd.read_excel(filepath_other_arr, sheet_name='Plot Mask(Full)', header=None)).to_numpy()
mask_expd = (pd.read_excel(filepath_other_arr, sheet_name='Plot Mask(Expd)', header=None)).to_numpy()

# endregion

# ==================================================
# region    Set Decimal Precision and Filename
# ==================================================
# Uses same precision from Initial_Psi_Convergence, imported from Excel
getcontext().prec = precision

filename_soln = 'Q for 2000 nm EV, err=0.001, L=25, 200x200 (5e-8, NEW)'  # input('Base filename for solution')
filename_soln_expd = filename_soln + "_EXPD"

# endregion

# ==================================================
# region    Parameters for Solving for Q
# ==================================================
# ----- Pre-set Parameters -----
viscosity_Pa_s = Decimal('0.06')  # Fluid viscosity in Pa-s
# convert to more convenient units
# UNITS: nN * s / micron^2
viscosity_nNs_micron2 = viscosity_Pa_s * (Decimal('1e-3'))

# ----- User Input Parameters -----
epsilon_stop = Decimal('0.001')  # Decimal(input('Convergence criteria (Q solver) as a decimal (ie if 1% input 0.01) = '))
iter_max = int(100)  # int(input('Max number of iterations to perform (Q solver) = '))
d_EV_nm = Decimal('2000')  # Decimal(input('Diameter of EV in nanometer (100-1000 nm) or half RBC is 4477= '))
RBC_Mem_T = 50  # Decimal(input('RBC Membrane Tension in nN/micron (1 nN/micron = 1 dyn/cm)'))

# ----- Calculated from User Input -----
# EV parameters
d_EV_micron = d_EV_nm / 1000  # Convert to microns
V_EV = Decimal(math.pi / 6) * (d_EV_micron ** 3)  # in cubic microns

# endregion

# ==================================================
# region  x on Elliptic corresponding to stair-step RBC nodes
# ==================================================
# This section finds all the x's on the elliptic geometry that correspond to a node on the approximated RBC
# Meaning each of these x's on the ellipse has a corresponding shear rate
# AND these x's on the ellipse should be our integration steps

# ----- Initialize array to hold ellipse x's -----
# holds x position in microns
RBC_Node_to_Ellip_x = np.full((1, 1), Decimal('0'), dtype=object)

# ----- Fill in the array for each approximated RBC node -----
for j in range(0, n_row_expanded + 1):
    for i in range(0, n_column_expanded + 1):
        if BC6_Expd[j, i] != 0 and i == 0:
            # ON boarder and ON first node of RBC (ie radius from line of symmetry = b)
            # Check for horizontal, vertical, or diagonal (should be horizontal always)
            # should just reset first column values, after which we will append the func_of_x arrays
            if BC6_Expd[j, i + 1] != 0:
                # On horizontal segment
                RBC_Node_to_Ellip_x[0, i] = i * h_expd_decimal
            elif BC6_Expd[j + 1, i] != 0:
                # On a vertical segment
                y_pos = j * h_expd_decimal
                # Find corresponding x on exact geometry
                x_exact = (a ** 2 * (1 - ((y_pos - H_expd) ** 2 / b ** 2)))**Decimal('0.5')
                RBC_Node_to_Ellip_x[0, i] = x_exact
            elif BC6_Expd[j + 1, i + 1] != 0:
                # On diagonal segment
                RBC_Node_to_Ellip_x[0, i] = i * h_expd_decimal
            else:
                print("Condition not coded for")
                sys.exit("Force Stop, Q Solver, line 163")
        elif BC6_Expd[j, i] != 0 and i != 0 and j != n_row_expanded:
            # ON boarder and NOT on any endpoint
            # Determine horizontal, vertical, or diagonal segment
            if BC6_Expd[j, i + 1] != 0:
                # On horizontal
                RBC_Node_to_Ellip_x = np.append(RBC_Node_to_Ellip_x, np.array([[i * h_expd_decimal]]), axis=1)
            elif BC6_Expd[j + 1, i] != 0:
                # On vertical
                y_pos = j * h_expd_decimal
                # Find corresponding x on exact geometry
                x_exact = (a ** 2 * (1 - ((y_pos - H_expd) ** 2 / b ** 2)))**Decimal('0.5')
                RBC_Node_to_Ellip_x = np.append(RBC_Node_to_Ellip_x, np.array([[x_exact]]), axis=1)
            elif BC6_Expd[j + 1, i + 1] != 0:
                # On diagonal
                RBC_Node_to_Ellip_x = np.append(RBC_Node_to_Ellip_x, np.array([[i * h_expd_decimal]]), axis=1)
            else:
                print("Condition not coded for")
                sys.exit("Force Stop, Q Solver, line 181")
        elif BC6_Expd[j, i] != 0 and j == n_row_expanded:
            # ON boarder and ON last node of RBC (ie radius from line of symmetry = 0)
            # we have to check in reverse direction for horizontal, vertical, or diagonal
            if BC6_Expd[j, i - 1] != 0:
                # ON horizontal
                RBC_Node_to_Ellip_x = np.append(RBC_Node_to_Ellip_x, np.array([[i * h_expd_decimal]]), axis=1)
            elif BC6_Expd[j - 1, i] != 0:
                # ON vertical
                # Check for backwards check, check vertical prior to diagonal
                # Sometime last node on j = n_bot will have two neighbors (one vertical and one diagonal)
                # Checking vertical first ensures we "draw" the shape correctly
                y_pos = j * h_expd_decimal
                # Find corresponding x on exact geometry
                x_exact = (a ** 2 * (1 - ((y_pos - H_expd) ** 2 / b ** 2)))**Decimal('0.5')
                RBC_Node_to_Ellip_x = np.append(RBC_Node_to_Ellip_x, np.array([[x_exact]]), axis=1)
            elif BC6_Expd[j - 1, i - 1] != 0:
                # ON diagonal
                RBC_Node_to_Ellip_x = np.append(RBC_Node_to_Ellip_x, np.array([[i * h_expd_decimal]]), axis=1)
            else:
                print("Condition not coded for")
                sys.exit("Force Stop, Q Solver, line 202")

# ----- Number of Steps on Ellipse -----
n_elliptic_RBC = np.size(RBC_Node_to_Ellip_x, axis=1) - 1

# endregion

# ==================================================
# region  Find EV Pinch Location: Elliptic RBC Geometry
# ==================================================

# ----- Solve for exact pinch location -----
# Given a known EV diameter (user defined) and therefore a known EV volume, Solve for the breakpoint
# Eqn for volume derived from parametric ellipse equations and a volume of revolution about the symmetry line
# Our volume of interest should be found in the range t=[0, pi/2], integration from some location to 0 yields EV vol
# Equation CANNOT be explicitly solved for t_EV so use SymPy to solve

# Define symbol for use in sympy
t_EV= sp.symbols('t_EV')
# Define the equation to be solved
eq = sp.Eq(V_EV,  # LHS
           (a * b**2 * sp.pi) * (Decimal(2/3) + ((sp.cos(t_EV)**3) / 3) - sp.cos(t_EV))  # RHS
           )
# input into nsolve
# sp.nsolve(equation to be solved, variable to solve for, initial guess range, solver = 'bisect')
t_EV_exact = sp.nsolve(eq, t_EV, (0, 1.58), solver='bisect')
# Convert t_EV_exact to x position
x_EV_exact = float(a) * math.cos(t_EV_exact)  # microns
y_EV_top_half_exact = -math.sqrt((1 - (x_EV_exact**2/float(a)**2)) * float(b)**2) + float(H_expd)  # microns
r_break_exact = float(b) * math.sin(t_EV_exact)  # microns

# t_EV_exact and x_EV_exact are the exact solution for the breakpoint which may or may not
# directly correspond to a node on the elliptic RBC

# ----- Determine we have more than one integration step -----
if RBC_Node_to_Ellip_x[0, n_elliptic_RBC-1] <= Decimal(x_EV_exact) <= RBC_Node_to_Ellip_x[0, n_elliptic_RBC]:
    print('WARNING: There is only one integration step for Tension and Force Calculations')
    print('Ideally there would be more than one step')
    cont = input('Continue? (Y/N) ')
    if cont == 'N':
        sys.exit("Force Stop")

# ----- Determine if x_EV_exact falls on a node or the neighboring points that do -----
# We want to integrate from known node points

# setting initial values for x_EV left and right locations outside the grid
# ie if they are > h_expd * (n_column_expanded) they have NOT been calculated/set
# x_EV_break and n_EV_Break_Ellipse are the values we will use to calculate tension and shear
# They may or may not be the exact solution determined above
x_EV_break = h_expd_decimal * (n_column_expanded + 2)
x_EV_left = h_expd_decimal * (n_column_expanded + 2)
x_EV_right = h_expd_decimal * (n_column_expanded + 2)
n_EV_Break_Ellipse = n_column_expanded + 2
n_EV_Break_Ellipse_left = n_column_expanded + 2
n_EV_Break_Ellipse_right = n_column_expanded + 2

for i in range(0, n_elliptic_RBC + 1):
    if RBC_Node_to_Ellip_x[0, i] == x_EV_exact:
        # x_EV_exact is already a node on elliptic RBC
        x_EV_break = x_EV_exact
        n_EV_Break_Ellipse = i
        break
    elif RBC_Node_to_Ellip_x[0, i] < x_EV_exact < RBC_Node_to_Ellip_x[0, i + 1]:
        x_EV_left = RBC_Node_to_Ellip_x[0, i]
        x_EV_right = RBC_Node_to_Ellip_x[0, i + 1]
        n_EV_Break_Ellipse_left = i
        n_EV_Break_Ellipse_right = i + 1
        break

# ----- Round break point to nearest elliptic node -----
# We cannot use round() as x-position is likely fraction and number of decimals needed depends on mesh size
if n_EV_Break_Ellipse > n_column_expanded and x_EV_break > (h_expd_decimal * n_column_expanded):
    # This means x_EV_exact did NOT correspond to an exact breakpoint
    # ie n_EV_Break_Ellipse and x_EV_break did NOT reset from initial values outside of grid
    # Determine rounded/estimated break point that does fall on a node
    Dist_left_node = (Decimal(x_EV_exact) - x_EV_left) / (x_EV_right - x_EV_left)
    Dist_right_node = (x_EV_right - Decimal(x_EV_exact)) / (x_EV_right - x_EV_left)
    # Quick confirmation that calculation is correct
    # if Dist_left_node + Dist_right_node != 1:
    #     print("Something wrong in rounded EV breakpoint")
    #     sys.exit("Force Stop, Q Solver, line 272")

    # Actually do the rounding now
    if Dist_left_node < 0.5:
        # round down
        x_EV_break = x_EV_left
        n_EV_Break_Ellipse = n_EV_Break_Ellipse_left
    elif Dist_right_node <= 0.5:
        # round up
        x_EV_break = x_EV_right
        n_EV_Break_Ellipse = n_EV_Break_Ellipse_right

# Double check that x_EV_break and n_EV_Break_Ellipse have been set
if x_EV_break > (h_expd_decimal * n_column_expanded) and n_EV_Break_Ellipse > n_column_expanded:
    print("EV break point for Tension and Force Calculations NOT Set")
    sys.exit("Force Stop, Q Solver, line 296")

# Ensure Breakpoint is not > 8.039259
if x_EV_break >= Decimal(8.039259) / 1:
    print("Break point is > 8.039259")
    print('last RBC node is set to 8.039259 NOT x=a=8.03926')
    print('to bypass undefined tangent slope issues')
    print('choose new pseudo RBC endpoint and rerun code for this mesh size')
    sys.exit("Force Stop, Q Solver, line 257")

# Convert x_EV_break to t_EV_break
t_EV_break = math.acos(float(x_EV_break)/float(a))

# Calculate y position for breakpoint to be used as well as radius at that point on the ellipse
y_EV_top_half_break = -math.sqrt((1 - (float(x_EV_break)**2/float(a)**2)) * float(b)**2) + float(H_expd)  # microns
r_break = float(b) * math.sin(t_EV_break)  # microns

# ----- Reset last node so that it doesn't cause undefined slopes -----
# Only doing this after we confirm x_EV_break is NOT > 8.039259
# NOTE: last node should be x = a = 8.0326, reset so that it is 8.0325
# IMPORTANT: All variables calculated as a function of x will be using the true values at x= a for this node
# Just use 8.0325 when we need the tangent slope eveything else uses true a
RBC_Node_to_Ellip_x[0, n_elliptic_RBC] = Decimal(8.039259) / 1

# ----- Volume of EV from rounded break point -----
# UNITS: micron ^ 3
V_EV_rnd = (float(a) * float(b)**2 * math.pi) * ((2/3) + ((math.cos(t_EV_break)**3) / 3) - math.cos(t_EV_break))

# ----- Diameter of EV from rounded break point -----
# UNITS: micron
d_EV_rnd_micron = ((6 * Decimal(V_EV_rnd)) / Decimal(math.pi)) ** (Decimal('1')/Decimal('3'))
# UNITS: nm
d_EV_rnd_nm = d_EV_rnd_micron * 1000

# endregion

# ==================================================
# region  Determine Force We Need to Overcome
# ==================================================
# UNITS: nN
F_needed = RBC_Mem_T * 2 * Decimal(math.pi) * Decimal(r_break)
F_half_needed = RBC_Mem_T * 2 * Decimal(math.pi) * b

# endregion

# ==================================================
# region   Set up Initial Guesses-Bisection Method
# ==================================================
# ----- Initial Guess for V_max_0 -----
# UNITS: micron/s
# V_max_0 will always be zero for bisection method (ie F_0 < F_needed)
print('Begin Iterations')
V_max_guess_0 = 0

# ----- Initial Guess for V_max_1 -----
# For bisection method we need v_max_1 to be too high (ie F_1 > F_needed)
# Initial guess based on OU micro-channel typical flow rate of 125 uL/min as a starting point
Q_uL_min_guess = Decimal('4000')  # Decimal(input('Initial Guess for Q (uL/min), should be larger than solution (Good starting point is 125 uL/min) = '))

# Convert to um^3/sec
Q_um3_sec_guess = Q_uL_min_guess * (Decimal('1e9') / Decimal('60'))

# Solve for v_max, assuming velocity does NOT vary in Z direction
# this assumption should be fine for an initial guess
# UNITS: micron/sec
V_max_guess_1 = (Decimal('3') / Decimal('8')) * (Q_um3_sec_guess / (L * D))

# first guess for iteration (midpoint between 0 and 1)
V_max_guess_2 = (V_max_guess_0 + V_max_guess_1) / 2

# ----- Initialize Arrays -----
# Set up 4, 1D arrays that will hold values throughout iterations
# One for V_max_guess values, one for corresponding F_EV, one for corresponding T_EV, and one for corresponding err
V_max_arr = np.array((V_max_guess_0, V_max_guess_1, V_max_guess_2), dtype=object)
F_EV_arr = np.array((0, 0), dtype=object)
T_EV_arr = np.array((0, 0), dtype=object)
err_F = np.array((0, 0), dtype=object)

# ----- Fill in arrays for initial guesses -----
for m in range(0, 2):
    # Solve velocity field given V_max
    # UNITS: Psi in micron^2/s, velocity in micron/s
    (Psi_Dimen_Expd, v_x_Expd, v_y_Expd,
     v_y_neg_Expd, v_mag_Expd, v_x_unit_Expd,
     v_y_neg_unit_Expd) = ZoomVel.Zoom_Resolve_Vel(row_expanded, column_expanded, n_row_expanded, n_column_expanded,
                                                   n_rbc_x_expd_approx, n_rbc_y_expd_approx, V_max_arr[m], L,
                                                   h_expd_decimal, Psi_Converged_Expd, In_or_Out_Expd, BC6_Expd,
                                                   precision)
    # Solve for shear rates along RBC
    # UNITS: 1/s
    (dvx_dx_expd, dvy_dy_expd, dvy_dx_expd,
     dvx_dy_expd, gamma_xx, gamma_yy,
     gamma_yx, gamma_xy, gamma_xx_func_of_x,
     gamma_yy_func_of_x, gamma_yx_func_of_x,
     gamma_xy_func_of_x) = ZoomShear.Shear_Rate(row_expanded, column_expanded, n_row_expanded, n_column_expanded,
                                                n_rbc_x_expd_approx, n_rbc_y_expd_approx, h_expd_decimal,
                                                v_x_Expd, v_y_Expd, In_or_Out_Expd, BC6_Expd, precision, a, b)
    # Solve for Tension and Force
    # UNITS: T_EV in nN/micron, F_EV in nN
    # IMPORTANT: viscosity input should be in nN*s/micron^2
    (T_EV_arr[m], F_EV_arr[m]) = TF.T_and_F(n_elliptic_RBC, n_EV_Break_Ellipse, viscosity_nNs_micron2, a, b,
                                            gamma_xx_func_of_x, gamma_yy_func_of_x, gamma_yx_func_of_x,
                                            gamma_xy_func_of_x, RBC_Node_to_Ellip_x, precision)

    # Calculate errors for first two guesses
    err_F[m] = (F_EV_arr[m] - F_needed) / F_needed

# Confirm first two errors have different signs (ie on either side of the root)
if err_F[0] * err_F[1] > 0:
    print("First two V_max guesses are not on either side of the solution, which does not work for bisection method")
    print('Increase initial guess for Q (corresponds to V_max_guess_1)')
    print('F_needed to pinch EV = ' + str(F_needed))
    print('Initial Guess for Q = ' + str(Q_uL_min_guess))
    print('V_max_guess_0 = ' + str(V_max_arr[0]))
    print('F_0 = ' + str(F_EV_arr[0]))
    print('V_max_guess_1 = ' + str(V_max_arr[1]))
    print('F_1 = ' + str(F_EV_arr[1]))
    sys.exit("Force Stop, Q_Iterative_Solver, line 425")
else:
    # set lower and upper index, used in solver
    lower_index = 0
    upper_index = 1

# endregion

# ==================================================
# region      Iteratively Solve for Q
# ==================================================
# ----- Set Up Variables Needed for Loop -----
epsilon = Decimal('1')  # starting error so we don't start "converged"
V_max_Solution = 0  # initially 0
# iteration counter, starting at 1 since V_max_1 has already been calculated
# Can also use to reference V_max_arr, T_EV_arr, F_EV_arr, and err_F
k = 1

# ----- Begin Iterations -----
while epsilon > epsilon_stop:
    # add 1 to iteration counter at beginning of each iteration
    # will start at k = 2
    k += 1
    # visual indicator that code hasn't frozen
    if k % 10 == 0:
        print("k_Q = " + str(k))

    # ----- Find velocity components with current velocity guess -----
    # UNITS: Psi in micron^2/s, velocity in micron/s
    (Psi_Dimen_Expd, v_x_Expd, v_y_Expd,
     v_y_neg_Expd, v_mag_Expd, v_x_unit_Expd,
     v_y_neg_unit_Expd) = ZoomVel.Zoom_Resolve_Vel(row_expanded, column_expanded, n_row_expanded, n_column_expanded,
                                                   n_rbc_x_expd_approx, n_rbc_y_expd_approx, V_max_arr[k], L,
                                                   h_expd_decimal, Psi_Converged_Expd, In_or_Out_Expd, BC6_Expd,
                                                   precision)

    # ----- Find shear rate at each node along RBC -----
    # UNITS: 1/s
    (dvx_dx_expd, dvy_dy_expd, dvy_dx_expd,
     dvx_dy_expd, gamma_xx, gamma_yy,
     gamma_yx, gamma_xy, gamma_xx_func_of_x,
     gamma_yy_func_of_x, gamma_yx_func_of_x,
     gamma_xy_func_of_x) = ZoomShear.Shear_Rate(row_expanded, column_expanded, n_row_expanded, n_column_expanded,
                                                n_rbc_x_expd_approx, n_rbc_y_expd_approx, h_expd_decimal,
                                                v_x_Expd, v_y_Expd, In_or_Out_Expd, BC6_Expd, precision, a, b)

    # ----- Find Tension and Force from breakpoint to nose -----
    # UNITS: T_EV in nN/micron, F_EV in nN
    (T_EV, F_EV) = TF.T_and_F(n_elliptic_RBC, n_EV_Break_Ellipse, viscosity_nNs_micron2, a, b,
                              gamma_xx_func_of_x, gamma_yy_func_of_x, gamma_yx_func_of_x,
                              gamma_xy_func_of_x, RBC_Node_to_Ellip_x, precision)


    # ----- Add Tension and Force to arrays, dependent on k -----
    T_EV_arr = np.append(T_EV_arr, T_EV)
    F_EV_arr = np.append(F_EV_arr, F_EV)

    # ----- Compare F to the F_needed -----
    # Find percent difference
    err_F = np.append(err_F, ((F_EV - F_needed) / F_needed))
    epsilon = abs(err_F[k])

    # ----- If Converged Break out of While Loop -----
    if epsilon <= epsilon_stop:
        # converged, break out of while
        V_max_Solution = V_max_arr[k]
        solution_index = k
        break

    # ----- If NOT Converged, update V_max_guess -----
    elif epsilon > epsilon_stop:
        # get a new V_max_guess
        if err_F[lower_index] * err_F[k] < 0:
            # new range is between lower_index and k (new upper index)
            upper_index = k
            v_max_new = (V_max_arr[lower_index] + V_max_arr[upper_index]) / 2
            V_max_arr = np.append(V_max_arr, v_max_new)
        elif err_F[upper_index] * err_F[k] < 0:
            # new range is between k (new lower index) and upper index
            lower_index = k
            v_max_new = (V_max_arr[lower_index] + V_max_arr[upper_index]) / 2
            V_max_arr = np.append(V_max_arr, v_max_new)

    # ----- Check iteration count against max iteration (failsafe for infinite loop) -----
    if (k - 1) >= iter_max:
        break

# ----- Print Statement Regarding Convergence -----
if (k - 1) >= iter_max:
    print('Q_solver reached max number of iterations allowed to prevent an infinite loop. Number of iterations = ' +
          str(k - 1))
    print('Last v_max_guess used = ' + str(V_max_arr[k]))
else:
    print('Q solver converged')
    print('Number of iterations for Q solver = ' + str(k - 1))
    print('V_max solution = ' + str(V_max_Solution))

# endregion

# ==================================================
# region       Flow Field for Solution
# ==================================================
# -----Convert V_max_solution to Q, assuming velocity does NOT vary in Z direction for now -----
# UNITS: micron^3/s
Q_um3_sec_soln = (Decimal('8')/Decimal('3')) * L * D * V_max_Solution
# UNITS: uL/min
Q_uL_min_soln = Q_um3_sec_soln * (Decimal('60') / (Decimal('1e9')))
# UNITS: uL/hr
Q_uL_hr_soln = Q_uL_min_soln * Decimal('60')

print('Q_soln (uL/min) assuming no z dependence = ' + str(Q_uL_min_soln))
print('Q_soln (uL/hr) assuming no z dependence = ' + str(Q_uL_hr_soln))

# ----- Get velocity field for solution: Full Grid -----
# UNITS: Psi in micron^2/s, velocity in micron/s
(Psi_Dimen_Full_soln, v_x_Full_soln, v_y_Full_soln,
 v_y_neg_Full_soln, v_mag_Full_soln, v_x_unit_Full_soln,
 v_y_neg_unit_Full_soln) = VelSol.Vel_Solve(n_top, n_bot, n_rbc_x_approx, n_rbc_y_approx,
                                            V_max_Solution, L, h_decimal, Psi_Converged_Full,
                                            In_or_Out_Full, BC6_Full, precision)

# ----- Get velocity field for solution: Zoom/Expd Grid -----
# UNITS: Psi in micron^2/s, velocity in micron/s
(Psi_Dimen_Expd_soln, v_x_Expd_soln, v_y_Expd_soln,
 v_y_neg_Expd_soln, v_mag_Expd_soln, v_x_unit_Expd_soln,
 v_y_neg_unit_Expd_soln) = ZoomVel.Zoom_Resolve_Vel(row_expanded, column_expanded, n_row_expanded, n_column_expanded,
                                                    n_rbc_x_expd_approx, n_rbc_y_expd_approx,
                                                    V_max_Solution, L, h_expd_decimal,
                                                    Psi_Converged_Expd, In_or_Out_Expd, BC6_Expd, precision)
# ----- Get shear rates for solution: Zoom/Expd Grid only -----
(dvx_dx_expd_soln, dvy_dy_expd_soln, dvy_dx_expd_soln,
 dvx_dy_expd_soln, gamma_xx_soln, gamma_yy_soln,
 gamma_yx_soln, gamma_xy_soln, gamma_xx_func_of_x_soln,
 gamma_yy_func_of_x_soln, gamma_yx_func_of_x_soln,
 gamma_xy_func_of_x_soln) = ZoomShear.Shear_Rate(row_expanded, column_expanded, n_row_expanded, n_column_expanded,
                                            n_rbc_x_expd_approx, n_rbc_y_expd_approx, h_expd_decimal,
                                            v_x_Expd_soln, v_y_Expd_soln, In_or_Out_Expd, BC6_Expd, precision, a, b)

# ----- Get Force and Tension at EV Breakpoint -----
T_EV_soln = T_EV_arr[solution_index]
F_EV_soln = F_EV_arr[solution_index]

# endregion

# ==================================================
# region Force/ Tension along RBC (each step, not summed)
# ==================================================
(tau_xx_func_of_x_soln, tau_yy_func_of_x_soln,
 tau_yx_func_of_x_soln, tau_xy_func_of_x_soln,
 T_func_of_x_soln, F_func_of_x_soln, m_tan_func_of_x_soln,
 mag_tau_parallel_Tm_func_of_x_soln, r_func_of_x_soln,
 Tension_Half_RBC_soln, Force_Half_RBC_soln) = TF.T_and_F_func_x(n_elliptic_RBC, viscosity_nNs_micron2, a, b, gamma_xx_func_of_x,
                                                       gamma_yy_func_of_x, gamma_yx_func_of_x, gamma_xy_func_of_x,
                                                       RBC_Node_to_Ellip_x, precision)

# endregion

# ==================================================
# region    Stagnation Pressure Contribution
# ==================================================
# ----- Convert V_max to m/s -----
# UNITS: m/s
V_max_Solution_m_s = V_max_Solution * Decimal('1e-6')
# ----- Stagnation Pressure -----
# UNITS: N/m^2
P_stag_N_m2 = Decimal('0.5') * 1000 * V_max_Solution_m_s

# UNITS: nN/um^2
P_stag = P_stag_N_m2 * Decimal('1e-3')

# ----- Portion of stagnation pressure acting parallel to membrane -----
# slope of tangent line at x_EV_break
m_tan_break = ((b / a**2) * x_EV_break) / ((1 - (x_EV_break**2 / a**2))**(Decimal(0.5)))
# Magnitude of P_stag component parallel to Tm
P_stag_para = P_stag / ((1 + (m_tan_break**2))**(Decimal(0.5)))

# ----- Tension from Stagnation Pressure -----
# UNITS: nN/um
# Tension is acting in positive x-direction
T_P_stag = (P_stag * Decimal(r_break)) / 2
# Component parallel to membrane tension (found from vector projection)
T_P_stag_para = (P_stag_para * Decimal(r_break)) / 2

# ----- Force from Stagnation Pressure -----
# UNITS: nN
# Force is acting in positive x-direction
F_P_stag = P_stag * Decimal(math.pi) * (Decimal(r_break) ** 2)
# Component parallel to membrane tension (found from vector projection)
F_P_stag_para = P_stag_para * Decimal(math.pi) * (Decimal(r_break) ** 2)

# endregion

# ==================================================
# region    Filenames and Filepaths
# ==================================================
# ----- Excel Output -----
filename_xlsx = filename_soln + '.xlsx'
filepath_xlsx = "C:\\Users\kylie\\OneDrive - University of Oklahoma\\Documents\\Graduate School\\Crossflow Elongational Stresses Model\\CrossFlow Python Excel File Outputs\\" + filename_xlsx

# ----- Velocity Plots -----
filename_vector_full_png = filename_soln + '_velocity quiver.png'
filename_vector_expd_png = filename_soln_expd + '_velocity quiver.png'
filename_vel_cmap_full_png = filename_soln + '_velocity cmap.png'
filename_vel_cmap_expd_png = filename_soln_expd + '_velocity cmap.png'
filepath_vector_full_png = "C:\\Users\kylie\\OneDrive - University of Oklahoma\\Documents\\Graduate School\\Crossflow Elongational Stresses Model\\CrossFlow Python Excel File Outputs\\" + filename_vector_full_png
filepath_vector_expd_png = "C:\\Users\kylie\\OneDrive - University of Oklahoma\\Documents\\Graduate School\\Crossflow Elongational Stresses Model\\CrossFlow Python Excel File Outputs\\" + filename_vector_expd_png
filepath_vel_cmap_full_png = "C:\\Users\kylie\\OneDrive - University of Oklahoma\\Documents\\Graduate School\\Crossflow Elongational Stresses Model\\CrossFlow Python Excel File Outputs\\" + filename_vel_cmap_full_png
filepath_vel_cmap_expd_png = "C:\\Users\kylie\\OneDrive - University of Oklahoma\\Documents\\Graduate School\\Crossflow Elongational Stresses Model\\CrossFlow Python Excel File Outputs\\" + filename_vel_cmap_expd_png

# endregion
# ==================================================
# region           Excel Output
# ==================================================
# ----- Parameter Arrays -----
Model_Criteria = np.array([['Variable Name', 'Variable Description', 'Variable Value'],
                           ['viscosity', 'fluid viscosity (Pa-s)', viscosity_Pa_s],
                           ['viscosity', 'fluid viscosity (nN-s/um^2)', viscosity_nNs_micron2],
                           # ['d_EV', 'Desired EV diameter (nm)', d_EV_nm],
                           ['d_EV', 'Desired EV diameter (um)', d_EV_micron],
                           ['V_EV', 'Volume of desired EV (um^3)', V_EV],
                           ['RBC_Mem_T', 'RBC Membrane breaking tension (nN/um = dyn/cm)', RBC_Mem_T],
                           ['epsilon_stop_Q', 'convergence criteria for Q (% as a decimal)', epsilon_stop],
                           ['iter_max_Q', 'max iterations allowed for Q convergence', iter_max]])
Solution_Parameters = np.array([['Variable Name', 'Variable Description', 'Variable Value'],
                                ['x_EV_exact', 'exact x position for EV pinch point on zoom/expd grid (um)', x_EV_exact],
                                ['x_EV_left', 'x position left of EV pinch point on zoom/expd grid (um)', x_EV_left],
                                ['x_EV_right', 'x position right of EV pinch point on zoom/expd grid (um)', x_EV_right],
                                ['n_EV_left', 'node on elliptic RBC left of pinch point on zoom/expd grid', n_EV_Break_Ellipse_left],
                                ['n_EV_right', 'node on elliptic RBC right of pinch point on zoom/expd grid', n_EV_Break_Ellipse_right],
                                ['x_EV_break', 'rounded x-position to EV pinch point (um) (used to solve Q)', x_EV_break],
                                ['n_EV_break', 'node on elliptic RBC corresponding to x_EV_break', n_EV_Break_Ellipse],
                                ['n_elliptic_RBC', 'steps along ellipse corresponding to calculated shear rates', n_elliptic_RBC],
                                ['d_EV_micron_model', 'EV Diameter from model rounded pinch location (um)', d_EV_rnd_micron],
                                ['V_EV_model', 'EV Volume from model rounded pinch location (um^3)', V_EV_rnd],
                                ['F_needed', 'Force to overcome membrane tension at rounded EV pinch location (nN)', F_needed],
                                ['Q_uL_min_soln', 'Q SOLUTION (uL/min)', Q_uL_min_soln],
                                ['Q_uL_hr_soln', 'Q SOLUTION (uL/hr)', Q_uL_hr_soln],
                                ['T_EV_soln', 'Tension at EV breakpoint from flow (nN/um)', T_EV_soln],
                                ['F_EV_soln', 'Force applied to EV from flow (nN)', F_EV_soln],
                                ['P_stag', 'Stagnation Pressure (nN/um^2)', P_stag],
                                ['T_P_stag', 'Tension at EV breakpoint from stagnation pressure (nN/um)', T_P_stag],
                                ['F_P_stag', 'Force applied to EV from stagnation pressure (nN)', F_P_stag],
                                ['P_stag_para', 'Magnitude of portion of stagnation pressure acting parallel to membrane tension (nN/um^2)', P_stag_para],
                                ['T_P_stag_para', 'Tension at EV breakpoint from stag pressure parallel to membrane tension (nN/um)', T_P_stag_para],
                                ['F_P_stag_para', 'Force applied to EV from stag pressure parallel to membrane tension (nN)', F_P_stag_para],
                                ['T_Half_RBC', 'Tension applied to half of RBC (nN/um)', Tension_Half_RBC_soln],
                                ['F_Half_RBC', 'Force applied ot half of RBC (nN)', Force_Half_RBC_soln],
                                ['F_Half_needed', 'Force needed to break RBC in half (nN)', F_half_needed],
                                ['k_Q', 'number of iterations (Q Solver)', k - 1]])

# ----- One Spreadsheet with all variables as a function of elliptic x position -----
# Add variable names to end of each array
RBC_Node_to_Ellip_x = np.append(RBC_Node_to_Ellip_x, np.array([['RBC x position (um)']]), axis=1)
r_func_of_x_soln = np.append(r_func_of_x_soln, np.array([['RBC radius (um)']]), axis=1)
m_tan_func_of_x_soln = np.append(m_tan_func_of_x_soln, np.array([['RBC tangent slope']]), axis=1)
tau_xx_func_of_x_soln = np.append(tau_xx_func_of_x_soln, np.array([['tau_xx (nN/um^2)']]), axis=1)
tau_yy_func_of_x_soln = np.append(tau_yy_func_of_x_soln, np.array([['tau_yy (nN/um^2)']]), axis=1)
tau_yx_func_of_x_soln = np.append(tau_yx_func_of_x_soln, np.array([['tau_yx (nN/um^2)']]), axis=1)
tau_xy_func_of_x_soln = np.append(tau_xy_func_of_x_soln, np.array([['tau_xy (nN/um^2)']]), axis=1)
mag_tau_parallel_Tm_func_of_x_soln = np.append(mag_tau_parallel_Tm_func_of_x_soln, np.array([['magnitude tau to tangent RBC (nN/um^2)']]), axis=1)
T_func_of_x_soln = np.append(T_func_of_x_soln, np.array([['Tension (nN/um)']]), axis=1)
F_func_of_x_soln = np.append(F_func_of_x_soln, np.array([['Force (nN)']]), axis=1)

# Combine into one array
Variables_func_x = np.append(RBC_Node_to_Ellip_x, r_func_of_x_soln, axis=0)
Variables_func_x = np.append(Variables_func_x, m_tan_func_of_x_soln, axis=0 )
Variables_func_x = np.append(Variables_func_x, tau_xx_func_of_x_soln, axis=0)
Variables_func_x = np.append(Variables_func_x, tau_yy_func_of_x_soln, axis=0)
Variables_func_x = np.append(Variables_func_x, tau_yx_func_of_x_soln, axis=0)
Variables_func_x = np.append(Variables_func_x, tau_xy_func_of_x_soln, axis=0)
Variables_func_x = np.append(Variables_func_x, mag_tau_parallel_Tm_func_of_x_soln, axis=0)
Variables_func_x = np.append(Variables_func_x, T_func_of_x_soln, axis=0)
Variables_func_x = np.append(Variables_func_x, F_func_of_x_soln, axis=0)


# ----- Convert numpy arrays to dataframes -----
GenParameters_df = pd.DataFrame(GenParameters)
FullParameters_df = pd.DataFrame(FullParameters)
ExpdParameters_df = pd.DataFrame(ExpdParameters)
Model_Criteria_df = pd.DataFrame(Model_Criteria)
Solution_Parameters_df = pd.DataFrame(Solution_Parameters)
# Psi_Converged_Full_df = pd.DataFrame(Psi_Converged_Full)
# Psi_Converged_Expd_df = pd.DataFrame(Psi_Converged_Expd)
Psi_Dimen_Full_soln_df = pd.DataFrame(Psi_Dimen_Full_soln)
Psi_Dimen_Expd_soln_df = pd.DataFrame(Psi_Dimen_Expd_soln)
v_x_Full_soln_df = pd.DataFrame(v_x_Full_soln)
v_y_Full_soln_df = pd.DataFrame(v_y_Full_soln)
v_x_Expd_soln_df = pd.DataFrame(v_x_Expd_soln)
v_y_Expd_soln_df = pd.DataFrame(v_y_Expd_soln)
dvx_dx_expd_soln_df = pd.DataFrame(dvx_dx_expd_soln)
dvy_dy_expd_soln_df = pd.DataFrame(dvy_dy_expd_soln)
dvy_dx_expd_soln_df = pd.DataFrame(dvy_dx_expd_soln)
dvx_dy_expd_soln_df = pd.DataFrame(dvx_dy_expd_soln)
gamma_xx_soln_df = pd.DataFrame(gamma_xx_soln)
gamma_yy_soln_df = pd.DataFrame(gamma_yy_soln)
gamma_yx_soln_df = pd.DataFrame(gamma_yx_soln)
gamma_xy_soln_df = pd.DataFrame(gamma_xy_soln)
Variables_func_x_df = pd.DataFrame(Variables_func_x)
V_max_arr_df = pd.DataFrame(V_max_arr)
F_EV_arr_df = pd.DataFrame(F_EV_arr)
T_EV_arr_df = pd.DataFrame(T_EV_arr)
err_F_df = pd.DataFrame(err_F)


# ----- Write to Excel -----
with pd.ExcelWriter(filepath_xlsx) as writer:
    GenParameters_df.to_excel(writer, sheet_name='GenParameter', header=False, index=False)
    FullParameters_df.to_excel(writer, sheet_name='FullParameter', header=False, index=False)
    ExpdParameters_df.to_excel(writer, sheet_name='EXPDParameter', header=False, index=False)
    Model_Criteria_df.to_excel(writer, sheet_name='QSolve Parameter', header=False, index=False)
    Solution_Parameters_df.to_excel(writer, sheet_name='Solution', header=False, index=False)
    # Psi_Converged_Full_df.to_excel(writer, sheet_name='Psi_star_full', header=False, index=False)
    # Psi_Converged_Expd_df.to_excel(writer, sheet_name='Psi_star_EXPD', header=False, index=False)
    Psi_Dimen_Full_soln_df.to_excel(writer, sheet_name='Psi_full(um^2_s)', header=False, index=False)
    Psi_Dimen_Expd_soln_df.to_excel(writer, sheet_name='Psi_EXPD(um^2_s)', header=False, index=False)
    v_x_Full_soln_df.to_excel(writer, sheet_name='vx_full(um_s)', header=False, index=False)
    v_y_Full_soln_df.to_excel(writer, sheet_name='vy_full(um_s)', header=False, index=False)
    v_x_Expd_soln_df.to_excel(writer, sheet_name='vx_EXPD(um_s)', header=False, index=False)
    v_y_Expd_soln_df.to_excel(writer, sheet_name='vy_EXPD(um_s)', header=False, index=False)
    dvx_dx_expd_soln_df.to_excel(writer, sheet_name='dvx_dx EXPD (1_s)', header=False, index=False)
    dvy_dy_expd_soln_df.to_excel(writer, sheet_name='dvy_dy EXPD (1_s)', header=False, index=False)
    dvy_dx_expd_soln_df.to_excel(writer, sheet_name='dvy_dx EXPD (1_s)', header=False, index=False)
    dvx_dy_expd_soln_df.to_excel(writer, sheet_name='dvx_dy EXPD (1_s)', header=False, index=False)
    gamma_xx_soln_df.to_excel(writer, sheet_name='gamma_xx (1_s)', header=False, index=False)
    gamma_yy_soln_df.to_excel(writer, sheet_name='gamma_yy (1_s)', header=False, index=False)
    gamma_yx_soln_df.to_excel(writer, sheet_name='gamma_yx (1_s)', header=False, index=False)
    gamma_xy_soln_df.to_excel(writer, sheet_name='gamma_xy (1_s)', header=False, index=False)
    Variables_func_x_df.to_excel(writer, sheet_name='Values along Ellipse RBC', header=False, index=False)
    V_max_arr_df.to_excel(writer, sheet_name='V_max guesses (um_s)', header=False, index=False)
    F_EV_arr_df.to_excel(writer, sheet_name='F(V_max_guess) (nN)', header=False, index=False)
    T_EV_arr_df.to_excel(writer, sheet_name='T(V_max_guess) (nN_um)', header=False, index=False)
    err_F_df.to_excel(writer, sheet_name='error F(V_max_guess)', header=False, index=False)

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
# region   Plotting Setup - Full Grid Boarders
# ==================================================
# ----- Initialize arrays -----
BC2_Full = np.zeros((n_bot + 1, n_bot + 1))
BC3_Full = np.zeros((n_bot + 1, n_bot + 1))
BC5_Full = np.zeros((n_bot + 1, n_bot + 1))
BC6_plot_Full = np.zeros((n_bot + 1, n_bot + 1))
BC6_h_Full = np.zeros((n_bot + 1, n_bot + 1))
BC6_v_Full = np.zeros((n_bot + 1, n_bot + 1))
BC7_Full = np.zeros((n_bot + 1, n_bot + 1))

# ----- Establish borders as binary matrix -----
for j in range(0, n_bot + 1):
    for i in range(0, n_bot + 1):
        # first take care of areas where boundaries meet
        if i == n_top and j == n_top:
            # corner where BC2 and BC3 meet
            BC2_Full[j, i] = 1
            BC3_Full[j, i] = 1
        elif i == 0 and j == n_rbc_y_approx:
            # corner where BC6 and BC7 meet
            BC6_plot_Full[j, i] = 1
            BC7_Full[j, i] = 1
            if BC6_Full[j, i] == 1:
                BC6_h_Full[j, i] = 1
            elif BC6_Full[j, i] == 2:
                BC6_v_Full[j, i] = 1
            else:
                pass
        elif i == n_rbc_x_approx and j == n_bot:
            # corner where BC5 and BC6 meet
            BC5_Full[j, i] = 1
            BC6_plot_Full[j, i] = 1
            if BC6_Full[j, i] == 1:
                BC6_h_Full[j, i] = 1
            elif BC6_Full[j, i] == 2:
                BC6_v_Full[j, i] = 1
            else:
                pass
        # next establish the rest of the boundaries
        elif i == n_top and 0 <= j < n_top:
            # BC2 location only
            BC2_Full[j, i] = 1
        elif n_top < i <= n_bot and j == n_top:
            # BC3 location only
            BC3_Full[j, i] = 1
        elif n_rbc_x_approx < i <= n_bot and j == n_bot:
            # BC5 location only
            BC5_Full[j, i] = 1
        elif BC6_Full[j, i] == 1:
            # BC6 horizontal plane
            BC6_plot_Full[j, i] = 1
            BC6_h_Full[j, i] = 1
        elif BC6_Full[j, i] == 2:
            # BC6 vertical plane
            BC6_plot_Full[j, i] = 1
            BC6_v_Full[j, i] = 1
        elif i == 0 and 0 <= j < n_rbc_y_approx:
            # BC7 location only
            BC7_Full[j, i] = 1
        else:
            pass

# ----- Convert binary matrix to coordinates -----
y_BC2_Full, x_BC2_Full = np.where(BC2_Full == 1)
y_BC3_Full, x_BC3_Full = np.where(BC3_Full == 1)
y_BC5_Full, x_BC5_Full = np.where(BC5_Full == 1)
y_BC6_Full, x_BC6_Full = np.where(BC6_plot_Full == 1)
y_BC6h_Full, x_BC6h_Full = np.where(BC6_h_Full == 1)
y_BC6v_Full, x_BC6v_Full = np.where(BC6_v_Full == 1)
y_BC7_Full, x_BC7_Full = np.where(BC7_Full == 1)

# endregion

# ==================================================
# region  Plotting Setup - Zoom/Expd Grid Boarders
# ==================================================
# ----- Initialize arrays -----
BC5_expd = np.zeros((row_expanded, column_expanded))
BC6_plot_expd = np.zeros((row_expanded, column_expanded))
BC6_h_expd = np.zeros((row_expanded, column_expanded))
BC6_v_expd = np.zeros((row_expanded, column_expanded))
BC7_expd = np.zeros((row_expanded, column_expanded))

# ----- Establish borders as binary matrix -----
for j in range(0, n_row_expanded + 1):
    for i in range(0, n_column_expanded + 1):
        # first take care of areas where boundaries meet
        if i == 0 and j == n_rbc_y_expd_approx:
            # corner where BC6 and BC7 meet
            BC6_plot_expd[j, i] = 1
            BC7_expd[j, i] = 1
            if BC6_Expd[j, i] == 1:
                BC6_h_expd[j, i] = 1
            elif BC6_Expd[j, i] == 2:
                BC6_v_expd[j, i] = 1
            else:
                pass
        elif i == n_rbc_x_expd_approx and j == n_row_expanded:
            # corner where BC5 and BC6 meet
            BC5_expd[j, i] = 1
            BC6_plot_expd[j, i] = 1
            if BC6_Expd[j, i] == 1:
                BC6_h_expd[j, i] = 1
            elif BC6_Expd[j, i] == 2:
                BC6_v_expd[j, i] = 1
            else:
                pass
        # next establish the rest of the boundaries
        elif n_rbc_x_expd_approx < i <= n_column_expanded and j == n_row_expanded:
            # BC5 location only
            BC5_expd[j, i] = 1
        elif BC6_Expd[j, i] == 1:
            # BC6 horizontal plane
            BC6_plot_expd[j, i] = 1
            BC6_h_expd[j, i] = 1
        elif BC6_Expd[j, i] == 2:
            # BC6 vertical plane
            BC6_plot_expd[j, i] = 1
            BC6_v_expd[j, i] = 1
        elif i == 0 and 0 <= j < n_rbc_y_expd_approx:
            # BC7 location only
            BC7_expd[j, i] = 1
        else:
            pass

# ----- Convert binary matrix to coordinates -----
y_BC5_expd, x_BC5_expd = np.where(BC5_expd == 1)
y_BC6_expd, x_BC6_expd = np.where(BC6_plot_expd == 1)
y_BC6h_expd, x_BC6h_expd = np.where(BC6_h_expd == 1)
y_BC6v_expd, x_BC6v_expd = np.where(BC6_v_expd == 1)
y_BC7_expd, x_BC7_expd = np.where(BC7_expd == 1)

# endregion

# ==================================================
# region    Plotting Setup: Node Spacing
# ==================================================
# Full Grid
x1_Full = np.linspace(0, n_bot, (n_bot + 1))
y1_Full = np.linspace(0, n_bot, (n_bot + 1))
x_Full, y_Full = np.meshgrid(x1_Full, y1_Full)

# Zoom/Expd Grid
x1_Expd = np.linspace(0, n_column_expanded, n_column_expanded + 1)
y1_Expd = np.linspace(0, n_row_expanded, n_row_expanded + 1)
x_Expd, y_Expd = np.meshgrid(x1_Expd, y1_Expd)

# endregion

# ==================================================
# region             Plot Velocity
# ==================================================
# ----- Convert Decimal.decimal to floats -----
# Full Grid
v_x_Full_soln_flt = v_x_Full_soln.astype(float)
v_y_neg_Full_soln_flt = v_y_neg_Full_soln.astype(float)
v_mag_Full_soln_flt = v_mag_Full_soln.astype(float)
v_x_unit_Full_soln_flt = v_x_unit_Full_soln.astype(float)
v_y_neg_unit_Full_soln_flt = v_y_neg_unit_Full_soln.astype(float)
# Zoom/Expd Grid
v_x_Expd_soln_flt = v_x_Expd_soln.astype(float)
v_y_neg_Expd_soln_flt = v_y_neg_Expd_soln.astype(float)
v_mag_Expd_soln_flt = v_mag_Expd_soln.astype(float)
v_x_unit_Expd_soln_flt = v_x_unit_Expd_soln.astype(float)
v_y_neg_unit_Expd_soln_flt = v_y_neg_unit_Expd_soln.astype(float)

# ----- Mask data to 'hide' parts outside grid or inside RBC -----
# Full Grid
mask_vx_full_soln = np.ma.masked_array(v_x_Full_soln_flt, mask=mask_full)
mask_vy_neg_full_soln = np.ma.masked_array(v_y_neg_Full_soln_flt, mask=mask_full)
mask_vmag_full_soln = np.ma.masked_array(v_mag_Full_soln_flt, mask=mask_full)
mask_vx_unit_full_soln = np.ma.masked_array(v_x_unit_Full_soln_flt, mask=mask_full)
mask_vy_neg_unit_full_soln = np.ma.masked_array(v_y_neg_unit_Full_soln_flt, mask=mask_full)

# Zoom/Expd Grid
mask_vx_expd_soln = np.ma.masked_array(v_x_Expd_soln_flt, mask=mask_expd)
mask_vy_neg_expd_soln = np.ma.masked_array(v_y_neg_Expd_soln_flt, mask=mask_expd)
mask_vmag_expd_soln = np.ma.masked_array(v_mag_Expd_soln_flt, mask=mask_expd)
mask_vx_unit_expd_soln = np.ma.masked_array(v_x_unit_Expd_soln_flt, mask=mask_expd)
mask_vy_neg_unit_expd_soln = np.ma.masked_array(v_y_neg_unit_Expd_soln_flt, mask=mask_expd)

# ----- Define Color Map -----
# cmap_velocity = plt.cm.YlOrRd
# custom_cmap_velocity = cmap_velocity.copy()
# custom_cmap_velocity.set_under(color='white')

# ----- Quiver Plot with boundaries: Full Grid -----
fig_quiv_Full, ax_quiv_Full = plt.subplots()
ax_quiv_Full.plot(x_BC2_Full, y_BC2_Full, color='black')
ax_quiv_Full.plot(x_BC3_Full, y_BC3_Full, color='black')
ax_quiv_Full.plot(x_BC5_Full, y_BC5_Full, color='black', linestyle='dashed')
ax_quiv_Full.plot(x_BC6_Full, y_BC6_Full, color='darkred')
# ax_quiv_Full.scatter(x_BC6h_Full, y_BC6h_Full, color='coral', marker='_', zorder=10)
# ax_quiv_Full.scatter(x_BC6v_Full, y_BC6v_Full, color='coral', marker='|', zorder=11)
ax_quiv_Full.plot(x_BC7_Full, y_BC7_Full, color='black', linestyle='dashed')
ax_quiv_Full.plot(x_ellipse, y_ellipse, color='red')
ax_quiv_Full.quiver(x_Full, y_Full, mask_vx_full_soln, mask_vy_neg_full_soln, color='blue', alpha=0.65, zorder=20)
ax_quiv_Full.set_xlim(-1, n_bot+2)
ax_quiv_Full.invert_yaxis()
# fig_quiv_Full.savefig(filepath_vector_full_png, dpi=400)

# ----- Quiver Plot with boundaries: Zoom/Expd Grid -----
fig_quiv_Expd, ax_quiv_Expd = plt.subplots()
ax_quiv_Expd.plot(x_BC5_expd, y_BC5_expd, color='black', linestyle='dashed')
ax_quiv_Expd.plot(x_BC6_expd, y_BC6_expd, color='darkred')
# ax_quiv_Expd.scatter(x_BC6h_expd, y_BC6h_expd, color='coral', marker='_', zorder=10)
# ax_quiv_Expd.scatter(x_BC6v_expd, y_BC6v_expd, color='coral', marker='|', zorder=11)
ax_quiv_Expd.plot(x_BC7_expd, y_BC7_expd, color='black', linestyle='dashed')
ax_quiv_Expd.plot(x_ellipse_expd, y_ellipse_expd, color='red')
ax_quiv_Expd.quiver(x_Expd, y_Expd, mask_vx_expd_soln, mask_vy_neg_expd_soln, color='blue', alpha=0.65, zorder=20)
ax_quiv_Expd.set_xlim(-1, n_column_expanded+4)
ax_quiv_Expd.invert_yaxis()
# fig_quiv_Expd.savefig(filepath_vector_expd_png, dpi=400)

# ----- Color Map: Full Grid -----
fig_cmap_Full, ax_cmap_Full = plt.subplots()
map_velocity_Full = ax_cmap_Full.imshow(mask_vmag_full_soln , cmap='turbo', vmin=0, vmax=V_max_Solution)
bar_velocity_Full = plt.colorbar(map_velocity_Full)
ax_cmap_Full.quiver(x_Full, y_Full, mask_vx_unit_full_soln, mask_vy_neg_unit_full_soln, zorder=10)
ax_cmap_Full.plot(x_ellipse, y_ellipse, color='red')
# fig_cmap_Full.savefig(filepath_vel_cmap_full_png, dpi=400)

# ----- Color Map: Zoom/Expd Grid -----
fig_cmap_EXPD, ax_cmap_EXPD = plt.subplots()
map_velocity_EXPD = ax_cmap_EXPD.imshow(mask_vmag_expd_soln, cmap='turbo', vmin=0, vmax=V_max_Solution)
bar_velocity_EXPD = plt.colorbar(map_velocity_EXPD)
ax_cmap_EXPD.quiver(x_Expd, y_Expd, mask_vx_unit_expd_soln, mask_vy_neg_unit_expd_soln, zorder=10)
ax_cmap_EXPD.plot(x_ellipse_expd, y_ellipse_expd, color='red')
# fig_cmap_EXPD.savefig(filepath_vel_cmap_expd_png, dpi=400)

# endregion
