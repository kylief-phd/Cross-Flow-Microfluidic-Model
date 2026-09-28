# ==================================================
# region      Script Description
# ==================================================
# Script is used to test how much "solution" is changing for different convergence criteria for a constant mesh size
# Import two different solutions and code exports the percent change between the solutions as each node
# After confirming that the Psi matrix has been solved with an appropriate convergence criteria
# Import data into Q_Iterative_Solver_V2.py

# endregion

# ==================================================
# region      Import Libraries/Scripts
# ==================================================
# Python Libraries
import numpy as np
import pandas as pd
import math
import sys
from decimal import Decimal, getcontext

# endregion

# ==================================================
# region    Import Data From Psi Convergence
# ==================================================
# ----- User Defined Filepath for Excel Location -----
print('Import two Excel files to examine how "solutions" change with different convergence criteria')
print('Make sure the two Excel files are for the SAME mesh size, L, and precision')

# File Path for Solution using 'looser' convergence criteria
filepath_loose = "C:\\Users\kylie\\OneDrive - University of Oklahoma\\Documents\\Graduate School\\Crossflow Elongational Stresses Model\\CrossFlow Python Excel File Outputs\\13_L=25, 200x200, err=1e-7(NEW), initial V3.xlsx"
# input('Filepath for Excel File containing converged Psi data at bigger (less conservative) convergence criteria:\n '
#                  'Locate Excel file to upload in File Explorer, copy filepath and paste here\n'
#                  'DO NOT forget to include filename as well: ')

# File Path for Solution using 'tighter' convergence criteria
filepath_tight = "C:\\Users\kylie\\OneDrive - University of Oklahoma\\Documents\\Graduate School\\Crossflow Elongational Stresses Model\\CrossFlow Python Excel File Outputs\\18_L=25, 200x200, err=5e-8(NEW), initial V3.xlsx"
# input('Filepath for Excel Workbook containing converged Psi data at smaller (more conservative) convergence criteria:\n '
#                        'Locate Excel file to upload in File Explorer, copy filepath and paste here\n'
#                        'DO NOT forget to include filename as well: ')

# File Path for the Excel sheet holding In_or_Out and BC6 matrix (from either first or second solution)

filepath_other_arr = "C:\\Users\kylie\\OneDrive - University of Oklahoma\\Documents\\Graduate School\\Crossflow Elongational Stresses Model\\CrossFlow Python Excel File Outputs\\18_L=25, 200x200, err=5e-8(NEW), initial V3 other arrays.xlsx"
# File Name for results
filename = 'L=25, 13v18_200x200, err 1e-7 v 5e-8 (NEW, Alt Compar)'
# input('Filename for results output: ')

# ----- Parameters for BOTH Full and EXPD Grids -----
GenParameters_loose = (pd.read_excel(filepath_loose, sheet_name='GenParameter', header=None, dtype=object)).to_numpy()
L_loose = int(GenParameters_loose[1, 2] / 2)  # microns
epsilon_loose = Decimal(GenParameters_loose[7, 2])
precision_loose = GenParameters_loose[10, 2]  # IMPORTANT: index will change depending on conv criteria method used

GenParameters_tight = (pd.read_excel(filepath_tight, sheet_name='GenParameter', header=None, dtype=object)).to_numpy()
L_tight = int(GenParameters_tight[1, 2] / 2)  # microns
epsilon_tight = Decimal(GenParameters_tight[7, 2])
precision_tight = GenParameters_tight[10, 2]  # IMPORTANT: index will change depending on conv criteria method used

# ----- Channel Dimensions Full Grid -----
FullParameters_loose = (pd.read_excel(filepath_loose, sheet_name='FullParameter', header=None, dtype=object)).to_numpy()
n_top_loose = FullParameters_loose[1, 2]
n_bot_loose = FullParameters_loose[2, 2]
h_decimal_loose = Decimal(FullParameters_loose[4, 2])  # microns

FullParameters_tight = (pd.read_excel(filepath_tight, sheet_name='FullParameter', header=None, dtype=object)).to_numpy()
n_top_tight = FullParameters_tight[1, 2]
n_bot_tight = FullParameters_tight[2, 2]
h_decimal_tight = Decimal(FullParameters_tight[4, 2])  # microns

# ----- Confirm imported data has same channel dimensions and mesh size -----
if L_loose == L_tight:
    L = L_loose
else:
    print('Uploaded files have different L')
    sys.exit("Force Stop")

if precision_loose == precision_tight:
    precision = precision_loose
else:
    print('Uploaded files used different precision value')
    sys.exit("Force Stop")

if n_top_loose == n_top_tight and n_bot_loose == n_bot_tight and h_decimal_loose == h_decimal_tight:
    n_top = n_top_loose
    n_bot = n_bot_loose
    h_decimal = h_decimal_loose
else:
    print('Uploaded files have different mesh sizes')
    sys.exit("Force Stop")

if epsilon_loose < epsilon_tight:
    print('Excel files were uploaded in incorrect order, epsilon_loose < epsilon_tight when it should be opposite')
    sys.exit("Force Stop")

# ----- Channel Dimensions Zoom/Expd Grid -----
ExpdParameters_loose = (pd.read_excel(filepath_loose, sheet_name='EXPDParameter', header=None, dtype=object)).to_numpy()
n_column_expanded = ExpdParameters_loose[1, 2]
n_row_expanded = ExpdParameters_loose[2, 2]
h_expd_decimal = Decimal(ExpdParameters_loose[4, 2])  # microns

# ----- Psi "Solutions" -----
# UNITS: dimensionless
# Full Grid as a np array full of strings
Psi_loose = (pd.read_excel(filepath_loose, sheet_name='Psi(Full)', header=None, dtype=object)).to_numpy()
# Convert to decimal.Decimal
for j in range(0, n_bot + 1):
    for i in range(0, n_bot + 1):
        Psi_loose[j, i] = Decimal(Psi_loose[j, i])

Psi_tight = (pd.read_excel(filepath_tight, sheet_name='Psi(Full)', header=None, dtype=object)).to_numpy()
# Convert to decimal.Decimal
for j in range(0, n_bot + 1):
    for i in range(0, n_bot + 1):
        Psi_tight[j, i] = Decimal(Psi_tight[j, i])

# Zoom/Expanded Grid as a np array full of strings
# UNITS: dimensionless
Psi_Expd_loose = (pd.read_excel(filepath_loose, sheet_name='Psi Expd', header=None, dtype=object)).to_numpy()
# Convert to decimal.Decimal
for j in range(0, n_row_expanded + 1):
    for i in range(0, n_column_expanded + 1):
        Psi_Expd_loose[j, i] = Decimal(Psi_Expd_loose[j, i])

Psi_Expd_tight = (pd.read_excel(filepath_tight, sheet_name='Psi Expd', header=None, dtype=object)).to_numpy()
# Convert to decimal.Decimal
for j in range(0, n_row_expanded + 1):
    for i in range(0, n_column_expanded + 1):
        Psi_Expd_tight[j, i] = Decimal(Psi_Expd_tight[j, i])

# ----- Arbitrary Velocity "Solutions" -----
# UNITS: micron/sec
# Full Grid as a np array full of strings
vx_loose = (pd.read_excel(filepath_loose, sheet_name='Arbitrary vx(Full)', header=None, dtype=object)).to_numpy()
# Convert to decimal.Decimal
for j in range(0, n_bot + 1):
    for i in range(0, n_bot + 1):
        vx_loose[j, i] = Decimal(vx_loose[j, i])

vx_tight = (pd.read_excel(filepath_tight, sheet_name='Arbitrary vx(Full)', header=None, dtype=object)).to_numpy()
# Convert to decimal.Decimal
for j in range(0, n_bot + 1):
    for i in range(0, n_bot + 1):
        vx_tight[j, i] = Decimal(vx_tight[j, i])

vy_loose = (pd.read_excel(filepath_loose, sheet_name='Arbitrary vy(Full)', header=None, dtype=object)).to_numpy()
# Convert to decimal.Decimal
for j in range(0, n_bot + 1):
    for i in range(0, n_bot + 1):
        vy_loose[j, i] = Decimal(vy_loose[j, i])

vy_tight = (pd.read_excel(filepath_tight, sheet_name='Arbitrary vy(Full)', header=None, dtype=object)).to_numpy()
# Convert to decimal.Decimal
for j in range(0, n_bot + 1):
    for i in range(0, n_bot + 1):
        vy_tight[j, i] = Decimal(vy_tight[j, i])


# Zoom/Expanded Grid as a np array full of strings
vx_Expd_loose = (pd.read_excel(filepath_loose, sheet_name='Arbitrary vx(Expd)', header=None, dtype=object)).to_numpy()
# Convert to decimal.Decimal
for j in range(0, n_row_expanded + 1):
    for i in range(0, n_column_expanded + 1):
        vx_Expd_loose[j, i] = Decimal(vx_Expd_loose[j, i])

vx_Expd_tight = (pd.read_excel(filepath_tight, sheet_name='Arbitrary vx(Expd)', header=None, dtype=object)).to_numpy()
# Convert to decimal.Decimal
for j in range(0, n_row_expanded + 1):
    for i in range(0, n_column_expanded + 1):
        vx_Expd_tight[j, i] = Decimal(vx_Expd_tight[j, i])

vy_Expd_loose = (pd.read_excel(filepath_loose, sheet_name='Arbitrary vy(Expd)', header=None, dtype=object)).to_numpy()
# Convert to decimal.Decimal
for j in range(0, n_row_expanded + 1):
    for i in range(0, n_column_expanded + 1):
        vy_Expd_loose[j, i] = Decimal(vy_Expd_loose[j, i])

vy_Expd_tight = (pd.read_excel(filepath_tight, sheet_name='Arbitrary vy(Expd)', header=None, dtype=object)).to_numpy()
# Convert to decimal.Decimal
for j in range(0, n_row_expanded+ 1):
    for i in range(0, n_column_expanded + 1):
        vy_Expd_tight[j, i] = Decimal(vy_Expd_tight[j, i])

# ----- In_or_Out & BC6 Arrays -----
# In_or_Out: 1 = outside RBC (inclusive of boarder), 0 = inside RBC
# BC6: 1/2 = on boarder, 0 = not on boarder

# Full Grid
In_or_Out_Full = (pd.read_excel(filepath_other_arr, sheet_name='In_Out(Full)', header=None)).to_numpy()
BC6_Full = (pd.read_excel(filepath_other_arr, sheet_name='BC6(Full)', header=None)).to_numpy()

# Zoom/Expanded Grid
In_or_Out_Expd = (pd.read_excel(filepath_other_arr, sheet_name='In_Out Expd', header=None)).to_numpy()
BC6_Expd = (pd.read_excel(filepath_other_arr, sheet_name='BC6 Expd', header=None)).to_numpy()


# endregion

# ==================================================
# region    Set Decimal Precision
# ==================================================
# Uses same precision from Initial_Psi_Convergence, imported from Excel
getcontext().prec = precision

# endregion

# ==================================================
# region     Compute Diff and Max Absolute Diff
# ==================================================
# Including out of bounds nodes and boarder because it won't contribute (differences will be zero)
# ----- Psi: FULL GRID -----
Psi_diff = np.subtract(Psi_tight, Psi_loose)
Psi_diff_abs = np.abs(Psi_diff)
Psi_Max_Abs_Diff = np.max(Psi_diff_abs)

# ----- Psi: Zoom/Expd Grid -----
Psi_Expd_diff = np.subtract(Psi_Expd_tight, Psi_Expd_loose)
Psi_Expd_diff_abs = np.abs(Psi_Expd_diff)
Psi_Expd_Max_Abs_Diff = np.max(Psi_Expd_diff_abs)

# ----- vx: FULL GRID -----
vx_diff = np.subtract(vx_tight, vx_loose)
vx_diff_abs = np.abs(vx_diff)
vx_Max_Abs_Diff = np.max(vx_diff_abs)

# ----- vx: Zoom/Expd Grid -----
vx_Expd_diff = np.subtract(vx_Expd_tight, vx_Expd_loose)
vx_Expd_diff_abs = np.abs(vx_Expd_diff)
vx_Expd_Max_Abs_Diff = np.max(vx_Expd_diff_abs)

# ----- vy: FULL GRID -----
vy_diff = np.subtract(vy_tight, vy_loose)
vy_diff_abs = np.abs(vy_diff)
vy_Max_Abs_Diff = np.max(vy_diff_abs)

# ----- vy: Zoom/Expd Grid -----
vy_Expd_diff = np.subtract(vy_Expd_tight, vy_Expd_loose)
vy_Expd_diff_abs = np.abs(vy_Expd_diff)
vy_Expd_Max_Abs_Diff = np.max(vy_Expd_diff_abs)

# endregion

# ==================================================
# region     Global Relative Changes
# ==================================================
# ----- Psi -----
GRC_Psi = math.sqrt(np.sum(Psi_diff**2)) / math.sqrt(np.sum(Psi_tight**2))
GRC_Psi_Expd = math.sqrt(np.sum(Psi_Expd_diff**2)) / math.sqrt(np.sum(Psi_Expd_tight**2))

# ----- vx -----
GRC_vx = math.sqrt(np.sum(vx_diff**2)) / math.sqrt(np.sum(vx_tight**2))
GRC_vx_Expd = math.sqrt(np.sum(vx_Expd_diff**2)) / math.sqrt(np.sum(vx_Expd_tight**2))

# ----- vy -----
GRC_vy = math.sqrt(np.sum(vy_diff**2)) / math.sqrt(np.sum(vy_tight**2))
GRC_vy_Expd = math.sqrt(np.sum(vy_Expd_diff**2)) / math.sqrt(np.sum(vy_Expd_tight**2))

# endregion

# ==================================================
# region     Root Mean Square (RMS)
# ==================================================
# ----- Full Grid: Find number of active nodes -----
N_Full = 0
for j in range(0, n_top + 1):
    for i in range(0, n_top + 1):
        # in step 1 area
        N_Full += 1
for j in range(n_top + 1, n_bot + 1):
    for i in range(0, n_bot + 1):
        # need conditional statements in step 2. The raster area includes areas where Psi is always 0
        # For example on RBC boarder and inside RBC, if we check convergence here we divide by 0
        if In_or_Out_Full[j, i] == 0 and BC6_Full[j, i] == 0:
            # This only happens inside the RBC, and we do not need to check convergence
            pass
        else:
            # all other areas should be where we want to check convergence
            N_Full += 1

# ----- Full Grid: Find number of active nodes -----
N_Expd = 0
for j in range(0, n_row_expanded + 1):
    for i in range(0, n_column_expanded + 1):
        # need conditional statements in step 2. The raster area includes areas where Psi is always 0
        # For example on RBC boarder and inside RBC, if we check convergence here we divide by 0
        if In_or_Out_Expd[j, i] == 0 and BC6_Expd[j, i] == 0:
            # This only happens inside the RBC, and we do not need to check convergence
            pass
        else:
            # all other areas should be where we want to check convergence
            N_Expd += 1

# ----- RMS Psi -----
RMS_Psi = math.sqrt(np.sum(Psi_diff**2) / N_Full)
RMS_Psi_Expd = math.sqrt(np.sum(Psi_Expd_diff**2) / N_Expd)

# ----- RMS vx -----
RMS_vx = math.sqrt(np.sum(vx_diff**2) / N_Full)
RMS_vx_Expd = math.sqrt(np.sum(vx_Expd_diff**2) / N_Expd)

# ----- RMS vy -----
RMS_vy = math.sqrt(np.sum(vy_diff**2) / N_Full)
RMS_vy_Expd = math.sqrt(np.sum(vy_Expd_diff**2) / N_Expd)

# endregion

# ==================================================
# region           Excel Output
# ==================================================
# ----- Filename and Path -----
filename_xlsx = filename + '.xlsx'
filepath_xlsx = "C:\\Users\kylie\\OneDrive - University of Oklahoma\\Documents\\Graduate School\\Crossflow Elongational Stresses Model\\CrossFlow Python Excel File Outputs\\" + filename_xlsx

# ----- Important Parameters -----
Parameters_Full = np.array([['Variable Name', 'Variable Description', 'Value'],
                            ['2L', 'Width of Channel', 2*L],
                            ['n_top', 'steps across top (center to wall)', n_top],
                            ['n_bot', 'steps across bottom (center point to exit)', n_bot],
                            ['epsilon_stop_loose', 'less conservative convergence criteria', epsilon_loose],
                            ['epsilon_stop_tight', 'more conservative convergence criteria', epsilon_tight],
                            ['N_Full', 'active nodes including boarders (used in RMS)', N_Full],
                            ['Psi_Max_Abs_Diff', 'Max Absolute Difference between Psi_tight and Psi_loose', Psi_Max_Abs_Diff],
                            ['vx_Max_Abs_Diff', 'Max Absolute Difference between vx_tight and vx_loose', vx_Max_Abs_Diff],
                            ['vy_Max_Abs_Diff', 'Max Absolute Difference between vy_tight and vy_loose', vy_Max_Abs_Diff],
                            ['GRC_Psi', 'Global Relative Change of Psi (% as decimal)', GRC_Psi],
                            ['GRC_vx', 'Global Relative Change of vx (% as decimal)', GRC_vx],
                            ['GRC_vy', 'Global Relative Change of vy (% as decimal)', GRC_vy],
                            ['RMS_Psi', 'Root Mean Square of Psi (avg absolute change per active node)', RMS_Psi],
                            ['RMS_vx', 'Root Mean Square of vx (avg absolute change per active node)', RMS_vx],
                            ['RMS_vy', 'Root Mean Square of vy (avg absolute change per active node)', RMS_vy]])

Parameters_Expd = np.array([['Variable Name', 'Variable Description', 'Value'],
                            ['n_column_expanded', 'steps across length of expanded grid', n_column_expanded],
                            ['n_row_expanded', 'steps across height of expanded grid', n_row_expanded],
                            ['epsilon_stop_loose', 'less conservative convergence criteria', epsilon_loose],
                            ['epsilon_stop_tight', 'more conservative convergence criteria', epsilon_tight],
                            ['N_Expd', 'active nodes including boarders (used in RMS)', N_Expd],
                            ['Psi_Expd_Max_Abs_Diff', 'Max Absolute Difference between Psi_tight and Psi_loose (Expd)', Psi_Expd_Max_Abs_Diff],
                            ['vx_Expd_Max_Abs_Diff', 'Max Absolute Difference between vx_tight and vx_loose (Expd)', vx_Expd_Max_Abs_Diff],
                            ['vy_Expd_Max_Abs_Diff', 'Max Absolute Difference between vy_tight and vy_loose (Expd)', vy_Expd_Max_Abs_Diff],
                            ['GRC_Psi_Expd', 'Global Relative Change of Psi_Expd (% as decimal)', GRC_Psi_Expd],
                            ['GRC_vx_Expd', 'Global Relative Change of vx_Expd (% as decimal)', GRC_vx_Expd],
                            ['GRC_vy_Expd', 'Global Relative Change of vy_Expd (% as decimal)', GRC_vy_Expd],
                            ['RMS_Psi_Expd', 'Root Mean Square of Psi_Expd (avg absolute change per active node)', RMS_Psi_Expd],
                            ['RMS_vx_Expd', 'Root Mean Square of vx_Expd (avg absolute change per active node)', RMS_vx_Expd],
                            ['RMS_vy_Expd', 'Root Mean Square of vy_Expd (avg absolute change per active node)', RMS_vy_Expd]])

# ----- Convert numpy to dataframe -----
Parameters_Full_df = pd.DataFrame(Parameters_Full)
Parameters_Expd_df = pd. DataFrame(Parameters_Expd)
Psi_diff_abs_df = pd.DataFrame(Psi_diff_abs)
vx_diff_abs_df = pd.DataFrame(vx_diff_abs)
vy_diff_abs_df = pd.DataFrame(vy_diff_abs)
Psi_Expd_diff_abs_df = pd.DataFrame(Psi_Expd_diff_abs)
vx_Expd_diff_abs_df = pd.DataFrame(vx_Expd_diff_abs)
vy_Expd_diff_abs_df = pd.DataFrame(vy_Expd_diff_abs)

# ----- Write to Excel -----
with pd.ExcelWriter(filepath_xlsx) as writer:
    Parameters_Full_df.to_excel(writer, sheet_name="Parameter(Full)", header=False, index=False)
    Parameters_Expd_df.to_excel(writer, sheet_name="Parameter(Expd)", header=False, index=False)
    Psi_diff_abs_df.to_excel(writer, sheet_name="Psi Absolute Changes(Full)", header=False, index=False)
    vx_diff_abs_df.to_excel(writer, sheet_name="vx Absolute Changes(Full)", header=False, index=False)
    vy_diff_abs_df.to_excel(writer, sheet_name="vy Absolute Changes(Full)", header=False, index=False)
    Psi_Expd_diff_abs_df.to_excel(writer, sheet_name="Psi Absolute Changes(Expd)", header=False, index=False)
    vx_Expd_diff_abs_df.to_excel(writer, sheet_name="vx Absolute Changes(Expd)", header=False, index=False)
    vy_Expd_diff_abs_df.to_excel(writer, sheet_name="vy Absolute Changes(Expd)", header=False, index=False)

# endregion
