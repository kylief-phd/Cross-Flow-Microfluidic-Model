# Cross-Flow-Microfluidic-Model
Python code for modeling the flow field in a cross-flow microchannel. The model assumes a prolate ellipsoid deformed RBC is trapped in the stagnation point. The code iteratively solves for the flowrate needed to pinch an EV off.


Script Order and description
	Initial_Psi_Convergence_Alt_Converg.py
	Description:  This script solves the flow field in ¼ of the cross-flow channel by using finite differences to solve the vorticity equation of the stream function. Once the Psi matrix has been solved, the script then uses an arbitrary max velocity to solve the velocity field.  This arbitrary velocity field is used to aid in visualization of the flow field and visually identify any obvious issues with convergence (ie convergence criteria was too large so the true solution has not been found even though convergence criteria has been met).
	User Input Needed: 
	Precision value: number of decimals to keep in calculations in Decimal.decimal values. Decimal.decimal helps eliminate floating point errors due to using a binary system that cannot precisely represent all decimal numbers. 
	filename
	filepath_initial: filepath for initial guess to use (if not starting from ground 0, ie importing previous solution as a starting point for a tighter convergence criteria)
	L: Half channel width in micron
	n_top: number of steps across half channel width (L)
	epsilon_stop: convergence criteria for global relative error as a decimal (ie 1% error input as 0.01)
	iter_max: max number of iterations to perform (failsafe to prevent infinite loop)
	Instructions:
	Run script with a starting convergence criteria to obtain solution. After solution has been found, run the script again with a tighter convergence (i.e. decrease by 1 order of magnitude).  Compare the two solutions using the next script (Psi_Convergence_Checker_Alt_Converg.py)
	.Repeat until an adequate convergence criteria is used (ie solution no longer significantly changes)
	This script calls the following scripts to operate.
	Psi_Solver_Function_Alt_Converg.py
	Description: This sets up a function that iteratively solves the matrix of dimensionless Psi values.  The initial guess matrix is set up in the main portion of Initial_Psi_Convergence_Alt_Converg.py
	Inputs: n_top, n_bot, n_rbc_x_approx, n_rbc_y_approx, epsilon_stop, iter_max, Psi_initial, In_or_Out, BC6_location, precision
	Outputs: Psi, err, k
	Psi_Solver_Function.py calls the following additional scripts:
	BC_Functions_Stair_Step_RBC.py
	Description: Sets up multiple functions defining ghost points along each boundary condition.
	Psi_Finite_Diff_Func_Stair_Step.py
	Description: Sets up a function for the centered, centered finite difference equation to be used with the dimensionless Psi values to solve for a given node using the surrounding nodes.
	Velocity_Solver_Function.py
	Description: This uses the converged Psi and an arbitrary max velocity to solve for the velocity components and overall magnitude at each node. It first dimensionalizes Psi and then computes the velocity components at each node.
	Inputs: n_top, n_bot, n_rbc_x_approx, n_rbc_y_approx, V_max, L, h_decimal, Psi_Converged, In_or_Out, BC6_location, precision
	Outputs: Psi_dimensions, v_x, v_y, v_y_neg, v_mag, v_x_unit, v_y_neg_unit
	Velocity_Solver_Function.py calls the following additional scripts:
	Velocity_Functions_Stair_Step.py
	Description: Sets up multiple functions for the different finite difference equations needed to convert Psi into velocity components.
	Zoom_and_resolve_Psi_Function_Alt_Converg.py
	Description: After converging the main Psi matrix, the model is designed to zoom into the stagnation point (ie where the RBC is) then add in new nodes between all the original ones, effectively creating more resolution along the RBC surface. This function solves the Psi matrix on this zoomed in portion since intermediate nodes are added.
	Inputs: row_delete, column_delete, row_expanded, column_expanded, n_row_expanded, n_column_expanded, n_rbc_x_expd_approx, epsilon_stop, iter_max, Psi_Converged, In_or_Out_zoom_expd_fill, BC6_zoom_expd_fill, precision
	Outputs: Psi_zoom_expd_original, Psi_initial_zoom_expd, Psi_zoom_expd_fill, err_zoom_expd, k
	Zoom_and_resolve_Psi_Function_Alt_Converg.py calls the following additional scripts:
	BC_Functions_Stair_Step_RBC.py
	Psi_Finite_Diff_Func_Stair_Step.py
	Zoom_and_resolve_Vel_Function.py
	Description: After solving the Psi matrix on the zoomed in portion, the same arbitrary max velocity is used to solve for the velocity components and overall magnitude at each node.
	Inputs: row_expanded, column_expanded, n_row_expanded, n_column_expanded, n_rbc_x_expd_approx, n_rbc_y_expd_approx, V_max, L, h_expd_decimal, Psi_Expd_converged, In_or_Out_zoom_expd_fill, BC6_zoom_expd_fill, precision
	Outputs: Psi_dimensions_zoom_expd, vx_zoom_expd, vy_zoom_expd, v_y_neg_expd, v_mag_zoom_expd, v_x_unit_expd, v_y_neg_unit_expd
	Zoom_and_resolve_Vel_Function.py calls the following additional scripts:
	Velocity_Functions_Stair_Step.py
	Convergence criteria used in Initial_Psi_Convergence_Alt_Converg.py
	The Global Relative Error (GRE) which is essentially the overall change between iterations (global absolute error) compared to the overall magnitude of the current approximation. In other words, the overall or average relative change.
GRE=‖Ψ^(k+1)-Ψ^k ‖_F/(‖Ψ^(k+1) ‖_F  )=‖ΔΨ‖_F/‖Ψ^(k+1) ‖_F 
where
Ψ^(k+1)=Current Approximtion
Ψ^k=Previous Approximation
‖┤‖_F=Frobenius norm of a matrix (a way to measure the matrix size)
* The Frobenius norm is the equivalent of the Euclidean norm of a vector (ie finding the magnitude of a vector)
	The GRE alone is not enough, as it is essentially an average. So there can be large/moderate changes and small changes leading to an acceptable GRE.  (ie it does not guarantee every node has converged). SO we will also use…
	The Max Absolute Error (MAE), which ensures that all nodes that are slower to converge have adequately small changes. 
MAE=‖Ψ^(k+1)-Ψ^k ‖_max=‖ΔΨ‖_max=max⁡(ΔΨ_(i,j))
	The convergence criteria for MAE is two orders of magnitude smaller than GRE. As stated before, GRE can converge for a matrix containing small, moderate, and large changes depending on the size of the current matrix (denominator).  So a smaller MAE ensures that the slowest converging node has adequately small changes
	Both GRE and MAE must be satisfied to converge.
	In addition to using GRE and MAE as the convergence criteria.  The  Root Mean Square (RMS) of the matrix will also be monitored as a double check. 
RMS=√(1/N ∑_i▒∑_j▒(Ψ^(k+1)-Ψ^k )^2 )=√(1/N ∑_i▒∑_j▒(ΔΨ_(i,j) )^2 )
	RMS is essentially a measure of the average absolute change between iterations per node.
	Psi_Convergence_Checker_Alt_Converg.py
	Description: Takes two converged Psi matrices and velocity fields (based on arbitrary max velocity) and compares them to test if the convergence criteria used was good enough to find the true solution.  This script performs this on both the full matrix and the zoomed matrix.
	IMPORTANT: The two imported matrices should be identical EXCEPT they should have two different convergence criteria. 
	User Input Needed: 
	filepath_loose: filepath to the excel file containing the converged Psi matrix using the looser (ie larger) convergence criteria.
	filepath_tight: filepath to the excel file containing the converged Psi matrix using the tighter (ie smaller) convergence criteria.
	filepath_other_arr: filepath to the excel file containing information like BC6 or In_or_Out which is needed to establish boundary locations along RBC.  Can use the other_arr excel sheet for either solution, the convergence criteria will not affect these. 
	filename: filename for excel sheet holding the results of the comparison for the two matrices
	INSTRUCTIONS: 
	Import the two solutions being compared by designating the correct filepaths.  Make sure that the solution using the looser criteria is imported into the filepath_loose and the tigher criteria into filepath_tight. 
	The code will compare the solutions with the following parameters:
	Matrix of differences between solutions at each node.
	Computes the max absolute difference between solutions (MAE)
	Computes global relative change
	Computes the root mean square (RMS)
	These parameters can be used to compare the matrices, but the official way to check convergence is to use the two matrices to compute the flow rate (Q) needed to produce varying sizes of EVs using the next script. Once the Q solutions stop changing, then an appropriate convergence criteria has been found.
	Q_Iterative_Solver_V2.py
	Description: This script uses the solved dimensionless Psi matrix and iteratively solves for the flow rate needed to cause the pinching off of an EV of a user defined size.  
	IMPORTANT: V2 of this script performs volume calculations and all integrations on the prolate ellipse geometry NOT on the stair step geometry.  V1 of the script only uses the stair step geometry
	User Input Needed:
	filepath_Psi: filepath to the excel file containing the converged Psi matrix for the desired mesh size and channel dimensions.
	filepath_other_arr: filepath to the excel file containing information like BC6 or In_or_Out which is needed to establish boundary locations along RBC.  
	filename_soln: filename for excel sheet holding the solution (ie flow rate, stresses, final velocity profiles, etc)
	viscosity_Pa_s: fluid viscosity in Pa-s (based on Korean group PVP viscosities)
	epsilon_stop: the convergence criteria
	iter_max: max number of iterations to perform to solve for Q (failsafe to prevent infinite loop)
	d_EV_nm: desired EV diameter in nanometers
	RBC_Mem_T: Critical membrane breaking tension (value from Jay is ~50 dyn/cm = 50 nN/um)
	Q_uL_min_guess: The initial guess for the volumetric flow rate, must be larger than the actual solution. Code will force stop and request a new value if this condition is not met.
	INSTRUCTIONS:
	Define the user inputs above.  Designate the file path of the Psi solution and the matrices governing the RBC boundary (other_arr). Define the filename for the excel file containing the solution. Define desired EV size (diameter, viscosity, and membrane breaking tension). Set the convergence criteria and the max number of iterations. 
	From this input the code does the following:
	Maps nodes on the stair step geometry to the elliptic geometry.
	Identifies EV pinch location (x and node) based on the EV volume given the user defined EV diameter.  Also checks if there is more than one integration step from EV pinch location to nose for the given grid size. 
	Rounds EV pinch location to an actual node (ie not a fractional node) and determines the EV volume and diameter given the rounded position
	Calculates the required force to break EV at that point.
	Begins iteratively solving using the bisection method given a user defined initial Q guess. Technically there are two initial guesses, 0 and the value defined by the user. If user defined value is too low (ie produces a force smaller than required) the code will force stop and request a larger value. It must be larger than the solution for the bisection method to work.
	For each iteration the code solves for the velocity flow field, computes velocity gradients and shear rates (including the normal stresses), and integrates along the RBC membrane to determine Tension and Force caused by the flow at the EV pinch location. 
	Once the solution had been found, the true velocity profiles, shear, tension, and force along the RBC are reported. It will also determine the stagnation pressure, the tension and force from the pressure at the break point. 
	This script calls the following scripts to operate.
	Zoom_and_resolve_Vel_Function.py
	Description: Used to find the velocity components in the flow field for each new Q guess during the iterative solver. Also used to find the actual flow field solution once converged. 
	Inputs: row_expanded, column_expanded, n_row_expanded, n_column_expanded, n_rbc_x_expd_approx, n_rbc_y_expd_approx, V_max, L, h_expd_decimal, Psi_Expd_converged, In_or_Out_zoom_expd_fill, BC6_zoom_expd_fill, precision
	Outputs: Psi_dimensions_zoom_expd, vx_zoom_expd, vy_zoom_expd, v_y_neg_expd, v_mag_zoom_expd, v_x_unit_expd, v_y_neg_unit_expd
	This script calls the following additional scripts:
	Velocity_Functions_Stair_Step.py
	Zoomed_Shear_Rate_Solver_Function.py
	Description: Takes the intermediate or final velocity profiles, calculates velocity gradients and shear rates along the stair step RBC membrane.  It then maps these values back to the prolate ellipse geometry to be used during integration steps. 
	Inputs: row_expanded, column_expanded, n_row_expanded, n_column_expanded, n_rbc_x_expd_approx, n_rbc_y_expd_approx, h_exp_decimal, vx_zoom_expd, vy_zoom_expd, In_or_Out_zoom_expd_fill, BC6_zoom_expd_fill, precision, a, b
	Outputs: dvx_dx_expd, dvy_dy_expd, dvy_dx_expd, dvx_dy_expd, gamma_xx, gamma_yy, gamma_yx, gamma_xy, gamma_xx_func_of_x, gamma_yy_func_of_x, gamma_yx_func_of_x, gamma_xy_func_of_x
	This script calls the following additional scripts:
	Shear_Rate_Func_Stair_Step.py
	Description: Sets up multiple functions for the different finite difference equations needed to calculate the needed velocity gradients. 
	T_and_F_Solver_Function_V2.py
	Description: This takes the shear rates, converts them to stresses, finds the contribution tangent to membrane (ie parallel to membrane tension) and integrates to find the tension and force applied to the RBC membrane due to fluid flow.  A second function is also defined that is used after the solution for Q has been found. This second function finds the Tension and Force along the membrane as a function of x. 
	Inputs: n_elliptic_RBC, n_EV_Break_Ellipse, viscosity, a, b, gamma_xx_func_of_x, gamma_yy_func_of_x, gamma_yx_func_of_x, gamma_xy_func_of_x, RBC_Node_to_Ellip_x, precision
	Outputs: tau_xx_func_of_x, tau_yy_func_of_x, tau_yx_func_of_x, tau_xy_func_of_x, T_func_of_x, F_func_of_x, m_tan_func_of_x, mag_tau_parallel_Tm_func_of_x, r_func_of_x, Tension_Half_RBC, Force_Half_RBC
	Velocity_Solver_Function.py
	Description: Calculates the velocity profile for the full grid once the solution for Q has been found. 
	Inputs: n_top, n_bot, n_rbc_x_approx, n_rbc_y_approx, V_max, L, h_decimal, Psi_Converged, In_or_Out, BC6_location, precision
	Outputs: Psi_dimensions, v_x, v_y, v_y_neg, v_mag, v_x_unit, v_y_neg_unit
Using solution for Q to check Psi convergence
	Use the two solved Psi matrices for the two convergence criterias (for Psi solution) being compared and run the Q_Iterative_Solver_V2.py script
	Find Q needed to produce a variety of EV diametners for both Psi matrices
	Compare the solutions (Q values) by calculculting the percent change between the two Psi matrices.
	Continue tighteining convergence criteria for Psi solution until Q have minimally acceptable percent changes
	Example for a 200x200 node grid using 
	Viscosity = 0.06 Pa-s
	Membrane Tension = 50 dyn/cm
Q convergence criteria = 0.001 (i.e. 0.1%)
