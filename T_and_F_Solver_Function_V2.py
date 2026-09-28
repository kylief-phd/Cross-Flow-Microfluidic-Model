# ==================================================
# region     Import Libraries/Scripts
# ==================================================
# Python Libraries
import numpy as np
import math
from decimal import Decimal, getcontext

# endregion

# ==================================================
# region     Description
# ==================================================
# IMPORTANT: All imported shear rates have NOT been multiplied by -1 or viscosity yet
# V1 of this script is for tension and force integration along the approximated RBC
# V2 of this script if for tension and force integration along the elliptic RBC
# viscosity imported as nN s/micron^2

# ==================================================
# region    Tension and Force Function: Sums
# ==================================================


def T_and_F(n_elliptic_RBC, n_EV_Break_Ellipse, viscosity, a, b,
            gamma_xx_func_of_x, gamma_yy_func_of_x, gamma_yx_func_of_x, gamma_xy_func_of_x,
            RBC_Node_to_Ellip_x, precision):
    # Function OUTPUTS: Tension, Force
    # ----- Set Decimal Precision -----
    getcontext().prec = precision

    # ----- Initialize Variables and Arrays -----
    # Arrays to hold results from each integration step
    # each entry corresponds to RBC_Node_to_Ellip_x which lists x positions (micron) along elliptic RBC
    T_func_of_x = np.full((1, n_elliptic_RBC + 1), Decimal('0'), dtype=object)
    F_func_of_x = np.full((1, n_elliptic_RBC + 1), Decimal('0'), dtype=object)
    m_tan_func_of_x = np.full((1, n_elliptic_RBC + 1), Decimal('0'), dtype=object)
    mag_tau_parallel_Tm_func_of_x = np.full((1, n_elliptic_RBC + 1), Decimal('0'), dtype=object)
    r_func_of_x = np.full((1, n_elliptic_RBC + 1), Decimal('0'), dtype=object)

    # ----- Convert shear rates to a single row with just shear rates -----
    # delete row 0 which are actual x positions which are also held in RBC_Node_to_Ellip_x
    gamma_xx_func_of_x = np.delete(gamma_xx_func_of_x, 0, axis=0)
    gamma_yy_func_of_x = np.delete(gamma_yy_func_of_x, 0, axis=0)
    gamma_yx_func_of_x = np.delete(gamma_yx_func_of_x, 0, axis=0)
    gamma_xy_func_of_x = np.delete(gamma_xy_func_of_x, 0, axis=0)

    # ----- Calculate tau_xx, tau_yy, tau_yx, and tau_xy -----
    # tau = -viscosity * shear rate
    # UNITS: nN/micron^2
    # each entry corresponds to RBC_Node_to_Ellip_x which lists x positions (micron) along elliptic RBC
    tau_xx_func_of_x = -viscosity * gamma_xx_func_of_x
    tau_yy_func_of_x = -viscosity * gamma_yy_func_of_x
    tau_yx_func_of_x = -viscosity * gamma_yx_func_of_x
    tau_xy_func_of_x = -viscosity * gamma_xy_func_of_x

    # ----- Calculate slope of tangent to ellipse at each defined x -----
    for i in range(0, n_elliptic_RBC + 1):
        # x in micron given by RBC_Node_to_Ellip_x[i]
        x = RBC_Node_to_Ellip_x[0, i]
        m_tan_func_of_x[0, i] = ((b / a**2) * x) / ((1 - (x**2 / a**2))**(Decimal(0.5)))

    # ----- Calculate magnitude of tau_parallel to Tm -----
    for i in range(0, n_elliptic_RBC + 1):
        slope = m_tan_func_of_x[0, i]
        tau_yx = tau_yx_func_of_x[0, i]
        tau_yy = tau_yy_func_of_x[0, i]
        tau_xx = tau_xx_func_of_x[0, i]
        tau_xy = tau_xy_func_of_x[0, i]
        mag_tau_parallel_Tm_func_of_x[0, i] = (tau_yx + (slope * (tau_yy - tau_xx - (slope * tau_xy)))) / (1 + slope**2)

    # ----- Calculate r for each x position -----
    for i in range(0, n_elliptic_RBC + 1):
        if i < n_elliptic_RBC:
            x = RBC_Node_to_Ellip_x[0, i]
            r_func_of_x[0, i] = ((1 - (x**2/a**2)) * b**2)**Decimal(0.5)
        elif i == n_elliptic_RBC:
            # having to do this so that the correct last node is used
            # instead of the altered one for undefined slope issue
            x = a
            r_func_of_x[0, i] = ((1 - (x ** 2 / a ** 2)) * b ** 2) ** Decimal(0.5)

    # ----- Calculate Tension and Force -----
    # UNITS: T in nN/micron; F in nN
    # Start raster at break point (breakpoint node on ellipse is n_EV_Break_Ellipse)
    for i in range(n_EV_Break_Ellipse, n_elliptic_RBC + 1):
        if i < n_elliptic_RBC:
            x1 = RBC_Node_to_Ellip_x[0, i]
            x2 = RBC_Node_to_Ellip_x[0, i + 1]
            tau_x1 = mag_tau_parallel_Tm_func_of_x[0, i]
            tau_x2 = mag_tau_parallel_Tm_func_of_x[0, i + 1]
            slope_x1 = m_tan_func_of_x[0, i]
            slope_x2 = m_tan_func_of_x[0, i + 1]
            r_x1 = r_func_of_x[0, i]
            r_x2 = r_func_of_x[0, i + 1]

            # Result of integration step held at x1 position of step
            T_func_of_x[0, i] = ((x2 - x1) / 2) * ((tau_x1 * ((1 + slope_x1**2)**Decimal(0.5))) +
                                                   (tau_x2 * ((1 + slope_x2**2)**Decimal(0.5))))
            F_func_of_x[0, i] = (x2 - x1) * Decimal(math.pi) * ((tau_x1 * r_x1 * ((1 + slope_x1**2)**Decimal(0.5))) +
                                                                (tau_x2 * r_x2 * ((1 + slope_x2**2)**Decimal(0.5))))
        elif i == n_elliptic_RBC:
            # at last node, no integration needed
            pass

    # ----- Sum up array holding values from each integration step -----
    # UNITS: T in nN/micron; F in nN
    Tension = np.sum(T_func_of_x)
    Force = np.sum(F_func_of_x)

    # ----- Function Outputs -----
    return Tension, Force

# endregion

# ==================================================
# region    Tension and Force Function: func of x
# ==================================================
# this function is used after Q solution has been found
# Designed to calcualte T an F values at every intergration step along RBC and store in array

def T_and_F_func_x(n_elliptic_RBC, viscosity, a, b,
                   gamma_xx_func_of_x, gamma_yy_func_of_x, gamma_yx_func_of_x, gamma_xy_func_of_x,
                   RBC_Node_to_Ellip_x, precision):
    # Function OUTPUTS: (tau_xx_func_of_x, tau_yy_func_of_x, tau_yx_func_of_x, tau_xy_func_of_x, T_func_of_x, F_func_of_x,
    #             m_tan_func_of_x, mag_tau_parallel_Tm_func_of_x, r_func_of_x, T_func_of_x, F_func_of_x,
    #             Tension_Half_RBC, Force_Half_RBC)
    # ----- Set Decimal Precision -----
    getcontext().prec = precision

    # ----- Initialize Variables and Arrays -----
    # Arrays to hold results from each integration step
    # each entry corresponds to RBC_Node_to_Ellip_x which lists x positions (micron) along elliptic RBC
    T_func_of_x = np.full((1, n_elliptic_RBC + 1), Decimal('0'), dtype=object)
    F_func_of_x = np.full((1, n_elliptic_RBC + 1), Decimal('0'), dtype=object)
    m_tan_func_of_x = np.full((1, n_elliptic_RBC + 1), Decimal('0'), dtype=object)
    mag_tau_parallel_Tm_func_of_x = np.full((1, n_elliptic_RBC + 1), Decimal('0'), dtype=object)
    r_func_of_x = np.full((1, n_elliptic_RBC + 1), Decimal('0'), dtype=object)

    # ----- Convert shear rates to a single row with just shear rates -----
    # delete row 0 which are actual x positions which are also held in RBC_Node_to_Ellip_x
    gamma_xx_func_of_x = np.delete(gamma_xx_func_of_x, 0, axis=0)
    gamma_yy_func_of_x = np.delete(gamma_yy_func_of_x, 0, axis=0)
    gamma_yx_func_of_x = np.delete(gamma_yx_func_of_x, 0, axis=0)
    gamma_xy_func_of_x = np.delete(gamma_xy_func_of_x, 0, axis=0)

    # ----- Calculate tau_xx, tau_yy, tau_yx, and tau_xy -----
    # tau = -viscosity * shear rate
    # UNITS: nN/micron^2
    # each entry corresponds to RBC_Node_to_Ellip_x which lists x positions (micron) along elliptic RBC
    tau_xx_func_of_x = -viscosity * gamma_xx_func_of_x
    tau_yy_func_of_x = -viscosity * gamma_yy_func_of_x
    tau_yx_func_of_x = -viscosity * gamma_yx_func_of_x
    tau_xy_func_of_x = -viscosity * gamma_xy_func_of_x

    # ----- Calculate slope of tangent to ellipse at each defined x -----
    for i in range(0, n_elliptic_RBC + 1):
        # x in micron given by RBC_Node_to_Ellip_x[i]
        x = RBC_Node_to_Ellip_x[0, i]
        m_tan_func_of_x[0, i] = ((b / a**2) * x) / ((1 - (x**2 / a**2))**(Decimal(0.5)))

    # ----- Calculate magnitude of tau_parallel to Tm -----
    for i in range(0, n_elliptic_RBC + 1):
        slope = m_tan_func_of_x[0, i]
        tau_yx = tau_yx_func_of_x[0, i]
        tau_yy = tau_yy_func_of_x[0, i]
        tau_xx = tau_xx_func_of_x[0, i]
        tau_xy = tau_xy_func_of_x[0, i]
        mag_tau_parallel_Tm_func_of_x[0, i] = (tau_yx + (slope * (tau_yy - tau_xx - (slope * tau_xy)))) / (1 + slope**2)

    # ----- Calculate r for each x position -----
    for i in range(0, n_elliptic_RBC + 1):
        if i < n_elliptic_RBC:
            x = RBC_Node_to_Ellip_x[0, i]
            r_func_of_x[0, i] = ((1 - (x**2/a**2)) * b**2)**Decimal(0.5)
        elif i == n_elliptic_RBC:
            # having to do this so that the correct last node is used
            # instead of the altered one for undefined slope issue
            x = a
            r_func_of_x[0, i] = ((1 - (x ** 2 / a ** 2)) * b ** 2) ** Decimal(0.5)

    # ----- Calculate Tension and Force -----
    # UNITS: T in nN/micron; F in nN
    # Start raster at break point (breakpoint node on ellipse is n_EV_Break_Ellipse)
    for i in range(0, n_elliptic_RBC + 1):
        if i < n_elliptic_RBC:
            x1 = RBC_Node_to_Ellip_x[0, i]
            x2 = RBC_Node_to_Ellip_x[0, i + 1]
            tau_x1 = mag_tau_parallel_Tm_func_of_x[0, i]
            tau_x2 = mag_tau_parallel_Tm_func_of_x[0, i + 1]
            slope_x1 = m_tan_func_of_x[0, i]
            slope_x2 = m_tan_func_of_x[0, i + 1]
            r_x1 = r_func_of_x[0, i]
            r_x2 = r_func_of_x[0, i + 1]

            # Result of integration step held at x1 position of step
            T_func_of_x[0, i] = ((x2 - x1) / 2) * ((tau_x1 * ((1 + slope_x1**2)**Decimal(0.5))) +
                                                   (tau_x2 * ((1 + slope_x2**2)**Decimal(0.5))))
            F_func_of_x[0, i] = (x2 - x1) * Decimal(math.pi) * ((tau_x1 * r_x1 * ((1 + slope_x1**2)**Decimal(0.5))) +
                                                                (tau_x2 * r_x2 * ((1 + slope_x2**2)**Decimal(0.5))))
        elif i == n_elliptic_RBC:
            # at last node, no integration needed
            pass

    # ----- Sum up array holding values from each integration step -----
    # UNITS: T in nN/micron; F in nN
    Tension_Half_RBC = np.sum(T_func_of_x)
    Force_Half_RBC = np.sum(F_func_of_x)

    # ----- Function Outputs -----
    return (tau_xx_func_of_x, tau_yy_func_of_x, tau_yx_func_of_x, tau_xy_func_of_x, T_func_of_x, F_func_of_x,
            m_tan_func_of_x, mag_tau_parallel_Tm_func_of_x, r_func_of_x, Tension_Half_RBC, Force_Half_RBC)

# endregion