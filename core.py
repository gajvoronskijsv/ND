import numpy as np
import math
from tqdm import tqdm

deltaSmall = 1e-8


# Flux through the CEM side, time > 0
def f_J_c(J_c, cH_D_new, cNa_D_new, DBL2, d31, DH_S_DBL2, k1_CEM, cNa_A_new, cH_A_new, k2_CEM, DNa_CEM):
    P = cH_D_new + cNa_D_new - J_c * DBL2 / d31
    up_CHs = 2 * d31 * (cH_D_new + cNa_D_new) * (DH_S_DBL2 * cH_D_new + J_c * DBL2) - J_c * DBL2 * J_c * DBL2
    down_CHs = 2 * DH_S_DBL2 * (d31 * (cH_D_new + cNa_D_new) - J_c * DBL2)
    CH_DBL2 = up_CHs / down_CHs
    CNa_DBL2 = P - CH_DBL2
    J_c_up = k1_CEM * (1 / (1 + (cNa_A_new / cH_A_new)) - 1 / (1 + (CNa_DBL2 / CH_DBL2)))
    J_c_down = k2_CEM * (1 / (1 + (cNa_A_new / cH_A_new)) + 1 / (1 + (CNa_DBL2 / CH_DBL2))) + 2 * DNa_CEM
    f_J_c = J_c_up / J_c_down - J_c
    return f_J_c


# derivative of f(J_c)
def df_J_c(J_c, cH_D_new, cNa_D_new, DBL2, d31, DH_S_DBL2, k1_CEM, cNa_A_new, cH_A_new, k2_CEM, DNa_CEM):
    return (f_J_c(J_c + deltaSmall, cH_D_new, cNa_D_new, DBL2, d31, DH_S_DBL2, k1_CEM, cNa_A_new, cH_A_new, k2_CEM,
                  DNa_CEM) - f_J_c(J_c - deltaSmall, cH_D_new, cNa_D_new, DBL2, d31, DH_S_DBL2, k1_CEM, cNa_A_new,
                                   cH_A_new, k2_CEM, DNa_CEM)) / (2 * deltaSmall)


# Flux through the AEM side, time > 0
def f_J_a(J_a, cOH_D_new, cCl_D_new, DBL3, d42, DOH_S_DBL3, k1_AEM, cCl_B_new, cOH_B_new, k2_AEM, DCl_AEM):
    Z = cOH_D_new + cCl_D_new - J_a * DBL3 / d42
    up_COHs = 2 * d42 * (cOH_D_new + cCl_D_new) * (DOH_S_DBL3 * cOH_D_new + J_a * DBL3) - J_a * DBL3 * J_a * DBL3
    down_COHs = 2 * DOH_S_DBL3 * (d42 * (cOH_D_new + cCl_D_new) - J_a * DBL3)
    COH_DBL3 = up_COHs / down_COHs
    CCl_DBL3 = Z - COH_DBL3
    J_a_up = k1_AEM * (1 / (1 + (cCl_B_new / cOH_B_new)) - 1 / (1 + (CCl_DBL3 / COH_DBL3)))
    J_a_down = k2_AEM * (1 / (1 + (cCl_B_new / cOH_B_new)) + 1 / (1 + (CCl_DBL3 / COH_DBL3))) + 2 * DCl_AEM
    f_J_a = J_a_up / J_a_down - J_a
    return f_J_a


# derivative of f(J_a)
def df_J_a(J_a, cOH_D_new, cCl_D_new, DBL3, d42, DOH_S_DBL3, k1_AEM, cCl_B_new, cOH_B_new, k2_AEM, DCl_AEM):
    df_J_a = (f_J_a(J_a + deltaSmall, cOH_D_new, cCl_D_new, DBL3, d42, DOH_S_DBL3, k1_AEM, cCl_B_new, cOH_B_new, k2_AEM,
                    DCl_AEM) - f_J_a(J_a - deltaSmall, cOH_D_new, cCl_D_new, DBL3, d42, DOH_S_DBL3, k1_AEM, cCl_B_new,
                                     cOH_B_new, k2_AEM, DCl_AEM)) / (2 * deltaSmall)
    return df_J_a


Tmax = 11800  # Duration of experiment, seconds(3 hours = 10800 sec)
dt = 20
M = int(np.floor(Tmax / dt))  # Number of time steps

def NDModel(cNaCl, cHCl, cNaOH, DBLs, res, output):
    # model paramters
    DH_CEM = 2.7e-6  # Difusion coefficient of H + into a CEM, cm2 / s
    DNa_CEM = 3.6e-7  # Difusion coefficient of Na + into a CEM, cm2 / s
    DOH_AEM = 9.6e-7  # Difusion coefficient of OH + into an AEM, cm2 / s
    DCl_AEM = 3.0e-7  # Difusion coefficient of Cl - into an AEM, cm2 / s
    # DBLs = 80
    DBL2_mkm = DBLs  # DBL thickness(from the right - hand side of CEM), microns
    DBL3_mkm = DBLs  # DBL thickness(from the left - hand side of AEM), microns
    DBL2 = DBL2_mkm / 10000  # DBL thickness(from the right - hand side of CEM), cm
    DBL3 = DBL3_mkm / 10000  # DBL thickness(from the left - hand side of AEM), cm
    # Parameters for the numerical solution
    dAEM_mkm = 140  # AEM thickness, microns
    dCEM_mkm = 170  # CEM thickness, microns
    dAEM = dAEM_mkm / 10000  # AEM thickness, cm
    dCEM = dCEM_mkm / 10000  # CEM thickness, cm
    # Diffusion coefficients of ions in solution at infinite dilution, cm2 / s
    DH_S = 9.31e-5
    DOH_S = 5.26e-5
    DNa_S = 1.34e-5
    DCl_S = 2.03e-5
    # precision
    eps = 1e-8
    # max iterations for Newton-Raphson
    maxIter = 1000
    # Initial concentrations(mmol / cm3) in Desalination(D), Base(B) and Acid(A)compartments
    cH_A = cHCl
    cOH_B = cNaOH
    cNa_D = cNaCl
    cCl_D = cNaCl
    Kw = 1e-14
    cH_D = 1e-7
    cOH_D = Kw / cH_D
    # Exchange capacity, mmol / cm3
    Xa = 1.28
    Xc = 1.43
    # Volume of alkali, base, desalination compartment, cm3 Membrane active surface area, cm2
    VA = 1500
    VB = 1500
    VD = 500
    S = 64
    # Faraday constant, C / mmol Gas constant, J / (mmol * K) Temperature, K
    F = 96485.34  # C / mol
    R = 8.314  # J / (mol * K)
    T = 298  # K
    # numeric solution
    cH_D_new = cH_D
    cOH_D_new = cOH_D
    cNa_D_new = cNa_D
    cCl_D_new = cCl_D
    cH_A_new = cH_A
    cOH_B_new = cOH_B
    cNa_A_new = 0
    cCl_B_new = 0
    k1_CEM = 2 * DH_CEM * DNa_CEM * Xc / dCEM
    k2_CEM = DH_CEM - DNa_CEM
    k1_AEM = 2 * DOH_AEM * DCl_AEM * Xa / dAEM
    k2_AEM = DOH_AEM - DCl_AEM
    # time - dependent problem
    tauk = np.zeros(M)
    kappa = np.zeros(M)
    pH = np.zeros(M)
    J_ct = np.zeros(M)
    J_at = np.zeros(M)
    J_c = 3.0e-5
    J_a = 2.7e-5
    E_D = dt * S / VD
    E_A = dt * S / VA
    E_B = dt * S / VB
    CH_DBL2_num = 0.0036
    CNa_DBL2_num = 0.0084
    COH_DBL3_num = 0.0046
    CCl_DBL3_num = 0.0121
    for k in range(0, M):
        tauk[k] = k * dt
        # Diff coef dependent on concentration
        # Activity factor at DBLs / membrane borders PROBLEM FOR C>0.02
        g_H_DBL2 = 0.7396 * CH_DBL2_num - 0.5184 * pow(CH_DBL2_num, 0.5) + 0.9977
        g_Na_DBL2 = 0.7396 * CNa_DBL2_num - 0.5184 * pow(CNa_DBL2_num, 0.5) + 0.9977
        g_OH_DBL3 = 0.7396 * COH_DBL3_num - 0.5184 * pow(COH_DBL3_num, 0.5) + 0.9977
        g_Cl_DBL3 = 0.7396 * CCl_DBL3_num - 0.5184 * pow(CCl_DBL3_num, 0.5) + 0.9977
        # Activity factor in Desalination chamber
        g_H_D = 0.7396 * cH_D_new - 0.5184 * pow(cH_D_new, 0.5) + 0.9977
        g_Na_D = 0.7396 * cNa_D_new - 0.5184 * pow(cNa_D_new, 0.5) + 0.9977
        g_OH_D = 0.7396 * cOH_D_new - 0.5184 * pow(cOH_D_new, 0.5) + 0.9977
        g_Cl_D = 0.7396 * cCl_D_new - 0.5184 * pow(cCl_D_new, 0.5) + 0.9977
        # New diffusion coefficients at DBLs / membrane borders
        # New diffusion coefficients in DBLs
        DH_S_DBL2 = DH_S * g_H_DBL2
        DNa_S_DBL2 = DNa_S * g_Na_DBL2
        DOH_S_DBL3 = DOH_S * g_OH_DBL3
        DCl_S_DBL3 = DCl_S * g_Cl_DBL3
        # New diffusion coefficients in Desalination chamber
        DH_S_D = DH_S * g_H_D
        DNa_S_D = DNa_S * g_Na_D
        DOH_S_D = DOH_S * g_OH_D
        DCl_S_D = DCl_S * g_Cl_D
        d31 = 2 * DH_S_DBL2 * DNa_S_DBL2 / (DH_S_DBL2 - DNa_S_DBL2)
        d42 = 2 * DOH_S_DBL3 * DCl_S_DBL3 / (DOH_S_DBL3 - DCl_S_DBL3)
        x0 = J_c
        y0 = J_a
        # initial point for Newton - Raphson
        J_c = x0
        J_c_old = x0
        for i in range(1, maxIter):
            J_c = J_c - f_J_c(J_c, cH_D_new, cNa_D_new, DBL2, d31, DH_S_DBL2, k1_CEM, cNa_A_new, cH_A_new, k2_CEM,
                              DNa_CEM) / df_J_c(J_c, cH_D_new, cNa_D_new, DBL2, d31, DH_S_DBL2, k1_CEM, cNa_A_new,
                                                cH_A_new, k2_CEM, DNa_CEM)
            if (abs(J_c - J_c_old) < eps):
                break
            J_c_old = J_c
        J_ct[k] = J_c * 100000
        P = cH_D_new + cNa_D_new - J_c * DBL2 / d31
        up_CHs = 2 * d31 * (cH_D_new + cNa_D_new) * (DH_S_DBL2 * cH_D_new + J_c * DBL2) - J_c * DBL2 * J_c * DBL2
        down_CHs = 2 * DH_S_DBL2 * (d31 * (cH_D_new + cNa_D_new) - J_c * DBL2)
        CH_DBL2_num = up_CHs / down_CHs
        CNa_DBL2_num = P - CH_DBL2_num
        # initial point for Newton - Raphson
        J_a = y0
        J_a_old = y0
        for i in range(1, maxIter):
            J_a = J_a - f_J_a(J_a, cOH_D_new, cCl_D_new, DBL3, d42, DOH_S_DBL3, k1_AEM, cCl_B_new, cOH_B_new, k2_AEM,
                              DCl_AEM) / df_J_a(J_a, cOH_D_new, cCl_D_new, DBL3, d42, DOH_S_DBL3, k1_AEM, cCl_B_new,
                                                cOH_B_new, k2_AEM, DCl_AEM)
            if (abs(J_a - J_a_old) < eps):
                break
            J_a_old = J_a
        J_at[k] = J_a * 100000
        Z = cOH_D_new + cCl_D_new - J_a * DBL3 / d42
        up_COHs = 2 * d42 * (cOH_D_new + cCl_D_new) * (DOH_S_DBL3 * cOH_D_new + J_a * DBL3) - J_a * DBL3 * J_a * DBL3
        down_COHs = 2 * DOH_S_DBL3 * (d42 * (cOH_D_new + cCl_D_new) - J_a * DBL3)
        COH_DBL3_num = up_COHs / down_COHs
        CCl_DBL3_num = Z - COH_DBL3_num
        # New concentrations into compartments
        # Acid compartment
        cH_A_new = cH_A_new - E_A * J_c
        cNa_A_new = cH_A - cH_A_new
        # Alkali compartment
        cOH_B_new = cOH_B_new - E_B * J_a
        cCl_B_new = cOH_B - cOH_B_new
        # Desalination compartment
        cCl_D_new = cCl_D_new - E_D * J_a
        B = cH_D_new - cOH_D_new + E_D * (J_c - J_a)
        cH_D_new = (B + pow((pow(B, 2) + 4 * Kw), 0.5)) / 2
        cOH_D_new = Kw / cH_D_new
        cNa_D_new = cCl_D_new + cOH_D_new - cH_D_new
        kappa[k] = ((pow(F, 2)) / (R * T)) * (
                    DNa_S_D * cNa_D_new + DCl_S_D * cCl_D_new + DH_S_D * cH_D_new + DOH_S_D * cOH_D_new)  # mS / cm
        pH[k] = -np.log10(cH_D_new)

    tau_opt_kappa = tauk[0]
    opt_kappa = kappa[0]


    for i in range(1, M):
        if (abs(kappa[i] - 1) < abs(opt_kappa - 1)):
            opt_kappa = kappa[i]
            tau_opt_kappa = tauk[i]

    tau_opt_pH1 = tauk[0]
    opt_pH1 = pH[0]
    tau_opt_pH2 = tauk[1]
    opt_pH2 = pH[1]
    for i in range(2, M):
        if (abs(pH[i] - 7.5) < abs(opt_pH1 - 7.5)):
            opt_pH2 = opt_pH1
            tau_opt_pH2 = tau_opt_pH1
            opt_pH1 = pH[i]
            tau_opt_pH1 = tauk[i]

    opt_pH = (opt_pH1 + opt_pH2) / 2
    tau_opt_pH = (tau_opt_pH1 + tau_opt_pH2) / 2
    if (abs(opt_pH1 - 7.5) < abs(opt_pH - 7.5)):
        opt_pH = opt_pH1
        tau_opt_pH = tau_opt_pH1

    if (abs(opt_pH2 - 7.5) < abs(opt_pH - 7.5)):
        opt_pH = opt_pH2
        tau_opt_pH = tau_opt_pH2

    res[0] = tau_opt_kappa / 60  # min
    res[1] = opt_kappa
    res[2] = tau_opt_pH / 60  # min
    res[3] = opt_pH
    if output is not None:
        for i in range(0, M):
            output[i][0] = round(tauk[i] / 60, 3)
            output[i][1] = round(kappa[i], 3)
            output[i][2] = round(pH[i], 3)
    del tauk
    del kappa
    del pH
    del J_ct
    del J_at

#
# Neural network solution
#

# Input 1
x1_step1xoffset = np.array([0.03, 0.03, 0.03, 50])
x1_step1gain = np.array([2.06185567010309, 2.06185567010309, 2.06185567010309, 0.02])
x1_step1ymin = -1

# Layer 1
b1 = np.array([27.177573996392236921, 14.880714188527033315, 162.136364782905531, 16.170005808795071545,
          1.4456195503804971647, 25.414569680758770431, -80.410560249748627371, 84.72768455662090048,
          47.375114644902375005, 20.414755758958829546])
IW1_1 = np.array([
    [28.948276715724706065, -0.0061373692956062245521, 0.0067456411302385284323, 0.27616634768642694953],
    [5.8222712116664672166, -0.036597357771013819261, 8.9395793959844169763, -0.00021075424544159603816],
    [84.136371104008148336, 88.591629908603522381, 8.4524315152753377589, -0.017355001204765664602],
    [5.8984822754741452755, -0.023871146978789423848, 9.718731856309492656, 0.0051531982654060429563],
    [-0.19202054871677753933, -0.0019790040082250590081, 0.00083267191435455203095, -0.00020315958011458200663],
    [13.486729136927468886, 11.694653430686432927, 0.002664135792939997284, -0.07080587877642186001],
    [1.0510498293890684351, 7.8072535985036584094, -88.110387913417881123, 0.018543578607810983633],
    [93.043674555249737068, 0.034638012445512303406, -7.9621165713730679414, -0.0002212799960255210465],
    [14.641597186431654976, 14.683631500350164956, 17.062916031579103304, -0.032730372233845568541],
    [-0.96724338317870173221, 0.11104230749084295637, 19.999635828142295679, -0.021964332497414348899]
])
# Layer 2
b2 = np.array([5.0702886369837454339, 22.330561664114494391, -3.3189996402548689325, 2.3818926534569992981])
LW2_1 = ([
    [19.919711992847840776, -22.792013664891815239, 1.4159573881902061121, 10.189364206314639461,
     -0.016454599042609058951, -3.73575126836330762, -1.1914518700845733168, -18.678440044772170125,
     9.8505980874818881432, -1.4168482567859468357],
    [0.00067577873089100129293, 0.20490594471528272846, -0.010040145562518885958, -0.28218860725914779453,
     -24.916801956004043461, 0.016178112501074266849, 0.019719928675780502153, -0.0048412087668278508401,
     -0.13073083248179054316, -0.011502663318395042746],
    [-2.687394135167166187, -8.6164570455333642229, -0.025352825912198458103, 20.669698821197520999,
     -0.0083984123365493379421, 0.5324512424928633525, -0.00051726664530761577807, 1.8897001068019918524,
     -9.4360322269250644922, -0.00080030025260362951979],
    [0.62535463371376653896, -1.9971049178043411843, 0.0025039271015711885016, 4.4391754869779642689,
     0.15448207158145418894, 0.023243567356829565596, -0.0032943670934626693297, -1.403231577396765406,
     -5.2029375818862968472, -0.0057154305808885235568]
])

# Output1
y1_step1ymin = -1
y1_step1gain = np.array([0.0205480155753958, 0.0134066187377365, 0.0101867745106528, 0.559442124313634])
y1_step1xoffset = np.array([99, 0.995954, 0, 4.52641])

def nnm(x,  y):
    # input1
    x1 = np.zeros(4)
    for i in range( 0 , 4):
        x1[i] = (x[i] - x1_step1xoffset[i]) * x1_step1gain[i] + x1_step1ymin
    # layer1
    layer1 = np.zeros(10)
    for i in range(0, 10):
        # signal = sum(inputs * weights) + b
        layer1[i] = 0
        for j in range(0, 4):
            layer1[i] += IW1_1[i][j] * x1[j]
            layer1[i] += b1[i]
        # activation sigmoid
        layer1[i] = 2 / (1 + np.exp(-2 * layer1[i])) - 1
    # layer2
    layer2 = np.zeros(4)
    for i in range(0 ,4):
        # signal = sum(inputs * weights) + b
        layer2[i] = 0
        for j in range(0, 10):
            layer2[i] += LW2_1[i][j] * layer1[j]
        layer2[i] += b2[i]
        # activation linear
    # Output1
    for i in range(0, 4):
        y[i] = (layer2[i] - y1_step1ymin) / y1_step1gain[i] + y1_step1xoffset[i]
    del layer1
    del layer2


useNN = None
cNaCl = None
cHClMin = None
cHClMax = None
cNaOHMin = None
cNaOHMax = None
DBLMin = None
DBLMax = None

def evalParams(x):
    # HCl
    if (x[1] < cHClMin):
        x[1] = cHClMin
    if (x[1] > cHClMax):
        x[1] = cHClMax
    # NaOH
    if (x[2] < cNaOHMin):
        x[2] = cNaOHMin
    if (x[2] > cNaOHMax):
        x[2] = cNaOHMax
    # DBL
    if (x[3] < DBLMin):
        x[3] = DBLMin
    if (x[3] > DBLMax):
        x[3] = DBLMax


def evalParams3d(x):
    # HCl
    if (x[0] < cHClMin):
        x[0] = cHClMin
    if (x[0] > cHClMax):
        x[0] = cHClMax
    # NaOH
    if (x[1] < cNaOHMin):
        x[1] = cNaOHMin
    if (x[1] > cNaOHMax):
        x[1] = cNaOHMax
    # DBL
    if (x[2] < DBLMin):
        x[2] = DBLMin
    if (x[2] > DBLMax):
        x[2] = DBLMax
        
# Optimisation(Fletcher - Rives)
eps1 = 1e-3
eps2 = 1e-3
eps3 = 1e-3
eps4 = 1e-3
n = 3
N = 1e3
deltaBig = 1e-1

def model(x, y):
    evalParams(x)
    if (useNN):
        nnm(x, y)
    else:
        NDModel(x[0], x[1], x[2], x[3], y, None)
    if (y[0] < 0):
        y[0] = 0
    if (y[1] < 0):
        y[1] = 0
    if (y[2] < 0):
        y[2] = 0
    if (y[3] < 0):
        y[3] = 0


def tf(y):
    # y0 = kappaTime
    # y1 = kappa
    # y2 = phTime
    # y3 = ph
    dev = 0
    dev += pow(y[0] - y[2], 2)
    # dev += (abs(y[0]) / 200 + abs(y[2]) / 200) / 2
    dev += pow(y[1] - 1, 2)
    dev += pow(y[3] - 7.5, 2)
    # if (abs(y[3] - 7.5) > 1) dev += abs(y[3] - 7.5) / 8
    # else dev += abs(y[3] - 7.5) / 80
    return dev


def model_tf(NaCl, param):
    x = np.array([NaCl, param[0], param[1], param[2]])
    y = np.zeros(4)
    model(x, y)
    return tf(y)

def gradf(x0, x, grad):
    for i in range(0, n):
        x[i] += deltaBig
        grad[i] = model_tf(x0, x)
        x[i] -= 2 * deltaBig
        grad[i] -= model_tf(x0, x)
        grad[i] /= (2 * deltaBig)
        x[i] += deltaBig
        

def norma(x1, x2):
    norma = abs(x1[0] - x2[0])
    for i in range(1, n):
        if (norma < abs(x1[i] - x2[i])): norma = abs(x1[i] - x2[i])
    return norma

def normaGrad(grad):
    max = abs(grad[0])
    for i in range(1, n):
        if (abs(grad[i]) > max): max = abs(grad[i])
    return max

def g(a,NaCl,  x0,  S):
    x1 = np.zeros(n)
    for i in range(0, n):
        x1[i] = x0[i] - a * S[i]
    return model_tf(NaCl, x1)


def localMin(x, y):
    Xcur = np.zeros(n) # Yk
    Xnew = np.zeros(n)# Yk + 1
    S = np.zeros(n)
    grad = np.zeros(n)
    NaCl = x[0]
    HCl = x[1]
    NaOH = x[2]
    DBL = x[3]
    Xcur[0] = HCl
    Xcur[1] = NaOH
    Xcur[2] = DBL
    # iteration0 (just like classic drop)
    gradf(NaCl, Xcur, grad)
    # saving grad norm for beta
    tmpNorm = normaGrad(grad)
    # So = antigrad(F(Xo))
    for i in range(0,n): S[i] = -grad[i]
    # alfa?
    l = 0
    r = 1
    while (abs(l - r) > eps4):
        alfa = (l + r) / 2
        if (g(alfa - deltaBig, NaCl, Xcur, S) < g(alfa + deltaBig, NaCl, Xcur, S)): r = alfa
        else: l = alfa
    alfa = (l + r) / 2
    # X1 = Xo + alfa * So
    for i in range(0, n):
        Xnew[i] = Xcur[i] + alfa * S[i]
    gradf(NaCl, Xnew, grad)
    count = 0
    # main cycle
    while (normaGrad(grad) > eps1 and
           norma(Xnew, Xcur) > eps2 and
           abs(model_tf(NaCl, Xnew) - model_tf(NaCl, Xcur)) > eps3 and
           count < N):
        for i in range(0, n):
            Xcur[i] = Xnew[i]
        # betak=norm(grad(F(Xk))) / norm(grad(F(Xk-1)))
        beta = pow(normaGrad(grad), 2) / pow(tmpNorm, 2)
        tmpNorm = normaGrad(grad)
        # Sk=-gradF(Xk)+betak * Sk-1
        for i in range(0, n): S[i] = -grad[i] + beta * S[i]
        # alfa?
        l = 0
        r = 1
        while (abs(l - r) > eps4):
            alfa = (l + r) / 2
            if (g(alfa - deltaBig, NaCl, Xcur, S) < g(alfa + deltaBig, NaCl, Xcur, S)): r = alfa
            else: l = alfa
        alfa = (l + r) / 2
        # Xk + 1 = Xk + alfa * Sk
        for i in range(0, n):
            Xnew[i] = Xcur[i] + alfa * S[i]
        evalParams3d(Xnew)
        gradf(NaCl, Xnew, grad)
        # print(endl
        # for (int i = 0 i < n ++i) print(Xnew[i] << " "
        count+=1
    x[0] = NaCl
    x[1] = Xnew[0]
    x[2] = Xnew[1]
    x[3] = Xnew[2]
    model(x, y)
    del Xcur
    del Xnew
    del S
    del grad


def execute(xMin, yMin):
    x = np.zeros(4)
    y = np.zeros(4)
    nodes = 4
    if (useNN):
        nodes = 6
    print("Число узлов: ", pow(nodes, 3))
    x[0] = cNaCl
    xMin[0] = x[0]
    xMin[1] = cHClMin
    xMin[2] = cNaOHMin
    xMin[3] = DBLMin
    localMin(xMin, yMin)
    for l in range(0, nodes):
        for m in range(0, nodes):
            for d in range(0, nodes):
                x[1] = cHClMin + (l) * (cHClMax - cHClMin) / (nodes + 1)
                x[2] = cNaOHMin + (m) * (cNaOHMax - cNaOHMin) / (nodes + 1)
                x[3] = DBLMin + (d) * (DBLMax - DBLMin) / (nodes + 1)
                localMin(x, y)
                if (tf(y) < tf(yMin)):
                    xMin[1] = x[1]
                    xMin[2] = x[2]
                    xMin[3] = x[3]
                    yMin[0] = y[0]
                    yMin[1] = y[1]
                    yMin[2] = y[2]
                    yMin[3] = y[3]
    del x
    del y


# функция вероятности принятия нового состояния системы
def h( deltaE, T):
    # точное значение
    #return 1 / (1 + np.exp(deltaE / T))
    # приближенное значение
    return np.exp(-deltaE / T)


# закон изменения температуры
def T(k, To):
    # Больцмановскйи отжиг
    return To / np.log(1 + k)


# порождающее семейство распределений
def G(x, T):
    n = 24
    max = np.array([cHClMax, cNaOHMax, DBLMax])
    min = np.array([cHClMin, cNaOHMin, DBLMin])
    rng = np.random.default_rng()
    for i in range(1, 4):
        #S = 0
        #for j in range(0, n):
            # накапливаем равномерно распределенные значения случайной величины в пределах[0, 1]
        #    S += (rand()) / RAND_MAX
        # нормализуем и получаем приближенное нормальное распределение
        #S = (S - n / 2) / sqrt(n / 12)
        S = rng.standard_normal()
        # получаем распределение g(0, T) (N(mu, rho ^ 2))
        S = np.sqrt(T) * S
        # придаем возмущение соотв.элементу вектора
        # x[i] += S
        x[i] = (max[i - 1] - min[i-1]) / 2 * S + x[i]
    evalParams(x)
    return 0


def monteCarlo(xMin, yMin):
    x = np.zeros(4)
    y = np.zeros(4)
    x[0] = cNaCl
    x[1] = cHClMin + (cHClMax - cHClMin) / 2
    x[2] = cNaOHMin + (cNaOHMax - cNaOHMin) / 2
    x[3] = DBLMin + (DBLMax - DBLMin) / 2
    xMin[0] = x[0]
    xMin[1] = x[1]
    xMin[2] = x[2]
    xMin[3] = x[3]
    localMin(xMin, yMin)
    # Энергия системы
    E = tf(yMin)
    # Начальная температура
    To = (1) / 12
    rng = np.random.default_rng()
    for k in range(1, int(1e3)):
        t = T(k, To)
        G(x, t)
        model(x, y)
        newE = tf(y)
        if (rng.uniform() < h(newE - E, t)):
            E = newE,
            xMin[0] = x[0]
            xMin[1] = x[1]
            xMin[2] = x[2]
            xMin[3] = x[3]
            yMin[0] = y[0]
            yMin[1] = y[1]
            yMin[2] = y[2]
            yMin[3] = y[3]


def main(float_input, bool_input):
    global cNaCl
    global cHClMin
    global cHClMax
    global cNaOHMin
    global cNaOHMax
    global DBLMin
    global DBLMax
    global useNN

    cNaCl = float_input[0]
    cHClMin = float_input[1]
    cHClMax = float_input[2]
    cNaOHMin = float_input[3]
    cNaOHMax = float_input[4]
    DBLMin = float_input[5]
    DBLMax = float_input[6]
    useNN = bool_input[0]
    useMonteCarlo = bool_input[1]
    print("Выбраны следующие начальные параметры:")
    print("Концентрация NaCl: " + str(cNaCl))
    print("Концентрация HCl: от " + str(cHClMin) + " до " + str(cHClMax))
    print("Концентрация NaOH: от " + str(cNaOHMin) + " до " + str(cNaOHMax))
    print("Толщина DBL: от " + str(DBLMin) + " до " + str(DBLMax))
    if (useNN):
        print("Метод моделирования: нейросетевой")
    else:
        print("Метод моделирования: численный")

    if (useMonteCarlo):
        print("Метод оптимизации: отжиг Больцмана")
    else:
        print("Метод оптимизации: быстрый спуск")

    xMin = np.zeros(4)
    yMin = np.zeros(4)
    if (useMonteCarlo):
        monteCarlo(xMin, yMin)
    else:
        execute(xMin, yMin)
    output1 = "\n"
    output1 += "\nНайдены оптимальные параметры:"
    output1 += "\nКонцентрация HCl: " + str(xMin[1])
    output1 += "\nКонцентрация NaOH: " + str(xMin[2])
    output1 += "\nТолщина DBL: " + str(xMin[3])
    output1 += "\nРезультаты моделирования:"
    output1 += "\nkappaTime=" + str(yMin[0])
    output1 += "\nkappa=" + str(yMin[1])
    output1 += "\npHTime=" + str(yMin[2])
    output1 += "\npH=" + str(yMin[3])
    output = np.zeros((M,3))
    NDModel(xMin[0], xMin[1], xMin[2], xMin[3], yMin, output)
    # print(endl << "time\tkappa\tpH" << endl
    # for (int i = 0 i < M ++i) {
                                   # print(output[i][0] << "\t" << output[i][1] << "\t" << output[i][2]
    #}
    if (useNN):
        output1 += "\nРезультаты численной проверки нейросети:"
        output1 += "\nkappaTime=" + str(yMin[0])
        output1 += "\nkappa=" + str(yMin[1])
        output1 += "\npHTime=" + str(yMin[2])
        output1 += "\npH=" + str(yMin[3])

    output2 = "t\tkappa\tpH"
    for i in range(0, M):
        output2 += "\n" + str(output[i][0])
        output2 += "\t" + str(output[i][1])
        output2 += "\t" + str(output[i][2])
    del output
    del xMin
    del yMin

    return output1, output2


def maintest():
    global cNaCl
    global cHClMin
    global cHClMax
    global cNaOHMin
    global cNaOHMax
    global DBLMin
    global DBLMax
    global useMonteCarlo
    global useNN

    cNaCl = 0.015
    cHClMin = 0.01
    cHClMax = 1
    cNaOHMin = 0.01
    cNaOHMax = 1
    DBLMin = 50
    DBLMax = 150
    useMonteCarlo = False
    useNN = False

    x = np.array([0.015, 0.3, 0.3, 80])
    y = np.zeros(4)
    execute(x, y)
    print(x)
    print(y)

