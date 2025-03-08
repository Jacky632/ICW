import numpy as np
import scipy.integrate
import matplotlib.pyplot as plt


def MLKF_1dof(m1, l1, k1, f1):

    """Return mass, damping, stiffness & force matrices for 1DOF system"""

    M = np.array([[m1]])
    L = np.array([[l1]])
    K = np.array([[k1]])
    F = np.array([f1])

    return M, L, K, F


def MLKF_2dof(m1, l1, k1, f1, m2, l2, k2, f2):

    """Return mass, damping, stiffness & force matrices for 2DOF system"""

    M = np.array([[m1, 0], [0, m2]])
    L = np.array([[l1+l2, -l2], [-l2, l2]])
    K = np.array([[k1+k2, -k2], [-k2, k2]])
    F = np.array([f1, f2])

    return M, L, K, F


def MLKF_ndof(m1, l1, k1, f1, m_t, n, o, do, l, f):

    """Return mass, damping, stiffness & force matrices for nDOF system"""
    m = m_t / n
    M = np.eye(n+1) * m
    M[0][0] = m1

    L = np.zeros((n+1, n+1))
    L[0, 0] = l1 + n*l
    for i in range(1, n+1):
        L[i, i] = l
        L[i, 0] = -l
        L[0, i] = -l

    K = np.zeros((n+1, n+1)) 
    o_i = o - do/(n-1) * (n//2)
    sumK = 0
    for i in range(1, n+1):
        k_i = (2*np.pi*o_i)**2 * m
        K[i, i] = k_i
        K[i, 0] = -k_i
        K[0, i] = -k_i
        sumK += k_i
        o_i += do/(n-1)
    K[0, 0] = k1 + sumK

    F = np.array([f1] + [f]*n)

    return M, L, K, F


def freq_response(w_list, M, L, K, F):

    """Return complex frequency response of system"""

    return np.array(
        [np.linalg.solve(-w*w * M + 1j * w * L + K, F) for w in w_list]
    )


def time_response(t_list, M, L, K, F):

    """Return time response of system"""

    mm = M.diagonal()

    def slope(t, y):
        xv = y.reshape((2, -1))
        a = (F - L@xv[1] - K@xv[0]) / mm
        s = np.concatenate((xv[1], a))
        return s

    solution = scipy.integrate.solve_ivp(
        fun=slope,
        t_span=(t_list[0], t_list[-1]),
        y0=np.zeros(len(mm) * 2),
        method='Radau',
        t_eval=t_list
    )

    return solution.y[0:len(mm), :].T


def last_nonzero(arr, axis, invalid_val=-1):

    """Return index of last non-zero element of an array"""

    mask = (arr != 0)
    val = arr.shape[axis] - np.flip(mask, axis=axis).argmax(axis=axis) - 1
    return np.where(mask.any(axis=axis), val, invalid_val)


def plot(ax, fig, hz, sec, M, L, K, F, cap):

    """Plot frequency and time domain responses"""

    # Generate response data

    f_response = freq_response(hz * 2*np.pi, M, L, K, F)
    f_amplitude = np.abs(f_response)
    t_response = time_response(sec, M, L, K, F)

    # leave only building response
    f_amplitude = [[row[0]] for row in f_amplitude]
    t_response = t_response[:, 0].reshape(-1, 1)

    # Determine suitable legends

    f_legends = [
        '{} peak {:.4g} mm at {:.4g} Hz'.format(
            cap,
            f_amplitude[m][i],
            hz[m]
        )
        for i, m in enumerate(np.argmax(f_amplitude, axis=0))
    ]

    equilib = np.abs(freq_response([0], M, L, K, F))[0]         # Zero Hz
    toobig = abs(100 * (t_response - equilib) / equilib) >= 2
    lastbig = last_nonzero(toobig, axis=0, invalid_val=len(sec)-1)

    t_legends = [
        '{} settled to 2% beyond {:.4g} sec'.format(
            cap,
            sec[lastbig[i]]
        )
        for i, _ in enumerate(t_response.T)
    ]

    # Create plot
    ax[0].plot(hz, f_amplitude, label=f_legends[0])
    ax[-1].plot(sec, t_response, label=t_legends[0])


def main():

    """Main program"""

    m1 = 3.94  # Building Mass
    l1 = 2.23  # Building Damping
    k1 = 2110  # Building Spring
    f1 = 300  # Building Force (N*10^-3)

    # frequency range
    hzstart = 0
    hzend = 5

    # time limit
    seclimit = 15

    # Generate frequency and time arrays
    hz = np.linspace(hzstart, hzend, 10001)
    sec = np.linspace(0, seclimit, 10001)

    fig, ax = plt.subplots(2, 1, figsize=(8, 8), gridspec_kw={'height_ratios': [2, 1]})

    ax[0].set_title('Amplitude of frequency domain response to sinusoidal force')
    ax[0].set_xlabel('Frequency/hertz')
    ax[0].set_ylabel('Amplitude/milimetre')
    ax[-1].set_title('Time domain response to step force')
    ax[-1].set_xlabel('Time/second')
    ax[-1].set_ylabel('Displacement/milimetre')

    # NO DAMPING --------------
    # Generate matrices describing the system.
    M, L, K, F = MLKF_1dof(
        m1, l1, k1, f1
    )
    plot(ax, fig, hz, sec, M, L, K, F, 'No Damping: ')

    # 10 DAMPER --------------
    m_t = 0.4  # total mass
    n = 10  # 10 absorbers
    l = 0.60 / n  # individual damping
    f = 0  # no external force
    o = 3.683  # main frequency to get rid of

    do = 1.55  # variation in tuned frequency
    M, L, K, F = MLKF_ndof(
        m1, l1, k1, f1,
        m_t, n, o, do, l, f
    )
    plot(ax, fig, hz, sec, M, L, K, F, "10 Dampers, Total Mass 0.08 kg: ")


    # Plot results

    fig.canvas.mpl_connect('resize_event', lambda x: fig.tight_layout(pad=2.5))
    ax[0].legend()
    ax[-1].set_xlim(0, 5)
    ax[-1].legend(loc="lower right")
    fig.tight_layout()
    plt.savefig('optimized10Dampers.png', dpi=300)
    plt.show()


if __name__ == '__main__':
    main()
