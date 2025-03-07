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
    ax[-1].plot(hz, t_response, label=t_legends[0])


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

    # 1 DAMPER -------------
    m2 = 0.15
    l2 = 0.92
    k2 = 80.33
    f2 = 0

    M, L, K, F = MLKF_2dof(
        m1, l1, k1, f1,
        m2, l2, k2, f2
    )
    plot(ax, fig, hz, sec, M, L, K, F, "1 Damper:     ")

    # 10 DAMPER --------------
    m_t = 0.15  # total mass
    


    # Plot results

    fig.canvas.mpl_connect('resize_event', lambda x: fig.tight_layout(pad=2.5))
    ax[0].legend()
    ax[-1].legend()
    fig.tight_layout()
    plt.savefig('1.png', dpi=300)
    plt.show()


if __name__ == '__main__':
    main()
