
import numpy as np
import scipy.constants as sc

AU = sc.astronomical_unit
c = sc.speed_of_light
pc = sc.parsec
AU_light_sec = AU/c
AU_pc = AU/pc

def solar_wind(toas, freqs, planetssb, sunssb, pos_t,
               n_earth=5, n_earth_bins=None,
               t_init=None, t_final=None):
    """
    Construct DM-Solar Model fourier design matrix.

    :param toas: vector of time series in seconds
    :param planetssb: solar system bayrcenter positions
    :param pos_t: pulsar position as 3-vector
    :param freqs: radio frequencies of observations [MHz]
    :param n_earth: The electron density from the solar wind at 1 AU.
    :param n_earth_bins: Number of binned values of n_earth for which to fit or
                an array or list of bin edges to use for binned n_Earth values.
                In the latter case the first and last edges must encompass all
                TOAs and in all cases it must match the size (number of
                elements) of n_earth.
    :param t_init: Initial time of earliest TOA in entire dataset, including
                all pulsars.
    :param t_final: Final time of last TOA in entire dataset, including all
                pulsars.

    :return dt_DM: Chromatic time delay due to solar wind
    """

    if n_earth_bins is None:
        theta, R_earth, _, _ = theta_impact(planetssb, sunssb, pos_t)
        dm_sol_wind = dm_solar(n_earth, theta, R_earth)
        dt_DM = (dm_sol_wind) * 4.148808e3 / freqs**2

    else:
        if isinstance(n_earth_bins, int) and (t_init is None
                                              or t_final is None):
            err_msg = 'Need to enter t_init and t_final '
            err_msg += 'to make binned n_earth values.'
            raise ValueError(err_msg)

        elif isinstance(n_earth_bins, int):
            edges, step = np.linspace(t_init, t_final, n_earth_bins,
                                      endpoint=True, retstep=True)

        elif isinstance(n_earth_bins, list) or isinstance(n_earth_bins,
                                                          np.ndarray):
            edges = n_earth_bins

        dt_DM = []

        for ii, bin in enumerate(edges[:-1]):

            bin_mask = np.logical_and(toas >= bin, toas <= edges[ii + 1])
            theta, R_earth, _, _ = theta_impact(planetssb,
                                                sunssb,
                                                pos_t)
            dm_sol_wind = dm_solar(n_earth[ii], theta[bin_mask],
                                   R_earth[bin_mask])
            if dm_sol_wind.size != 0:
                dt_DM.extend((dm_sol_wind)
                             * 4.148808e3 / freqs[bin_mask]**2)
            else:
                pass

        dt_DM = np.array(dt_DM)
        if dt_DM.size!=toas.size:
            err_msg = 'dt_DM ({0}) does not '.format(dt_DM.size)
            err_msg +='match number of TOAs ({0})!!!'.format(toas.size)
            raise ValueError(err_msg)

    return dt_DM


def createfourierdesignmatrix_solar_dm(toas, freqs, planetssb, sunssb, pos_t,
                                       modes=None, nmodes=30,
                                       Tspan=None, logf=True, fmin=None,
                                       fmax=None):
    """
    Construct DM-Solar Model fourier design matrix.

    :param toas: vector of time series in seconds
    :param planetssb: solar system bayrcenter positions
    :param pos_t: pulsar position as 3-vector
    :param nmodes: number of fourier coefficients to use
    :param freqs: radio frequencies of observations [MHz]
    :param Tspan: option to some other Tspan
    :param logf: use log frequency spacing
    :param fmin: lower sampling frequency
    :param fmax: upper sampling frequency

    :return: F: SW DM-variation fourier design matrix
    :return: f: Sampling frequencies
    """

    # get base fourier design matrix and frequencies
    F, Ffreqs = createfourierdesignmatrix_red(toas, nmodes=nmodes,
                                                    modes=modes,
                                                    Tspan=Tspan, logf=logf,
                                                    fmin=fmin, fmax=fmax)
    theta, R_earth, _, _ = theta_impact(planetssb, sunssb, pos_t)
    dm_sol_wind = dm_solar(1.0, theta, R_earth)
    dt_DM = dm_sol_wind * 4.148808e3 /(freqs**2)

    return F * dt_DM[:, None], Ffreqs



def _dm_solar_close(n_earth, r_earth):
    return (n_earth * AU_light_sec * AU_pc / r_earth)


def _dm_solar(n_earth, theta, r_earth):
    return ((np.pi - theta) *
            (n_earth * AU_light_sec * AU_pc
             / (r_earth * np.sin(theta))))


def dm_solar(n_earth, theta, r_earth):
    """
    Calculates Dispersion measure due to 1/r^2 solar wind density model.
    ::param :n_earth Solar wind proto/electron density at Earth (1/cm^3)
    ::param :theta: angle between sun and line-of-sight to pulsar (rad)
    ::param :r_earth :distance from Earth to Sun in (light seconds).
    See You et al. 2007 for more details.
    """
    return np.where(np.pi - theta >= 1e-5,
                    _dm_solar(n_earth, theta, r_earth),
                    _dm_solar_close(n_earth, r_earth))


def theta_impact(planetssb, sunssb, pos_t):
    """
    Use the attributes of an enterprise Pulsar object to calculate the
    solar impact angle.

    ::param :planetssb Solar system barycenter time series supplied with
        enterprise.Pulsar objects.
    ::param :sunssb Solar system sun-to-barycenter timeseries supplied with
        enterprise.Pulsar objects.
    ::param :pos_t Unit vector to pulsar position over time in ecliptic
        coordinates. Supplied with enterprise.Pulsar objects.

    returns: Solar impact angle (rad), Distance to Earth (R_earth),
             impact distance (b), perpendicular distance (z_earth)
    """
    earth = planetssb[:, 2, :3]
    sun = sunssb[:, :3]
    earthsun = earth - sun
    R_earth = np.sqrt(np.einsum('ij,ij->i', earthsun, earthsun))
    Re_cos_theta_impact = np.einsum('ij,ij->i', earthsun, pos_t)

    theta_impact = np.arccos(-Re_cos_theta_impact / R_earth)
    b = np.sqrt(R_earth**2 - Re_cos_theta_impact**2)

    return theta_impact, R_earth, b, -Re_cos_theta_impact



def createfourierdesignmatrix_red(
    toas, nmodes=30, Tspan=None, logf=False, fmin=None, fmax=None, pshift=False, modes=None, pseed=None
):
    """
    Construct fourier design matrix from eq 11 of Lentati et al, 2013
    :param toas: vector of time series in seconds
    :param nmodes: number of fourier coefficients to use
    :param freq: option to output frequencies
    :param Tspan: option to some other Tspan
    :param logf: use log frequency spacing
    :param fmin: lower sampling frequency
    :param fmax: upper sampling frequency
    :param pshift: option to add random phase shift
    :param pseed: option to provide phase shift seed
    :param modes: option to provide explicit list or array of
                  sampling frequencies

    :return: F: fourier design matrix
    :return: f: Sampling frequencies
    """

    T = Tspan if Tspan is not None else toas.max() - toas.min()

    # define sampling frequencies
    if modes is not None:
        nmodes = len(modes)
        f = modes
    elif fmin is None and fmax is None and not logf:
        # make sure partially overlapping sets of modes
        # have identical frequencies
        f = 1.0 * np.arange(1, nmodes + 1) / T
    else:
        # more general case

        if fmin is None:
            fmin = 1 / T

        if fmax is None:
            fmax = nmodes / T

        if logf:
            f = np.logspace(np.log10(fmin), np.log10(fmax), nmodes)
        else:
            f = np.linspace(fmin, fmax, nmodes)

    # if requested, add random phase shift to basis functions
    if pshift or pseed is not None:
        if pseed is not None:
            # use the first toa to make a different seed for every pulsar
            seed = int(toas[0] / 17) + int(pseed)
            np.random.seed(seed)

        ranphase = np.random.uniform(0.0, 2 * np.pi, nmodes)
    else:
        ranphase = np.zeros(nmodes)

    Ffreqs = np.repeat(f, 2)

    N = len(toas)
    F = np.zeros((N, 2 * nmodes))

    # The sine/cosine modes
    F[:, ::2] = np.sin(2 * np.pi * toas[:, None] * f[None, :] + ranphase[None, :])
    F[:, 1::2] = np.cos(2 * np.pi * toas[:, None] * f[None, :] + ranphase[None, :])

    return F, Ffreqs