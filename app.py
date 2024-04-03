import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.cm import ScalarMappable
from matplotlib.colors import ListedColormap
from matplotlib.cm import ScalarMappable
import jsm_SHMR


mass_example = np.load("mass_example.npy")
red_example = np.load("redshift_example.npy")
halo_masses = np.linspace(8,12,100) # just for the model

# Define function to generate plot
def generate_plot(theta, mock=False, redshift=False):

    model_color="cornflowerblue"

    stellar_example = jsm_SHMR.general(theta, mass_example, red_example, 1)
    theta_det = theta[:2] + [0, 0] + theta[4:]

    fig, ax = plt.subplots(figsize=(8, 8))

    if redshift==True:
        z_max = 7
        z_array = np.arange(z_max)

        cmap = plt.get_cmap('RdBu_r')  # You can choose a different colormap
        colors = cmap(np.linspace(0, 1, z_array.shape[0]))
        custom_cmap = ListedColormap(colors)

        sigma = theta[2] + theta[3] * (halo_masses - 12)
        sigma[sigma < 0] = 0.0

        deterministic_early = jsm_SHMR.general(theta_det, halo_masses, z_max, Nsamples=1)
        deterministic_presentday = jsm_SHMR.general(theta_det, halo_masses, 0, Nsamples=1)

        if deterministic_early[0] > deterministic_presentday[0]:
            ax.fill_between(halo_masses, deterministic_early + sigma, deterministic_presentday - sigma, color=model_color, alpha=0.3, zorder=0)
            ax.fill_between(halo_masses, deterministic_early + 2 * sigma, deterministic_presentday - 2 * sigma, color=model_color, alpha=0.2, zorder=0)
            ax.fill_between(halo_masses, deterministic_early + 3 * sigma, deterministic_presentday - 3 * sigma, color=model_color, alpha=0.1, zorder=0)

        else:
            ax.fill_between(halo_masses, deterministic_presentday + sigma, deterministic_early - sigma, color=model_color, alpha=0.3, zorder=0)
            ax.fill_between(halo_masses, deterministic_presentday + 2 * sigma, deterministic_early - 2 * sigma, color=model_color, alpha=0.2, zorder=0)
            ax.fill_between(halo_masses, deterministic_presentday + 3 * sigma, deterministic_early - 3 * sigma, color=model_color, alpha=0.1, zorder=0)

        for i in z_array:   
            ax.plot(halo_masses, jsm_SHMR.general(theta_det, halo_masses, i, 1), lw=1, color=colors[i])
        
        norm = plt.Normalize(0, z_max-1)
        sm = ScalarMappable(cmap=custom_cmap, norm=norm)
        sm.set_array([])

        cax = ax.inset_axes([0.57, 0.12, 0.4, 0.04])
        colorbar = fig.colorbar(sm, cax=cax, ticks=z_array, orientation='horizontal')
        colorbar.ax.tick_params(axis='both', which='major', labelsize=12)
        colorbar.set_label('$z_{\mathrm{acc}}$', fontsize=15)

    else:
        det = jsm_SHMR.general(theta_det, halo_masses, 0, 1)
        ax.plot(halo_masses, det, color=model_color, label="Model", lw=1)

        sigma = theta[2] + theta[3]*(halo_masses-12)
        
        ax.fill_between(halo_masses, det - sigma, det + sigma, color=model_color, alpha=0.3) 
        ax.fill_between(halo_masses, det - 2*sigma, det + 2*sigma, color=model_color, alpha=0.2) 
        ax.fill_between(halo_masses, det - 3*sigma, det + 3*sigma, color=model_color, alpha=0.1) 

    if mock==True:
        mock_lgMh, mock_lgMs = mass_example, stellar_example
        mass_cut = mock_lgMs > 6.5
        ax.axhline(6.5, ls="--", label="SAGA Magnitude Limit", lw=1, color="grey")
        ax.scatter(mock_lgMh[mass_cut], mock_lgMs[mass_cut], marker="*", color="black", label="mock survey", zorder=5)


    ax.plot(halo_masses, jsm_SHMR.lgMs_B13(halo_masses, 0), color="darkorange", ls="--", label="$\mathrm{Behroozi\ 2013}$", lw=1, zorder=5)
    ax.plot(halo_masses, jsm_SHMR.lgMs_RP17(halo_masses, 0), color="darkmagenta", ls="-.", label="$\mathrm{Rodriguez-Puebla\ 2017}$", lw=1, zorder=5)
    ax.set_ylabel("$\log M_{*}\ [\mathrm{M}_{\odot}]$", fontsize=15)
    ax.set_xlabel("$\log M_{\mathrm{acc}}\ [\mathrm{M}_{\odot}]$", fontsize=15)
    ax.set_ylim(5.5, 10.5)
    ax.set_xlim(9.2, 12)
    ax.legend(fontsize=12, loc=2)

    st.pyplot(fig)

# Main function
def main():
    st.title('SHMR_JSM')

    # Define default values for parameters
    default_theta = [10.5, 2.0, 0.2, 0.0, 0.0, 0.0]
    limits = [[10.0,11.0], [1.0,4.0], [0.0,1.5], [-1.0,0.0], [-3.0,2.0], [0.0,3.0]]
    param_labels = ["$M_{*}$", "$\\alpha$", "$\\sigma$"," $\\gamma$", "$\\beta$", "$\\tau$"]
    spacing = [0.1, 0.01, 0.01, 0.01, 0.01, 0.1]

    with st.sidebar:
        st.title('Parameters ($\\vec{\\theta}$)')
        # Define sliders for parameters
        theta = [st.slider(param_labels[i], limits[i][0], limits[i][1], default_theta[i], spacing[i]) for i in range(len(default_theta))]
        mock = st.button('Toggle Mock Survey')
        redshift = st.button('Toggle Redshift Curves')

    # Generate and display the plot
    generate_plot(theta, mock=mock, redshift=redshift)

if __name__ == '__main__':
    main()