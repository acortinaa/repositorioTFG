from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

def plot_tracks_and_truth_2x2(tracks, trayectorias_truth):
    fig = plt.figure(figsize=(10, 8))
    fig.suptitle('Comparación de Tracks y Trayectorias Truth', fontsize=16)

    # Tracks 2D (XY)
    ax1 = fig.add_subplot(221)
    ax1.set_title('Tracks (XY)')
    for track in tracks:
        ax1.plot(track[:, 0], track[:, 1], alpha=0.6, color='tab:red')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')

    # Truth 2D (XY)
    ax2 = fig.add_subplot(222)
    ax2.set_title('Trayectorias Truth (XY)')
    for tray in trayectorias_truth:
        ax2.plot(tray[:, 0], tray[:, 1], alpha=0.6, color='tab:blue')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')

    # Tracks 3D
    ax3 = fig.add_subplot(223, projection='3d')
    ax3.set_title('Tracks (3D)')
    for track in tracks:
        ax3.plot(track[:, 0], track[:, 1], track[:, 2], alpha=0.6, color='tab:red')
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_zlabel('Z')

    # Truth 3D
    ax4 = fig.add_subplot(224, projection='3d')
    ax4.set_title('Trayectorias Truth (3D)')
    for tray in trayectorias_truth:
        ax4.plot(tray[:, 0], tray[:, 1], tray[:, 2], alpha=0.6, color='tab:blue')
    ax4.set_xlabel('X')
    ax4.set_ylabel('Y')
    ax4.set_zlabel('Z')

    plt.tight_layout()
    plt.show()
