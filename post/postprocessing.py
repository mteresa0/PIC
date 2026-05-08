import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import argparse

# COLLISIONS 

def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df

def plots_P_coll(data: pd.DataFrame) -> None:
    print(data.describe())

    # Create a figure with 1 row and 3 columns
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Plot 1: P_coll
    sns.histplot(data['P_coll'], kde=True, ax=axes[0], color='skyblue')
    axes[0].set_title("P_coll Distribution")
    axes[0].set_xlabel("P_coll")

    # Plot 2: sigma/sigma_cr_max
    sns.histplot(data['sigma/sigma_cr_max'], kde=True, ax=axes[1], color='salmon')
    axes[1].set_title("$\sigma / \sigma_{cr,max}$ Distribution")
    axes[1].set_xlabel("Ratio")

    # Plot 3: F_correction
    sns.histplot(data['F_correction'], kde=True, ax=axes[2], color='seagreen')
    axes[2].set_title("F_correction Distribution")
    axes[2].set_xlabel("F_correction")

    # Automatically adjust spacing so titles/labels don't overlap
    plt.tight_layout()
    plt.show()

def plots_n_coll(data: pd.DataFrame):
    print(data.describe())

def collisions_postprocess():
    csv_path = "../outputs/collisions_it.csv"
    df = load_data(csv_path)
    plots_P_coll(df)

    csv_path = "../outputs/n_coll.csv" 
    df = load_data(csv_path)
    plots_n_coll(df)

# KINETIC ENERGY

def plot_ke(csv_path):
    df = pd.read_csv(csv_path)

    plt.figure()
    plt.plot(df["time"], df["total_kinetic_energy"], label="total kinetic energy")
    plt.plot(df["time"], df["sum_particles_kinetic_energy"], label="sum particles KE")

    plt.xlabel("time")
    plt.ylabel("energy")
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.show()

def ke_postprocess():
    plot_ke("../outputs/total_kinetic_energy.csv")

# velocities
def plot_vels(csv_path):
    df = pd.read_csv(csv_path)

    # We check if 'vz' exists to handle both 2D and 3D outputs
    if 'vz' in df.columns:
        df['v_mag'] = np.sqrt(df['vx']**2 + df['vy']**2 + df['vz']**2)
    else:
        df['v_mag'] = np.sqrt(df['vx']**2 + df['vy']**2)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # --- velocity distribution ---
    final_time = df['time'].max()
    final_data = df[df['time'] == final_time]
    
    sns.histplot(data=final_data, x='v_mag', hue='species index', 
                 element="step", kde=True, ax=ax1, palette='viridis', 
                 stat="density", common_norm=False)
    ax1.set_title(f'Velocity Distribution at t = {final_time}')
    ax1.set_xlabel('Magnitude $|v|$')
    ax1.set_ylabel('Density')

    # --- mean velocities ---
    mean_vel = df.groupby(['time', 'species index'])['v_mag'].mean().reset_index()
    
    sns.lineplot(data=mean_vel, x='time', y='v_mag', 
                 hue='species index', ax=ax2, palette='viridis')
    ax2.set_title('Mean Velocity Evolution')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Mean Magnitude $\langle |v| \\rangle$')

    plt.tight_layout()
    plt.show()

def vel_postprocess():
    plot_vels("../outputs/velocities.csv")

# MAIN

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("task", choices=["collisions", "ke", "vels", "all"])
    args = parser.parse_args()
    
    if args.task == "collisions":
        collisions_postprocess()
    elif args.task == "ke":
        ke_postprocess()
    else:
        collisions_postprocess()
        ke_postprocess()
        vel_postprocess()
    
