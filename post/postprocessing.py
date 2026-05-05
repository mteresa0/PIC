import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import argparse

# COLLISIONS 

def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df

def plots_P_coll(data: pd.DataFrame) -> None:
    print(data.describe())

    plt.figure()
    sns.histplot(data['P_coll'], kde=True)
    plt.title("P_coll distribution")
    plt.show(block=False)

    plt.figure()
    sns.histplot(data['sigma/sigma_cr_max'], kde=True)
    plt.show(block = False)

    plt.figure()
    sns.histplot(data['F_correction'], kde=True)
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
    
# MAIN

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("task", choices=["collisions", "ke", "all"])
    args = parser.parse_args()
    
    if args.task == "collisions":
        collisions_postprocess()
    elif args.task == "ke":
        ke_postprocess()
    else:
        collisions_postprocess()
        ke_postprocess()
    
