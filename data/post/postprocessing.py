import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def load_data(path: str) -> pd.DataFrame:
    """Load a CSV file into a DataFrame."""
    df = pd.read_csv(path)
    return df


def plots_P_coll(data: pd.DataFrame) -> None:
    """Generate plots from the DataFrame."""
    # Example: line plot of one column vs another
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

    # Add more plots here as needed

def plots_n_coll(data: pd.DataFrame):
    print(data.describe())

def main():
    csv_path = "../outputs/collisions_it.csv"   # adjust this
    df = load_data(csv_path)
    plots_P_coll(df)

    csv_path = "../outputs/n_coll.csv"   # adjust this
    df = load_data(csv_path)
    plots_n_coll(df)


if __name__ == "__main__":
    main()
