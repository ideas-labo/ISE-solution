import matplotlib.pyplot as plt
import pandas as pd
import os


def visualize_search_results(results_folder, dataset_name, visualization_folder):
    """
    Visualize the search results from stored CSV results.

    Parameters:
        results_folder (str): Folder containing the search results CSV files.
        dataset_name (str): Name of the dataset to visualize (without extension).
        visualization_folder (str): Folder to save the visualization images.
    """
    # Construct the file paths
    csv_file = os.path.join(results_folder, f"{dataset_name}_search_results.csv")
    output_image = os.path.join(visualization_folder, f"{dataset_name}_visualization.png")

    # Check if the CSV file exists
    if not os.path.exists(csv_file):
        print(f"Error: The results file {csv_file} does not exist.")
        return

    # Load the search results
    search_df = pd.read_csv(csv_file)

    # Find the best performance value and its index
    best_performance = search_df["Performance"].min() if "---" not in dataset_name.lower() else search_df["Performance"].max()
    best_index = search_df[search_df["Performance"] == best_performance].index[0]

    # Create the plot
    plt.figure(figsize=(10, 6))

    # Plot the performance values
    plt.plot(search_df.index, search_df["Performance"], marker="o", linestyle="-", label="Performance")

    # Highlight the best point
    plt.plot(best_index, best_performance, marker="*", color="red", markersize=12, label="Best Point")

    # Add labels and title
    plt.xlabel("Search Iteration", fontsize=14)
    plt.ylabel("Performance", fontsize=14)
    plt.title(f"Search Results Visualization for {dataset_name}", fontsize=16)
    plt.legend()

    # Save and show the plot
    os.makedirs(visualization_folder, exist_ok=True)
    plt.savefig(output_image)
    plt.show()


def main():
    """
    Main function to generate visualizations for all datasets in the results folder.
    """
    results_folder = "search_results"
    visualization_folder = "visualization_results"

    if not os.path.exists(results_folder):
        print(f"Error: The folder {results_folder} does not exist.")
        return

    for file_name in os.listdir(results_folder):
        if file_name.endswith("_search_results.csv"):
            dataset_name = file_name.replace("_search_results.csv", "")
            visualize_search_results(results_folder, dataset_name, visualization_folder)


if __name__ == "__main__":
    main()
