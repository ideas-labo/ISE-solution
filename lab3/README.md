## Project Overview
This project provides a framework for performing random search on configurable systems and visualizing the results. The tool allows users to evaluate configurations across different datasets, identify optimal configurations, and generate visualizations to aid in performance analysis.

## Features
- Random search for exploring configurations in datasets.
- Automatic handling of maximization and minimization problems.
- Visualization of search results with performance trends and optimal points.
- Clear organization of datasets, search results, and visualization outputs.

## Installation

### 1. Clone the Repository
```bash
git clone https://github.com/ideas-labo/ISE-s.git
cd lab3
```

### 2. Install Dependencies
Ensure `pip` is up-to-date and install required packages:
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

## Dataset Description
The `datasets` folder contains CSV files representing different configurable systems. Each CSV file has the following structure:

- **Columns (1 to n-1):** Configuration parameters (discrete or continuous values).

- **Column n:** Performance objective (a numeric value).

### Included Systems
| System       | Optimization Type  |
|--------------|--------------------|
| 7z           | Minimization       |
| Apache       | Minimization       |
| Brotli       | Minimization       |
| LLVM         | Minimization       |
| PostgreSQL   | Minimization       |
| Spear        | Minimization       |
| Storm        | Minimization       |
| x264         | Minimization       |

- **Minimization Problems:** Lower performance values are better.
- **Maximization Problems:** Higher performance values are better.

## Usage

### 1. Perform Random Search
Run the main script to perform random search on all datasets:
```bash
python main.py
```
The search results for each dataset will be stored in the `search_results` folder as CSV files.

### 2. Visualize Search Results
Run the visualization script to generate performance plots for all datasets:
```bash
python visualize_search_results.py
```
The visualizations are automatically generated and stored in the `visualization_results` folder after running script. Each dataset's visualization shows:

- Performance trends across search iterations.

- The optimal performance point highlighted in red *.

### 3. Execute via IDE (Optional)
If you prefer using an IDE like PyCharm, you can:

- Run `main.py` to execute the random search process.

- Run `visualize_search_results.py` to generate visualizations of the results.

### 4. Customize Search Budget
You can adjust the search budget by modifying the `budget` variable in the main script.



## Project Structure
```
project-folder/
├── datasets/               # Contains input datasets (CSV files).
├── search_results/         # Stores search results after running the script.
├── visualization_results/  # Stores visualizations of search results.
├── main.py                 # Main script to run random search and generate results.
├── visualize_search_results.py  # Script for generating visualizations.
├── requirements.txt        # Python dependencies.
└── README.md               # Project documentation.
```

## Notes
- For non-existing configurations during the search, the tool assigns:
  - **Minimization Problems:** Twice the maximum performance value in the dataset.
  - **Maximization Problems:** Half the minimum performance value in the dataset.

- Ensure datasets are formatted correctly with valid configuration and performance columns.


