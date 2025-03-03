# Requirements
1.Install any versions of Python 3
2.Install any required packages using pip in terminal, including pandas, numpy，tensorflow and scikit-learn 

# How to run the code
1. **Open the Python file with any IDE**:
   - Launch your preferred Python IDE (e.g., PyCharm, VSCode, etc.).
   - Open the `main.py` file.

2. **Change any parameters to adjust the experiment settings**:
   - **Change the dataset file**:
     - Update the dataset file path ('dataset/processed_adult.csv') in the code to point to the dataset.
   - **Change the DNN model**:
     - Replace the DNN model with the appropriate **serialized model** file.
     - Ensure that the model is loaded using the correct method (e.g., `keras.models.load_model` for Keras models).
     ```python
     from keras.models import load_model
     model = load_model('DNN/model_processed_adult.h5')
     ```
   - **Modify `load_and_preprocess_data` function**:
     - In the `load_and_preprocess_data` function, update the **target variable** to match the target feature of the selected dataset.
     - For example, if the adult dataset has a target variable named `Class-label`, change the function to load `label` as the target:
       ```python
       target = 'Class-label'  # Replace with your dataset's target column name
       ```

3. **Click 'Run' button**:
   - Click the **Run** button in the IDE (typically a green play button).
   - The results will be printed in the console or terminal window.
