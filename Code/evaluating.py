import pandas as pd
from sklearn.metrics import precision_score, recall_score, accuracy_score

# Read the Excel file
data = pd.read_excel('results_test.xlsx')

# Assuming 'output' and 'target' are the column names containing 0s and 1s
output_column = data['results']
target_column = data['target']

# Compute accuracy
total_samples = len(output_column)
correct_predictions = sum(output_column == target_column)
accuracy = correct_predictions / total_samples

precision = precision_score(target_column, output_column)
recall = recall_score(target_column, output_column)

print(f"Accuracy: {accuracy * 100:.2f}%")
