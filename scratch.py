from collections import Counter

# Given list of types
types = ['phishing', 'benign', 'benign', 'defacement', 'defacement', 'benign', 'benign', 'benign', 'defacement', 'benign', 'benign', 'defacement', 'benign', 'defacement', 'defacement', 'defacement', 'benign', 'benign', 'benign', 'defacement', 'benign', 'phishing', 'benign', 'benign', 'benign', 'benign', 'benign', 'benign', 'phishing', 'benign', 'benign', 'benign', 'benign', 'benign', 'benign', 'defacement', 'benign', 'benign', 'malware', 'defacement', 'phishing', 'benign', 'benign', 'benign', 'benign', 'benign', 'defacement', 'benign', 'benign', 'benign', 'defacement', 'defacement', 'benign', 'benign', 'defacement', 'benign', 'benign', 'benign', 'benign', 'benign', 'phishing', 'defacement', 'benign', 'benign', 'benign', 'defacement', 'benign', 'benign', 'defacement']

# Calculate the count and percentage of each type
type_counts = Counter(types)
total_samples = len(types)

percentage_per_type = {key: (value / total_samples) * 100 for key, value in type_counts.items()}

# Display the results
print("Type\t\tCount\t\tPercentage")
print("-----------------------------------")
for key, value in type_counts.items():
    percentage = percentage_per_type[key]
    print(f"{key}\t\t{value}\t\t{percentage:.2f}%")

# Display the overall sum
print("\nOverall Sum:")
print(f"Total Samples: {total_samples}")
