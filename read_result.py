import csv
import statistics

def calculate_avg_std(filename):
    with open(filename, 'r') as csv_file:
        reader = csv.reader(csv_file)
        headers = next(reader)  # Read the headers
        headers = headers[1:]  # Remove the first column header

        data = [[] for _ in range(len(headers))]  # Create empty lists for each column

        for row in reader:
            for i, value in enumerate(row[1:]):
                data[i].append(float(value))  # Assuming all values are numeric, convert to float

        print("Column\t\tAverage\t\tStandard Deviation")
        print("--------------------------------------------")
        for i, column in enumerate(data):
            avg = statistics.mean(column)
            std = statistics.stdev(column)
            print(f"{headers[i]:<20}\t\t{avg:.4f}\t\t{std:.4f}")

# Example usage
import sys
calculate_avg_std(sys.argv[1])
