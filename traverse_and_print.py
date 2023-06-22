import os
import csv
import statistics

def calculate_avg_std(filename):
    with open(filename, 'r') as csv_file:
        reader = csv.reader(csv_file)
        headers = next(reader)  # Read the headers
        headers = headers[1:]  # Remove the first column header

        data = [[] for _ in range(len(headers))]  # Create empty lists for each column

        k = 0
        for row in reader:
            for i, value in enumerate(row[1:]):
                try:
                    data[i].append(float(value))  # Assuming all values are numeric, convert to float
                except:
                    data[i].append(0.0)
            k += 1

        print(f"number of lines: {k}")
        if k < 2:
            print("not enough data")
            return

        print("Column\t\tAverage\t\tStandard Deviation")
        print("--------------------------------------------")
        for i, column in enumerate(data):
            avg = statistics.mean(column)
            std = statistics.stdev(column)
            print(f"{headers[i]:<20}\t\t{avg:.4f}\t\t{std:.4f}")


def traverse_files(directory):
    result_list = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            if (
                True
                # file_path[8] == '_' 
                and "mushroom" in file_path 
                and "_nf" in file_path
                and "MULTIGBAG" in file_path
                and "DAGBAG"  not in file_path
                and "JASGBAG"  not in file_path
                and "JASDAGBAG" not in file_path
                ):
                result_list.append(file_path)

    result_list.sort()
    for file_path in result_list:
        print(file_path)
        calculate_avg_std(file_path)  # Or perform any other operation on the file

# Usage example
directory_path = 'results'
traverse_files(directory_path)