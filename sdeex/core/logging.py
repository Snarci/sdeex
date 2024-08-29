 
import os
import csv 

def log_to_csv(log_dir, log_file, data, header=None):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_file = os.path.join(log_dir, log_file)

    # Write the header if needed
    if not os.path.exists(log_file):
        with open(log_file, 'w', newline='') as f:
            if header:
                writer = csv.writer(f)
                writer.writerow(header)

    # Append data to the file
    with open(log_file, 'a', newline='') as f:
        writer = csv.writer(f)
        current_data = [data[key] for key in header]
        writer.writerow(current_data)



def log_to_console(data):
    #write to console in one row 
    acc_str = ''
    for key, value in data.items():
        acc_str += f'--{key}: {value}--'
    print(acc_str)


if __name__ == '__main__':
    log_dir = 'logs'
    log_file = 'log.csv'
    data = {'loss': 0.1, 'acc': 0.9}
    header = ['loss', 'acc']
    log_to_csv(log_dir, log_file, data, header)
    log_to_console(data)