import pandas as pd
import matplotlib.pyplot as plt


def plot_total_timing(csv_file):
    data = pd.read_csv(csv_file)

    data['Total_time'] = data.sum(axis=1)
    # plt.figure(figs)
    plt.plot(data['Total_time'], marker='o')
    plt.title('Total Time per Iteration')
    plt.xlabel('Iteration')
    plt.ylabel('Total time (ms)')
    plt.grid(True)
    plt.savefig('total_timings_plot.png')
    plt.show()


def plot_means_bar_chart(csv_file):
    data = pd.read_csv(csv_file)
    means = data.iloc[:,:].mean()
    # means = data.iloc[:, 1:].mean()
    # labels = ["Kalman Filter", "Player Tracking", "Team Assignment"]
    labels = ["Homography Estimation", "Kalman Filter", "Player Tracking", "Team Assignment"]
    plt.figure(figsize=(8, 6))
    bars = means.plot(kind='bar')
    plt.title('Average Time of Components Execution (ms)')
    plt.grid(True)
    plt.xlabel('Columns')
    plt.ylabel('Average Time')
    plt.xticks(ticks=range(len(means)), labels=labels, rotation=0)  # Set custom labels

    for bar in bars.patches:
        bars.annotate(format(bar.get_height(), '.4f'),
                      (bar.get_x() + bar.get_width() / 2,
                       bar.get_height()), ha='center', va='center',
                      size=10, xytext=(0, 8),
                      textcoords='offset points')
    plt.savefig('bar_with.png')
    plt.show()


def plot_average_pie_chart(csv_file):
    # Read the CSV file
    data = pd.read_csv(csv_file)
    labels = [ "Kalman Filter", "Player Tracking", "Team Assignment"]
    # Calculate the average time for the specified columns (2nd to 4th)
    averages = data.iloc[:, 1:].mean()

    # Plotting a pie chart
    plt.figure(figsize=(8, 8))
    plt.pie(averages, labels=labels, autopct='%1.1f%%')
    plt.title('Average Time of Columns 2 to 4')
    plt.savefig('average_pie_chart.png')
    plt.show()

# Call the function
plot_means_bar_chart("../evaldata/component_timings.csv")