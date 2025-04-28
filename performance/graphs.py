import matplotlib.pyplot as plt

def parse_counters(filename):
    times = []
    with open(filename, "r") as file:
        for line in file:
            line = line.strip()
            times.append(float(line))

    return times

def graph_counters(seq_file, basic_file, fg_file):
    seq_times = parse_counters(seq_file)
    basic_times = parse_counters(basic_file)
    fg_times = parse_counters(fg_file)
    num_counters = [2 ** i for i in range(1, 6)]

    basic_speedup = [seq_times[i] / basic_times[i] for i in range(len(seq_times))]
    fg_speedup = [seq_times[i] / fg_times[i] for i in range(len(seq_times))]

    plt.figure()
    plt.plot(num_counters, basic_speedup, marker='o', linestyle='-', color='r', label="Basic GPU")
    plt.plot(num_counters, fg_speedup, marker='o', linestyle='-', color='b', label="Fine-Grain GPU")
    plt.xlabel("Number of Shared Counters")
    plt.ylabel("Speedup (T_sequential / T_parallel)")
    plt.xscale("log", basex=2)
    plt.xticks(num_counters, num_counters, rotation=45)
    plt.legend()
    plt.grid(True)
    plt.title("Speedup Comparison for Shared Counters Mini-Benchmark")
    plt.tight_layout()
    plt.savefig("shared_counters_speedup")
    plt.close()

graph_counters("shared_counters/seq.txt", "shared_counters/basic.txt", "shared_counters/fg.txt")