
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from collections import Counter

def distrib_fam_sizes(dataset):
    # Plot the distribution of family sizes
    f, ax = plt.subplots(figsize=(8, 5))
    sorted_targets = dataset.label.groupby(dataset.label).size().sort_values(ascending=False)
    sns.histplot(sorted_targets.values, kde=True, log_scale=True, ax=ax)
    plt.title("Distribution of family sizes for the 'train' split")
    plt.xlabel("Family size (log scale)")
    plt.ylabel("# Families")
    plt.savefig("images/family_size.png")
    # plt.show()

def distr_sequ_length(dataset):
    # Plot the distribution of sequences' lengths
    f, ax = plt.subplots(figsize=(8, 5))
    sequence_lengths = dataset.data.str.len()
    median = sequence_lengths.median()
    mean = sequence_lengths.mean()
    sns.histplot(sequence_lengths.values, kde=True, log_scale=True, bins=60, ax=ax)
    ax.axvline(mean, color='r', linestyle='-', label=f"Mean = {mean:.1f}")
    ax.axvline(median, color='g', linestyle='-', label=f"Median = {median:.1f}")
    plt.title("Distribution of sequence lengths")
    plt.xlabel("Sequence' length (log scale)")
    plt.ylabel("# Sequences")
    plt.legend(loc="best")
    plt.savefig("images/sequence_len.png")
    # plt.show()

def get_amino_acid_frequencies(data):

    aa_counter = Counter()
    for sequence in data:
        aa_counter.update(sequence)
    return pd.DataFrame({'AA': list(aa_counter.keys()), 'Frequency': list(aa_counter.values())})

def distrib_AA_freq(dataset):
    # Plot the distribution of AA frequencies
    f, ax = plt.subplots(figsize=(8, 5))
    amino_acid_counter = get_amino_acid_frequencies(dataset.data)
    sns.barplot(x='AA', y='Frequency', data=amino_acid_counter.
    sort_values(by=['Frequency'], ascending=False), ax=ax)
    plt.title("Distribution of AAs' frequencies in the 'train' split")
    plt.xlabel("Amino acid codes")
    plt.ylabel("Frequency (log scale)")
    plt.yscale("log")
    plt.savefig("images/aa_freq.png")
    # plt.show()