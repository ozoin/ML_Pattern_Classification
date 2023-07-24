import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import re
import pandas as pd


def non_zero_sequences(lst):
    result = []
    sequence = []
    for num in lst:
        if num != 0:
            sequence.append(num)
        elif sequence:
            result.append(sequence)
            sequence = []
    if sequence:
        result.append(sequence)
    return result


class BirdsData:
    def __init__(self, project_folder):
        self.project_folder = project_folder

    # simply retrieve the features and corresponding label for given sample of specified feature
    def get_sound_label(self, species, sample):
        folder_path = os.path.join(self.project_folder, species)
        file_list = os.listdir(folder_path)
        for file in file_list:
            if file.endswith('.npy'):
                file_path = os.path.join(folder_path, sample)
                data = np.load(f'{file_path}.npy')
                label = np.load(f'{file_path}.labels.npy')
                return np.concatenate((data, label[:, 0].reshape(label.shape[0], 1)), axis=1)

    # method to create one huge dataset

    def united_dataset(self):

        birds = [os.path.join(self.project_folder, b)
                 for b in os.listdir(self.project_folder)]
        all_files = [os.path.join(bird, f)
                     for bird in birds for f in os.listdir(bird)]
        all_files = sorted([f for f in all_files if not 'labels' in f])

        comcuc = [f for f in all_files if 'comcuc' in f]
        comcuc_features = [
            str(re.search(r"\d+(?=.npy)", s).group(0)) for s in comcuc]
        comcuc_ars = []
        for f in comcuc_features:
            comcuc_ars.append(
                BirdsData(self.project_folder).get_sound_label('comcuc', f))
        comcuc_ars = np.concatenate(comcuc_ars)

        cowpig1 = [f for f in all_files if 'cowpig1' in f]
        cowpig1_features = [
            str(re.search(r"\d+(?=.npy)", s).group(0)) for s in cowpig1]
        cowpig1_ars = []
        for f in cowpig1_features:
            cowpig1_ars.append(
                BirdsData(self.project_folder).get_sound_label('cowpig1', f))
        cowpig1_ars = np.concatenate(cowpig1_ars)

        eucdov = [f for f in all_files if 'eucdov' in f]
        eucdov_features = [
            str(re.search(r"\d+(?=.npy)", s).group(0)) for s in eucdov]
        eucdov_ars = []
        for f in eucdov_features:
            eucdov_ars.append(
                BirdsData(self.project_folder).get_sound_label('eucdov', f))
        eucdov_ars = np.concatenate(eucdov_ars)

        eueowl1 = [f for f in all_files if 'eueowl1' in f]
        eueowl1_features = [
            str(re.search(r"\d+(?=.npy)", s).group(0)) for s in eueowl1]
        eueowl1_ars = []
        for f in eueowl1_features:
            eueowl1_ars.append(
                BirdsData(self.project_folder).get_sound_label('eueowl1', f))
        eueowl1_ars = np.concatenate(eueowl1_ars)

        grswoo = [f for f in all_files if 'grswoo' in f]
        grswoo_features = [
            str(re.search(r"\d+(?=.npy)", s).group(0)) for s in grswoo]
        grswoo_ars = []
        for f in grswoo_features:
            grswoo_ars.append(
                BirdsData(self.project_folder).get_sound_label('grswoo', f))
        grswoo_ars = np.concatenate(grswoo_ars)

        tawowl1 = [f for f in all_files if 'tawowl1' in f]
        tawowl1_features = [
            str(re.search(r"\d+(?=.npy)", s).group(0)) for s in tawowl1]
        tawowl1_ars = []
        for f in tawowl1_features:
            tawowl1_ars.append(
                BirdsData(self.project_folder).get_sound_label('tawowl1', f))
        tawowl1_ars = np.concatenate(tawowl1_ars)

        all_birdiiies = np.concatenate(
            (comcuc_ars, cowpig1_ars, eucdov_ars, eueowl1_ars, grswoo_ars, tawowl1_ars))

        return all_birdiiies

    def get_dataframe(self):
        dataframe = pd.DataFrame(BirdsData('ptichki').united_dataset())
        with open('feature_names.txt', 'r') as f:
            lines = [line.rstrip() for line in f.readlines()]
        cols = []
        for line in lines:
            cols.append(line)
        cols.append('target')
        dataframe.columns = cols

        targets = {
            0: 'other',
            1: 'comcuc',
            2: 'cowpig1',
            3: 'eucdov',
            4: 'eueowl1',
            5: 'grswoo',
            6: 'tawowl1'
        }

        dataframe.loc[:, 'target'] = dataframe.loc[:, 'target'].map(targets)

        return dataframe

    def labels_distribution(self):
        dataset = BirdsData(self.project_folder).united_dataset()
        others_count = len(dataset[dataset[:, -1] == 0])
        class_counts = []
        for i in range(0, 7):
            class_counts.append(len(dataset[dataset[:, -1] == i]))
        x = ['other', 'comcuc', 'cowpig1', 'eucdov',
             'eueowl1', 'grswoo', 'tawowl1']
        y = class_counts
        plt.figure(figsize=(10, 7))
        plt.plot(x, y)
        plt.title('Plot of labels distribution')
        plt.xlabel('Species (labels)')
        plt.ylabel('Samples')
        plt.show()

        return class_counts

    # potential method to complete "annotators agreement" part. Returns general agreement, positive agreement (how
    # many annotators think positive class to be positive, negative agreement (how many annotators think "others" part
    # to be "others"

    def compute_agreement(self, species):
        folder_path = os.path.join(self.project_folder, species)
        file_list = os.listdir(folder_path)
        labels = [f for f in file_list if 'labels' in f]
        labels_paths = [os.path.join(folder_path, i) for i in labels]
        labels_ar_list = []
        for i in range(len(labels_paths)):
            labels_ar_list.append(np.load(labels_paths[i]))
        general_agreements = 0
        for i in range(len(labels_ar_list)):
            for j in range(0, 100):
                # index of label which occurs the most
                general_agreements += np.mean(labels_ar_list[i][j] == np.argmax(
                    np.bincount(labels_ar_list[i][j][1:])))

        positive_agreements = 0
        count_p = 0
        negative_agreements = 0
        count_n = 0
        for i in range(len(labels_ar_list)):
            for j in range(0, 100):
                if labels_ar_list[i][j][0] == 0:
                    negative_agreements += np.mean(
                        labels_ar_list[i][j] == np.argmax(np.bincount(labels_ar_list[i][j][1:])))
                    count_n += 1
                else:
                    positive_agreements += np.mean(
                        labels_ar_list[i][j] == np.argmax(np.bincount(labels_ar_list[i][j][1:])))
                    count_p += 1

        general_agreements = general_agreements / (len(labels) * 100)
        positive_agreements = positive_agreements / count_p
        negative_agreements = negative_agreements / count_n
        return np.array([general_agreements, positive_agreements, negative_agreements])

    # Method below might be of use to further calculate which features are important and which are not.

    def compute_feature_correlations_for_species(self, species):
        # folder_path = f'{self.project_folder}/{species}'
        # file_list = os.listdir(f'{self.project_folder}/{species}')
        folder_path = os.path.join(self.project_folder, species)
        file_list = os.listdir(folder_path)
        features = [f for f in file_list if not 'labels' in f]
        cor_mat_list = []
        file_paths = [os.path.join(folder_path, i) for i in features]
        for i in range(len(file_paths)):
            cor_mat_list.append(np.corrcoef(np.load(file_paths[i]).T))
        return np.mean(cor_mat_list, axis=0)

    def plot_cor_distribution(self, species, save=False):
        plt.figure(figsize=(14, 10))
        y = np.unique(BirdsData(
            self.project_folder).compute_feature_correlations_for_species(species))
        plt.plot(y)
        plt.grid()
        plt.show()
        if save:
            plt.savefig(self.project_folder)

    def species_calls(self, species):
        folder_path = os.path.join(self.project_folder, species)
        file_list = os.listdir(folder_path)
        labels = [f for f in file_list if 'labels' in f]
        labels_paths = [os.path.join(folder_path, i) for i in labels]
        labels_ar_list = []
        for i in range(len(labels_paths)):
            labels_ar_list.append(np.load(labels_paths[i]))
        labels_list = []
        for i in range(len(labels_ar_list)):
            for j in range(0, 100):
                labels_list.append(labels_ar_list[i][j][0])
        nonz = non_zero_sequences(labels_list)
        for i in range(len(nonz)):
            for j in range(len(nonz[i])):
                nonz[i][j] = 0.2
        for l in range(len(nonz)):
            nonz[l] = round(sum(nonz[l]), 3)

        return nonz

    def species_call_distribution(self):
        species = ['comcuc', 'cowpig1', 'eucdov',
                   'eueowl1', 'grswoo', 'tawowl1']
        plt.figure(figsize=(12, 7))
        data = [BirdsData('ptichki').species_calls(b) for b in species]
        plt.boxplot(data, showmeans=True)
        plt.xticks([1, 2, 3, 4, 5, 6], species)
        plt.ylabel('Duration of call/drumming(sec.)')
        plt.title('Distribution of duration of call/drumming for each species')
        plt.show()

    def plot_duration_means(self):
        species = ['comcuc', 'cowpig1', 'eucdov',
                   'eueowl1', 'grswoo', 'tawowl1']
        plt.figure(figsize=(12, 7))
        data = [np.mean(BirdsData('ptichki').species_calls(b))
                for b in species]
        plt.plot(species, data)
        plt.title('Average duration of drumming for each bird(sec.)')
        plt.ylabel('Duration(sec.)')
        plt.grid()
        plt.show()

    def cor_feat_label(self):
        c = BirdsData('ptichki')
        X = c.united_dataset()[:, 0:548]
        y = c.united_dataset()[:, -1]
        cors = [np.corrcoef(x=X[:, i], y=y)[1, 0] for i in range(548)]
        plt.figure(figsize=(14, 10))
        plt.plot(cors)
        plt.xlabel('features')
        plt.ylabel('correlation with target')
        plt.title('Feature correletion rates with targets')
        idcs = np.argsort(cors)
        feat_names = BirdsData(
            self.project_folder).get_dataframe().columns[0:-1]
        top_10 = feat_names[idcs[538:]]
        print(top_10[::-1])
        plt.show()
        return top_10[::-1]

    def find_classification_distibution(self, species):
        folder_path = os.path.join(self.project_folder, species)
        file_list = os.listdir(folder_path)
        labels = [f for f in file_list if 'labels' in f]
        labels_paths = [os.path.join(folder_path, i) for i in labels]
        labels_ar_list = []
        for i in range(len(labels_paths)):
            labels_ar_list.append(np.load(labels_paths[i]))
        return np.array(labels_ar_list)

    def parallel_coordinates(self):
        df = BirdsData(self.project_folder).get_dataframe()
        plt.figure(figsize=(14, 10))
        pd.plotting.parallel_coordinates(df, 'target', axvlines=False)
        plt.show()

    def feature_distriburutions(self):

        bird = BirdsData('ptichki')
        df = bird.get_dataframe()
        df = df.loc[:, df.columns != 'target']

        medians = []
        for i in range(0, 548):
            medians.append(np.median(df.iloc[:, i]))

        means = []
        for i in range(0, 548):
            means.append(np.mean(df.iloc[:, i]))

        stds = []
        for i in range(0, 548):
            stds.append(np.var(df.iloc[:, i]))

        fig = plt.figure(figsize=(10, 9))

        fig.add_subplot(3, 1, 1)
        plt.plot(medians)
        plt.title('Median of each feature')
        plt.ylabel('Median')

        fig.add_subplot(3, 1, 2)
        plt.plot(means)
        plt.title('Arithmetic mean each feature')
        plt.ylabel('Arithmetic mean')

        fig.add_subplot(3, 1, 3)
        plt.plot(stds)
        plt.title('Standard deviation of each feature')
        plt.ylabel('Standard deviation')

        fig.tight_layout(pad=5.0)

        plt.plot()

        plt.show()


bird = BirdsData('bird_data')
df = bird.get_dataframe()
df = df.loc[:, df.columns != 'target']


def get_redundant_pairs(df):
    '''Get diagonal and lower triangular pairs of correlation matrix'''
    pairs_to_drop = set()
    cols = df.columns
    for i in range(0, df.shape[1]):
        for j in range(0, i+1):
            pairs_to_drop.add((cols[i], cols[j]))
    return pairs_to_drop


def get_top_abs_correlations(df, n=5):
    au_corr = df.corr().abs().unstack()
    labels_to_drop = get_redundant_pairs(df)
    au_corr = au_corr.drop(labels=labels_to_drop).sort_values(ascending=True)
    return au_corr[0:n]


bird.feature_distriburutions()
