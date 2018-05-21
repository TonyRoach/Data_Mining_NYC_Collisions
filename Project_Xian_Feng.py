"""
author: Xian Feng
"""

import csv
import random
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
import math
import pandas as pd
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
import time

def read_csv(filename):
    """
    read data
    :param filename
    :return: data
    """
    data = pd.read_csv(filename,usecols=[0,1,4,5])
    data.head()

    return data

def select_data(data, start_year, end_year):
    # clean missing data
    data.dropna(axis=0, how='any', subset=['LATITUDE', 'LONGITUDE'], inplace=True)
    data.reset_index(drop=True, inplace=True)

    # choose subset data
    latitude = []
    longitude = []
    if end_year - start_year == 6:
        latitude = np.array(data["LATITUDE"])
        longitude = np.array(data["LONGITUDE"])
    else:
        for i in range(len(data["DATE"])):
            if start_year <= int(str(data["DATE"][i]).strip().split('/')[2]) <= end_year:
                latitude.append(data["LATITUDE"][i])
                longitude.append(data["LONGITUDE"][i])

        latitude = np.array(latitude)
        longitude = np.array(longitude)

    # remove outlier by using means and std.dev.
    x_upper = latitude.mean() + latitude.std() / 2
    x_lower = latitude.mean() - latitude.std() / 2

    y_upper = longitude.mean() + 0.5
    y_lower = longitude.mean() - 0.5

    a = []
    b = []
    for i in range(len(latitude)):

        if x_lower <= latitude[i] <= x_upper:
            if y_lower <= longitude[i] <= y_upper:
                a.append(latitude[i])
                b.append(longitude[i])


    X = pd.DataFrame({'LATITUDE': a, 'LONGITUDE': b})

    # find the best value of k
    k = find_best_K(X)

    # plot graph
    l(X, k, start_year, end_year)

def find_best_K(X):
    """
    using sample data to find k
    :param X: data
    :return: k
    """
    if len(X) <= 100000:
        k, d = find_K(X)
        plot_best_k(k,d)
        return k

    else:

        tol_K = []
        distortions=[]

        if len(X) > 500000:
            new_len = 5
        else:
            new_len = len(X)//100000

        for i in range(new_len):
            a = X.sample(100000)
            k, d = find_K(a)
            tol_K.append(k)
            distortions.append(d)

        cnt = Counter(tol_K)

        idx = 0
        for i in range(len(tol_K)):
            if tol_K[i] == cnt.most_common(1)[0][0]:
                idx = i
        plot_best_k(cnt.most_common(1)[0][0], distortions[idx])
        return cnt.most_common(1)[0][0]

def plot_best_k(k,d):
    # Plot best_k graph
    distortions = d
    K = range(1,10)
    plt.clf()
    plt.plot(K, distortions, 'bx-')
    plt.xlabel('k')
    plt.ylabel('SSE')
    plt.title('Find the best k')
    plt.savefig("Best_k.png")

def find_K(X):
    """
    find best k value from 1 tp 10
    :param X: data
    :return: k
    """

    distortions = []
    K = range(1, 10)

    for k in K:

        kmeans_model = KMeans(n_clusters=k).fit(X)

        labels = kmeans_model.labels_

        interia = kmeans_model.inertia_
        distortions.append(interia)

    # normalize
    distortions = [i / sum(distortions) for i in distortions]
    K = [i / sum(K) for i in K]

    dist = []
    for i in range(len(K)):
        dist.append(math.sqrt((K[i]) ** 2 + (distortions[i]) ** 2))
    k = dist.index(min(dist))
    return k, distortions

def l(X,k, start_year, end_year):
    """
    run k-means and plot it out
    :param X: data
    :param k: value
    :param start_year:
    :param end_year:
    :return:
    """
    # run KMeans
    id_n = k
    kmeans = KMeans(n_clusters=id_n, random_state=0).fit(X)
    id_label = kmeans.labels_

    plt.clf()
    ptsymb = np.array(['b.', 'r.', 'm.', 'g.', 'c.', 'k.', 'b*', 'r*', 'm*', 'r^'])
    plt.figure(figsize=(12, 12))
    plt.ylabel('Longitude', fontsize=12)
    plt.xlabel('Latitude', fontsize=12)
    plt.title('Accident Frequency on NYC from '+str(start_year)+' to '+str(end_year), fontsize=12)
    for i in range(id_n):
        cluster = np.where(id_label == i)[0]
        plt.plot(X['LATITUDE'][cluster].values, X['LONGITUDE'][cluster].values, ptsymb[i])

    plt.savefig('Accident_local_on_NYC.png')

def d(data, type):
    """
    analysis data against period of time
    :param data: data
    :param type: y-axis
    :return: none
    """
    date = np.array(data['DATE'])
    if type == 'd':

        total_date=[]
        for j in date:
            j = str(j).strip().split('/')
            total_date.append(int(j[1]))

        bins = np.arange(0, 32) + 0.5
        plt.hist(total_date, bins=bins ,edgecolor='black', linewidth=1.2)
        plt.xticks(range(1,32,2))
        plt.title("Daily of Accidents Frequency from 2012 to 2018")
        plt.ylabel('Accidents Frequency')
        plt.xlabel('Days')
        plt.savefig("Daily_Accidents.png")

    elif type == 'm':
        total_date = []
        for j in date:
            j = str(j).strip().split('/')
            total_date.append(int(j[0]))
        bins = np.arange(0, 13) + 0.5
        plt.clf()
        plt.hist(total_date, bins=bins, edgecolor='black', linewidth=1.2)
        plt.xticks(range(0, 13))
        plt.title("Monthly of Accidents Frequency from 2012 to 2018")
        plt.ylabel('Accidents Frequency')
        plt.xlabel('Month')
        plt.savefig("Month_Accidents.png")

    elif type == 'y':
        total_date = []
        for j in date:
            j = str(j).strip().split('/')
            total_date.append(int(j[2]))
        bins = np.arange(2011,2019)+0.5
        plt.clf()
        plt.hist(total_date, bins=bins, edgecolor='black', linewidth=1.2)
        plt.xticks(range(2011, 2019))
        plt.title("Yearly of Accidents Frequency from 2012 to 2018")
        plt.ylabel('Accidents Frequency')
        plt.xlabel('Years')
        plt.savefig("Yearly_Accidents.png")

    elif type == 'h':
        total_date = []
        Time = data['TIME']
        for j in Time:
            j = str(j).strip().split(':')
            total_date.append(int(j[0]))
        bins = np.arange(0,26,2)
        plt.hist(total_date, bins=bins, edgecolor='black', linewidth=1.2)
        plt.xticks(range(0, 26, 2))
        plt.title("Hourly of Accidents Frequency from 2012 to 2018")
        plt.ylabel('Accidents Frequency')
        plt.xlabel('Hours')
        plt.savefig("Hourly_Accidents.png")

def main():
    data = read_csv("NYPD_Motor_Vehicle_Collisions.csv")

    print("Starting Analysing accidents frequency against time period...")
    data_type = input("Enter the x-axis for time period(h for hourly, d for daily, m for monthly, y for yearly):")
    d(data, data_type)
    print("Result was plotted as a graph.")
    print("Starting Analysing accidents frequency against location...")
    start_year = int(input("Enter the start year of data set(from 2012 to 2018):"))
    end_year = int(input("Enter the end year of data set(from 2012 to 2018):"))

    if start_year > end_year:
        raise Exception("Start year larger than End year.")
    if 2012 > int(start_year) > 2018 or 2012 > int(end_year) > 2018:
        raise Exception("Input out of range.")

    select_data(data,start_year,end_year)
    print("Finish")


if __name__ == '__main__':
    s = time.time()
    main()
    e = time.time()
    print("time cost:",e-s)