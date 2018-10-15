#!/usr/bin/python
"""
@author: David Strickland
Course: CS5402 -Intro to Data Mining
Assignemnt: HW5
"""

import math
import random
import time
import tkinter as tk

######################################################################
# This section contains functions for loading CSV (comma separated values)
# files and convert them to a dataset of instances.
# Each instance is a tuple of attributes. The entire dataset is a list
# of tuples.
######################################################################
# Loads a CSV files into a list of tuples.
# Ignores the first row of the file (header).
# Numeric attributes are converted to floats, nominal attributes
# are represented with strings.
# Parameters:
#   fileName: name of the CSV file to be read
# Returns: a list of tuples
def loadCSV(fileName):
    fileHandler = open(fileName, "rt")
    lines = fileHandler.readlines()
    fileHandler.close()
    del lines[0] # remove the header
    dataset = []
    for line in lines:
        instance = lineToTuple(line)
        dataset.append(instance)
    return dataset

# Converts a comma separated string into a tuple
# Parameters
#   line: a string
# Returns: a tuple
def lineToTuple(line):
    # remove leading/trailing witespace and newlines
    cleanLine = line.strip()
    # get rid of quotes
    cleanLine = cleanLine.replace('"', '')
    # separate the fields
    lineList = cleanLine.split(",")
    # convert strings into numbers
    stringsToNumbers(lineList)
    lineTuple = tuple(lineList)
    return lineTuple

# Destructively converts all the string elements representing numbers
# to floating point numbers.
# Parameters:
#   myList: a list of strings
# Returns None
def stringsToNumbers(myList):
    for i in range(len(myList)):
        if (isValidNumberString(myList[i])):
            myList[i] = float(myList[i])
            
# Checks if a given string can be safely converted into a positive float.
# Parameters:
#   s: the string to be checked
# Returns: True if the string represents a positive float, False otherwise
def isValidNumberString(s):
    if len(s) == 0:
        return False
    if  len(s) > 1 and s[0] == "-":
        s = s[1:]
    for c in s:
        if c not in "0123456789.":
            return False
    return True

######################################################################
# This section contains functions for clustering a dataset
# using the k-means algorithm.
######################################################################
def euclideanDistance(instance1, instance2):
    if instance1 == None or instance2 == None:
        return float("inf")
    sumOfSquares = 0
    for i in range(1, len(instance1)):
        #print(i)
        #print("%s:  %s" % (type(instance1[i]), instance1[i]))
        #print("%s:  %s" % (type(instance2[i]), instance2[i]))
        sumOfSquares += (instance1[i] - instance2[i])**2
        #print(sumOfSquares)
    return math.sqrt(sumOfSquares)

def manhattanDistance(instance1, instance2):
    if instance1 == None or instance2 == None:
        return float("inf")
    absSum = 0
    for i in range(1, len(instance1)):
        absSum += (abs(instance1[i] - instance2[i]))
        #print(absSum)
    return absSum

def jaccardDistance(instance1, instance2):
    if instance1 == None or instance2 == None:
        return float("inf")
    sumJaccard = 0
    minSumJaccard = 0
    maxSumJaccard = 0
    for i in range(1, len(instance1)):
        minSumJaccard += min(instance1[i], instance2[i])
        maxSumJaccard += max(instance1[i], instance2[i])
    sumJaccard = minSumJaccard / maxSumJaccard
    return (1 - sumJaccard)

def cosineDistance(instance1, instance2):
    if instance1 == None or instance2 == None:
        return float("inf")
    sumOfSquaresA = 0
    sumOfSquaresB = 0
    sumOfAB = 0
    for i in range(1, len(instance1)):
        sumOfSquaresA += (instance1[i])**2
        sumOfSquaresB += (instance2[i])**2
        sumOfAB += (instance1[i] * instance2[i])
    cosineDist = (sumOfAB / (math.sqrt(sumOfSquaresA) * math.sqrt(sumOfSquaresB)))
    return(1 - cosineDist)

def meanInstance(name, instanceList):
    numInstances = len(instanceList)
    if (numInstances == 0):
        return
    numAttributes = len(instanceList[0])
    means = [name] + [0] * (numAttributes-1)
    for instance in instanceList:
        for i in range(1, numAttributes):
                means[i] += instance[i]
    for i in range(1, numAttributes):
        means[i] /= float(numInstances)
    return tuple(means)

def assign(instance, centroids, distance):
    if distance == "Euclidean":
        minDistance = euclideanDistance(instance, centroids[0])
        minDistanceIndex = 0
        for i in range(1, len(centroids)):
            d = euclideanDistance(instance, centroids[i])
            if (d < minDistance):
                minDistance = d
                minDistanceIndex = i

    elif distance == "Manhattan":
        minDistance = manhattanDistance(instance, centroids[0])
        minDistanceIndex = 0
        for i in range(1, len(centroids)):
            d = manhattanDistance(instance, centroids[i])
            if (d < minDistance):
                minDistance = d
                minDistanceIndex = i

    elif distance == "Jaccard":
        minDistance = jaccardDistance(instance, centroids[0])
        minDistanceIndex = 0
        for i in range(1, len(centroids)):
            d = jaccardDistance(instance, centroids[i])
            if (d < minDistance):
                minDistance = d
                minDistanceIndex = i

    elif distance == "Cosine":
        minDistance = cosineDistance(instance, centroids[0])
        minDistanceIndex = 0
        for i in range(1, len(centroids)):
            d = cosineDistance(instance, centroids[i])
            if (d < minDistance):
                minDistance = d
                minDistanceIndex = i

    return minDistanceIndex

def createEmptyListOfLists(numSubLists):
    myList = []
    for i in range(numSubLists):
        myList.append([])
    return myList

def assignAll(instances, centroids, distance):
    clusters = createEmptyListOfLists(len(centroids))
    for instance in instances:
        clusterIndex = assign(instance, centroids, distance)
        clusters[clusterIndex].append(instance)
    return clusters

def computeCentroids(clusters):
    centroids = []
    for i in range(len(clusters)):
        name = "centroid" + str(i)
        centroid = meanInstance(name, clusters[i])
        centroids.append(centroid)
    return centroids

def kmeans(instances, k, animation=False, initCentroids=None, distance="Euclidean"):
    result = {}
    if (initCentroids == None or len(initCentroids) < k):
        # randomly select k initial centroids
        random.seed(time.time())
        centroids = random.sample(instances, k)
    else:
        centroids = initCentroids
    prevCentroids = []
    if animation:
        delay = 1.0 # seconds
        canvas = prepareWindow(instances)
        clusters = createEmptyListOfLists(k)
        clusters[0] = instances
        paintClusters2D(canvas, clusters, centroids, "Initial centroids")
        time.sleep(delay)
    iteration = 0
    while (centroids != prevCentroids):
        iteration += 1
        clusters = assignAll(instances, centroids, distance)
        if animation:
            paintClusters2D(canvas, clusters, centroids, "Assign %d" % 
    iteration)
            time.sleep(delay)
        prevCentroids = centroids
        centroids = computeCentroids(clusters)
        computeSSE(clusters, centroids, distance)
        withinss = computeWithinss(clusters, centroids, distance)
        if animation:
            paintClusters2D(canvas, clusters, centroids,
        "Update %d, withinss %.1f" % (iteration, 
        withinss))
            time.sleep(delay)
    result["clusters"] = clusters
    result["centroids"] = centroids
    result["withinss"] = withinss
    return result

def computeSSE(clusters, centroids, distance):
    result = 0
    if distance == "Euclidean":
        for i in range(len(centroids)):
            result = 0
            centroid = centroids[i]
            cluster = clusters[i]
            for instance in cluster:
                result += (euclideanDistance(instance, centroid))**2
            print("Euclidean SSE, Cluster %s: %s" % (i, result))
                
    elif distance == "Manhattan":
        for i in range(len(centroids)):
            result = 0
            centroid = centroids[i]
            cluster = clusters[i]
            for instance in cluster:
                result += (manhattanDistance(instance, centroid))**2
            print("Manhattan SSE, Cluster %s: %s" % (i, result))
            
    elif distance == "Jaccard":
        for i in range(len(centroids)):
            result = 0
            centroid = centroids[i]
            cluster = clusters[i]
            for instance in cluster:
                result += (jaccardDistance(instance, centroid))**2
            print("Jaccard SSE, Cluster %s: %s" % (i, result))
            
    elif distance == "Cosine":
        for i in range(len(centroids)):
            result = 0
            centroid = centroids[i]
            cluster = clusters[i]
            for instance in cluster:
                result += (cosineDistance(instance, centroid))**2
            print("Cosine SSE, Cluster %s: %s" % (i, result))
    return result
                
def computeWithinss(clusters, centroids, distance):
    result = 0
    if distance == "Euclidean":
        for i in range(len(centroids)):
            centroid = centroids[i]
            cluster = clusters[i]
            for instance in cluster:
                result += euclideanDistance(centroid, instance)
                
    elif distance == "Manhattan":
        for i in range(len(centroids)):
            centroid = centroids[i]
            cluster = clusters[i]
            for instance in cluster:
                result += manhattanDistance(centroid, instance)
    
    elif distance == "Jaccard":
        for i in range(len(centroids)):
            centroid = centroids[i]
            cluster = clusters[i]
            for instance in cluster:
                result += jaccardDistance(centroid, instance)
        
    elif distance == "Cosine":
        for i in range(len(centroids)):
            centroid = centroids[i]
            cluster = clusters[i]
            for instance in cluster:
                result += cosineDistance(centroid, instance)
    return result

# Repeats k-means clustering n times, and returns the clustering
# with the smallest withinss
def repeatedKMeans(instances, k, n):
    bestClustering = {}
    bestClustering["withinss"] = float("inf")
    for i in range(1, n+1):
        print("k-means trial %d," % i)
        trialClustering = kmeans(instances, k)
        print("withinss: %.1f" % trialClustering["withinss"])
        if trialClustering["withinss"] < bestClustering["withinss"]:
            bestClustering = trialClustering
            minWithinssTrial = i
    print("Trial with minimum withinss:", minWithinssTrial)
    return bestClustering

######################################################################
# This section contains functions for visualizing datasets and
# clustered datasets.
######################################################################
def printTable(instances):
    for instance in instances:
        if instance != None:
            line = instance[0] + "\t"
            for i in range(1, len(instance)):
                line += "%.2f " % instance[i]
            print(line)

def extractAttribute(instances, index):
    result = []
    for instance in instances:
        #print(instances)
        #print(index)
        result.append(instance[index])
    return result

def paintCircle(canvas, xc, yc, r, color):
    canvas.create_oval(xc-r, yc-r, xc+r, yc+r, outline=color)

def paintSquare(canvas, xc, yc, r, color):
    canvas.create_rectangle(xc-r, yc-r, xc+r, yc+r, fill=color)

def drawPoints(canvas, instances, color, shape):
    random.seed(0)
    width = canvas.winfo_reqwidth()
    height = canvas.winfo_reqheight()
    margin = canvas.data["margin"]
    minX = canvas.data["minX"]
    minY = canvas.data["minY"]
    maxX = canvas.data["maxX"]
    maxY = canvas.data["maxY"]
    scaleX = float(width - 2*margin) / (maxX - minX)
    scaleY = float(height - 2*margin) / (maxY - minY)
    for instance in instances:
        x = 5*(random.random()-0.5)+margin+(instance[1]-minX)*scaleX
        y = 5*(random.random()-0.5)+height-margin-(instance[2]-
        minY)*scaleY
        if (shape == "square"):
            paintSquare(canvas, x, y, 5, color)
        else:
            paintCircle(canvas, x, y, 5, color)
    canvas.update()

def connectPoints(canvas, instances1, instances2, color):
    width = canvas.winfo_reqwidth()
    height = canvas.winfo_reqheight()
    margin = canvas.data["margin"]
    minX = canvas.data["minX"]
    minY = canvas.data["minY"]
    maxX = canvas.data["maxX"]
    maxY = canvas.data["maxY"]
    scaleX = float(width - 2*margin) / (maxX - minX)
    scaleY = float(height - 2*margin) / (maxY - minY)
    for p1 in instances1:
        for p2 in instances2:
            x1 = margin + (p1[1]-minX)*scaleX
            y1 = height - margin - (p1[2]-minY)*scaleY
            x2 = margin + (p2[1]-minX)*scaleX
            y2 = height - margin - (p2[2]-minY)*scaleY
            canvas.create_line(x1, y1, x2, y2, fill=color)
    canvas.update()

def mergeClusters(clusters):
    result = []
    for cluster in clusters:
        result.extend(cluster)
    return result

def prepareWindow(instances):
    width = 500
    height = 500
    margin = 50
    root = tk.Tk()
    canvas = tk.Canvas(root, width=width, height=height, background="white")
    canvas.pack()
    canvas.data = {}
    canvas.data["margin"] = margin
    setBounds2D(canvas, instances)
    paintAxes(canvas)
    canvas.update()
    return canvas

def setBounds2D(canvas, instances):
    attributeX = extractAttribute(instances, 1)
    attributeY = extractAttribute(instances, 2)
    canvas.data["minX"] = min(attributeX)
    canvas.data["minY"] = min(attributeY)
    canvas.data["maxX"] = max(attributeX)
    canvas.data["maxY"] = max(attributeY)

def paintAxes(canvas):
    width = canvas.winfo_reqwidth()
    height = canvas.winfo_reqheight()
    margin = canvas.data["margin"]
    minX = canvas.data["minX"]
    minY = canvas.data["minY"]
    maxX = canvas.data["maxX"]
    maxY = canvas.data["maxY"]
    canvas.create_line(margin/2, height-margin/2, width-5, height-
    margin/2,
    width=2, arrow=tk.LAST)
    canvas.create_text(margin, height-margin/4,
    text=str(minX), font="Sans 11")
    canvas.create_text(width-margin, height-margin/4,
    text=str(maxX), font="Sans 11")
    canvas.create_line(margin/2, height-margin/2, margin/2, 5,
    width=2, arrow=tk.LAST)
    canvas.create_text(margin/4, height-margin,
    text=str(minY), font="Sans 11", anchor=tk.W)
    canvas.create_text(margin/4, margin,
    text=str(maxY), font="Sans 11", anchor=tk.W)
    canvas.update()

def showDataset2D(instances):
    canvas = prepareWindow(instances)
    paintDataset2D(canvas, instances)

def paintDataset2D(canvas, instances):
    canvas.delete(tk.ALL)
    paintAxes(canvas)
    drawPoints(canvas, instances, "blue", "circle")
    canvas.update()

def showClusters2D(clusteringDictionary):
    clusters = clusteringDictionary["clusters"]
    centroids = clusteringDictionary["centroids"]
    withinss = clusteringDictionary["withinss"]
    canvas = prepareWindow(mergeClusters(clusters))
    paintClusters2D(canvas, clusters, centroids,
    "Withinss: %.1f" % withinss)

def paintClusters2D(canvas, clusters, centroids, title=""):
    canvas.delete(tk.ALL)
    paintAxes(canvas)
    colors = ["blue", "red", "green", "brown", "purple", "orange"]
    for clusterIndex in range(len(clusters)):
        color = colors[clusterIndex%len(colors)]
        instances = clusters[clusterIndex]
        centroid = centroids[clusterIndex]
        drawPoints(canvas, instances, color, "circle")
        if (centroid != None):
            drawPoints(canvas, [centroid], color, "square")
        connectPoints(canvas, [centroid], instances, color)
    width = canvas.winfo_reqwidth()
    canvas.create_text(width/2, 20, text=title, font="Sans 14")
    canvas.update()



def main():
    ######################################################################
    # Test code
    ######################################################################
    dataset = loadCSV("HW5_1.csv")
    #dataset = loadCSV("/home/davide/15-110/datasets/tshirts-H.csv")
    #dataset = loadCSV("/home/davide/15-110/datasets/tshirts-I.csv")
    #dataset = loadCSV("/home/davide/15-110/datasets/tshirts-J.csv")
    #dataset = loadCSV("data/tshirts-G-nooutliers.csv")
    showDataset2D(dataset)
    centroids1_1 = [[0,4,6],[1,5,4]]
    centroids1_3 = [[0,3,3],[1,8,3]]
    centroids1_4 = [[0,3,2],[1,4,8]]
    clustering1 = kmeans(dataset, 2, False, centroids1_1, "Euclidean")
    clustering2 = kmeans(dataset, 2, False, centroids1_1, "Manhattan")
    clustering3 = kmeans(dataset, 2, False, centroids1_3, "Euclidean")
    clustering4 = kmeans(dataset, 2, False, centroids1_4, "Euclidean")
    printTable(clustering1["centroids"])
    printTable(clustering2["centroids"])
    printTable(clustering3["centroids"])
    printTable(clustering4["centroids"])
    
    irisDataset = loadCSV("../HW4/iris.csv")
    newIrisDataset = []
    for row in irisDataset:
        rowList = list(row)
        del rowList[4]
        newIrisDataset.append(rowList)
    
    #print(newIrisDataset)
    clusteringEuclidean = kmeans(newIrisDataset, 4, False,distance = "Euclidean")
    clusteringCosine = kmeans(newIrisDataset, 4, False, distance = "Cosine")
    clusteringJaccard = kmeans(newIrisDataset, 4, False, distance = "Jaccard")
    
    #print(clusteringEuclidean)
    printTable(clusteringEuclidean["centroids"])
    printTable(clusteringCosine["centroids"])
    printTable(clusteringJaccard["centroids"])
if __name__ == "__main__":
    main()
    
    