# Euclidean distance
def distance_euclidean(x_train, x_test_point):
    distances = []
    for row in range(len(x_train)):
        current_train_point = x_train[row]
        current_distance = np.sum((current_train_point - x_test_point) ** 2)
        distances.append(np.sqrt(current_distance))
    
    dist = pd.DataFrame(data=distances, columns=['dist'])
    return dist

# Manhattan distance
def distance_manhattan(x_train, x_test_point):
    distances = []
    for row in range(len(x_train)):
        current_train_point = x_train[row]
        current_distance = np.sum(np.abs(current_train_point - x_test_point))
        distances.append(current_distance)
    return pd.DataFrame(data=distances, columns=['dist'])

# Minkowski distance
def distance_minkowski(x_train, x_test_point, p=3):
    distances = []
    for row in range(len(x_train)):
        current_train_point = x_train[row]
        current_distance = np.sum(np.abs(current_train_point - x_test_point) ** p)
        distances.append(np.power(current_distance, 1/p))  # Minkowski mesafesi
    return pd.DataFrame(data=distances, columns=['dist'])

# Finding the nearest neighbors
def nearest_neighbors(distance_point, K):
    df_nearest = distance_point.sort_values(by=['dist'], axis=0)
    df_nearest = df_nearest[:K]
    return df_nearest

# Finding the most common class in the neighbors
def voting(df_nearest, y_train):
    neighbors_classes = y_train[df_nearest.index] # Finding the class of the neighbors
    frequency = {}
    for label in neighbors_classes:
        if label in frequency:
            frequency[label] += 1
        else:
            frequency[label] = 1
    
    vote = max(frequency, key=frequency.get) # finding the most common
    return vote


# k-NN Algorithm
def KNN_from_scratch(x_train, y_train, x_test, K, distance_metric='euclidean'):
    y_pred = []
    for x_test_point in x_test:
        if distance_metric == 'euclidean':
            distance_point = distance_euclidean(x_train, x_test_point)
        elif distance_metric == 'manhattan':
            distance_point = distance_manhattan(x_train, x_test_point)
        elif distance_metric == 'minkowski':
            distance_point = distance_minkowski(x_train, x_test_point)

        df_nearest = nearest_neighbors(distance_point, K)
        y_pred_point = voting(df_nearest, y_train)
        y_pred.append(y_pred_point)

    return y_pred