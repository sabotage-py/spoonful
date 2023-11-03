import os
import cv2 as cv
import networkx as nx

from helpers import is_transparent, is_white_bg, get_front_area, sort_tiltness


def get_num_matches(descriptors_1, descriptors_2):
    """input: SIFT descriptors"""
    bf = cv.BFMatcher()
    matches = bf.knnMatch(descriptors_1, descriptors_2, k=2)
    good_matches = []
    # apply ratio test
    for match1, match2 in matches:
        if match1.distance < 0.65 * match2.distance:
            good_matches.append([match1])
            # 0.65 can be replaced with other values
            # but then comparison_constant in cluster_similar_images
            # will need to be changed as well
    return len(good_matches)


def cluster_similar_images(path):
    """input: path location
    runs the deduplication algo on all the images in the path and
    returns a subset of the largest cluster of similar images in sorted
    order, based on the criteria of stage 3
    """
    comparison_constant = 0.25  
    # this value is used to compare the images.
    # 0.25 has been selected after many manual experiments
    red = os.listdir(path)
    graph = nx.Graph()
    edges = []
    images_orig = {i: 
        cv.imread(os.path.join(path, img_name), cv.IMREAD_UNCHANGED) 
        for i, img_name in enumerate(red)
    }
    images_gray = {i:
        cv.imread(os.path.join(path, img_name), cv.IMREAD_GRAYSCALE)
        for i, img_name in enumerate(red)
    }
    sift = cv.SIFT_create()
    keypoints_descriptors = {i:
        sift.detectAndCompute(images_gray[i], None) 
        for i in range(len(red))
    }

    # compare all the images with each other
    # and get 'similarity value' between them
    for i in range(len(red)-1):
        keypoints_i, descriptors_i = keypoints_descriptors[i]
        num_keypoints_i = len(keypoints_i)
        for j in range(i+1, len(red)):
            keypoints_j, descriptors_j = keypoints_descriptors[j]
            num_keypoints_j = len(keypoints_j)
            min_keypoints = min(num_keypoints_i, num_keypoints_j)
            ratio_i2j = get_num_matches(descriptors_i, descriptors_j) \
                / min_keypoints
            ratio_j2i = get_num_matches(descriptors_j, descriptors_i) \
                / min_keypoints
            if (ratio_i2j > comparison_constant 
                    or ratio_j2i > comparison_constant):
                edges.append([i, j])
                # the value 0.25 has been selected after
                # many manual experiments
    # create a graph with indices as nodes
    graph.add_edges_from(edges)
    clique_list = list(nx.find_cliques(graph))  # list of cliques
    original_cliques = clique_list[:]  # create a copy for later use.
    clique_list.sort(key=lambda w: len(w), reverse=True)
    if not clique_list:
        return None
    nodes = list(graph.nodes)
    new_edges = []

    # add an edge between two disconnected nodes
    # if they have significant number of common neighbours
    for i in range(len(red) - 1):
        if i not in nodes:
            continue
        nbrs = list(graph.neighbors(i))
        for j in range(i+1, len(red)):
            if j in nbrs or (j not in nodes): 
                continue
            if (len(list(nx.common_neighbors(graph, i, j)))
                    >= len(clique_list[0]) // 2):
                new_edges.append([i, j])
    graph.add_edges_from(new_edges)

    # get cliques of the updated graph
    clique_list = list(nx.find_cliques(graph))
    clique_list.sort(key=lambda w: len(w), reverse=True)
    prediction = clique_list[0]  # this is our required cluster 
    #                              of similar images
    threshold_size = 299
    temp_pred = []
    del_pred = []
    for p in prediction:
        img = images_orig[p]
        x, y = img.shape[:2]
        if max(x, y) > threshold_size and not is_transparent(img):
            temp_pred.append(p)
        else:
            # these do not satisfy our criteria
            del_pred.append(p)
    prediction = temp_pred[:]  # the ones that satisfy criteria
    not_white, temp_pred = [], []
    for p in prediction:
        img = images_orig[p]
        if is_white_bg(img):
            temp_pred.append(p)
        else:
            not_white.append(p)
    prediction = temp_pred[:]
    if not prediction:
        return None
    ncliques = dict()  
    # ncliques[p] = no. of diff cliques p is part of in the orig graph
    for p in prediction:
        count_cliques = 0
        for cliq in original_cliques:
            if p in cliq:
                count_cliques += 1
        ncliques[p] = count_cliques
    max_cliques = max(ncliques.values())
    new_pred = [
        p for p in prediction if ncliques[p] == max_cliques
    ]
    small_area = []
    prediction = []
    front_area = dict()
    for p in new_pred:
        img = images_gray[p]
        front_area[p] = get_front_area(img)
    largest_area = max(front_area.values())
    for p in new_pred:
        if front_area[p]/largest_area >= 0.67:
            prediction.append(p)
        else:
            small_area.append(p)
    if prediction:
        sorted_new_pred = sort_tiltness(path, prediction)
        return sorted_new_pred
    return None


def get_candidate(path, candidates_original, label='back'):
    """input: path location and indices of the images
    from which the ideal image needs to be selected.
    """
    candidates = candidates_original.copy()
    red = os.listdir(path)
    images_orig = {candidate: cv.imread(
        os.path.join(path, red[candidate]), cv.IMREAD_UNCHANGED) 
        for candidate in candidates
    }
    threshold_size = 299
    new_candidates, transparent_candidates = [], []
    for c in candidates:
        img = images_orig[c]
        if isinstance(img, type(None)):
            continue
        x, y = img.shape[:2]
        if max(x, y) > threshold_size and not is_transparent(img):
            new_candidates.append(c)
        elif max(x, y) > threshold_size:
            transparent_candidates.append(c)
    candidates = new_candidates[:]
    are_candidates_transparent = False
    if not candidates:
        # there are no images with non-transparent background
        candidates = transparent_candidates[:]
        are_candidates_transparent = True
    if not (are_candidates_transparent or label == 'back'):
        # if the candidates do not have transparent bg
        # then check for white bg
        not_white, new_candidates = [], []
        for c in candidates:
            img = images_orig[c]
            if is_white_bg(img):
                new_candidates.append(c)
            else:
                not_white.append(c)
        candidates = new_candidates[:]
        if not candidates:
            candidates = not_white[:]
    if candidates:
        # sort_reverse = True if label == 'back' else False
        sort_reverse = False
        candidates = sort_tiltness(path, candidates, 
                                   reverse=sort_reverse)
        return candidates
    return None
