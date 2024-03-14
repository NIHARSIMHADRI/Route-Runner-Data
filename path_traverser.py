import requests
from math import radians, sin, cos, sqrt, atan2
from sklearn.neighbors import BallTree
import numpy as np
import random
from scipy.stats import gaussian_kde
from scipy.spatial import Voronoi, voronoi_plot_2d
from rtree import index
from multiprocessing import Pool
import math
from concurrent.futures import ProcessPoolExecutor
import time
from api_code import api_weather_passcode

def calculate_bounds(latitude, longitude, distance_miles):
    """
    Calculate the latitude and longitude bounds for a given distance from a point.
    
    Parameters:
    - latitude (float): The starting latitude in degrees.
    - longitude (float): The starting longitude in degrees.
    - distance_miles (float): The distance in miles one is willing to travel from the starting point.
    
    Returns:
    - A tuple containing two tuples: ((latitude_lower_bound, latitude_upper_bound), (longitude_lower_bound, longitude_upper_bound))
    """
    miles_per_degree_latitude = 69  # Approximate miles per degree of latitude
    
    # Calculate latitude bounds
    latitude_bound_upper = latitude + (distance_miles / miles_per_degree_latitude)
    latitude_bound_lower = latitude - (distance_miles / miles_per_degree_latitude)
    
    # Convert latitude to radians for cosine calculation
    latitude_radians = radians(latitude)
    
    # Calculate longitude bounds, taking into account the cosine of latitude for longitude distance calculation
    longitude_bound_upper = longitude + (distance_miles / (miles_per_degree_latitude * cos(latitude_radians)))
    longitude_bound_lower = longitude - (distance_miles / (miles_per_degree_latitude * cos(latitude_radians)))
    
    return ((latitude_bound_lower, latitude_bound_upper), (longitude_bound_lower, longitude_bound_upper))


def find_pedestrian_areas(min_lat, min_lon, max_lat, max_lon):
    overpass_url = "http://overpass-api.de/api/interpreter"
    overpass_query = f"""
        [out:json];
        (
            way["highway"~"footway|path|sidewalk|pedestrian|steps|track|crossing|bridleway"]({min_lat},{min_lon},{max_lat},{max_lon});
            node["highway"~"footway|path|sidewalk|pedestrian|steps|track|crossing|bridleway"]({min_lat},{min_lon},{max_lat},{max_lon});
        );
        out geom;
    """
    response = requests.get(overpass_url, params={'data': overpass_query})
    data = response.json()
    return data


def haversine(lat1, lon1, lat2, lon2):
    # Radius of the Earth in feet
    earth_radius_feet = 20902231.92

    # Convert coordinates from degrees to radians
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])

    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    distance_feet = earth_radius_feet * c

    return distance_feet


def find_all_coords(pedestrian_areas, groupid_pairings, scores_arr):
    path_mapper = {}
    path_mapper['footway'] = 0.75
    path_mapper['path'] = 0.5
    path_mapper['cycleway'] = 0.25
    
    # Iterate over the elements and print details
    for element in pedestrian_areas['elements']:
        if element['type'] == 'way':
            if 'geometry' in element:
                # Extract coordinates from each node in the geometry
                way_coords = [(node['lat'], node['lon']) for node in element['geometry']]
                geom_pos = 0
    
                groupid_pairings[element['id']] = way_coords
    
                for coord in way_coords:
                    #if coord not in loc_pairs:
                        #loc_pairs.add(coord)
    
                    init_score = 0
                    if element['tags'].get('highway', 'Unknown') in path_mapper:
                        init_score += path_mapper[element['tags'].get('highway', 'Unknown')]
    
                    # location, score, geom_id, pos_in_geom
                    scores_arr.append([coord, init_score, element['id'], geom_pos])
                    geom_pos += 1


def find_water_fountains(min_lat, min_lon, max_lat, max_lon):
    overpass_url = "https://lz4.overpass-api.de/api/interpreter"
    overpass_query = f"""
        [out:json];
        node["amenity"="drinking_water"]({min_lat},{min_lon},{max_lat},{max_lon});
        out;
    """
    response = requests.get(overpass_url, params={'data': overpass_query})

    if response.status_code == 200:
        data = response.json()
        return data
    else:
        print("Error fetching data:", response.status_code)
        return None


def extract_coordinates(data):
    coordinates = []
    if 'elements' in data:
        for element in data['elements']:
            if 'lat' in element and 'lon' in element:
                lat = element['lat']
                lon = element['lon']
                coordinates.append((lat, lon))
    return coordinates


def find_closest_fountain(point, fountain_locations):
    # Convert fountain locations to numpy array
    fountain_array = np.array(fountain_locations)

    # Build a k-d tree index
    tree = BallTree(fountain_array, leaf_size=40)

    # Query the nearest neighbor
    dist, ind = tree.query([point], k=1)

    # Retrieve the index of the closest fountain
    closest_index = ind[0][0]

    # Retrieve the coordinates of the closest fountain
    closest_fountain = fountain_locations[closest_index]

    return closest_fountain

def min_max_scaling(fountain_distances):
    min_distance = min(fountain_distances)
    max_distance = max(fountain_distances)
    
    scaled_distances = [(max_distance - d) / (max_distance - min_distance) for d in fountain_distances]
    
    return scaled_distances


def closest_fountain_dists(scores_arr, water_fountains_coordinates):
    
    fountain_distances = []
    
    for i in range(len(scores_arr)):
        closest_fountain = find_closest_fountain(scores_arr[i][0], water_fountains_coordinates)
        distance = haversine(scores_arr[i][0][0], scores_arr[i][0][1], closest_fountain[0], closest_fountain[1])
        fountain_distances.append(distance)

    scaled_distances = min_max_scaling(fountain_distances)

    for i in range(len(scaled_distances)):
        scores_arr[i][1] += scaled_distances[i]


def generate_random_lat_long(min_lat, max_lat, min_long, max_long):
    """
    Generate a random latitude and longitude pair within specified ranges.
    
    Args:
        min_lat (float): Minimum latitude.
        max_lat (float): Maximum latitude.
        min_long (float): Minimum longitude.
        max_long (float): Maximum longitude.
        
    Returns:
        tuple: Random latitude and longitude pair as (latitude, longitude).
    """
    latitude = random.uniform(min_lat, max_lat)
    longitude = random.uniform(min_long, max_long)
    return latitude, longitude


def safety_data(scores_arr, min_latitude, min_longitude, max_latitude, max_longitude):

    safety_items = ['assault', 'robbery', 'stalking', 'gangs', 'drugs', 'vandalism', 'discrimination', 'noise']

    safety_incidents = {}
    safety_incidents['assault'] = 1
    safety_incidents['robbery'] = 0.875
    safety_incidents['stalking'] = 0.75
    safety_incidents['gangs'] = 0.625
    safety_incidents['drugs'] = 0.5
    safety_incidents['vandalism'] = 0.375
    safety_incidents['discrimination'] = 0.25
    safety_incidents['noise'] = 0.125

    crime_data = []
    weights = []

    # randomly select 200 crime incidents
    for i in range(200):
        crime_data.append(generate_random_lat_long(min_latitude, max_latitude, min_longitude, max_longitude))
        random_crime = random.randint(0, 7)
        weights.append(safety_incidents[safety_items[random_crime]])
    
    weights = np.array(weights)

    # Convert list of tuples to a NumPy array
    crime_data_array = np.array(crime_data)

    # Perform kernel density estimation
    kde = gaussian_kde(crime_data_array.T, weights=weights)

    area_safety = []

    for i in range(len(scores_arr)):
        area_safety.append(kde(scores_arr[i][0]))
    
    scaled_safeties = min_max_scaling(area_safety)
    
    # decrease score if higher hotspot density
    for i in range(len(scores_arr)):
        scores_arr[i][1] -= scaled_safeties[i][0]


def generate_points(min_lat, max_lat, min_lon, max_lon, num_points):
    # Calculate the area of the bounding box
    area = (max_lat - min_lat) * (max_lon - min_lon)
    
    # Calculate the dimensions of a square grid cell to achieve equal area
    cell_area = area / num_points
    cell_side_length = np.sqrt(cell_area)
    
    # Calculate the number of points in latitude and longitude directions
    num_lat_points = int(np.ceil((max_lat - min_lat) / cell_side_length))
    num_lon_points = int(np.ceil((max_lon - min_lon) / cell_side_length))
    
    # Generate latitude-longitude points within each grid cell
    points = []
    for i in range(num_lat_points):
        for j in range(num_lon_points):
            lat = min_lat + i * cell_side_length
            lon = min_lon + j * cell_side_length
            points.append((lat, lon))
    
    # Trim excess points to ensure exactly num_points
    points = points[:num_points]
    
    return points


def get_weather(api_key, latitude, longitude):
    url = f"https://api.openweathermap.org/data/2.5/weather?lat={latitude}&lon={longitude}&appid={api_key}&units=metric"

    response = requests.get(url)

    if response.status_code == 200:
        data = response.json()
        return data
    else:
        print("Error fetching data:", response.status_code)
        return None


def rate_factors(factor, ideal_factor=15, lower_bound=10, upper_bound=25):
    # Calculate the rating based on the temperature's proximity to the ideal temperature
    if lower_bound <= factor <= upper_bound:
        rating = 1 - abs(factor - ideal_factor) / (upper_bound - ideal_factor)
    else:
        rating = 0
    
    return max(0, min(1, rating))


def rate_weather_type(data_id):
    rating = 0

    semiacceptable_categories = [300, 301, 310, 311, 500, 520, 600, 612, 615, 620]

    group = data_id / 100

    if group == 3 or group == 7:
        rating = 0
    elif data_id in semiacceptable_categories:
        rating = 0.25
    else:
        rating = 1

    return rating


def get_weather_scores(weather_ratings, weather_data):

    num_factors = 4.0
    
    for data in weather_data:
        rating = 0
    
        # temp rating
        ideal_temp = 15
        lower_bound_temp = 10
        upper_bound_temp = 25
    
        rating += rate_factors(data['main']['temp'], ideal_temp, lower_bound_temp, upper_bound_temp)
    
        # humidity rating
        ideal_humidity = 50
        lower_bound_humidity = 30
        upper_bound_humidity = 70
    
        rating += rate_factors(data['main']['humidity'], ideal_humidity, lower_bound_humidity, upper_bound_humidity)
    
        # wind rating
        ideal_wind = 4.5
        lower_bound_wind = 2
        upper_bound_wind = 7
    
        rating += rate_factors(data['wind']['speed'], ideal_wind, lower_bound_wind, upper_bound_wind)
    
        # weather type
        rating += rate_weather_type(data['weather'][0]['id'])
    
        rating = rating / num_factors
    
        weather_ratings.append(rating)


def calc_edges(cell_bbox_vertices):
    # Calculate minimum and maximum coordinates
    min_x = np.min(cell_bbox_vertices[:, 0])
    max_x = np.max(cell_bbox_vertices[:, 0])
    min_y = np.min(cell_bbox_vertices[:, 1])
    max_y = np.max(cell_bbox_vertices[:, 1])

    return (min_x, min_y, max_x, max_y)


def closest_voronoi_cells(scores_arr, weather_points, weather_and_ratings):
    weather_rating_map = {}

    for each_pair in weather_and_ratings:
        weather_rating_map[tuple(each_pair[0])] = each_pair[1]
        
    weather_points = np.array(weather_points)
    
    # Compute Voronoi diagram
    vor = Voronoi(weather_points)
    
    voronoi_cells = []
    for region_index in vor.point_region:
        vertices_indices = vor.regions[region_index]
        cell_vertices = vor.vertices[vertices_indices]
        voronoi_cells.append(calc_edges(cell_vertices))

    # Create an R-tree index
    idx = index.Index()
    
    # Insert bounding boxes of Voronoi cells into the R-tree index
    for i, cell_bbox in enumerate(voronoi_cells):
        idx.insert(i, cell_bbox)

    for each_point in scores_arr:

        # Given a query point `query_point`, find the index of the nearest cell in the R-tree
        nearest_indices = list(idx.nearest(list(each_point[0]), 1))
    
        # Retrieve the corresponding Voronoi cell
        nearest_cell_index = nearest_indices[0]
        nearest_cell = weather_points[nearest_cell_index]
    
        each_point[1] += weather_rating_map[tuple(nearest_cell)]


def points_scores_and_frequencies(count_points, lat_long_score_dict, access_group, scores_arr):
    
    for point in scores_arr:
        if point[0] not in count_points:
            count_points[point[0]] = 1
        else:
            count_points[point[0]] += 1
    
        if point[0] not in lat_long_score_dict:
            lat_long_score_dict[point[0]] = point[1]

    for row in scores_arr:
        pos = row[0]
        group_num = row[2]
        pos_in_geom = row[3]
    
        if pos not in access_group:
            access_group[pos] = []
    
        access_group[pos].append((group_num, pos_in_geom))


def calculate_distance(coord1, coord2):
    lat1, lon1 = coord1
    lat2, lon2 = coord2
    R = 6371.0  # Radius of the Earth in kilometers
    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)

    dlon = lon2_rad - lon1_rad
    dlat = lat2_rad - lat1_rad

    a = math.sin(dlat / 2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    distance_km = R * c  # Distance in kilometers
    distance_ft = distance_km * 3280.84  # Convert kilometers to feet (1 kilometer = 3280.84 feet)
    return distance_ft


def construct_path(random_groupid, count_points, access_group, groupid_pairings):

    #print(random_groupid)
    #print()
    group_lat_longs = groupid_pairings[random_groupid]
    #print(group_lat_longs)
    
    path_follower = []

    for coord in group_lat_longs:
        if count_points[coord] == 1:
            path_follower.append((coord, -1))
        else:
            path_follower.append((coord, access_group[coord]))

    return path_follower

def find_optimal_path(random_groupid, goal_distance, lat_long_score_dict, count_points, access_group, groupid_pairings):

    used_groups = set()
    used_groups.add(random_groupid)

    path_follower = construct_path(random_groupid, count_points, access_group, groupid_pairings)
    
    front = bool(random.getrandbits(1))
    curr_index = -1
    
    if front:
        curr_index = 0
    else:
        curr_index = len(path_follower) - 1
        
    points = []
    # we don't want a loop in the path
    no_repeats = set()
    curr_score = 0
    curr_distance = 0
    
    curr_group = random_groupid
    
    # append initial point
    no_repeats.add(path_follower[curr_index][0])
    points.append((path_follower[curr_index][0], lat_long_score_dict[path_follower[curr_index][0]]))
    curr_score += lat_long_score_dict[path_follower[curr_index][0]]

    #print(path_follower)

    while curr_distance < goal_distance:
        # we can jump to one or more groups from this point
        if path_follower[curr_index][1] != -1:
            # randomly choose group to continue with
            if ((curr_index == 0 and front == False) or (curr_index == len(path_follower)-1 and front == True)):
                # if at the end of current group force to switch to different group
                chosen_groupid = curr_group
                iter = 1
                group_possibs = path_follower[curr_index][1]

                many_groups_exist = set()

                for pair in group_possibs:
                    many_groups_exist.add(pair[0])

                # make sure jumping to a different group when at the end is possible
                if len(many_groups_exist) > 1:
                    while chosen_groupid == curr_group:
                        chosen_group = random.randint(0, len(path_follower[curr_index][1])-1)
                        chosen_groupid = path_follower[curr_index][1][chosen_group][0]
                # jump anywhere
                else:
                    chosen_group = random.randint(0, len(path_follower[curr_index][1])-1)
                    chosen_groupid = path_follower[curr_index][1][chosen_group][0]
            # can continue with same group possibly
            else:
                chosen_group = random.randint(0, len(path_follower[curr_index][1])-1)
                chosen_groupid = path_follower[curr_index][1][chosen_group][0]
        
            # if continuing with the same group don't change the direction
            if path_follower[curr_index][1][chosen_group][0] != curr_group:
                
                curr_group = path_follower[curr_index][1][chosen_group][0]
                used_groups.add(curr_group)
                curr_index = path_follower[curr_index][1][chosen_group][1]
    
                # need to do process of reconstructing the path
                path_follower = construct_path(curr_group, count_points, access_group, groupid_pairings)
        
                # check if index is in front, back or middle
                if curr_index == 0:
                    front = True
                elif curr_index == len(path_follower) - 1:
                    front = False
                else:
                    # in middle case choose direction randomly
                    front = bool(random.getrandbits(1))
    

        old_index = curr_index
        next_point = 0
        
        if front:
            next_point = 1
        else:
            next_point = -1
        
        curr_index += next_point
        
        # end loop if can no longer continue
        if curr_index < 0 or curr_index >= len(path_follower) or old_index < 0 or old_index >= len(path_follower):
            return (points, curr_score, curr_distance, used_groups)
        else:
            # no repeat points
            if path_follower[curr_index][0] in no_repeats:
                return (points, curr_score, curr_distance, used_groups)
            #print(path_follower[curr_index][0])
            points.append((path_follower[curr_index][0], lat_long_score_dict[path_follower[curr_index][0]]))
            curr_score += lat_long_score_dict[path_follower[curr_index][0]]
            curr_distance += calculate_distance(path_follower[old_index][0], path_follower[curr_index][0])
            no_repeats.add(path_follower[curr_index][0])

    return (points, curr_score, curr_distance, used_groups)


def find_good_parents(fitnesses, index):

    these_points = fitnesses[index][0]
    curr_path_bools = []

    for x in range(len(fitnesses)):
        if x == index:
            curr_path_bools.append(False)
        else:
            other_points = fitnesses[x][0]
            curr_path_bools.append(intersection_exists(these_points, other_points))
    return curr_path_bools
    

def intersection_exists(list1, list2):
    set1, set2 = set(list1), set(list2)
    # Iterate over the smaller set for efficiency
    if len(set1) > len(set2):
        set1, set2 = set2, set1
    for element in set1:
        if element in set2:
            return True  # Early exit
    return False

def parents_roulette(good_parents_test, taken_sample, max_pairs_num):

    mateable_parents = []

    for i in range(len(good_parents_test)):
        if any(good_parents_test[i]):
            mateable_parents.append((good_parents_test[i], taken_sample[i]))

    total_score = [score[1][1] for score in mateable_parents]
    total_score = sum(total_score)

    ratios = [score[1][1]/total_score for score in mateable_parents]
    ratios = np.array(ratios)

    cumulative_sums = np.cumsum(ratios)

    # try to get parents a maximum of 200 times up to max_pairs_num
    total_tries = 0
    num_parent_pairs = 0
    indexes_used = set()
    parent_pairs = []
    completely_useless = set()

    while total_tries < 200 and num_parent_pairs < max_pairs_num:
        rand = np.random.rand()  # Generate a random number between 0 and 1
        # Find the index where this random number would fit in the cumulative sums
        first_index = np.searchsorted(cumulative_sums, rand)

        if first_index not in completely_useless:
            search_through_first = good_parents_test[first_index]
    
            for i in range(len(search_through_first)):
                if search_through_first[i] == True:
                    if first_index not in indexes_used and i not in indexes_used:
                        num_parent_pairs += 1
                        indexes_used.add(first_index)
                        indexes_used.add(i)
                        parent_pairs.append((taken_sample[first_index], taken_sample[i]))
                        break
                    if i == len(search_through_first)-1:
                        completely_useless.add(first_index)


        total_tries += 1

    return parent_pairs


def produce_parents(parent_pairs):

    for pair in parent_pairs:
        intersecting_points = set()
        existing_points_first = set()

        first_parent_points = pair[0][0]
        second_parent_points = pair[1][0]

        # fine to include points with 
        for point in first_parent_points:
            if point not in existing_points_first:
                existing_points_first.add(point)

        for point in second_parent_points:
            if point in existing_points_first:
                intersecting_points.add(point)

        # get the best path to each intersecting point
        traverse_intersections = list(intersecting_points)

        all_intersection_paths = []

        for intersection in traverse_intersections:
            first_parent_index = first_parent_points.index(intersection)
            second_parent_index = second_parent_points.index(intersection)

            complete_path = set()

            # check better path backwards from the indexes
            curr_first_index_back = first_parent_index

            first_back_distance = 0
            first_back_score = 0
            first_back_path = []
            first_back_prev = -1
            while (first_parent_points[curr_first_index_back] not in intersecting_points or first_parent_points[curr_first_index_back] == intersection) and curr_first_index_back >= 0:
                first_back_path.insert(0, first_parent_points[curr_first_index_back])
                first_back_score += first_parent_points[curr_first_index_back][1]
                if first_back_prev != -1:
                    first_back_distance += calculate_distance(first_parent_points[curr_first_index_back], first_parent_points[first_back_prev])
                first_back_prev = curr_first_index_back
                curr_first_index_back -= 1

            
            curr_second_index_back = second_parent_index

            second_back_distance = 0
            second_back_score = 0
            second_back_path = []
            second_back_prev = -1
            while (second_parent_points[curr_second_index_back] not in intersecting_points or second_parent_points[curr_second_index_back] == intersection) and curr_second_index_back >= 0:
                second_back_path.insert(0, second_parent_points[curr_second_index_back])
                second_back_score += second_parent_points[curr_second_index_back][1]
                if second_back_prev != -1:
                    second_back_distance += calculate_distance(second_parent_points[curr_second_index_back], second_parent_points[second_back_prev])
                second_back_prev = curr_second_index_back
                curr_second_index_back -= 1

            # decide which path to append
            if (first_back_score/first_back_distance > second_back_score/second_back_distance):
                complete_path.add(first_back_path)
            else:
                complete_path.add(second_back_path)


            # check better path forwards from the indexes
            curr_first_index_front = first_parent_index

            first_front_distance = 0
            first_front_score = 0
            first_front_path = []
            first_front_prev = -1
            while (first_parent_points[curr_first_index_front] not in intersecting_points or first_parent_points[curr_first_index_front] == intersection) and curr_first_index_front < len(first_parent_points):
                first_front_path.append(first_parent_points[curr_first_index_front])
                first_front_score += first_parent_points[curr_first_index_front][1]
                if first_front_prev != -1:
                    first_front_distance += calculate_distance(first_parent_points[curr_first_index_front], first_parent_points[first_front_prev])
                first_front_prev = curr_first_index_front
                curr_first_index_front += 1


            curr_second_index_front = second_parent_index

            second_front_distance = 0
            second_front_score = 0
            second_front_path = []
            second_front_prev = -1
            while (second_parent_points[curr_second_index_front] not in intersecting_points or second_parent_points[curr_second_index_front] == intersection) and curr_second_index_front < len(second_parent_points):
                second_front_path.append(second_parent_points[curr_second_index_front])
                second_front_score += second_parent_points[curr_second_index_front][1]
                if second_front_prev != -1:
                    second_front_distance += calculate_distance(second_parent_points[curr_second_index_front], second_parent_points[second_front_prev])
                second_front_prev = curr_second_index_front
                curr_second_index_front += 1
        

            # decide which path to append
            if (first_front_score/first_front_distance > second_front_score/second_front_distance):
                complete_path.add(first_front_path)
            else:
                complete_path.add(second_front_path)


            all_intersection_paths.append(list(complete_path))

    return all_intersection_paths


def traverse_endpoint(follow_path, front_adder, total_path, total_distance, total_score, distance_target, blacklisted_groups, count_points, lat_long_score_dict, access_group, groupid_pairings):
    check_next_index = None
    chosen_group_and_index = None
    curr_group = None

    while (total_distance < distance_target):
        # first iteration step
        if check_next_index == None:
            chosen_group_and_index = random.choice(follow_path)
        else:
            all_options = follow_path[check_next_index][1]
            valid_options = [group for group in all_options if group[0] not in blacklisted_groups]
            if len(valid_options) == 0:
                return (total_path, total_score, total_distance)           
            chosen_group_and_index = random.choice(valid_options)
            # if switched groups make sure we can't go back to a previous group (no loops)
            if chosen_group_and_index[0] != curr_group:
                blacklisted_groups.add(curr_group)
                
        curr_group = chosen_group_and_index[0]
        curr_index = chosen_group_and_index[1]
        follow_path = construct_path(curr_group, count_points, access_group, groupid_pairings)

        next_index = 0
        if curr_index == 0:
            next_index = 1
        elif curr_index == len(follow_path) - 1:
            next_index = -1
        else:
            next_index = random.choice([1,-1])

        check_next_index = curr_index + next_index
        path_iter = 0
        group_change = False

        while ((0 <= check_next_index < len(follow_path)) and (total_distance < distance_target)):
            # gave chance to switch
            if count_points[follow_path[check_next_index][0]] > 1 and path_iter != 0:
                group_change = True
                break

            if front_adder:
                total_path.append((follow_path[check_next_index][0], lat_long_score_dict[follow_path[check_next_index][0]]))
            else:
                total_path.insert(0, (follow_path[check_next_index][0], lat_long_score_dict[follow_path[check_next_index][0]]))

            total_distance += calculate_distance(follow_path[curr_index][0], follow_path[check_next_index][0])
            total_score += lat_long_score_dict[follow_path[check_next_index][0]]
            curr_index = check_next_index
            check_next_index += next_index
            path_iter += 1

        if group_change:
            continue

        return (total_path, total_score, total_distance)

    return (total_path, total_score, total_distance)


def extend_group(pair, total_path, total_distance, total_score, distance_target, count_points, lat_long_score_dict, access_group, groupid_pairings):

    # make sure doesn't coninue on either path previous routes
    first_parent_groups = pair[0][3]
    second_parent_groups = pair[1][3]
    blacklisted_groups = first_parent_groups.union(second_parent_groups)

    # each point has its list of valid group and index tuples
    endpoints = {}
    
    # add front and end
    endpoints[(total_path[0][0], "front")] = []
    front_access_groups = access_group[total_path[0][0]]
    for group in front_access_groups:
        if group[0] not in blacklisted_groups:
            endpoints[(total_path[0][0], "front")].append((group[0], group[1]))

    # add end if we can
    if len(total_path) >= 2:
        endpoints[(total_path[-1][0], "end")] = []
        end_access_groups = access_group[total_path[-1][0]]
        for group in end_access_groups:
            if group[0] not in blacklisted_groups:
                endpoints[(total_path[-1][0], "end")].append((group[0], group[1]))

    while (len(endpoints) > 0):
        # randomly select either the front or end to append to
        chosen_endpoint = random.choice(list(endpoints.keys()))
        
        # randomly select a valid group
        group_options = endpoints[chosen_endpoint]
        group_options = [group for group in group_options if group[0] not in blacklisted_groups]
        if len(group_options) == 0:
            del endpoints[chosen_endpoint]
            continue
        else:
            front_or_end = chosen_endpoint[1]
            # either append or prepend
            front_adder = True
            if front_or_end == "front":
                front_adder = False

            total_path, total_score, total_distance = traverse_endpoint(group_options, front_adder, total_path, total_distance, total_score, distance_target, blacklisted_groups, count_points, lat_long_score_dict, access_group, groupid_pairings)

            del endpoints[chosen_endpoint]

    return (total_path, total_score, total_distance, blacklisted_groups)


def find_max_score_path(points, max_distance, access_group):

    dp_arr = []

    # Let dp_arr[i][d] represent the maximum score achievable up to point i using distance d
    
    biggest_index = 0
    biggest_score = 0
    
    # score, list of coords
    for a in range(len(points)):
        curr_point_path = []
        for b in range(max_distance+1):
            # first row has score of first point
            if a == 0:
                if points[a][1] > biggest_score:
                    biggest_score = points[a][1]
                    biggest_index = (a, 1)
                curr_point_path.append([points[a][1], [points[a][0]], 0])
            else:
                curr_point_path.append([0, [points[a][0]], 0])
        dp_arr.append(curr_point_path)
    
    # we can at least start at first point for distance of 0
    # if len(points) >= 1:
    #     dp_arr[0][0] = points[0][1]

    for i in range(1, len(points)):
        for d in range(max_distance+1):
            find_dist = int(calculate_distance(points[i][0], points[i-1][0]))
            if d - find_dist > 0:
                dp_arr[i][d][0] = dp_arr[i-1][d-find_dist][0] + points[i][1]
                dp_arr[i][d][1].extend(dp_arr[i-1][d-find_dist][1])
                #dp_arr[i][d][2] += calculate_distance(points[i][0], points[i-1][0])
                #print(dp_arr[i][d][2])
                # if (dp_arr[i][d][0] > dp_arr[i-1][d-find_dist][0] + points[i][1]):
                #     dp_arr[i][d] = (dp_arr[i][d][0], dp_arr[i][d][1] + [dp_arr[i][d][0]])
                # else:
                #     dp_arr[i][d] = (dp_arr[i-1][d-find_dist][0] + points[i][1], dp_arr[i][d][1] + [dp_arr[i][d][0]])
            else:
                dp_arr[i][d][0] = points[i][1]

            if dp_arr[i][d][0] > biggest_score:
                biggest_score = dp_arr[i][d][0]
                biggest_index = (i, d)
        #dp_arr[i][d] = max(dp_arr[i][d], dp_arr[i-1][d-find_dist] + points[i][1])

    l, r = biggest_index
    path = dp_arr[l][r][1]

    total_dist = 0
    for i in range(len(path) - 1):
        coord1 = path[i]
        coord2 = path[i + 1]
        total_dist += calculate_distance(coord1, coord2)

    add_groups = {}
    blacklisted_groups = set()
    # add group to blacklisted if it appears more than once
    for i in range(len(path)):
        groups_per_point = access_group[path[i]]
        for group in groups_per_point:
            if group[0] not in add_groups:
                add_groups[group[0]] = 1
            else:
                add_groups[group[0]] += 1
                blacklisted_groups.add(group[0])
        
    return (dp_arr[l][r][1], dp_arr[l][r][0], total_dist, blacklisted_groups)


def produce_parents(pair, distance_target, count_points, lat_long_score_dict, access_group, groupid_pairings):

    intersecting_points = []
    existing_points_first = set()

    first_parent_points = pair[0][0]
    second_parent_points = pair[1][0]

    # fine to include points with 
    for point in first_parent_points:
        if point not in existing_points_first:
            existing_points_first.add(point)

    for point in second_parent_points:
        if point in existing_points_first:
            intersecting_points.append(point)

    merge_intersection_paths = []

    for intersection in intersecting_points:
        first_parent_index = first_parent_points.index(intersection)
        second_parent_index = second_parent_points.index(intersection)

        complete_path = []

        # check better path backwards from the indexes
        curr_first_index_back = first_parent_index

        first_back_distance = 0
        first_back_score = 0
        first_back_path = []
        first_back_prev = -1
        while curr_first_index_back >= 0 and (first_parent_points[curr_first_index_back] not in intersecting_points or first_parent_points[curr_first_index_back] == intersection):
            first_back_path.insert(0, first_parent_points[curr_first_index_back])
            first_back_score += first_parent_points[curr_first_index_back][1]
            if first_back_prev != -1:
                first_back_distance += calculate_distance(first_parent_points[curr_first_index_back][0], first_parent_points[first_back_prev][0])
            first_back_prev = curr_first_index_back
            curr_first_index_back -= 1

        if curr_first_index_back >= 0:
            if first_back_prev != -1:
                first_back_distance += calculate_distance(first_parent_points[curr_first_index_back][0], first_parent_points[first_back_prev][0])

        
        curr_second_index_back = second_parent_index

        second_back_distance = 0
        second_back_score = 0
        second_back_path = []
        second_back_prev = -1
        while curr_second_index_back >= 0 and (second_parent_points[curr_second_index_back] not in intersecting_points or second_parent_points[curr_second_index_back] == intersection):
            second_back_path.insert(0, second_parent_points[curr_second_index_back])
            second_back_score += second_parent_points[curr_second_index_back][1]
            if second_back_prev != -1:
                second_back_distance += calculate_distance(second_parent_points[curr_second_index_back][0], second_parent_points[second_back_prev][0])
            second_back_prev = curr_second_index_back
            curr_second_index_back -= 1

        if curr_second_index_back >= 0:
            if second_back_prev != -1:
                second_back_distance += calculate_distance(second_parent_points[curr_second_index_back][0], second_parent_points[second_back_prev][0])

        # decide which path to append
        if first_back_distance != 0 and second_back_distance != 0:
            if (first_back_score/first_back_distance > second_back_score/second_back_distance):
                complete_path.append((tuple(first_back_path), first_back_score, first_back_distance, "back"))
            else:
                complete_path.append((tuple(second_back_path), second_back_score, second_back_distance, "back"))
        elif first_back_distance != 0 and second_back_distance == 0:
            complete_path.append((tuple(first_back_path), first_back_score, first_back_distance, "back"))
        elif first_back_distance == 0 and second_back_distance != 0:
            complete_path.append((tuple(second_back_path), second_back_score, second_back_distance, "back"))
        else:
            #print("None of the backs are valid")
            complete_path.append(((), 0, 0, "back"))


        # check better path forwards from the indexes
        curr_first_index_front = first_parent_index

        first_front_distance = 0
        first_front_score = 0
        first_front_path = []
        first_front_prev = -1
        while curr_first_index_front < len(first_parent_points) and (first_parent_points[curr_first_index_front] not in intersecting_points or first_parent_points[curr_first_index_front] == intersection):
            first_front_path.append(first_parent_points[curr_first_index_front])
            first_front_score += first_parent_points[curr_first_index_front][1]
            if first_front_prev != -1:
                first_front_distance += calculate_distance(first_parent_points[curr_first_index_front][0], first_parent_points[first_front_prev][0])
            first_front_prev = curr_first_index_front
            curr_first_index_front += 1

        if curr_first_index_front < len(first_parent_points):
            if first_front_prev != -1:
                first_front_distance += calculate_distance(first_parent_points[curr_first_index_front][0], first_parent_points[first_front_prev][0])
            
        
        curr_second_index_front = second_parent_index

        second_front_distance = 0
        second_front_score = 0
        second_front_path = []
        second_front_prev = -1
        while curr_second_index_front < len(second_parent_points) and (second_parent_points[curr_second_index_front] not in intersecting_points or second_parent_points[curr_second_index_front] == intersection):
            second_front_path.append(second_parent_points[curr_second_index_front])
            second_front_score += second_parent_points[curr_second_index_front][1]
            if second_front_prev != -1:
                second_front_distance += calculate_distance(second_parent_points[curr_second_index_front][0], second_parent_points[second_front_prev][0])
            second_front_prev = curr_second_index_front
            curr_second_index_front += 1

        # capture distance to terminating point but make sure didn't end on last point
        if curr_second_index_front < len(second_parent_points):
            if second_front_prev != -1:
                second_front_distance += calculate_distance(second_parent_points[curr_second_index_front][0], second_parent_points[second_front_prev][0])
            
    

        # decide which path to append

        # protect against case where distance is 0 if it's the last point
        if first_front_distance != 0 and second_front_distance != 0:
            if (first_front_score/first_front_distance > second_front_score/second_front_distance):
                complete_path.append((tuple(first_front_path), first_front_score, first_front_distance, "front"))
            else:
                complete_path.append((tuple(second_front_path), second_front_score, second_front_distance, "front"))
        elif first_front_distance != 0 and second_front_distance == 0:
            complete_path.append((tuple(first_front_path), first_front_score, first_front_distance, "front"))
        elif first_front_distance == 0 and second_front_distance != 0:
            complete_path.append((tuple(second_front_path), second_front_score, second_front_distance, "front"))
        else:
            #print("None of the fronts are valid")
            complete_path.append(((), 0, 0, "front"))


        merge_intersection_paths.append((complete_path, intersection))

    total_path = []
    total_score = 0
    total_distance = 0

    # back score

    repeating_distance = set()


    for i in range(len(merge_intersection_paths)):

        # add back path
        for x in range(len(merge_intersection_paths[i][0][0][0])):
            total_path.append(merge_intersection_paths[i][0][0][0][x])

        total_score += merge_intersection_paths[i][0][0][1]

        # make sure not to double count distance
        if merge_intersection_paths[i][0][0][2] not in repeating_distance:
            total_distance += merge_intersection_paths[i][0][0][2]
            repeating_distance.add(merge_intersection_paths[i][0][0][2])

        # add front path
        for x in range(1, len(merge_intersection_paths[i][0][1][0])):
            total_path.append(merge_intersection_paths[i][0][1][0][x])

        total_score += merge_intersection_paths[i][0][1][1]

        # make sure not to double count distance
        if merge_intersection_paths[i][0][1][2] not in repeating_distance:
            total_distance += merge_intersection_paths[i][0][1][2]
            repeating_distance.add(merge_intersection_paths[i][0][1][2])

        # decrease intersection overlap score
        total_score -= merge_intersection_paths[i][1][1]
    
    # if distance too big then use dp to find optimal path within it
    if total_distance > distance_target:
        return find_max_score_path(total_path, distance_target, access_group)
    # if can add more distance extend from either end point
    else:
        return extend_group(pair, total_path, total_distance, total_score, distance_target, count_points, lat_long_score_dict, access_group, groupid_pairings)
        


if __name__ == "__main__":
    

    # pass in points like (a, b), c, d where (a,b) represents current location as lat, long pair, c is maximum path distance in feet
    # and d is allowed distance away from (a,b) in miles
    curr_location_str = args[1].strip("()")  # Remove the parentheses
    curr_location = tuple(map(float, curr_location_str.split(',')))  # Split by comma and convert to float
    
    # The next arguments are floats
    path_distance_feet = float(args[2])
    distance_away_miles = float(args[3])

    (min_latitude, max_latitude), (min_longitude, max_longitude) = calculate_bounds(curr_location[0], curr_location[1], distance_away_miles)
    
    pedestrian_areas = find_pedestrian_areas(min_latitude, min_longitude, max_latitude, max_longitude)
    scores_arr = []
    groupid_pairings = {}
    
    find_all_coords(pedestrian_areas, groupid_pairings, scores_arr)
    
    water_fountains_data = find_water_fountains(min_latitude, min_longitude, max_latitude, max_longitude)
    water_fountains_coordinates = extract_coordinates(water_fountains_data)
    
    closest_fountain_dists(scores_arr, water_fountains_coordinates)
    
    safety_data(scores_arr, min_latitude, min_longitude, max_latitude, max_longitude)

    num_points = 15
    weather_points = generate_points(min_latitude, max_latitude, min_longitude, max_longitude, num_points)
    
    weather_data = []
    
    api_key = api_weather_passcode
    
    for point in weather_points:
        weather_data.append(get_weather(api_key, point[0], point[1]))
    
    weather_ratings = []
    
    get_weather_scores(weather_ratings, weather_data)

    weather_and_ratings = []

    for i in range(len(weather_data)):
        weather_and_ratings.append(([weather_points[i][0], weather_points[i][1]], weather_ratings[i]))
    
    closest_voronoi_cells(scores_arr, weather_points, weather_and_ratings)
    
    count_points = {}
    lat_long_score_dict = {}
    access_group = {}
    
    points_scores_and_frequencies(count_points, lat_long_score_dict, access_group, scores_arr)
    
    all_groupids = list(groupid_pairings.keys())    

    random_group_id_values = []

    for i in range(1000):
        random_groupid = random.choice(all_groupids)
        random_group_id_values.append(random_groupid)

    arg_values = []

    for i in range(len(random_group_id_values)):
        arg_values.append((random_group_id_values[i], path_distance_feet, lat_long_score_dict, count_points, access_group, groupid_pairings))
    
    # Use pool.starmap to apply the fitness function with multiple arguments
    with Pool() as pool:
        fitnesses = pool.starmap(find_optimal_path, arg_values)

    # order fitnesses by decreasing score
    fitnesses.sort(key=lambda points_scores: points_scores[1], reverse=True)

    # take top 100 fitnesses
    taken_sample = fitnesses[:101]

    #print(taken_sample)

    parent_args = []
    for i in range(len(taken_sample)):
        parent_args.append((taken_sample, i))

    with Pool() as pool:
        good_parents_test = pool.starmap(find_good_parents, parent_args)

    mateable_parents = []

    for i in range(len(good_parents_test)):
        if any(good_parents_test[i]):
            mateable_parents.append((good_parents_test[i], taken_sample[i]))

    # try to make 25 couples
    parent_pairs = parents_roulette(good_parents_test, taken_sample, 25)

    children_paths = []

    for parent_pair in parent_pairs:
        children_paths.append(produce_parents(parent_pair, path_distance_feet, count_points, lat_long_score_dict, access_group, groupid_pairings))

    # order children by decreasing score
    children_paths.sort(key=lambda points_scores: points_scores[1], reverse=True)

    # take half the parents and half the children and see their next generation
    next_gen_paths = []
    next_gen_paths += taken_sample[:len(taken_sample)//2]
    next_gen_paths += children_paths[:len(children_paths)//2]
    next_gen_paths.sort(key=lambda points_scores: points_scores[1], reverse=True)

    second_parent_args = []
    for i in range(len(next_gen_paths)):
        second_parent_args.append((next_gen_paths, i))

    with Pool() as pool:
        second_good_parents_test = pool.starmap(find_good_parents, second_parent_args)

    second_mateable_parents = []

    for i in range(len(second_good_parents_test)):
        if any(second_good_parents_test[i]):
            second_mateable_parents.append((second_good_parents_test[i], next_gen_paths[i]))

    # try to make 10 couples
    second_parent_pairs = parents_roulette(second_good_parents_test, next_gen_paths, 10)

    final_children_paths = []

    for parent_pair in second_parent_pairs:
        final_children_paths.append(produce_parents(parent_pair, path_distance_feet, count_points, lat_long_score_dict, access_group, groupid_pairings))

    # order children by decreasing score
    final_children_paths.sort(key=lambda points_scores: points_scores[1], reverse=True)

    # take half the parents and conclude for now (just 2 iterations of genetic algorithm)
    final_gen_paths = []
    final_gen_paths += next_gen_paths[:len(next_gen_paths)//2]
    final_gen_paths += final_children_paths[:len(final_children_paths)//2]
    final_gen_paths.sort(key=lambda points_scores: points_scores[1], reverse=True)

    print(list(final_gen_paths[0]))
    