import igl
import numpy as np
import pygeodesic.geodesic as geodesic
import vedo
import time
from scipy.spatial.distance import cdist
from scipy.spatial import Delaunay
import lang as lang
import meshplot as mp


########################################################################
#### Mesh to sample
########################################################################

def distance_2points(a:np.array, b:np.array):
    """ calculats the Eucliean distance between two 3 dimensional points

    Parameters
    ----------
    a: numpy.array 
        a 1x3 np.array
    b: numpy.array
        a 1x3 numpy array
    Returns
    -------
    distance
        float of the Euclidean distance
    """
   

    distance = np.sqrt(pow(a[0]-b[0], 2) +
                    pow(a[1]-b[1], 2) +
                    pow(a[2]-b[2], 2)* 1.0) 
    return distance

def distance_source_all(sourcePoint, targetPoints):
    """calculates the distance from one to many points

    Parameters
    ----------
    sourcePoint: numpy.array
        1x3 numpy array
    targetPoints: numpy.array
        Nx3 numpy array

    Returns
    -------
    distances
        numpy array of float distances from the source point
    
    """
    try: 
        targetPoints.shape[1]
        
    except IndexError:
        
        targetPoints = np.array([targetPoints])
        

    distances = np.zeros(targetPoints.shape[0])
    for i, point in enumerate(targetPoints):
        distances[i] = distance_2points(sourcePoint, point)
    
    return distances


def get_points_in_next_row(isoPoints:np.array , sourceIndices, targetIndices, stitchwidth:float):
    """calculates the points that are close enough to be in the next row

    Parameters
    ----------
    isoPoints: numpy array
        all points on isoline
    
    sourceIndices: 
        indices of points in current row into isoPoints
    
    targetIndices:
        indices of unsorted points into isoPoints

    stitchwidth: float
        here distance between rows

    Returns
    -------
    next_row
        numpy array of indexes for the next row
     
    """
    epsilon = stitchwidth * 0.4
    smaller = []
    bigger = []
    
    for source in sourceIndices:
        distance = cdist([isoPoints[source]], isoPoints[targetIndices])
        smaller += (list(targetIndices[distance[0] < stitchwidth+epsilon]))
        bigger += (list(targetIndices[distance[0] > stitchwidth-epsilon]))

            
    point_set = set(smaller).intersection(set(bigger))
    next_row = np.fromiter(point_set, int, len(point_set))
        
    return next_row

def get_path_iso_intersec_points(linePoints, pathPoints):
    """find two points in two groups of points each that are closest to each other

    Parameters
    ----------
    linePoints: numpy array
        Nx3 numpy array of points
    
    pathPoits: numpy array
        Mx3 numpy array of points

    Returns
    -------
    close_dist
        distances between two different points of two different groups as float
    close_iso_point
        two points from linePoints
    close_path_point
        two Points from pathPoints
     
    """

    close_iso_point = np.zeros((2,3))
    close_path_point = np.zeros((2,3))
    close_dist = np.zeros((2))
    close_dist.fill(100)
    
    for i, point in enumerate(pathPoints):
        distances = distance_source_all(point, linePoints)
        arg_min = np.argsort(distances)[:2]
        mini = np.sort(distances)[:2]
        
        if len(mini)>0:
            if (mini[0] < close_dist[0] or (len(mini)>1 and
                    mini[1] < close_dist[1])):
                
                for i, min in enumerate(arg_min):
                    if close_dist[0] > close_dist[1]:
                        close_dist[0] = mini[i]
                        close_iso_point[0] = linePoints[min]
                        close_path_point[0] = point
                    else:
                        close_dist[1] = mini[i]
                        close_iso_point[1] = linePoints[min]
                        close_path_point[1] = point
       
    return close_dist, close_iso_point, close_path_point
    

def calc_intersec_point(point1,point2,project_point):
    """ Project a point to line between two other points

    Parameters
    ----------
    point1: numpy array
        1x3 point on isoline

    point2: numpy array
        1x3 point on isoline

    project_point: numpy array
        1x3 path point

    Returns
    -------
    projection
        numpy array of koordinates for the pbrojection
        
    """
    #from https://stackoverflow.com/questions/61341712/calculate-projected-point-location-x-y-on-given-line-startx-y-endx-y
    dist = np.sum((point1-point2)**2)
    
    
    if dist == 0:
        return point1
    #The line extending the segment is parameterized as p1 + t (p2 - p1).
    #The projection falls where t = [(p3-p1) . (p2-p1)] / |p2-p1|^2

    #if you need the point to project on line extention connecting p1 and p2
    t = np.sum((project_point - point1) * (point2 - point1)) / dist
    
    #if you need the point to project on line segment between p1 and p2 or closest point of the line segment
    t = max(0, min(1, np.sum((project_point - point1) * (point2- point1)) / dist))
    
    projection = point1 + t * (point2 - point1)
    return projection

def get_all_starting_points(iso_points, isolines:dict, path_points):
    """Calculate all starting points

    Parameters
    ----------
    iso_points: numpy array
        all points on the isolines

    isolines: dict
        dictionary with indeces of isoPoints for each row
    
    path_points: numpy array
        N+3 array of points

    Returns
    -------
    all_ciso
        array of iso points close to path points
    all_cpath
        array of path points close to isolines
    all_start
        array of starting points

    """
    all_ciso = np.zeros((len(isolines.keys()), 2,3))
    all_cpath = np.zeros((len(isolines.keys()),2, 3))
    all_dist = np.zeros((len(isolines.keys()),2))
    all_start = np.zeros((len(isolines.keys()),3))

    #get 2 points closest to each isoline + 2 points on isoine
    for key in isolines.keys():
        
        cdist, ciso_point, cpath_point = get_path_iso_intersec_points(iso_points[isolines[key]], 
                                                                      path_points)
        all_ciso[key] = ciso_point
        all_cpath[key] = cpath_point
        all_dist[key] = cdist
        a = ciso_point[0]
        b = ciso_point[1]
        p = cpath_point[np.argmin(cdist)]
        all_start[key] = calc_intersec_point(a,b,p)
        
    return all_ciso, all_cpath, all_start 


def sample_points_on_isoline(iso_line, start_point, stitch_width:float):
    """ samples points on one isoline by calculating the next point with a distance = stitchwidth

    Parameters
    ----------
    iso_line: numpy array
        all points on the isoline

    start_point: numpy array
        1x3 array for one point

    stitch_width: float
        desired distance between points

    Returns
    -------
        np.array(sample_points)
            array of 3D points coordinates
    """
    sample_points = [start_point]
    endpoint = start_point
    epsilon = 0 #0.2
    checked_all_points = False
    
    while not checked_all_points:  #points left
        
        distances = distance_source_all(start_point, iso_line) # cdist
        arg_sort = np.argsort(distances)
        d_sort = np.sort(distances)
        prev_d = 0
        

        if len(d_sort) == 0 or d_sort[-1] < (stitch_width-epsilon):
            try:
                if not np.array_equal(iso_line[-1], endpoint):
                    
                    iso_line = np.append(iso_line, [endpoint], axis = 0)
                    
                    distances = distance_source_all(start_point, iso_line)
                    arg_sort = np.argsort(distances)
                    d_sort = np.sort(distances)
                    continue
                    
                else:
                    break
            
            except IndexError:
                print("Index Error: Array is empty, try a smaller Stitch width")
                raise
            
        
        for i, d in enumerate(d_sort):
            

            if ((d >= stitch_width-epsilon) and (d <= stitch_width+epsilon)):
                
                start_point = iso_line[arg_sort[i]]
                sample_points.append(start_point)
                
                iso_line = np.delete(iso_line, arg_sort[:i], axis = 0)
                
                break

            elif(i == 0 and (d > stitch_width+epsilon)):
                
                point_d = d - prev_d
                stitch_remain = stitch_width - prev_d
                t = stitch_remain / point_d
                r_vec = iso_line[arg_sort[i]]- start_point
                start_point = start_point + t * r_vec
                sample_points.append(start_point)
                iso_line = np.delete(iso_line, arg_sort[:i], axis = 0)
                


            elif ((prev_d < stitch_width-epsilon) and (d > stitch_width+epsilon)):
                point_d = d - prev_d
                stitch_remain = stitch_width - prev_d
                t = stitch_remain / point_d
                r_vec = iso_line[arg_sort[i]] - iso_line[arg_sort[i-1]] 
                start_point = iso_line[arg_sort[i-1]] + t * r_vec
                sample_points.append(start_point)
                iso_line = np.delete(iso_line, arg_sort[:i], axis = 0)
                break
            
            prev_d = d
        
        #delete last point if to close to first
        d_ends = distance_2points(sample_points[0], sample_points[-1])
        if (len(sample_points)> 1 and d_ends/stitch_width < 0.5 ):
            sample_points = sample_points[:-1]


    return np.array(sample_points)

def mesh_to_sample(v,f, start_index:int, stitch_width: float):
    """ Master function to sample points on 3D mesh surface

    calculates furstes point, isolines, and sample points

    Parameters
    ----------
    v: 3D mesh vertices

    f: 3D mesh faces

    start_index: int
        index into v
    
    stitch_width: float
        distance between rows and sample points    

    Returns
    -------
        sample_points
            dict of points in each line

        iso_lines
            sorted isoline points
        g_v
            all isoline points

        g_e
            isoline edges

        iso_lines
            sorted isoline points
        times
            array of durations of steps in function

    """
    times = np.zeros(6)
    start = time.time()

    #get point furthest point
    source_indices = np.array([start_index])
    target_indices = None
    geoalg = geodesic.PyGeodesicAlgorithmExact(v, f)
    distances, _ = geoalg.geodesicDistances(source_indices, target_indices)
    sourceIndex = source_indices[0]
    targetIndex = distances.argmax()
    distance, path = geoalg.geodesicDistance(sourceIndex, targetIndex)

    end = time.time()
    times[0] = end-start
    start = end

    #calculate iso lines
    n_iso = int(np.round(distance / stitch_width))
    g_v, g_e = igl.isolines(v, f, distances, n_iso)

    end = time.time()
    times[1] = end-start
    start = end
    
    
    #sort points into isolines
    iso_lines = {} #number of lines
    iso_lines[0] = np.where((g_v == v[sourceIndex]).all(axis=1))[0]
    indeces_left = np.arange(len(g_v))
    mask = np.ones(len(g_v), dtype=bool)
    mask[iso_lines[0]] = False
    for key in range(0, n_iso):
        if key == 0:
            mask[iso_lines[key]] = False
            #indeces_left = indeces_left[~(np.in1d(indeces_left, iso_lines[key]))]
            continue
        
        c_p = get_points_in_next_row(g_v, iso_lines[key-1], indeces_left[mask], stitch_width)
        iso_lines[key] = c_p
        mask[iso_lines[key]] = False
        #indeces_left = indeces_left[~(np.in1d(indeces_left, iso_lines[key]))]

    end = time.time()
    
    times[2] = end-start
    start = end
    
    #Calculate intersections
    all_ciso, all_cpath, proj = (get_all_starting_points(g_v, iso_lines, path))

    end = time.time()
    times[3] = end-start
    start = end
    
    #Sample points
    sample_points = {}
    for key in iso_lines.keys():
        sample_points[key] = sample_points_on_isoline(g_v[iso_lines[key]], proj[key], stitch_width)

    #reverse if point in previous line [1] is closer to line[-1] than line[1]
    for key in sample_points.keys():
        if key == 0 or key == 1:
            continue
        try:
            if distance_2points(sample_points[key-1][1], sample_points[key][1]) > distance_2points(sample_points[key-1][1], sample_points[key][-1]):
                sample_points[key] = np.flip(sample_points[key], 0)
        except IndexError:
            continue

    end = time.time()
    times[4] = end-start
    times[5] = np.sum(times[:-1])

    return sample_points, iso_lines, g_v, g_e, iso_lines, times #dict of coordinate

##################################################################
### Sample to graph
###################################################################

def get_row_edges(samples: dict): #TODO: not used?
    """ connect points in one array

    Parameters
    ----------
    samples: dict
        dictionary of points in differen rows 

    Returns
    -------
        edges_rows
            dict of edges in each row
    """
    edges_rows = {x: np.zeros((len(samples[x])-1, 2), dtype=int) for x in range(len(samples.keys()))}
    for key in samples.keys():
        if key == 0:
            edges_rows[key] = np.array([[0,0]])

        for i in range(len(samples[key])-1):
            edges_rows[key][i] = [i, i+1]

    return edges_rows

#dynamic time warping
def dtw(dist_mat):
    """
    from https://github.com/kamperh/lecture_dtw_notebook/blob/main/dtw.ipynb

    Find minimum-cost path through matrix `dist_mat` using dynamic programming.


    The cost of a path is defined as the sum of the matrix entries on that
    path. See the following for details of the algorithm:

    - http://en.wikipedia.org/wiki/Dynamic_time_warping
    - https://www.ee.columbia.edu/~dpwe/resources/matlab/dtw/dp.m

    The notation in the first reference was followed, while Dan Ellis's code
    (second reference) was used to check for correctness. Returns a list of
    path indices and the cost matrix.

    Parameters
    ----------
    dist_mat: array
        NxM matrix of distances

    Returns
    -------
    np.array(path[::-1])
        cheapest pth through the matrix

    np.array(action_list[::-1])
        list of connected edges

    cost_mat
        cost for different paths
        
    """

    N, M = dist_mat.shape
    
    # Initialize the cost matrix
    cost_mat = np.zeros((N + 1, M + 1))
    for i in range(1, N + 1):
        cost_mat[i, 0] = np.inf
    for i in range(1, M + 1):
        cost_mat[0, i] = np.inf

    # Fill the cost matrix while keeping traceback information
    traceback_mat = np.zeros((N, M))
    for i in range(N):
        for j in range(M):
            penalty = [
                cost_mat[i, j],      # sc (0)
                cost_mat[i, j + 1],  # dec (1)
                cost_mat[i + 1, j]]  # inc (2)
            i_penalty = np.argmin(penalty)
            cost_mat[i + 1, j + 1] = dist_mat[i, j] + penalty[i_penalty]
            traceback_mat[i, j] = i_penalty

    # Traceback from bottom right
    i = N - 1
    j = M - 1
    path = [[i, j]]
    action_list = []

    while i > 0 or j > 0:
        tb_type = traceback_mat[i, j]
        if tb_type == 0:
            # Match (single crochet)
            i = i - 1
            j = j - 1
            
        elif tb_type == 1:
            # Decrease
            i = i - 1
        elif tb_type == 2:
            # Increase
            j = j - 1
            
        action_list.append(tb_type)
        path.append([i, j])
        
    action_list.append(traceback_mat[0, 0])
    # Strip infinity edges from cost_mat before returning
    cost_mat = cost_mat[1:, 1:]
    return (np.array(path[::-1]), np.array(action_list[::-1]), cost_mat)

def calculate_dist_mat(source_points, target_points):
    """ Calculate distance matrix

    Parameters
    ----------
    source_points: numpy array
        array of points used as source
    
    target_points: numpy array
        array of target points

    Returns
    -------
        dist_mat
            distances between source and target points
    """
    
    N = source_points.shape[0]
    M = target_points.shape[0]
    dist_mat = np.zeros((N, M))
    
    for i in range(N) :
        for j in range(M):
            dist_mat[i, j] = distance_2points(source_points[i], target_points[j])
    
    return dist_mat

#edges between lines
def get_edges_and_actions(samples, iso_indeces):
    """
    Calculate the edges between different rows and the actions to get to the next row

    Parameters
    ----------
    samples: numpy arra
        Nx3 all sample points

    iso_indeces: dict
        dict of indeces inot samples

    Returns
    -------
        column_edges
            dict of edges into the next row

        action_lines
            dict of actions to turn one row into next row
    """

    #if number of points the same -> [1,1], [2,2]
    column_edges = {}
    action_lines = {}
    for key in iso_indeces.keys():
        try:
            dist_mat = calculate_dist_mat(samples[iso_indeces[key]], samples[iso_indeces[key+1]])
            path,action,  cost = dtw(dist_mat)
            edges = np.zeros(path.shape, dtype=int)
            action_lines[key] = action

            #translate path to path on all points
            for ind ,[i, j] in enumerate(path):
                edges[ind] = [iso_indeces[key][i], iso_indeces[key+1][j]]

            column_edges[key] = edges
        except KeyError :
            continue
        
    return column_edges, action_lines

def num_to_stitch(num: int, count = 1):
    """
    turn an integer into a string

    Parameters
    ----------
    num: int
        indectes the kind of stitch {0,1,2}

    count: int
        indecated the number of stitche

    Returns
    -------
        'sc' / 'inc{count}' / 'dec{count}'
            translation of number into Americn crochet stitch notation 
    """

    if num == 0:
        return 'sc'
    elif num == 1:
        return f'dec{count}'
    elif num == 2:
        return f'inc{count}'
    else:
        return f"{num} not translated"

def make_instructions(action_rows:dict):
    """
    Translate action row into American crochet terminologies

    Parameters
    ----------
    action_rows: dict
        dict of list of integers

    Returns
    -------
    instr
        dict of list of translated instructions
    """
    instr = {}    
    for key in action_rows.keys():
            
        line = []
        count = 1
        
        for ind, num in enumerate(action_rows[key]):

            is_last = ind == len(action_rows[key])-1

            if ind > 0:
                prev = action_rows[key][ind-1]
            else:
                prev = 0
            
            try:
                next = action_rows[key][ind + 1]
            except IndexError:
                next = None

            if is_last:
                if num == 0:
                    if prev != 0:
                        line.append(num_to_stitch(prev, count)) 
                    
                    line.append(num_to_stitch(num))
                elif num != 0:
                    count +=1
                    line.append(num_to_stitch(num, count))

            else:
                
                if num == 0:
                    if prev != 0:
                        
                        line.append(num_to_stitch(prev, count))
                    
                    
                    if next == 0:
                        line.append(num_to_stitch(num))
                    
                    count = 1
                else:
                    count += 1

        instr[key] = line
    return instr


def window_sliding(row):
    """
    Shortens row of instruction with window slidind
    Parameters
    ----------
    row: list 
        list of instructions


    Returns
    -------
    readable
        list of lists and tuples of instructions
    """

    
    if len(row)==0:
        return
    if len(row)<= 2:
        
        try:
            if np.array_equal(row[0], row[1]):
                return (len(row), row[0])
            else:
                return (1, row[0]), (1, row[1])
        except IndexError:
            return (1,row[0])
    
    
    for windowsize in range((len(row))//2, 0, -1):
        readable = []
        start = 0
        end = windowsize
        w_end = end + windowsize
        
            
        while end < len(row): 
            w_start = end
            w_end = end + windowsize
            count = 1
            
            if np.array_equal(row[start: end], row[w_start: w_end]):
                #reset readable
                readable = []
                count+=1
                pattern = row[start: end]
                if start > 0:
                    
                    readable.append(window_sliding(row[0: start]))
                
                while (w_end) < len(row): #(w_end + windowsize) < len(row):
                    #move window
                    w_start = w_end
                    w_end = w_end + windowsize
                    
                    if np.array_equal(row[start: end], row[w_start: w_end]):
                        count += 1

                    else:
                        #reset
                        w_start = w_start -windowsize
                        w_end = w_end- windowsize
                        break
                
                new_pattern = window_sliding(pattern)
                
                if isinstance(new_pattern, tuple) and isinstance(new_pattern[1], str):
                    
                    readable.append((count * new_pattern[0], new_pattern[1]))

                else:
                    readable.append((count, new_pattern))
                
                if len(row[w_end:]) > 0:
                    readable.append(window_sliding(row[w_end:]))
                
                if len(readable)== 1:
                    return readable[0]
                return(readable)
                
            else:
                readable.append((1, row[start]))
                start += 1
                end += 1
                
        for instr in row[start:end]:
            readable.append((1, instr))       
                
    return readable

def make_3d(points):
    """
    create faces based on points
    """
    faces = Delaunay(points)
    
    return points, faces.convex_hull

def sample_to_graph(sample_points:dict):
    """
    Master function of creating crochet instructions based on sample points

    Parameter
    ---------
    sample_points: dict
        dict of samplepoints sorted by row

    Returns
    -------
    shortend
        fiished instructions

    all_points
        array of all sample points 

    faces
        array of faces

    times
        array of duration fo calculation steps

    row_edges
        edges inside rows

    column_edges
        edges between rows

    """
    times = np.zeros(4)

    #put all sample points in oe array 
    sample_indeces = {}
    all_points = sample_points[0]
    for key in sample_points.keys():
        if key == 0 :
            sample_indeces[key] = np.array([0], dtype=int)
            continue
        sample_indeces[key] = np.arange(len(all_points), len(all_points) + len(sample_points[key]), dtype=int)
        
        all_points = np.concatenate((all_points, sample_points[key]), 0)

    #get row edges
    row_edges = {}
    for key in sample_indeces.keys():
        edges = []
        for ind, val in enumerate(sample_indeces[key]):
            edges.append(np.array([sample_indeces[key][ind -1], val]))
        
        row_edges[key] = np.array(edges)

    start = time.time()
    
    #get column edges
    column_edges, action_rows = get_edges_and_actions (all_points, sample_indeces)
    vertices, faces = make_3d(all_points)

    end = time.time()
    times[0] = end - start
    
    start = end

    # make instructions
    instr = make_instructions(action_rows)

    end = time.time()
    times[1] = end - start
    start = end

    # instruction to human readable
    shortend = {x: np.array(["sew togeteher"], dtype=str) for x in instr.keys()}
    
    for key in shortend.keys():
        sliding = window_sliding(instr[key])
        result, error = lang.run('<stdin>', str(sliding))
        shortend[key] = result

    end = time.time()
    times[2] = end - start
    start = end
    times[3] = np.sum(times[:-1])

    return shortend, all_points, faces, times, row_edges, column_edges



def run(file_path:str, stitch_width:float, startIndex:int = None):
    """
    Master function to create a crochet pattern

    Parameter
    ---------
    file_path:   str

    stitch_width: float
        distance between rows and points

    start_index: int
        index that exists in the file when read

    Returns
    --------
        instructions
            dict of instructions 

        all_points
            all sample points used for instructions

        sample_points
            dict of sampl points by row

        faces
            array of indeces into all points

        row_edges
            dict of edges in rows

        column_edges
            dict of edges between rows

        g_v
            all isoline points

        g_e
            all isoline edges
        
        isolines
            dict of all isoline points per row
    """

    v, f = igl.read_triangle_mesh(file_path)
    #find start index if non given
    if startIndex == None:
        try:
            startIndex = np.where((v == [0,0,0]).all(axis=1))[0][0]
        except IndexError:
            startIndex = 0
    else:
        startIndex = startIndex
    
    
    sample_points, isolines, g_v, g_e, iso_lines, times1 = mesh_to_sample(v, f, startIndex, stitch_width)
    
    instructions, all_points, faces, times2, row_edges, column_edges = sample_to_graph(sample_points)

    return instructions, all_points, sample_points, faces, row_edges, column_edges, g_v, g_e, isolines

def create_crochet_pattern(file_path:str, stitch_width:float, startIndex:int = None):
    """
    Create a full pattern

    Parameter
    ---------
    file_path:   str

    stitch_width: float
        distance between rows and points

    start_index: int
        index that exists in the file when read

    Returns
    --------
        instructions
            dict of instructions 

        sample_points
            dict of sampl points by row


    """

    v, f = igl.read_triangle_mesh(file_path)
    #find start index if non given
    if startIndex == None:
        try:
            startIndex = np.where((v == [0,0,0]).all(axis=1))[0][0]
        except IndexError:
            startIndex = 0
    else:
        startIndex = startIndex
    
    
    sample_points, isolines, g_v, g_e, iso_lines, times1 = mesh_to_sample(v, f, startIndex, stitch_width)
    
    instructions, all_points, faces, times2, row_edges, column_edges = sample_to_graph(sample_points)

    return instructions, sample_points