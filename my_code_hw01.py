#-- my_code_hw01.py
#-- hw01 GEO1015.2021
#-- [YOUR NAME]
#-- [YOUR STUDENT NUMBER] 
#-- [YOUR NAME]
#-- [YOUR STUDENT NUMBER] 


#-- import outside the standard Python library are not allowed, just those:
import math
import numpy as np
import scipy.spatial
import startinpy 
#-----


#-- Constant value should not be changed
nodata_value = -9999
#-----

import matplotlib.pyplot as plt # developing


def points2D(list_pts_3d):
    """
    Function that converts the 3d point list to 2d for computing convex hull, bbox, etc.
    Can get z value(elevation) by the point index.
    Input:
        list_pts_3d: the list of the input points (in 3D)
    Return:
        2d tuple, ((x1,y1),(x2,y2)...(xn,yn)) shouldn't change the sample point
        if need to, convert it into list/ndarray first
    """
    mypoints = []
    for item in list_pts_3d:
        mypoints.append((item[0],item[1]))
    return tuple(mypoints)


def point_in_hull(point, hull, tolerance=1e-8):
    """
    Function that identifies whether a point is in the convex hull.
    Input:
        point: a single point(x,y)
        hull: constructed hull object
        tolerance: 1e-8, if tolerance is infinity, all the points are considered inside the convex hull
    """
    return all((np.dot(eq[:-1], point) + eq[-1] <= tolerance) for eq in hull.equations)


def bounding_box(list_pts_3d):
    """
    Function that constructs the bounding box and return the points(lowleft and upright).
    Input:
        list_pts_3d: the list of the input points (in 3D)
    Return:
        (lowleft,upright):((min_x,min_y),(max_x,max_y))
    """
    points = points2D(list_pts_3d)
    min_x, min_y = np.min(points, axis=0)
    max_x, max_y = np.max(points, axis=0)
    lowleft = (min_x,min_y)
    upright = (max_x,max_y)
    return (lowleft,upright)


def get_size(list_pts_3d, jparams):
    """
    Function that gets the rows and cols of output raster.
    Input:
        list_pts_3d: the list of the input points (in 3D)
        jparams    : the parameters of the input for specific method
    Return:
        (nrows,ncols)
    """
    cellsize = jparams['cellsize'] # get cellsize from the json file
    lowleft = bounding_box(list_pts_3d)[0]
    upright = bounding_box(list_pts_3d)[1]  
    cal_row = (upright[0]-lowleft[0])/cellsize # x-axis
    cal_col = (upright[1]-lowleft[1])/cellsize # y-axis
    nrows = math.ceil(cal_row) # round-up
    ncols = math.ceil(cal_col) # round-up
    return (nrows,ncols)


def rowcol_to_xy(cur_row, cur_col, lowleft, nrows, cellsize):
    """
    Function that converts the row-col coordinate to xy-center coordinate.
    Input:    
    Return:
        the center coordinate of the cell: (x,y)
    """
    x = lowleft[0] + (cur_col+0.5)*cellsize
    y = lowleft[1] + (nrows-cur_row-0.5)*cellsize
    return (x,y)


def output_raster(raster, list_pts_3d, jparams):
    """
    Function for outputing raster to a standard  '.asc' file
    Return:
        raster: 2D ndarray
        file_nm: file name according to the json file
    """
    ncols = len(raster[0])
    nrows = len(raster)
    xllcorner = bounding_box(list_pts_3d)[0][0]
    yllcorner = bounding_box(list_pts_3d)[0][1]
    cellsize = jparams['cellsize']
    file_nm = jparams['output-file']

    # output attribute information
    with open(file_nm, 'w') as fh:
        fh.writelines('NCOLS' + ' ' + str(ncols) + '\n')
        fh.writelines('NROWS' + ' ' + str(nrows) + '\n')
        fh.writelines('XLLCORNER' + ' ' + str(xllcorner) + '\n')
        fh.writelines('YLLCORNER' + ' ' + str(yllcorner) + '\n')
        fh.writelines('CELLSIZE' + ' ' + str(cellsize) + '\n')
        fh.writelines('NODATA_VALUE' + ' ' + str(nodata_value) + '\n')

        # output raster cell values
        for row in raster:
            for col in row:
                fh.writelines(str(col)+' ')
            fh.writelines('\n')


def nn(list_pts_3d, jparams):
    """
    Function for nearest neighbour interpolation.  
    Return:
        raster: 2D ndarray
    """
    cellsize = jparams['cellsize']
    lowleft = bounding_box(list_pts_3d)[0]
    nrows = get_size(list_pts_3d, jparams)[0]
    ncols = get_size(list_pts_3d, jparams)[1]
    
    points = points2D(list_pts_3d)
    hull = scipy.spatial.ConvexHull(points)
    kd = scipy.spatial.KDTree(points)
    raster = np.zeros((nrows, ncols))
    
    for i in range(nrows):
        for j in range(ncols):
            center_pt = rowcol_to_xy(i, j, lowleft, nrows, cellsize)
            index = kd.query(center_pt,p=2, k=1)[1] # return the index of the nearest neighbour point
            value = list_pts_3d[index][2] # get the z value of the nn point
            raster[i][j] = value if point_in_hull(center_pt, hull) else nodata_value # assign the value
    return raster


def nn_interpolation(list_pts_3d, jparams):
    """
    !!! TO BE COMPLETED !!!
     
    Function that writes the output raster with nearest neighbour interpolation
     
    Input:
        list_pts_3d: the list of the input points (in 3D)
        jparams:     the parameters of the input for "nn"
    Output:
        (output file written to disk)
 
    """  
    # print("cellsize:", jparams['cellsize'])

    #-- to speed up the nearest neighbour us a kd-tree
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.KDTree.html#scipy.spatial.KDTree
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.KDTree.query.html#scipy.spatial.KDTree.query
    # kd = scipy.spatial.KDTree(list_pts)
    # d, i = kd.query(p, k=1)

    #raster = nn(list_pts_3d, jparams)
    #output_raster(raster, list_pts_3d, jparams)
    print("File written to", jparams['output-file'])


def idw_point_dist(pt_a, pt_b):
    """
    Function for calculating the distance between two points.
    Return: float
    """
    return math.sqrt((pt_b[0]-pt_a[0])**2 + (pt_b[1]-pt_a[1])**2)


def idw_circle_cal(center_pt, points, list_pts_3d, radius, power, kd):
    """
    Function for calculating the z value of the center point using IDW.
    Search shape: Circle
    Return: float
    !!Need to be finished with radius2, max and min points!!
    """
    nearby_pts_id = [] # index of nearby points
    for id in kd.query_ball_point(center_pt, radius):
        nearby_pts_id.append(id)

    if(len(nearby_pts_id)==0): return 0 # if no points within the given radius return 0
    else:
        weight_sum = 0
        value_sum = 0
        for id in nearby_pts_id:
            weight_sum += math.pow(idw_point_dist(center_pt, points[id]), -power)
            value_sum += math.pow(idw_point_dist(center_pt, points[id]), -power) * list_pts_3d[id][2]
        return (value_sum/weight_sum) if weight_sum != 0 else 0


def idw(list_pts_3d, jparams):
    """
    Function for idw interpolation.  
    Return:
        raster: 2D ndarray
    """
    cellsize = jparams['cellsize']
    radius = jparams['radius1'] # test radius2 also
    power = jparams['power']
    lowleft = bounding_box(list_pts_3d)[0]
    nrows = get_size(list_pts_3d, jparams)[0]
    ncols = get_size(list_pts_3d, jparams)[1]
    
    points = points2D(list_pts_3d)
    hull = scipy.spatial.ConvexHull(points)
    kd = scipy.spatial.KDTree(points)
    raster = np.zeros((nrows, ncols))

    for i in range(nrows):
        for j in range(ncols):
            center_pt = rowcol_to_xy(i, j, lowleft, nrows, cellsize)
            value = idw_circle_cal(center_pt, points, list_pts_3d, radius, power, kd)
            raster[i][j] = value if point_in_hull(center_pt, hull) else nodata_value # assign the value
    return raster


def idw_interpolation(list_pts_3d, jparams):
    """
    !!! TO BE COMPLETED !!!
     
    Function that writes the output raster with IDW
     
    Input:
        list_pts_3d: the list of the input points (in 3D)
        jparams:     the parameters of the input for "idw"
    Output:
        (output file written to disk)
 
    """  
    # print("cellsize:", jparams['cellsize'])
    # print("radius:", jparams['radius1'])
    # ...
    # print("radius:", jparams['angle'])

    #-- to speed up the nearest neighbour us a kd-tree
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.KDTree.html#scipy.spatial.KDTree
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.KDTree.query.html#scipy.spatial.KDTree.query
    # kd = scipy.spatial.KDTree(list_pts)
    # i = kd.query_ball_point(p, radius)

    #raster = idw(list_pts_3d, jparams)
    #output_raster(raster, list_pts_3d, jparams)
    print("File written to", jparams['output-file'])


def tin_area_triangle(center_pt, pt_a, pt_b):
    """
    Area of this triangle, using Heron's formula.
    Input:
        3 points. Format:(x,y)
    Return:
        float
    """
    a, b, c = idw_point_dist(center_pt, pt_a), idw_point_dist(center_pt, pt_b), idw_point_dist(pt_a, pt_b)
    if(a+b<=c or a+c<=b or b+c<=a): return 0.0 # check whether the triangle is "legal"
    if(a-b>=c or a-c>=b or b-c>=a): return 0.0
    s = (a+b+c)/2
    sum = math.sqrt(s*(s-a)*(s-b)*(s-c))
    return sum if (a>1e-8 and b>1e-8 and c>1e-8) else 0.0


def tin_cal(center_pt, list_pts_3d):
    """
    Calculate the z value of the unknown point using linear interpolation(TIN).
    """
    points = points2D(list_pts_3d)
    dt = scipy.spatial.Delaunay(points)
    if(len(dt.simplices)==0):
        print("Delaunay triangulation of input dataset constructed error")
        return None
    id = scipy.spatial.Delaunay.find_simplex(dt,center_pt)
    if(id==-1): return nodata_value # point outside of the tin

    a0 = tin_area_triangle(center_pt, points[dt.simplices[id][1]], points[dt.simplices[id][2]])
    a1 = tin_area_triangle(center_pt, points[dt.simplices[id][2]], points[dt.simplices[id][0]])
    a2 = tin_area_triangle(center_pt, points[dt.simplices[id][0]], points[dt.simplices[id][1]])
    total_value = 0
    total_value += list_pts_3d[dt.simplices[id][0]][2]*a0
    total_value += list_pts_3d[dt.simplices[id][1]][2]*a1
    total_value += list_pts_3d[dt.simplices[id][2]][2]*a2

    return total_value/(a0+a1+a2)


def tin(list_pts_3d, jparams):
    cellsize = jparams['cellsize']
    lowleft = bounding_box(list_pts_3d)[0]
    nrows = get_size(list_pts_3d, jparams)[0]
    ncols = get_size(list_pts_3d, jparams)[1]
    
    points = points2D(list_pts_3d)
    hull = scipy.spatial.ConvexHull(points)
    kd = scipy.spatial.KDTree(points)
    raster = np.zeros((nrows, ncols))

    for i in range(nrows):
        for j in range(ncols):
            center_pt = rowcol_to_xy(i, j, lowleft, nrows, cellsize)
            value = tin_cal(center_pt, list_pts_3d)
            raster[i][j] = value if point_in_hull(center_pt, hull) else nodata_value # assign the value
    return raster
    
    
def tin_interpolation(list_pts_3d, jparams):
    """
    !!! TO BE COMPLETED !!!
     
    Function that writes the output raster with linear in TIN interpolation
     
    Input:
        list_pts_3d: the list of the input points (in 3D)
        jparams:     the parameters of the input for "tin"
    Output:
        (output file written to disk)
 
    """  
    #-- example to construct the DT with scipy
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.Delaunay.html#scipy.spatial.Delaunay
    # dt = scipy.spatial.Delaunay([])

    #-- example to construct the DT with startinpy
    # minimal docs: https://github.com/hugoledoux/startinpy/blob/master/docs/doc.md
    # how to use it: https://github.com/hugoledoux/startinpy#examples
    # you are *not* allowed to use the function for the tin linear interpolation that I wrote for startinpy
    # you need to write your own code for this step
    # but you can of course read the code [dt.interpolate_tin_linear(x, y)]
    
    #raster = tin(list_pts_3d, jparams)
    #output_raster(raster, list_pts_3d, jparams)
    print("File written to", jparams['output-file'])


def circum_circle(dt,tri_ids):
    """
    return the coordinate of the circum circle of the triangle: (x,y)
    Input: 
        dt: the Delaunay triangulation object
        tri_ids [a1,a2,a3] the indices of the incident triangle of the inserted point
    the first indice, tri_ids[0] is the inserted interpolated point
    """
    p0, p1, p2 = dt.get_point(tri_ids[0]), dt.get_point(tri_ids[1]), dt.get_point(tri_ids[2])
    if(tin_area_triangle(p0,p1,p2)==0):
        print("No triangle, please check the conditions of laplace function")
        return None

    ax, ay = p0[0], p0[1] # x, y
    bx, by = p1[0], p1[1]
    cx, cy = p2[0], p2[1]

    # calculate center of circumcircle
    D = 2.0 * (ax*(by-cy) + bx*(cy-ay) + cx*(ay-by))
    if(D!=0):
        ux = ((ax*ax+ay*ay)*(by-cy) + (bx*bx+by*by)*(cy-ay) + (cx*cx+cy*cy)*(ay-by))/D
        uy = ((ax*ax+ay*ay)*(cx-bx) + (bx*bx+by*by)*(ax-cx) + (cx*cx+cy*cy)*(bx-ax))/D
        center = (ux,uy)
    else:
        print("Calculating error, please check the circum_circle function")
        return None
    return center


def laplace(list_pts_3d, jparams):
    dt = startinpy.DT()
    dt.insert(list_pts_3d)
    if(len(dt.all_triangles())==0): return None

    tolerance = dt.get_snap_tolerance()
 
    # dist of insert point and closest point<tolerance snap

    print("# vertices:", dt.number_of_vertices())
    print("# triangles:", dt.number_of_triangles())
    
    dt.insert_one_pt(0.5, 0.5, 20)
    print(dt.all_vertices())
    insert_pt = dt.all_vertices()[-1] # get coordinate of insert_pt
    insert_id = dt.number_of_vertices() # i.e. inital: 1,2,3 insert: 4

    # get the triangle list(may include parts of the infinity triangle )
    tri_id_list = dt.incident_triangles_to_vertex(insert_id)
   
    j = 0 
    for i in range(len(tri_id_list)):
        if 0 in tri_id_list[j]:
            tri_id_list.pop(j)
        else:
            j += 1
    print(tri_id_list)

    # identify whther the insert point is located on the boundary of the convexhull




    dt.remove(insert_id) # delete the insert point from DT 
    print(dt.all_vertices())
    print(dt.all_triangles())


    #tri_ids = dt.all_triangles()[0]
    #print(circum_circle(dt,tri_ids))


    #alltr = t.all_triangles()
    #print(alltr) # [[1,2,3],[2,4,3]]
    #print(t.get_point(4))
    #t.insert_one_pt(0.5, 0.5, 20)
    #print("# vertices:", t.number_of_vertices())
    #print(t.all_triangles())
    #t.remove(5)
    #print(t.all_triangles())

    #t.remove(4)
    #print(t.all_triangles())
    #t.insert_one_pt(0.5, 0.5, 20)
    #print(t.all_triangles())

    ##print(t.adjacent_vertices_to_vertex(1))
    #print(t.incident_triangles_to_vertex(4))
    #print(t.incident_triangles_to_vertex(1))

    # 获得邻近点的索引时，需要去掉索引=0的点; 索引=0的点代表-9999
    # 最邻近的点是从索引0开始，或者从右侧第一个点开始
    # 当插入点周围有三角形时，邻近点的索引中没有0
    # 对当前点的临近点: 加入数组，若数组中存在索引0，将其去除后再进行下一步操作

    # t.adjacent_vertices_to_vertex(2) [0, 4, 3, 1]
    # t.incident_triangles_to_vertex(2) [[2, 0, 4], [2, 4, 3], [2, 3, 1], [2, 1, 0]]
    # print(t.is_triangle([1, 0, 2])) # 三角形的三个顶点必须是CCW才可以返回true
    # 找到这个点的事件三角形，然后过滤含索引0的项，将过滤后的三角形索引列表作为计算edge值和两点距离的依据

    # 过滤列表中的特定值: 目前两种想法(1)list.pop 双指针 (2)维护一个新的列表 双列表
    #list_i = [[1,2],[2,0],[2,0],[3,1],[4,2],[3,0],[5,1]]
    #j = 0 
    #for i in range(len(list_i)):
    #    if 0 in list_i[j]:
    #        list_i.pop(j)
    #    else:
    #        j += 1
    #print(list_i)


def laplace_interpolation(list_pts_3d, jparams):
    """
    !!! TO BE COMPLETED !!!
     
    Function that writes the output raster with Laplace interpolation
     
    Input:
        list_pts_3d: the list of the input points (in 3D)
        jparams:     the parameters of the input for "laplace"
    Output:
        (output file written to disk)
 
    """  
    #-- example to construct the DT with scipy
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.Delaunay.html#scipy.spatial.Delaunay
    # dt = scipy.spatial.Delaunay([])

    #-- example to construct the DT with startinpy
    # minimal docs: https://github.com/hugoledoux/startinpy/blob/master/docs/doc.md
    # how to use it: https://github.com/hugoledoux/startinpy#examples
    # you are *not* allowed to use the function for the laplace interpolation that I wrote for startinpy
    # you need to write your own code for this step
    
    laplace(list_pts_3d, jparams)
    print("File written to", jparams['output-file'])
