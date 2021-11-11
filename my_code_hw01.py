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


#-- from the standard Python library and allowed external library
from scipy.spatial import ConvexHull
import math
#-----


#-- Constant value should not be changed
nodata_value = -9999
#-----

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
    hull = ConvexHull(points)
    raster = np.zeros((nrows, ncols))
    
    for i in range(nrows):
        for j in range(ncols):
            p = rowcol_to_xy(i, j, lowleft, nrows, cellsize)
            raster[i][j] = 1.0 if point_in_hull(p, hull) else nodata_value
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

    raster = nn(list_pts_3d, jparams)
    output_raster(raster, list_pts_3d, jparams)
    print("File written to", jparams['output-file'])


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

    print("File written to", jparams['output-file'])


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
    
    print("File written to", jparams['output-file'])



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
    
    print("File written to", jparams['output-file'])
