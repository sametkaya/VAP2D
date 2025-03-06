from collections import deque

import bm4d as bm4d
import cv2
import numpy as np
import skimage
from skimage.morphology import disk
from skimage.util import img_as_float
from GraphicItems.VAP_Vein import VAP_Vein
from GraphicItems.VAP_Point import VAP_Point_Type, VAP_Point
from System import FilterKernels


def GetImageFormats(ImagePath):
    image_raw = cv2.imread(ImagePath)
    image_gray = cv2.cvtColor(image_raw, cv2.COLOR_BGR2GRAY)

    return image_raw, image_gray

def GetVascularAreaFractionValues(image_byte):
    pixelCount = image_byte.size
    whitePixelCount = np.sum(image_byte == 255)
    blackPixelCount = pixelCount-whitePixelCount
    ratio = (whitePixelCount * 100) /  pixelCount
    return ratio, whitePixelCount, blackPixelCount

def DenoiseImage(image_byte):
    image_fload = img_as_float(image_byte)
    imagef_bm4d = bm4d.bm4d(image_fload, sigma_psd=0.1)
    image_byte = ((imagef_bm4d - imagef_bm4d.min()) * (1 / (imagef_bm4d.max() - imagef_bm4d.min()) * 255)).astype('uint8')
    return image_byte

def Segment(image_byte):
    image_float = skimage.filters.gaussian(image_byte, sigma=1)
    image_isodata_tresh = skimage.filters.threshold_isodata(image_float)
    image_isodata_uint8 = ((image_float > image_isodata_tresh) * 255).astype('uint8')
    return image_isodata_uint8

def BinaryClosing(image_byte):
    image_bool = 0 < image_byte
    footprint = disk(2)
    image_binary_closed_bool = skimage.morphology.binary_closing(image_bool, footprint)
    image_binary_closed_uint8 = (image_binary_closed_bool * 255).astype('uint8')
    return image_binary_closed_uint8

def RemoveSmallObject(image_byte):
    image_bool = 0 < image_byte
    image_removed_bool = skimage.morphology.remove_small_objects(image_bool, 40)
    image_removed_uint8 = (image_removed_bool * 255).astype('uint8')
    return image_removed_uint8

def Skeletonize(image_byte):
    image_bool = 0 < image_byte
    image_skel_bool = skimage.morphology.skeletonize(image_bool)
    image_skel_uint8 = (image_skel_bool * 255).astype('uint8')
    return image_skel_uint8

def find_branch_pts(image_skel_uint8):
    branch_points_kernels = FilterKernels.BranchPointKernels()
    #image_skel_padded = np.pad(image_skel_uint8, pad_width=1)
    image_skel_padded = np.pad(image_skel_uint8, pad_width=1, mode='constant', constant_values=0)

    image_branch_pts_padded = np.zeros(image_skel_padded.shape[:2], dtype=int)
    # Store branch points
    for branch_point_kernel in branch_points_kernels:
        image_branch_pts_padded = np.logical_or(cv2.morphologyEx(image_skel_padded, op=cv2.MORPH_HITMISS, kernel=branch_point_kernel), image_branch_pts_padded)

    image_branch_pts_padded = image_branch_pts_padded.astype(np.uint8) * 255

    branch_objects, _ = cv2.findContours(image_branch_pts_padded, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[-2:]
    vap_branch_list = []
    for i, branch in enumerate(branch_objects):
        x, y = branch.ravel()[:2]
        vap_branch_list.append(VAP_Point(y-1, x-1, VAP_Point_Type.BRANCH))
    return vap_branch_list
def find_tips(image_skel_uint8):

    tip_points_kernels = FilterKernels.TipPointKernels()
    image_skel_padded = np.pad(image_skel_uint8, pad_width=1, mode='constant', constant_values=1)
    image_tip_pts = np.zeros(image_skel_padded.shape[:2], dtype=int)
    #image_skel_padded = np.pad(skel_img, pad_width=1)
    for tip_point_kernel in tip_points_kernels:
        image_tip_pts = np.logical_or(cv2.morphologyEx(image_skel_padded, op=cv2.MORPH_HITMISS, kernel=tip_point_kernel), image_tip_pts)

    image_tip_pts = image_tip_pts.astype(np.uint8) * 255

    tip_objects, _ = cv2.findContours(image_tip_pts, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[-2:]

    vap_tip_list= []
    for i, tip in enumerate(tip_objects):
        x, y = tip.ravel()[:2]
        vap_tip_list.append(VAP_Point(y-1, x-1, VAP_Point_Type.TIP))

    return vap_tip_list

def find_veins(image_skel_uint8):

    branch_points_kernels = FilterKernels.BranchPointKernels()

    tip_points_kernels = FilterKernels.TipPointKernels()

    image_skel_padded = np.pad(image_skel_uint8, pad_width=1)

    branch_pts_img_padded = np.zeros(image_skel_padded.shape[:2], dtype=np.uint8)
    image_skel_padded = np.pad(image_skel_uint8, pad_width=1)
    # Store branch points
    for kernel in branch_points_kernels:
        branch_pts_img_padded = np.logical_or(cv2.morphologyEx(image_skel_padded, op=cv2.MORPH_HITMISS, kernel=kernel), branch_pts_img_padded)

    branch_points = np.transpose(np.nonzero(branch_pts_img_padded.astype(np.uint8)))

    # Store tip points
    tip_pts_img_padded = np.zeros(image_skel_padded.shape[:2], dtype=np.uint8)
    for kernel in tip_points_kernels:
        tip_pts_img_padded = np.logical_or(cv2.morphologyEx(image_skel_padded, op=cv2.MORPH_HITMISS, kernel=kernel,borderType=cv2.BORDER_CONSTANT, borderValue=0), tip_pts_img_padded)

    #tip_pts_img_padded_uint8= tip_pts_img_padded.astype(np.uint8)
    tip_points=np.transpose(np.nonzero(tip_pts_img_padded.astype(np.uint8)))

    #none 0, path 1, tip 2, brach 3
    image_points_padded = np.logical_or(image_skel_padded, image_skel_padded).astype(np.uint8) + branch_pts_img_padded.astype(np.uint8)* 2 + tip_pts_img_padded.astype(np.uint8)
    points= np.concatenate((branch_points,tip_points))

    drawed_skel_img_padded = np.zeros(image_skel_padded.shape[:2], dtype=np.uint8)
    #All 8 directions
    #delta = [(-1, -1), (-1, 0), (-1, 1),
    #         (0, -1), (0, 1),
    #         (1, -1), (1, 0), (1, 1)]

    delta = [(-1, 0), (0, 1), (0, -1), (1, 0), (-1, -1), (-1, 1), (1, -1), (1, 1)]
    vap_vein_list = []
    point_id = 1
    for point in points:
        y, x = point

        if (drawed_skel_img_padded[y][x] == VAP_Point_Type.NONE.value):
            scan_veins_bfs = deque([])
            if (image_points_padded[y][x] == VAP_Point_Type.BRANCH.value): # if branch point
                vap_branch_point = VAP_Point(y-1, x-1, VAP_Point_Type.BRANCH)
                for dy, dx in delta:
                    yy, xx = y + dy, x + dx
                    # If the next position hasn't already been looked at and it's white
                    if image_points_padded[yy][xx] > 0 and drawed_skel_img_padded[yy][xx] == 0:
                        #vap_vein = VAP_Vein()
                        scan_vein = VAP_Vein(point_id)
                        point_id = point_id + 1
                        vap_point = VAP_Point(yy - 1, xx - 1, image_points_padded[yy][xx])
                        scan_vein.branch_points.append(vap_branch_point)
                        scan_vein.vap_point_list.append(vap_point)
                        drawed_skel_img_padded[yy][xx] = scan_vein.idn
                        scan_veins_bfs.append(scan_vein)

            elif (image_points_padded[y][x] == VAP_Point_Type.TIP.value): # if tip point
                vap_point = VAP_Point(y-1, x-1, VAP_Point_Type.TIP)
                scan_vein = VAP_Vein(point_id)
                point_id = point_id + 1
                scan_vein.vap_point_list.append(vap_point)
                scan_vein.tip_points.append(vap_point)
                scan_veins_bfs.append(scan_vein)
                drawed_skel_img_padded[y][x] = VAP_Point_Type.TIP.value
            else:
                vap_point = VAP_Point(y-1, x-1, VAP_Point_Type.PATH)
                # vap_vein = VAP_Vein()
                scan_vein = VAP_Vein(point_id)
                point_id = point_id + 1
                scan_vein.vap_point_list.append(vap_point)
                drawed_skel_img_padded[y][x] = scan_vein.idn
                scan_veins_bfs.append(scan_vein)

            while len(scan_veins_bfs) > 0:
                scan_vein = scan_veins_bfs.popleft()
                vap_point = scan_vein.vap_point_list[0]
                padded_point = VAP_Point(vap_point.y+1, vap_point.x+1, vap_point.vp_type)
                vap_point_bfs = deque([padded_point])
                while len(vap_point_bfs) > 0:

                    p = vap_point_bfs.popleft()

                    for dy, dx in delta: # directions
                        yy, xx = p.y + dy, p.x + dx

                        next_vap_point = VAP_Point(yy, xx, VAP_Point_Type(image_points_padded[yy][xx]))

                        if(image_points_padded[yy][xx]>0 and drawed_skel_img_padded[yy][xx]==0 ):

                            if (image_points_padded[yy][xx] == VAP_Point_Type.PATH.value):

                                drawed_skel_img_padded[yy][xx] = scan_vein.idn
                                vap_point = VAP_Point(yy-1, xx-1, VAP_Point_Type.PATH)
                                scan_vein.vap_point_list.append(vap_point)
                                vap_point_bfs.append(next_vap_point)
                                break
                            elif(image_points_padded[yy][xx] == VAP_Point_Type.TIP.value):
                                drawed_skel_img_padded[yy][xx] = VAP_Point_Type.TIP.value
                                vap_point = VAP_Point(yy-1, xx-1, VAP_Point_Type.TIP)
                                if (vap_point in scan_vein.tip_points):
                                    continue
                                scan_vein.tip_points.append(vap_point)
                                scan_vein.vap_point_list.append(vap_point)
                                break
                            elif (image_points_padded[yy][xx] == VAP_Point_Type.BRANCH.value):

                                vap_point = VAP_Point(yy-1, xx-1, VAP_Point_Type.BRANCH)
                                if (vap_point in scan_vein.branch_points and len(scan_vein.vap_point_list) < 2):
                                    continue
                                scan_vein.branch_points.append(vap_point)
                                break
                scan_vein.Build_Up()
                if (len(scan_vein.vap_point_list) > 1) or (len(scan_vein.branch_points) > 1):
                    scan_vein.idn = len(vap_vein_list) + 1
                    vap_vein_list.append(scan_vein)

    return vap_vein_list


def Analyze(image_skel_uint8):

    branch_points_kernels = FilterKernels.BranchPointKernels()

    tip_points_kernels = FilterKernels.TipPointKernels()

    image_skel_padded = np.pad(image_skel_uint8, pad_width=1)

    branch_pts_img_padded = np.zeros(image_skel_padded.shape[:2], dtype=np.uint8)
    image_skel_padded = np.pad(image_skel_uint8, pad_width=1)
    # Store branch points
    for kernel in branch_points_kernels:
        branch_pts_img_padded = np.logical_or(cv2.morphologyEx(image_skel_padded, op=cv2.MORPH_HITMISS, kernel=kernel), branch_pts_img_padded)

    branch_points = np.transpose(np.nonzero(branch_pts_img_padded.astype(np.uint8)))

    # Store tip points
    tip_pts_img_padded = np.zeros(image_skel_padded.shape[:2], dtype=np.uint8)
    for kernel in tip_points_kernels:
        tip_pts_img_padded = np.logical_or(cv2.morphologyEx(image_skel_padded, op=cv2.MORPH_HITMISS, kernel=kernel,borderType=cv2.BORDER_CONSTANT, borderValue=0), tip_pts_img_padded)

    #tip_pts_img_padded_uint8= tip_pts_img_padded.astype(np.uint8)
    tip_points=np.transpose(np.nonzero(tip_pts_img_padded.astype(np.uint8)))

    #none 0, path 1, tip 2, brach 3
    image_points_padded = np.logical_or(image_skel_padded, image_skel_padded).astype(np.uint8) + branch_pts_img_padded.astype(np.uint8)* 2 + tip_pts_img_padded.astype(np.uint8)
    points= np.concatenate((branch_points,tip_points))

    drawed_skel_img_padded = np.zeros(image_skel_padded.shape[:2], dtype=np.uint8)
    #All 8 directions
    #delta = [(-1, -1), (-1, 0), (-1, 1),
    #         (0, -1), (0, 1),
    #         (1, -1), (1, 0), (1, 1)]

    delta = [(-1, 0), (0, 1), (0, -1), (1, 0), (-1, -1), (-1, 1), (1, -1), (1, 1)]
    vap_vein_list = []
    point_id = 1
    for point in points:
        y, x = point

        if (drawed_skel_img_padded[y][x] == VAP_Point_Type.NONE.value):
            scan_veins_bfs = deque([])
            if (image_points_padded[y][x] == VAP_Point_Type.BRANCH.value): # if branch point
                vap_branch_point = VAP_Point(y-1, x-1, VAP_Point_Type.BRANCH)
                for dy, dx in delta:
                    yy, xx = y + dy, x + dx
                    # If the next position hasn't already been looked at and it's white
                    if image_points_padded[yy][xx] > 0 and drawed_skel_img_padded[yy][xx] == 0:
                        #vap_vein = VAP_Vein()
                        scan_vein = VAP_Vein(point_id)
                        point_id = point_id + 1
                        vap_point = VAP_Point(yy - 1, xx - 1, image_points_padded[yy][xx])
                        scan_vein.branch_points.append(vap_branch_point)
                        scan_vein.vap_point_list.append(vap_point)
                        drawed_skel_img_padded[yy][xx] = scan_vein.idn
                        scan_veins_bfs.append(scan_vein)

            elif (image_points_padded[y][x] == VAP_Point_Type.TIP.value): # if tip point
                vap_point = VAP_Point(y-1, x-1, VAP_Point_Type.TIP)
                scan_vein = VAP_Vein(point_id)
                point_id = point_id + 1
                scan_vein.vap_point_list.append(vap_point)
                scan_vein.tip_points.append(vap_point)
                scan_veins_bfs.append(scan_vein)
                drawed_skel_img_padded[y][x] = VAP_Point_Type.TIP.value
            else:
                vap_point = VAP_Point(y-1, x-1, VAP_Point_Type.PATH)
                # vap_vein = VAP_Vein()
                scan_vein = VAP_Vein(point_id)
                point_id = point_id + 1
                scan_vein.vap_point_list.append(vap_point)
                drawed_skel_img_padded[y][x] = scan_vein.idn
                scan_veins_bfs.append(scan_vein)

            while len(scan_veins_bfs) > 0:
                scan_vein = scan_veins_bfs.popleft()
                vap_point = scan_vein.vap_point_list[0]
                padded_point = VAP_Point(vap_point.y+1, vap_point.x+1, vap_point.vp_type)
                vap_point_bfs = deque([padded_point])
                while len(vap_point_bfs) > 0:

                    p = vap_point_bfs.popleft()

                    for dy, dx in delta: # directions
                        yy, xx = p.y + dy, p.x + dx

                        next_vap_point = VAP_Point(yy, xx, VAP_Point_Type(image_points_padded[yy][xx]))

                        if(image_points_padded[yy][xx]>0 and drawed_skel_img_padded[yy][xx]==0 ):

                            if (image_points_padded[yy][xx] == VAP_Point_Type.PATH.value):

                                drawed_skel_img_padded[yy][xx] = scan_vein.idn
                                vap_point = VAP_Point(yy-1, xx-1, VAP_Point_Type.PATH)
                                scan_vein.vap_point_list.append(vap_point)
                                vap_point_bfs.append(next_vap_point)
                                break
                            elif(image_points_padded[yy][xx] == VAP_Point_Type.TIP.value):
                                drawed_skel_img_padded[yy][xx] = VAP_Point_Type.TIP.value
                                vap_point = VAP_Point(yy-1, xx-1, VAP_Point_Type.TIP)
                                if (vap_point in scan_vein.tip_points):
                                    continue
                                scan_vein.tip_points.append(vap_point)
                                scan_vein.vap_point_list.append(vap_point)
                                break
                            elif (image_points_padded[yy][xx] == VAP_Point_Type.BRANCH.value):

                                vap_point = VAP_Point(yy-1, xx-1, VAP_Point_Type.BRANCH)
                                if (vap_point in scan_vein.branch_points and len(scan_vein.vap_point_list) < 2):
                                    continue
                                scan_vein.branch_points.append(vap_point)
                                break
                scan_vein.Build_Up()
                if (len(scan_vein.vap_point_list) > 1) or (len(scan_vein.branch_points) > 1):
                    scan_vein.idn = len(vap_vein_list) + 1
                    vap_vein_list.append(scan_vein)

        #convert tip point to vap_point
        tip_pts_img_padded = tip_pts_img_padded.astype(np.uint8) * 255

        tip_objects, _ = cv2.findContours(tip_pts_img_padded, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[-2:]

        vap_tip_list = []
        for i, tip in enumerate(tip_objects):
            x, y = tip.ravel()[:2]
            vap_tip_list.append(VAP_Point(y - 1, x - 1, VAP_Point_Type.TIP))

        #convert branch point to vap_point
        branch_pts_img_padded = branch_pts_img_padded.astype(np.uint8) * 255

        branch_objects, _ = cv2.findContours(branch_pts_img_padded, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[-2:]
        vap_branch_list = []
        for i, branch in enumerate(branch_objects):
            x, y = branch.ravel()[:2]
            vap_branch_list.append(VAP_Point(y - 1, x - 1, VAP_Point_Type.BRANCH))

    return vap_vein_list, vap_branch_list, vap_tip_list

