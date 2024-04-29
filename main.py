from flask import Flask, request, jsonify, make_response, send_file
from flask_cors import CORS
from PIL import Image 
import PIL
import base64
from io import BytesIO
import requests
import depthEstimation as DE
import open3d as o3d
import traceback
import numpy as np
import json
import os
import threading
import time
app = Flask(__name__)
CORS(app)

depthAI = DE.DepthEstimator()

@app.route('/getPCD', methods = ['POST','OPTIONS'])
def generatePCD():
    if request.method == 'OPTIONS':
        return cors_flight_response() #returns allow headers to avoid cors violations
    elif request.method == 'POST':
        try:
            req_data = request.json
            req_data = req_data['img']
            originalImg = postToImg(req_data) #Get pil image from b64 string
            depthMap = depthAI.predictDepthMap(originalImg) #Produce depth map
            return pcdGen(originalImg,depthMap),200 #return point cloud with ok status 200
        except Exception as e:
            traceback.print_exc() 
            return jsonify({'status': 'error', 'message': str(e)}), 400
    else:
         raise RuntimeError("Unable to handle method {}".format(request.method))

def postToImg(imgData):
    try:
        base_string = ""
        if "data:image/jpeg;base64," in imgData:
            base_string = imgData.replace("data:image/jpeg;base64,", "")
        elif "data:image/png;base64," in imgData:
            base_string = imgData.replace("data:image/png;base64,", "")
        decoded_img = base64.b64decode(base_string)
        img = PIL.Image.open(BytesIO(decoded_img)).convert("RGB")
        img.thumbnail((500,500))
        return img
    except Exception as e:
        status = "Error! = " + str(e)
                
def pcdGen(inputImg, depthMap):
    width, height = inputImg.size
    inputImg = np.array(inputImg)

    depthMap3d = o3d.geometry.Image(np.ascontiguousarray(depthMap))
    input3d = o3d.geometry.Image(np.ascontiguousarray(inputImg))
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(input3d, depthMap3d,convert_rgb_to_intensity=False)

    camIntrinsic = o3d.camera.PinholeCameraIntrinsic()
    camIntrinsic.set_intrinsics(width, height, 500, 500, width/2, height/2)

    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, camIntrinsic)

    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=30.0)
    pcd = pcd.select_by_index(ind)
    
    pcd.estimate_normals()
    
    pcd.orient_normals_to_align_with_direction([0,0,-1])
    pcd.rotate(pcd.get_rotation_matrix_from_xyz((np.pi,0,0)))

    resp = jsonify({'points': np.array(pcd.points).flatten().tolist(),
                    'point_normals':np.array(pcd.normals).flatten().tolist(),
                    'colors': np.array(pcd.colors).flatten().tolist()})
    resp.headers.add("Access-Control-Allow-Origin", "*")
    return resp

@app.route('/getMesh', methods = ['POST','OPTIONS'])
def getMeshFile():
    if request.method == 'OPTIONS':
        return cors_flight_response()
    elif request.method == 'POST':
        try:
            req_data = request.json
            filepath = req_data["filename"]
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(np.reshape(req_data["points"],(-1,3)))
            pcd.colors = o3d.utility.Vector3dVector(np.reshape(req_data["colors"],(-1,3)))
            pcd.normals = o3d.utility.Vector3dVector(np.reshape(req_data["point_normals"],(-1,3)))
            if filepath[:-4] == '.pcd':
                pcdToFile(pcd, filepath)
            else:
                pcdToMeshFile(pcd, filepath )
            resp = send_file(filepath, as_attachment=True)
            delete_thread = threading.Thread(target=deleteFile, args=(filepath,))
            delete_thread.start()
            return resp, 200
        except Exception as e:
            traceback.print_exc() 
            return jsonify({'status': 'error', 'message': str(e)}), 400
    else:
         raise RuntimeError("Weird - don't know how to handle method {}".format(request.method))

def deleteFile(filepath):
    time.sleep(120)
    os.remove(filepath)

def pcdFile(pcd, filepath):
    o3d.io.write_point_cloud(filepath,pcd, write_ascii=True)

def pcdToMeshFile(pcd,filepath):
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=10, linear_fit = False, n_threads=1)
    vertices_to_remove = densities < np.quantile(densities, 0.025)
    mesh.remove_vertices_by_mask(vertices_to_remove)
    mesh.filter_smooth_simple(number_of_iterations=5)
    
    o3d.io.write_triangle_mesh(filepath,mesh, write_ascii=True)
    

def cors_flight_response():
    response = make_response()
    response.headers.add("Access-Control-Allow-Origin", "*")
    response.headers.add('Access-Control-Allow-Headers', "*")
    response.headers.add('Access-Control-Allow-Methods', "*")
    return response


        
if __name__ == '__main__':
    app.run(port = 9999, host='0.0.0.0')
