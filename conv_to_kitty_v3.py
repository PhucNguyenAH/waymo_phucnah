import os

import glob

import pandas as pd
import numpy as np
import tensorflow as tf
from waymo_open_dataset import v2
#from pandas_tfrecords import pd2tf
from waymo_open_dataset.v2.perception.utils import lidar_utils
import pyarrow.parquet as pq
from IPython.display import display
import hashlib
import traceback

from tqdm import tqdm
import re

def read_by_row_group(file_name):
    parquet_file = pq.ParquetFile(file_name)
    for row_group in range(parquet_file.num_row_groups):
        yield parquet_file.read_row_group(row_group).to_pandas()

class Converter:
    def __init__(self, root='/data/aaji/Waymo/data'):
        self.root = root
        self.train_path = os.path.join(root, "training")
        self.val_path = os.path.join(root, "validation")
        self.test_path = os.path.join(root, "testing")

        self.lidar_tr_pths = sorted(glob.glob(self.train_path+"/lidar/*.parquet"))
        #print(self.lidar_tr_pths)
        self.lidar_vl_pths = sorted(glob.glob(self.val_path+"/lidar/*.parquet"))
        self.lidar_ts_pths = sorted(glob.glob(self.test_path+"/lidar/*.parquet"))

        self.lseg_tr_pths = sorted(glob.glob(self.train_path+"/lidar_segmentation/*.parquet"))
        self.lseg_val_pths = sorted(glob.glob(self.val_path+"/lidar_segmentation/*.parquet"))
        self.lseg_ts_pths = sorted(glob.glob(self.test_path+"/lidar_segmentation/*.parquet"))

        self.lpose_tr_pths = sorted(glob.glob(self.train_path+"/lidar_pose/*.parquet"))
        self.lpose_val_pths = sorted(glob.glob(self.val_path+"/lidar_pose/*.parquet"))
        self.lpose_ts_pths = sorted(glob.glob(self.test_path+"/lidar_pose/*.parquet"))

        self.v_pose_tr_pths = sorted(glob.glob(self.train_path+"/vehicle_pose/*.parquet"))
        self.v_pose_val_pths = sorted(glob.glob(self.val_path+"/vehicle_pose/*.parquet"))
        self.v_pose_ts_pths = sorted(glob.glob(self.test_path+"/vehicle_pose/*.parquet"))

        self.calib_tr_pths = sorted(glob.glob(self.train_path+"/lidar_calibration/*.parquet"))
        self.calib_val_pths = sorted(glob.glob(self.val_path+"/lidar_calibration/*.parquet"))
        self.calib_ts_paths = sorted(glob.glob(self.test_path+"/lidar_calibration/*.parquet"))

    def __getstate__(self):
        # Define how to serialize the object
        return {'root': self.root}

    def __setstate__(self, state):
        # Define how to deserialize the object
        self.root = state['root']
        self.__init__(self.root)

    def __hash__(self):
        # Provide a deterministic hash based on the relevant attributes
        return int(hashlib.md5(self.root.encode('utf-8')).hexdigest(), 16)

    def gather(self, idx, subset='tr'):
        if subset == 'tr':
            lidar_tr_df = pd.concat(read_by_row_group(self.lidar_tr_pths[idx]))
            lseg_tr_df = pd.concat(read_by_row_group(self.lseg_tr_pths[idx]))
            lpose_tr_df = pd.concat(read_by_row_group(self.lpose_tr_pths[idx]))
            vpos_tr_df = pd.concat(read_by_row_group(self.v_pose_tr_pths[idx]))
            lcalib_tr_df = pd.concat(read_by_row_group(self.calib_tr_pths[idx]))
    
            train_df = v2.merge(lidar_tr_df, lseg_tr_df)
            train_df = v2.merge(train_df, vpos_tr_df)
            train_df = v2.merge(train_df, lcalib_tr_df)
            train_df = v2.merge(train_df, lpose_tr_df, right_group=True, left_group=True)
    
            return train_df
    
        elif subset == 'vl':
            lidar_vl_df = pd.concat(read_by_row_group(self.lidar_vl_pths[idx]))
            lseg_vl_df = pd.concat(read_by_row_group(self.lseg_val_pths[idx]))
            lpose_vl_df = pd.concat(read_by_row_group(self.lpose_val_pths[idx]))
            vpos_vl_df = pd.concat(read_by_row_group(self.v_pose_val_pths[idx]))
            lcalib_vl_df = pd.concat(read_by_row_group(self.calib_val_pths[idx]))
    
            val_df = v2.merge(lidar_vl_df, lseg_vl_df)
            val_df = v2.merge(val_df, vpos_vl_df)
            val_df = v2.merge(val_df, lcalib_vl_df)
            val_df = v2.merge(val_df, lpose_vl_df, right_group=True, left_group=True)
    
            return val_df
    
        elif subset == 'ts':
            #print(f"\nThis is the test set base path:\n {os.path.basename(self.lidar_ts_pths[idx])}\n")
            lidar_ts_df = pd.concat(read_by_row_group(self.lidar_ts_pths[idx]))
            #lseg_ts_df = pd.concat(read_by_row_group(self.lseg_test_pths[idx]))
            lpose_ts_df = pd.concat(read_by_row_group(self.lpose_ts_pths[idx]))
            vpos_ts_df = pd.concat(read_by_row_group(self.v_pose_ts_pths[idx]))
            lcalib_ts_df = pd.concat(read_by_row_group(self.calib_ts_paths[idx]))
            
            test_df = v2.merge(lidar_ts_df, vpos_ts_df)
            test_df = v2.merge(test_df, lcalib_ts_df)
            test_df = v2.merge(test_df, lpose_ts_df, right_group=True, left_group=True)

            return test_df

    def apply_point_clouds(self, row):
        lidar = v2.LiDARComponent.from_dict(row)
        range_imgs = lidar.range_image_returns
        calib_comp = v2.LiDARCalibrationComponent.from_dict(row)
        vpose_comp = v2.VehiclePoseComponent.from_dict(row)
        lpose_comp = v2.LiDARPoseComponent.from_dict(row)
        lpose_comp = lpose_comp.range_image_return1
        
        point_cld1 = v2.convert_range_image_to_point_cloud(range_image=range_imgs[0], calibration=calib_comp, frame_pose=vpose_comp, pixel_pose=lpose_comp, keep_polar_features=True).numpy()
        #print("Point Cloud 1 Shape: ", point_cld1.shape)
        #point_cld1 = np.concatenate(point_cld1.tolist(), axis=0)
        #print("Point Cloud 1 Shape after concat: ", point_cld1.shape)

        point_cld2 = v2.convert_range_image_to_point_cloud(range_image=range_imgs[1], calibration=calib_comp, frame_pose=vpose_comp, pixel_pose=lpose_comp, keep_polar_features=True).numpy()
        #print("Point Cloud 2 Shape: ", point_cld2.shape)
        #point_cld2 = np.concatenate(point_cld2.tolist(), axis=0)
        #print("Point Cloud 2 Shape after concat: ", point_cld2.shape)

        point_cld = np.concatenate([point_cld1, point_cld2], axis=0)
        #print("Final Combined Point Cloud Shape: ", point_cld.shape)
        velodyne = np.c_[point_cld[:, 3:6], point_cld[:, 1]]
        velodyne = velodyne.reshape((velodyne.shape[0] * velodyne.shape[1]))
        
        return velodyne
    
    def apply_labels(self, row):
        lidar = v2.LiDARComponent.from_dict(row)
        range_imgs = lidar.range_image_returns
        segmentation = v2.LiDARSegmentationLabelComponent(key=lidar.key,range_image_return1=range_imgs[0], range_image_return2=range_imgs[1])
        segments = segmentation.range_image_returns
       
        l_tensor1 = lidar.range_image_returns[0].tensor
        l_tensor2 = lidar.range_image_returns[1].tensor

        l_mask1 = l_tensor1[..., 0] > 0
        l_mask2 = l_tensor2[..., 0] > 0

        s_tensor1 = segments[0].tensor
        s_tensor2 = segments[1].tensor

        final_tensor1 = tf.gather_nd(s_tensor1, tf.where(l_mask1))
        final_tensor2 = tf.gather_nd(s_tensor2, tf.where(l_mask2))
        final_labelled_points = tf.concat([final_tensor1, final_tensor2], axis=0)
        final_labelled_points = final_labelled_points.numpy()

        return final_labelled_points

    def create_point_clouds_and_labels(self, subset, idx):
        df = self.gather(idx, subset)
        df['pcl'] = df.apply(self.apply_point_clouds, axis=1)
        if subset != 'ts':
            df['labels'] = df.apply(self.apply_labels, axis=1)
            return df[['pcl', 'labels']]
        else:
            return df 
        
    
    def process_files(self, output_path="./data_preproc"):
        subsets = [ 'tr', 'vl','ts']
        # make output directory regardless if it already exists
        os.makedirs(output_path, exist_ok=True)
        # make the training, validation and test sub-directories
        os.makedirs(os.path.join(output_path, 'training'), exist_ok=True)
        #os.makedirs(os.path.join(output_path, 'validation'), exist_ok=True)
        os.makedirs(os.path.join(output_path, 'testing'), exist_ok=True)
        # make a labels sub-folder within training and validation to store labels
        os.makedirs(os.path.join(output_path, 'training', 'labels'), exist_ok=True)
        os.makedirs(os.path.join(output_path, 'validation', 'labels'), exist_ok=True)
        
        for subset in subsets:
            if subset == 'tr':
                num_files = len(self.lidar_tr_pths)
            elif subset == 'vl':
                num_files = len(self.lidar_vl_pths)
            elif subset == 'ts':
                num_files = len(self.lidar_ts_pths)
            
            count = 0
            for idx in tqdm(range(num_files), desc=f'Processing subset: {subset}'):
                try:
                    pcl_and_labels = self.create_point_clouds_and_labels(subset, idx)
                    pcl = pcl_and_labels['pcl']
                    
                    #print("Point Cloud: /n", pcl)
                    pcl_list = pcl.to_numpy().tolist()
        
                    if subset != 'ts':
                        labels = pcl_and_labels['labels']
                        labels_list = labels.to_numpy().tolist()
                    
                    if subset == 'tr':
                        save_path = os.path.join(output_path, 'training')
                    elif subset == 'vl':
                        save_path = os.path.join(output_path, 'validation')
                    elif subset == 'ts':
                        save_path = os.path.join(output_path, 'testing')
                    
                    # Ensure the 'velodyne' directory exists
                    os.makedirs(os.path.join(save_path, 'velodyne'), exist_ok=True)
                    if subset != 'ts':
                        # Ensure the 'labels' directory exists
                        os.makedirs(os.path.join(save_path, 'labels'), exist_ok=True)
                        
                        for pcl, labels in zip(pcl_list, labels_list):
                            file_idx = "0" * (6 - len(str(count))) + str(count)
                            pcl.tofile(os.path.join(save_path, 'velodyne', f'{file_idx}.bin'))
                            labels.tofile(os.path.join(save_path, 'labels', f'{file_idx}.label'))
                            count += 1
                    else:
                        for pcl in pcl_list:
                            file_idx = "0" * (6 - len(str(count))) + str(count)
                            pcl.tofile(os.path.join(save_path, 'velodyne', f'{file_idx}.bin'))
                            count += 1

                except Exception as e:
                    print("Exception occured: \n", e)
                    traceback.print_exc()
                    continue                
def test():
    print("Running test...")
    #client = Client()
    data_converter = Converter()
    data_converter.create_point_clouds_and_labels('tr', 0)
    
    print("Test Ran Successfully!")
    #client.close()

if __name__ == "__main__":
    test()
    print("Processing all files...\n")
    #client = Client()
    data_converter = Converter()
    data_converter.process_files()
    #client.close()
    print("Files processed successfully!")