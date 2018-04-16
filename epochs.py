"""Code for (1) parsing DICOMS and contour files and (2) generating batches/epochs for training a convolutional neural network"""

import pydicom
import numpy as np
import pandas as pd
import os
from PIL import Image, ImageDraw

class create_epoch():
    """
    This creates a new epoch and returns the batches.
    :param all_files_dir: top-level folder containing the images and contours
    :return: numpy arrays containing the images and the target masks
    """
    def __init__(self,all_files_dir,batch_size=8):
        self.all_files_dir = all_files_dir
        self.file_info_df = create_file_info_df(all_files_dir)
        self.shuffled_df = self.file_info_df.sample(frac=1).copy()
        self.batch_size = batch_size
        self.batch_start = 0
        self.random_state = 0

    def get_current_batch(self):
        if self.batch_start >= self.file_info_df.shape[0]:
            return None,None
        else:
            current_batch = self.shuffled_df.iloc[self.batch_start:np.min((self.batch_start+self.batch_size,self.file_info_df.shape[0]))].copy().reset_index()
            images = []
            targets = []
            for idx, row in current_batch.iterrows():
                dicom_img = parse_dicom_file(row.dicom_file_name_with_path)
                img_mask = poly_to_mask(row.o_coords, dicom_img.shape[0], dicom_img.shape[1])
                images.append(dicom_img)
                targets.append(img_mask)
            images = np.array(images)
            targets = np.array(targets)
            self.batch_start += self.batch_size
            return images, targets

    def new_epoch(self):
        self.random_state+=1
        self.shuffled_df = self.file_info_df.sample(frac=1,random_state=self.random_state).copy()
        self.batch_start = 0

    def get_shuffled_df(self):
        return self.shuffled_df


def create_file_info_df(all_files_dir):
    """
    This creates a dataframe containing the contours matched with the patient names.
    :param all_files_dir: top-level directory containing the contours and dicoms
    :return file_info_df: dataframe containing the filenames and coordinates of the contours and dicoms
    """
    link_file = os.path.join(all_files_dir, 'link_edited.csv')
    try:
        link = pd.read_csv(link_file)
    except:
        print('Error: link file does not exist')
        return None
    if not os.path.isdir(os.path.join(all_files_dir, 'masks')):
        os.mkdir(os.path.join(all_files_dir, 'masks'))


    file_info_df = pd.DataFrame(columns=['image_num'])#,
                                         # 'patient_num'
                                         # 'original_id',
                                         # 'patient_id',
                                         # 'o_contour_file_name_with_path',
                                         # 'i_contour_file_name_with_path',
                                         # 'dicom_file_name_with_path',
                                         # 'img_size'
                                         # 'o_coords',
                                         # 'i_coords'])
    for idx, row in link.iterrows():
        patient_num = int(row.patient_id.split('SCD0000')[1].split('01')[0])
        # check for errors with the files and reduce the dataset depending on these errors
        try:
            # first check for the o-contour, then check for the i-contour
            o_contourdir = os.path.join(all_files_dir, 'contourfiles', row.original_id, 'o-contours')
            o_contour_files = os.listdir(o_contourdir)
            dicomdir = os.path.join(all_files_dir, 'dicoms', row.patient_id)
            i_contourdir = os.path.join(all_files_dir, 'contourfiles', row.original_id, 'i-contours')
            for filename in o_contour_files:
                image_nums = []
                patient_nums = []
                original_id = []
                patient_id = []
                o_contour_file_names_with_path = []
                i_contour_file_names_with_path = []
                dicom_file_names_with_path = []
                o_coords = []
                i_coords = []

                if 'IM-0001' in filename:
                    try:
                        image_num = int(filename.split('-')[2])
                        image_nums.append(image_num)

                        patient_nums.append(patient_num)
                        original_id.append(row.original_id)
                        patient_id.append(row.patient_id)

                        o_contour_file_names_with_path.append(os.path.join(o_contourdir, filename))
                        i_contour_filename = filename.replace('ocontour','icontour')
                        i_contour_file_names_with_path.append(os.path.join(i_contourdir, i_contour_filename))

                        dicomfile = os.path.join(dicomdir, str(image_num) + '.dcm')
                        dicom_file_names_with_path.append(dicomfile)

                        o_coord = parse_contour_file(os.path.join(o_contourdir, filename))
                        o_coords.append(o_coord)

                        i_coord = parse_contour_file(os.path.join(i_contourdir, i_contour_filename))
                        i_coords.append(i_coord)
                    except:
                        print('Warning: '+filename+' not read')

                temp_df = pd.DataFrame([image_nums], columns=['image_num'])
                temp_df['patient_num'] = patient_nums
                temp_df['original_id'] = original_id
                temp_df['patient_id'] = patient_id
                temp_df['o_contour_file_name_with_path'] = o_contour_file_names_with_path
                temp_df['i_contour_file_name_with_path'] = i_contour_file_names_with_path
                temp_df['dicom_file_name_with_path'] = dicom_file_names_with_path
                temp_df['o_coords'] = o_coords
                temp_df['i_coords'] = i_coords
                file_info_df = pd.concat((file_info_df, temp_df), axis=0)
        except:
            print('Error: contour and dicom matching')
    return file_info_df

# def parse_link_file(link_df):
#     """
#     Make sure that the contours in the link file match based on their numbers.
#     :param link_df:
#     :return:
#     """
#     'SCD0000501, SC-HF-I-6'
#     bad_idx = []
#     for idx, row in link_df.iterrows():
#         try:
#             patient_num = row.patient_id.replace('SCD0000','').replace('01','')
#             original_num = row.original_id.split('-')[-1]
#             if original_num != patient_num:
#                 bad_idx.append(idx)
#         except:
#             bad_idx.append(idx)
#     return link_df.drop(bad_idx,axis=0).copy()



def parse_contour_file(filename):
    """Parse the given contour filename

    :param filename: filepath to the contourfile to parse
    :return: list of tuples holding x, y coordinates of the contour
    """

    coords_lst = []

    with open(filename, 'r') as infile:
        for line in infile:
            coords = line.strip().split()

            x_coord = float(coords[0])
            y_coord = float(coords[1])
            coords_lst.append((x_coord, y_coord))
    return coords_lst


def parse_dicom_file(filename):
    """Parse the given DICOM filename

    :param filename: filepath to the DICOM file to parse
    :return: array containing DICOM image data
    """

    try:
        dcm = pydicom.read_file(filename)
        dcm_image = dcm.pixel_array

        try:
            intercept = dcm.RescaleIntercept
        except AttributeError:
            intercept = 0.0
        try:
            slope = dcm.RescaleSlope
        except AttributeError:
            slope = 0.0

        if intercept != 0.0 and slope != 0.0:
            dcm_image = dcm_image*slope + intercept
        return dcm_image
    except:
        print('Not a valid dicom file')
        return None


def poly_to_mask(polygon, width, height):
    """Convert polygon to mask

    :param polygon: list of pairs of x, y coords [(x1, y1), (x2, y2), ...]
     in units of pixels
    :param width: scalar image width
    :param height: scalar image height
    :return: Boolean mask of shape (height, width)
    """

    # http://stackoverflow.com/a/3732128/1410871
    img = Image.new(mode='L', size=(width, height), color=0)
    ImageDraw.Draw(img).polygon(xy=polygon, outline=0, fill=1)
    mask = np.array(img).astype(bool)
    return mask

if __name__=="__main__":
    file_info_df = create_file_info_df('final_data')
    file_info_df.to_csv('file_info_df_april16_2018.csv')
    # epoch = create_epoch('final_data')
    # images,targets = epoch.get_current_batch()
    # while images is not None:
    #     images, targets = epoch.get_current_batch()