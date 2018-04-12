import sys
sys.path.insert(0,'../')
import os
import shutil
import matplotlib.pyplot as plt
import numpy as np
import cv2

from epochs import create_epoch, create_file_info_df, parse_dicom_file, parse_contour_file, poly_to_mask

class check_file_info_outputs():
    """
    This checks that the file inputs and outputs are as expected. Currently, this uses visual
    inspection of the outputs to verify that the contours are what they should be. With more time,
    a testing framework with setup/teardown would have examples of images with known masks
    (e.g. a square region) and compare the values created by the mask.
    """
    def __init__(self, all_files_dir):
        self.all_files_dir = all_files_dir
        self.file_info_df = create_file_info_df(all_files_dir)

    def save_input_and_target_files(self):
        """
        This function saves the input and target files in directories with matching image names, in order to check that
        the inputs and target files are as expected. Once the files are exported, these can be visually inspected.
        :return: folders containing input and target files
        """
        input_dir = os.path.join(self.all_files_dir,'inputs')
        target_dir = os.path.join(self.all_files_dir,'targets')

        if os.path.isdir(input_dir):
            shutil.rmtree(input_dir)
        os.mkdir(input_dir)

        if os.path.isdir(target_dir):
            shutil.rmtree(target_dir)
        os.mkdir(target_dir)

        for idx, row in self.file_info_df.iterrows():
            dicom_img = parse_dicom_file(row.dicom_file_name_with_path)
            img_mask = poly_to_mask(row.data_coords, dicom_img.shape[0], dicom_img.shape[1])
            cv2.imwrite(os.path.join(input_dir,'patient'+str(row.patient_num)+'image'+str(row.image_num)+'.png'),dicom_img)
            cv2.imwrite(os.path.join(target_dir, 'patient' + str(row.patient_num) + 'image' + str(row.image_num) + '.png'),255*img_mask.astype(int))

    def save_images_with_masks(self,annotated_images_dir='test_image_matching'):
        """
        This function saves the input and target images alongside
        each other in order to verify that they match.
        :param annotated_images_dir: Directory containing the annotated images
        :return:images outlined with the contours alongside the masks
        """
        saved_images_dir = annotated_images_dir
        if os.path.isdir(saved_images_dir):
            shutil.rmtree(saved_images_dir)
        os.mkdir(saved_images_dir)

        for idx, row in self.file_info_df.iterrows():
            dicom_img = parse_dicom_file(row.dicom_file_name_with_path)
            img_mask = poly_to_mask(row.data_coords, dicom_img.shape[0], dicom_img.shape[1])
            fig, ax = plt.subplots(1, 2, figsize=(10, 5))
            ax[0].imshow(dicom_img, cmap='gray')
            ax[0].plot(np.array(row.data_coords)[:, 0], np.array(row.data_coords)[:, 1], 'r')
            ax[0].axis('off')
            ax[1].imshow(img_mask, cmap='gray')
            ax[1].axis('off')
            plt.savefig(os.path.join(saved_images_dir, 'patient' + str(row.patient_id) + 'image' + str(int(row.image_num)) + '.png'))
            plt.close()

    def export_image_info(self,csv_filename='file_info.csv'):
        self.file_info_df.to_csv(csv_filename)

def check_epochs(filename,out_dir = 'test_epoch_dir'):
    """
    This checks that the batches are being generated properly by
    saving the batches into folders with images and masks. Once the files
    are exported, these can be visually inspected.
    :param filename: top-level directory name
    :param out_dir: output directory containing the image/mask files
    """
    if os.path.isdir(out_dir):
        shutil.rmtree(out_dir)
    os.mkdir(out_dir)
    epoch = create_epoch(filename)
    images, targets = epoch.get_current_batch()
    batch_num = 0
    while images is not None:
        images, targets = epoch.get_current_batch()
        if images is not None:
            os.mkdir(os.path.join(out_dir,str(batch_num)))
            os.mkdir(os.path.join(out_dir, str(batch_num),'images'))
            os.mkdir(os.path.join(out_dir, str(batch_num),'targets'))
            for image_num in range(images.shape[0]):
                cv2.imwrite(os.path.join(out_dir, str(batch_num),'images','image'+str(image_num)+'.jpg'),images[image_num,:,:])
                cv2.imwrite(os.path.join(out_dir, str(batch_num), 'targets', 'target' + str(image_num) + '.jpg'),
                            targets[image_num, :, :]*255)
            batch_num += 1

if __name__=="__main__":
    C = check_file_info_outputs('../final_data')
    C.save_images_with_masks()
    C.save_input_and_target_files()
    C.export_image_info()
    check_epochs('../final_data')
