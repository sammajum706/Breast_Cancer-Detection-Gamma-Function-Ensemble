import imgaug.augmenters as iaa
import os
import cv2
import shutil
import splitfolders

def AugData(input_folder):
    targetnames = ['Benign','Malignant']
    # Splitting the dataset of a particluar magnification in 0.8-0.2 ratio into train and test set
    splitfolders.ratio(input_folder, output="Split_Data", seed=1337, ratio=(0.8,0.2), group_prefix=None)
    os.rename("Split_Data/val","Split_Data/test")
    # Augmenting images and storing them in temporary directories 
    for img_class in targetnames:

        #creating temporary directories
        # creating a base directory
        aug_dir = 'aug_dir/'
        os.mkdir(aug_dir)
        # creating a subdirectory inside the base directory for images of the same class
        img_dir = os.path.join(aug_dir, 'img_dir')
        os.mkdir(img_dir)

        img_list = os.listdir('Split_Data/train/' + img_class)

        # Copy images from the class train dir to the img_dir 
        for file_name in img_list:

            # path of source image in training directory
            source = os.path.join('Split_Data/train/' + img_class, file_name)

            # creating a target directory to send images 
            target = os.path.join(img_dir, file_name)

            # copying the image from the source to target file
            shutil.copyfile(source, target)

        # Temporary augumented dataset directory.
        source_path = aug_dir

        # Augmented images will be saved to training directory
        save_path = 'Split_Data/train/' + img_class
        os.chdir(save_path)
        # Creating Image Data Generator to augment images

        seq = iaa.Sequential([iaa.Resize(224),
          iaa.Fliplr(p=0.5),              
          iaa.Affine(scale=(0.9,1.1),translate_percent={"x": (-0.125, 0.125),
                                                        "y": (-0.125,0.125)},fit_output=True, 
                                                         cval=0,mode='edge'),                          
          iaa.Resize(224,interpolation=["linear"]),
          ], random_order= False)

        # Generate the augmented images
        aug_images = 2000 
        data=os.listdir(img_dir)
        num_files = len(data)
        num_batches = int(np.ceil((aug_images - num_files) /num_files))
        # creating 8000 augmented images per class
        j=0
        for d in data:
            img=cv2.imread(d)
            l=[img]
            i=0
            for i in range(num_batches):
                augmented_images = seq(images=l)
                cv2.imwrite(d+'_'+str(j)+'.jpg', augmented_images[0])
                j+=1
        # delete temporary directory 
        shutil.rmtree(aug_dir)
    # Splitting the augmented training data into train and validation sets
    input_folder="Split_Data/train/"
    splitfolders.ratio(input_folder, output="Final_Split", seed=1337, ratio=(0.9,0.1), group_prefix=None)
    # Moving the test folder to Final_Split folder
    shutil.move("/Split_Data/Test","/Final_Split/")