import cv2
import xml.etree.ElementTree as ET
import os

def crop_and_save(image_path, annotation_path, output_folder):
    tree = ET.parse(annotation_path)
    root = tree.getroot()

    xmin = int(root.find('object/bndbox/xmin').text)
    ymin = int(root.find('object/bndbox/ymin').text)
    xmax = int(root.find('object/bndbox/xmax').text)
    ymax = int(root.find('object/bndbox/ymax').text)

    image = cv2.imread(image_path)

    cropped_image = image[ymin:ymax, xmin:xmax]

    output_path = os.path.join(output_folder, os.path.basename(image_path))

    cv2.imwrite(output_path, cropped_image)
    print(f'Zapisano wycięty obszar do: {output_path}')

image_folder = 'dataset/JPEGImages'
annotation_folder = 'dataset/XMLAnnotations'
output_folder = 'dataset/output'

os.makedirs(output_folder, exist_ok=True)

for filename in os.listdir(annotation_folder):
    if filename.endswith(".xml"):
        image_filename = filename.replace('.xml', '.jpg')
        image_path = os.path.join(image_folder, image_filename)
        annotation_path = os.path.join(annotation_folder, filename)
        crop_and_save(image_path, annotation_path, output_folder)
