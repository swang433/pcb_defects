import xml.etree.ElementTree as ET
import os
from pathlib import Path
from PIL import Image

def convert_voc_to_yolo(xml_file, img_width, img_height, class_names):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    
    yolo_labels = []
    
    for obj in root.findall('object'):
        class_name = obj.find('name').text
        if class_name not in class_names:
            continue
            
        class_id = class_names.index(class_name)
        
        bbox = obj.find('bndbox')
        xmin = float(bbox.find('xmin').text)
        ymin = float(bbox.find('ymin').text)
        xmax = float(bbox.find('xmax').text)
        ymax = float(bbox.find('ymax').text)
        
        # Convert to YOLO format (normalized)
        x_center = ((xmin + xmax) / 2) / img_width
        y_center = ((ymin + ymax) / 2) / img_height
        width = (xmax - xmin) / img_width
        height = (ymax - ymin) / img_height
        
        yolo_labels.append(f"{class_id} {x_center} {y_center} {width} {height}")
    
    return yolo_labels

# Paths
annotations_dir = r"D:\Python\defect_det\PCB_DATASET\Annotations"
images_dir = r"D:\Python\defect_det\PCB_DATASET\images"
output_labels_dir = r"D:\Python\defect_det\PCB_DATASET\labels"

# Create output directory
os.makedirs(output_labels_dir, exist_ok=True)

classes = ['missing_hole', 'mouse_bite', 'open_circuit', 'short', 'spur', 'spurious_copper']

# Find all XML files in subdirectories
xml_files = list(Path(annotations_dir).rglob("*.xml"))
print(f"Found {len(xml_files)} XML files")

converted = 0
failed = 0

for xml_file in xml_files:
    # Get defect type folder name
    defect_type = xml_file.parent.name
    img_name = xml_file.stem
    
    # Look for image in corresponding subfolder
    img_path = None
    for ext in ['.jpg', '.JPG', '.png', '.PNG']:
        test_path = Path(images_dir) / defect_type / (img_name + ext)
        if test_path.exists():
            img_path = test_path
            break
    
    if img_path is None:
        print(f"Image not found for {xml_file.name}")
        failed += 1
        continue
    
    # Get image dimensions
    img = Image.open(img_path)
    img_width, img_height = img.size
    
    # Convert annotations
    yolo_labels = convert_voc_to_yolo(str(xml_file), img_width, img_height, classes)
    
    if len(yolo_labels) == 0:
        print(f"No objects found in {xml_file.name}")
        failed += 1
        continue
    
    # Save to txt file
    output_file = Path(output_labels_dir) / (xml_file.stem + ".txt")
    with open(output_file, 'w') as f:
        f.write('\n'.join(yolo_labels))
    
    converted += 1
    if converted % 100 == 0:
        print(f"Converted {converted} files...")

print(f"\nConversion complete!")
print(f"Successfully converted: {converted}")
print(f"Failed: {failed}")