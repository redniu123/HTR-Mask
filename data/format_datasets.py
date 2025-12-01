#  Copyright Université de Rouen Normandie (1), INSA Rouen (2),
#  tutelles du laboratoire LITIS (1 et 2)
#  contributors :
#  - Denis Coquenet
#
#
#  This software is a computer program written in XXX whose purpose is XXX.
#
#  This software is governed by the CeCILL-C license under French law and
#  abiding by the rules of distribution of free software.  You can  use,
#  modify and/ or redistribute the software under the terms of the CeCILL-C
#  license as circulated by CEA, CNRS and INRIA at the following URL
#  "http://www.cecill.info".
#
#  As a counterpart to the access to the source code and  rights to copy,
#  modify and redistribute granted by the license, users are provided only
#  with a limited warranty  and the software's author,  the holder of the
#  economic rights,  and the successive licensors  have only  limited
#  liability.
#
#  In this respect, the user's attention is drawn to the risks associated
#  with loading,  using,  modifying and/or developing or reproducing the
#  software by the user in light of its specific status of free software,
#  that may mean  that it is complicated to manipulate,  and  that  also
#  therefore means  that it is reserved for developers  and  experienced
#  professionals having in-depth computer knowledge. Users are therefore
#  encouraged to load and test the software's suitability as regards their
#  requirements in conditions enabling the security of their systems and/or
#  data to be ensured and,  more generally, to use and operate it in the
#  same conditions as regards security.
#
#  The fact that you are presently reading this means that you have had
#  knowledge of the CeCILL-C license and that you accept its terms.

import os
import shutil
import xml.etree.ElementTree as ET
import tarfile, zipfile
import pickle
import json
import numpy as np
from PIL import Image


def format_IAM_line():
    """
    Format the IAM dataset at line level using .ln files for split and xml.tgz for labels.
    Expected files in ./iam/:
        - lines.tgz: line images
        - xml.tgz: XML files with labels
        - train.ln, val.ln, test.ln: dataset split (image filename lists)
    """
    source_folder = "./iam"
    target_folder = "./iam/lines"

    # Check required files
    lines_tgz = os.path.join(source_folder, "lines.tgz")
    xml_tgz = os.path.join(source_folder, "xml.tgz")
    for f in [lines_tgz, xml_tgz]:
        if not os.path.isfile(f):
            print("error - {} not found".format(f))
            exit(-1)

    # Clean and create target folder
    if os.path.isdir(target_folder):
        shutil.rmtree(target_folder)
    os.makedirs(target_folder)

    # Extract archives
    line_folder_path = os.path.join(target_folder, "lines_temp")
    xml_folder_path = os.path.join(target_folder, "xml_temp")

    print("Extracting lines.tgz...")
    tar = tarfile.open(lines_tgz)
    tar.extractall(line_folder_path)
    tar.close()

    print("Extracting xml.tgz...")
    tar = tarfile.open(xml_tgz)
    tar.extractall(xml_folder_path)
    tar.close()

    # Build label dictionary from XML files
    print("Building label dictionary from XML files...")
    labels_dict = {}  # line_id -> text
    for xml_file in os.listdir(xml_folder_path):
        if not xml_file.endswith(".xml"):
            continue
        xml_path = os.path.join(xml_folder_path, xml_file)
        try:
            xml_root = ET.parse(xml_path).getroot()
            # Find all line elements in handwritten-part
            for hw_part in xml_root.findall("handwritten-part"):
                for line in hw_part.findall("line"):
                    line_id = line.attrib.get("id")
                    text = line.attrib.get("text")
                    if line_id and text:
                        labels_dict[line_id] = text
        except Exception as e:
            print(f"Warning: Failed to parse {xml_file}: {e}")

    print(f"Loaded {len(labels_dict)} labels from XML files")

    # Process each split
    set_files = {"train": "train.ln", "valid": "val.ln", "test": "test.ln"}
    gt = {"train": dict(), "valid": dict(), "test": dict()}
    charset = set()

    for set_name, ln_file in set_files.items():
        ln_path = os.path.join(source_folder, ln_file)
        if not os.path.isfile(ln_path):
            print(f"Warning: {ln_path} not found, skipping {set_name}")
            continue

        current_folder = os.path.join(target_folder, set_name)
        os.makedirs(current_folder, exist_ok=True)

        # Read image list from .ln file
        with open(ln_path, "r") as f:
            image_list = [line.strip() for line in f if line.strip()]

        print(f"Processing {set_name}: {len(image_list)} images")

        idx = 0
        for img_filename in image_list:
            # img_filename format: a02-102-00.png
            # line_id: a02-102-00
            # form_id: a02-102
            # source path: lines_temp/a02/a02-102/a02-102-00.png
            line_id = img_filename.replace(".png", "")
            parts = line_id.split("-")
            if len(parts) < 3:
                print(f"Warning: Invalid filename format: {img_filename}")
                continue

            folder1 = parts[0]  # a02
            folder2 = "-".join(parts[:2])  # a02-102

            src_path = os.path.join(line_folder_path, folder1, folder2, img_filename)

            if not os.path.isfile(src_path):
                print(f"Warning: Image not found: {src_path}")
                continue

            # Get label
            label = labels_dict.get(line_id)
            if label is None:
                print(f"Warning: No label for {line_id}")
                continue

            # Copy image and record
            new_img_name = f"{set_name}_{idx}.png"
            dst_path = os.path.join(current_folder, new_img_name)
            shutil.copy2(src_path, dst_path)

            gt[set_name][new_img_name] = {"text": label}
            charset = charset.union(set(label))
            idx += 1

        print(f"  Processed {idx} images for {set_name}")

    # Clean up temp folders
    shutil.rmtree(line_folder_path)
    shutil.rmtree(xml_folder_path)

    # Save labels
    with open(os.path.join(target_folder, "labels.pkl"), "wb") as f:
        pickle.dump(
            {
                "ground_truth": gt,
                "charset": sorted(list(charset)),
            },
            f,
        )

    print("IAM dataset formatting complete!")


def format_READ2016_line():
    """
    Format the READ 2016 dataset at line level with the official split (8,349 for training, 1,040 for validation and 1,138 for test)
    """
    source_folder = "./read2016"
    target_folder = "./read2016/lines"
    if os.path.isdir(target_folder):
        shutil.rmtree(target_folder)
    os.makedirs(target_folder)

    tar_filenames = ["Test-ICFHR-2016.tgz", "Train-And-Val-ICFHR-2016.tgz"]
    tar_paths = [os.path.join(source_folder, name) for name in tar_filenames]
    for tar_path in tar_paths:
        if not os.path.isfile(tar_path):
            print("error - {} not found".format(tar_path))
            exit(-1)
        tar = tarfile.open(tar_path)
        tar.extractall(target_folder)
        tar.close()

    os.rename(
        os.path.join(target_folder, "PublicData", "Training"),
        os.path.join(target_folder, "train"),
    )
    os.rename(
        os.path.join(target_folder, "PublicData", "Validation"),
        os.path.join(target_folder, "valid"),
    )
    os.rename(
        os.path.join(target_folder, "Test-ICFHR-2016"),
        os.path.join(target_folder, "test"),
    )
    os.rmdir(os.path.join(target_folder, "PublicData"))
    for set_name in [
        "train",
        "valid",
    ]:
        for filename in os.listdir(os.path.join(target_folder, set_name, "Images")):
            filepath = os.path.join(target_folder, set_name, "Images", filename)
            if os.path.isfile(filepath):
                os.rename(filepath, os.path.join(target_folder, set_name, filename))
        os.rmdir(os.path.join(target_folder, set_name, "Images"))

    gt = {"train": dict(), "valid": dict(), "test": dict()}

    charset = set()
    for set_name in ["train", "valid", "test"]:
        img_fold_path = os.path.join(target_folder, set_name)
        # train/valid 的 XML 在 page/page 目录下，test 的 XML 在 page 目录下
        if set_name in ["train", "valid"]:
            xml_fold_path = os.path.join(target_folder, set_name, "page", "page")
        else:
            xml_fold_path = os.path.join(target_folder, set_name, "page")
        i = 0
        for xml_file_name in sorted(os.listdir(xml_fold_path)):
            if xml_file_name.split(".")[-1] != "xml":
                continue
            filename = xml_file_name.split(".")[0]
            img_path = os.path.join(img_fold_path, filename + ".JPG")
            xml_file_path = os.path.join(xml_fold_path, xml_file_name)
            xml_root = ET.parse(xml_file_path).getroot()
            img = np.array(Image.open(img_path))
            for text_region in xml_root[1][1:]:
                if text_region.tag.split("}")[-1] != "TextRegion":
                    continue
                for balise in text_region:
                    if balise.tag.split("}")[-1] != "TextLine":
                        continue
                    for sub in balise:
                        if sub.tag.split("}")[-1] == "Coords":
                            points = sub.attrib["points"].split(" ")
                            x_points, y_points = list(), list()
                            for p in points:
                                y_points.append(int(p.split(",")[1]))
                                x_points.append(int(p.split(",")[0]))
                        elif sub.tag.split("}")[-1] == "TextEquiv":
                            line_label = sub[0].text
                    if line_label is None:
                        continue
                    top, bottom, left, right = (
                        np.min(y_points),
                        np.max(y_points),
                        np.min(x_points),
                        np.max(x_points),
                    )
                    new_img_name = "{}_{}.jpeg".format(set_name, i)
                    new_img_path = os.path.join(img_fold_path, new_img_name)
                    curr_img = img[top : bottom + 1, left : right + 1]
                    Image.fromarray(curr_img).save(new_img_path)
                    gt[set_name][new_img_name] = {
                        "text": line_label,
                    }
                    charset = charset.union(line_label)
                    i += 1
                    line_label = None
            os.remove(img_path)
        # 删除整个 page 目录（train/valid 的 page 目录包含嵌套的 page/page）
        page_folder = os.path.join(target_folder, set_name, "page")
        shutil.rmtree(page_folder)

    with open(os.path.join(target_folder, "labels.pkl"), "wb") as f:
        pickle.dump(
            {
                "ground_truth": gt,
                "charset": sorted(list(charset)),
            },
            f,
        )


def pkl2txt(dataset_name):
    for i in ["train", "valid", "test"]:
        with open((f"./{dataset_name}/lines/labels.pkl"), "rb") as f:
            a = pickle.load(f)
            for k, v in a["ground_truth"][i].items():
                head = k.split(".")[0]
                text = v["text"].replace("¬", "")
                with open(
                    f"./{dataset_name}/lines/{head}.txt", "a", encoding="utf-8"
                ) as t:
                    t.write(text)


def move_files_and_delete_folders(parent_folder):
    """
    Move all files from train, valid, and test folders to the parent folder and delete the empty folders.

    Args:
    parent_folder (str): The directory containing the train, valid, and test folders.
    """

    # Define the folders to be moved
    folders = ["train", "valid", "test"]

    for folder in folders:
        folder_path = os.path.join(parent_folder, folder)

        # Check if the folder exists
        if not os.path.isdir(folder_path):
            print(f"{folder} folder does not exist.")
            continue

        # Move files from the subfolder to the parent folder
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            if os.path.isfile(file_path):
                # Move the file to the parent folder
                shutil.move(file_path, os.path.join(parent_folder, filename))

        # Remove the empty folder
        os.rmdir(folder_path)
        print(f"Moved all files from {folder} and deleted the folder.")


if __name__ == "__main__":
    # format_READ2016_line()
    # pkl2txt("read2016")
    # move_files_and_delete_folders("./read2016/lines")

    format_IAM_line()
    pkl2txt("iam")
    move_files_and_delete_folders("./iam/lines")

    # format_LAM_line()
