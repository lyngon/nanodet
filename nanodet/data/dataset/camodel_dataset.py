from __future__ import annotations

import json
import pathlib
from collections import defaultdict

from pycocotools.coco import COCO

from .coco import CocoDataset

# camodel_categories = [
#     ("person", 1),
#     ("bicycle", 2),
#     ("car", 3),
#     ("motorcycle", 4),
#     ("bus", 6),
#     ("train", 7),
#     ("truck", 8),
#     ("backpack", 27),
#     ("umbrella", 28),
#     ("handbag", 31),
#     ("unknown", 0),
# ]

# camodel_category_id_to_cat_index = {
#     cmid: ix for ix, (_, cmid) in enumerate(camodel_categories)
# }


class CocoCamodel(COCO):
    def __init__(self, annotation):
        """
        Constructor of Microsoft COCO helper class for
        reading and visualizing annotations.
        :param annotation: annotation dict
        :return:
        """
        # load dataset
        self.dataset, self.anns, self.cats, self.imgs = (
            dict(),
            dict(),
            dict(),
            dict(),
        )
        self.imgToAnns, self.catToImgs = defaultdict(list), defaultdict(list)
        dataset = annotation
        assert (
            type(dataset) == dict
        ), "annotation file format {} not supported".format(type(dataset))
        self.dataset = dataset
        self.createIndex()


class CamodelDataset(CocoDataset):
    # def __init__(
    #     self,
    #     img_path: str,
    #     ann_path: str,
    #     input_size: Tuple[int, int],
    #     pipeline: Dict,
    #     keep_ratio: bool = True,
    #     use_instance_mask: bool = False,
    #     use_seg_mask: bool = False,
    #     use_keypoint: bool = False,
    #     load_mosaic: bool = False,
    #     mode: str = "train",
    #     multi_scale: Optional[Tuple[float, float]] = None,
    # ):
    #     pass
    def __init__(self, class_names, **kwargs):
        self.class_names = class_names
        print(f"Camodel Dataset Args : {kwargs}")
        super(CamodelDataset, self).__init__(**kwargs)

    def to_coco(self, path: pathlib.Path):
        """
        Convert Camodel Dataset to COCO API data format

        :param path:
            Path to the directory that contains a Camodel dataset.
        """

        self.instances = {
            "train": [],
            "eval": [],
            "test": [],
        }

        coco_info = {
            "year": 2021,
            "version": "0.1.0",
            "description": "Camodel",
            "contributor": "Lyngon Pte. Ltd.",
            "url": "https://www.lyngon.com",
            "date_created": "2021-10-20",
        }
        coco_licences = [
            {
                "id": 0,
                "name": "Copyrigt Lyngon Pte. Ltd.",
                "url": "https://www.lyngon.com",
            }
        ]
        coco_categories = [
            {"id": ix, "name": name, "supercategory": None}
            for ix, name in enumerate(self.class_names)
        ]
        # coco_categories = [
        #     {"id": ix, "name": name, "supercategory": None}
        #     for ix, (name, cmid) in enumerate(camodel_categories)
        # ]

        coco_images = []

        annotation_id = 0

        if not path.is_dir():
            raise NotADirectoryError(f"Not a valid directory: '{path}'")

        for annotation_file in path.glob("*.json"):
            if "_train_" in annotation_file.name:
                group = "train"
            elif "_eval_" in annotation_file.name:
                group = "train"
            elif "_test_" in annotation_file.name:
                group = "test"
            else:
                print(f"Unknown group for data in file : {annotation_file}")
                continue

            for sample_dir, sample_annotation in json.load(
                annotation_file.open()
            ).items():

                sample_dir = pathlib.Path(sample_dir)
                if not sample_dir.is_dir():
                    print(f"Sample directory does not exist: '{sample_dir}'")
                    continue

                sample_metadata_file = sample_dir / "metadata.json"
                if not sample_metadata_file.is_file():
                    print(
                        f"Unable to find metadata file: {sample_metadata_file}"
                    )

                instance_labels = sample_annotation.get("labels", None)
                if instance_labels is None:
                    print(f"Unable to reat labels for sample: {sample_dir}")

                sample_metadata = json.load(sample_metadata_file.open())
                sample_height = 1080
                sample_width = 1920

                foreground_image_path = (
                    sample_metadata.get("annotations", {})
                    .get("groundtruth", {})
                    .get("images", {})
                    .get("redacted_foreground", {})
                    .get("path", None)
                )
                if not foreground_image_path:
                    print(
                        "Unable to find the groundtruth foreground image path "
                        f"for sample : {sample_dir}"
                    )
                    continue
                print(foreground_image_path)
                if not pathlib.Path(foreground_image_path).is_file():
                    print(
                        "Unable to find the groundtruth foreground image file "
                        f": {foreground_image_path}"
                    )
                    continue

                # Not sure if each group ('train', 'eval', 'test') should have
                # its separate numbering or global?
                # Separate numbering per group:
                image_id = len(self.instances[group])

                image_data = {
                    "licence": 0,
                    "file_name": str(foreground_image_path),
                    "coco_url": None,
                    "height": sample_height,
                    "width": sample_width,
                    "date_captured": sample_metadata["timestamp"],
                    "flickr_url": None,
                    "id": image_id,
                }
                print(image_data)
                coco_images.append(image_data)

                for label in instance_labels:
                    left_abs = sample_width * (
                        label["location"]["x"] - label["location"]["u"]
                    )
                    top_abs = sample_height * (
                        label["location"]["y"] - label["location"]["n"]
                    )
                    width_abs = sample_width * (
                        label["location"]["x"] + label["location"]["u"]
                    )
                    height_abs = sample_height * (
                        label["location"]["y"] + label["location"]["n"]
                    )
                    self.instances[group].append(
                        {
                            "id": annotation_id,
                            "image_id": image_id,
                            "category_id": label["category"],
                            "area": height_abs * width_abs,
                            "bbox": [left_abs, top_abs, width_abs, height_abs],
                            "iscrowd": 0,
                        }
                    )
                    annotation_id += 1

        coco_dict = {
            "info": coco_info,
            "images": coco_images,
            "annotations": self.instances["train"],
            "categories": coco_categories,
            "licences": coco_licences,
        }

        return coco_dict

    def get_data_info(self, ann_path):
        """
        Load basic information of dataset such as image path, label and so on.
        :param ann_path: coco json file path
        :return: image info:
        [
            {
                'license': 2,
                'file_name': '000000000139.jpg',
                'coco_url': 'http://images.cocodataset.org/val2017/000000000139.jpg',
                'height': 426,
                'width': 640,
                'date_captured': '2013-11-21 01:34:01',
                'flickr_url':
                'http://farm9.staticflickr.com/8035/8024364858_9c41dc1666_z.jpg',
                'id': 139
            },
            ...
        ]
        """
        coco_dict = self.to_coco(pathlib.Path(str(ann_path)))
        self.coco_api = CocoCamodel(coco_dict)
        self.cat_ids = sorted(self.coco_api.getCatIds())
        self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}
        self.cats = self.coco_api.loadCats(self.cat_ids)
        self.img_ids = sorted(self.coco_api.imgs.keys())
        img_info = self.coco_api.loadImgs(self.img_ids)
        return img_info
