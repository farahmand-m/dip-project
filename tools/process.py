import argparse
import configparser
import json
import os
import pathlib
from xml.etree import ElementTree

import cv2
import tqdm.auto as tqdm


# noinspection PyAttributeOutsideInit
class Video:
    video_filename = 'video.mp4'
    annotations_filename = 'gt.xml'
    images_dirname = 'images'
    images_extension = '.jpg'
    annotations_dirname = 'gt'

    def __init__(self, source_dir, target_dir, args):
        self.args = args
        self.source_dir = source_dir
        self.target_dir = target_dir
        self.load_video(source_dir / self.video_filename)
        self.load_annotations(source_dir / self.annotations_filename)

    @property
    def video_name(self):
        head, tail = os.path.split(self.source_dir)
        return tail

    def load_video(self, filepath):
        self.capture = cv2.VideoCapture(str(filepath))
        assert self.capture.isOpened(), 'Failed to open "%s".' % filepath

        self.frame_rate = int(self.capture.get(cv2.CAP_PROP_FPS))
        self.width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.num_frames = int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT))

        self.ratio = min(args.max_width / self.width, 1)
        self.target_width = int(self.width * self.ratio)
        self.target_height = int(self.height * self.ratio)

    @property
    def target_size(self):
        return self.target_width, self.target_height

    def load_annotations(self, filepath):
        if not filepath.exists():
            self.annotations = None
            return
        with open(filepath, 'r') as stream:
            contents = stream.read()
        self.annotations = ElementTree.fromstring(contents)
        element = self.annotations
        for tag in ('meta', 'task', 'segments', 'segment', 'stop'):
            element = element.find(tag)
        num_ann_frames = int(element.text)
        self.annotations_offset = self.num_frames - num_ann_frames

    def dump_info(self):
        info = configparser.ConfigParser()
        info['Sequence'] = {
            'Name': self.video_name,
            'SeqLength': self.num_frames,
            'ImWidth': self.target_width,
            'ImHeight': self.target_height,
            'ImDir': self.images_dirname,
            'ImExt': self.images_extension,
            'FrameRate': self.frame_rate / (self.args.skip_frames + 1),
        }
        os.makedirs(self.target_dir, exist_ok=True)
        with open(self.target_dir / 'info.ini', 'w') as stream:
            info.write(stream)

    def extract_frames(self, video_id, image_id_offset):
        self.image_ids = {}
        frames = []

        output_dir = self.target_dir / self.images_dirname
        os.makedirs(output_dir, exist_ok=True)

        with tqdm.tqdm(total=self.num_frames, desc='* Frames') as progress:
            read, frame = self.capture.read()
            frame_count = 1
            frame_id = 1
            while read:
                if (frame_id - 1) % (self.args.skip_frames + 1) == 0:
                    if self.ratio < 1:
                        frame = cv2.resize(frame, self.target_size)
                    output_filepath = output_dir / f'{frame_id:06d}{self.images_extension}'
                    cv2.imwrite(str(output_filepath), frame, [cv2.IMWRITE_JPEG_QUALITY, self.args.quality])
                    image_id = image_id_offset + frame_count
                    di = {'id': image_id,
                          'video_id': video_id,
                          'frame_id': frame_count,
                          'file_name': str(output_filepath.relative_to(self.args.destination)),
                          'width': self.target_width, 'height': self.target_height}
                    self.image_ids[frame_id] = image_id
                    frames.append(di)
                    frame_count += 1
                frame_id += 1
                progress.update(1)
                read, frame = self.capture.read()
        assert frame_id > 1, 'Failed to read the frames.'

        return frames

    def extract_boxes(self, image_id_offset, track_id_offset, box_id_offset):
        boxes = []

        if self.annotations is not None:

            output_dir = self.target_dir / self.annotations_dirname
            os.makedirs(output_dir, exist_ok=True)

            with open(output_dir / 'gt.txt', 'w') as stream:
                box_id = 1
                track_id = 1
                for element in tqdm.tqdm(self.annotations, desc='* Tracks'):
                    if element.tag == 'track':
                        had_at_least_one = False
                        for child in element:
                            if child.tag == 'box':
                                attributes = child.attrib
                                frame_id = int(attributes['frame']) + self.annotations_offset
                                if frame_id in self.image_ids:
                                    left, top = float(attributes['xtl']), float(attributes['ytl'])
                                    right, bottom = float(attributes['xbr']), float(attributes['ybr'])
                                    assert right > left and bottom > top, 'Invalid Bounding Box'
                                    width, height = right - left, bottom - top
                                    bounding_box = left, top, width, height
                                    bounding_box = (el * self.ratio for el in bounding_box)
                                    bounding_box = (round(el) for el in bounding_box)
                                    bounding_box = tuple(bounding_box)
                                    area = round(width * height * self.ratio ** 2)
                                    rest = 1, 0, 0, 0  # Confidence, 3D Coordinates
                                    line = frame_id, track_id, *bounding_box, *rest
                                    line = ','.join(str(el) for el in line)
                                    stream.write(f'{line}\n')
                                    di = {'id': box_id_offset + box_id,
                                          'image_id': self.image_ids[frame_id],
                                          'track_id': track_id_offset + track_id,
                                          'bbox': bounding_box,
                                          'area': area,
                                          'conf': 1,
                                          'category_id': 1}
                                    had_at_least_one = True
                                    boxes.append(di)
                                    box_id += 1
                        if had_at_least_one:
                            track_id += 1

        return boxes, track_id


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', default='data/raw_data')
    parser.add_argument('--destination', default='data/processed')
    parser.add_argument('--max-width', type=int, default=1920)
    parser.add_argument('--skip-frames', type=int, default=1)
    parser.add_argument('--quality', type=int, default=80)
    args = parser.parse_args()

    source = pathlib.Path(args.source)
    destination = pathlib.Path(args.destination)

    directories = {split: os.listdir(source / split) for split in os.listdir(source)}

    coco_ann_dir = destination / 'annotations'
    os.makedirs(coco_ann_dir, exist_ok=True)

    categories = [{'id': 1, 'name': 'Bird'}]

    for subset, videos in directories.items():

        images_li, annotations_li = [], []

        current_video_index = 0
        current_frame_index = 0
        current_boxes_index = 0
        current_track_index = 0

        for video in videos:
            print('Processing', video)

            source_dir = source / subset / video
            target_dir = destination / subset / video

            instance = Video(source_dir, target_dir, args)
            instance.dump_info()
            current_video_index += 1

            frames = instance.extract_frames(current_video_index, current_frame_index)
            images_li.extend(frames)
            current_frame_index += len(frames)

            boxes, num_tracks = instance.extract_boxes(current_frame_index, current_track_index, current_boxes_index)
            annotations_li.extend(boxes)
            current_boxes_index += len(boxes)
            current_track_index += num_tracks

        with open(coco_ann_dir / f'{subset}.json', 'w') as stream:
            subset_di = {'categories': categories, 'images': images_li, 'annotations': annotations_li}
            json.dump(subset_di, stream, indent=2)
