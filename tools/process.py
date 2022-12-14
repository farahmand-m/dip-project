import argparse
import configparser
import os
import pathlib
from xml.etree import ElementTree

import cv2
from tqdm.auto import tqdm


def extract_images(filepath, output_dir, target_width, skip_frames, quality):
    
    os.makedirs(output_dir, exist_ok=True)
    head, tail = os.path.split(output_dir)
    
    video = cv2.VideoCapture(str(filepath))
    assert video.isOpened(), 'Failed to open the video file.'

    frame_rate = video.get(cv2.CAP_PROP_FPS)
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    must_resize, ratio = width > target_width, 1.0

    if must_resize:
        ratio = target_width / width
        target_height = int(height * ratio)
        width, height = target_width, target_height
        print('* Frames will be rescaled to', target_width, 'x', target_height, flush=True)

    with tqdm(total=num_frames, desc='* Frames') as progress:
        read, frame = video.read()
        frame_no = 1
        while read:
            if frame_no % skip_frames == 0:
                if must_resize:
                    frame = cv2.resize(frame, (width, height))
                output_filepath = str(output_dir / f'{frame_no:06d}.jpg')
                cv2.imwrite(output_filepath, frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
            frame_no += 1
            progress.update(1)
            read, frame = video.read()
    assert frame_no > 1, 'Failed to read anything from %s' % filepath

    info = configparser.ConfigParser()
    info['Sequence'] = {
        'FrameRate': frame_rate / skip_frames,
        'SeqLength': num_frames,
        'ImHeight': height,
        'ImWidth': width,
        'ImDir': 'images',
        'ImExt': '.jpg',
        'Name': tail,
    }
    with open(os.path.join(head, 'info.ini'), 'w') as stream:
        info.write(stream)

    video.release()

    return ratio


def parse_annotations(filepath, output_dir, scale_ratio):

    os.makedirs(output_dir, exist_ok=True)

    with open(filepath, 'r') as stream:
        contents = stream.read()
    annotations = ElementTree.fromstring(contents)

    count = 0

    with open(output_dir / 'gt.txt', 'w') as stream:
        track_id = 0
        for element in annotations:
            if element.tag == 'track':
                track_id += 1
                for child in element:
                    if child.tag == 'box':
                        attributes = child.attrib
                        frame_no = int(attributes['frame']) + 1
                        left, top = float(attributes['xtl']), float(attributes['ytl'])
                        right, bottom = float(attributes['xbr']), float(attributes['ybr'])
                        assert right > left and bottom > top, 'Invalid Bounding Box'
                        width, height = right - left, bottom - top
                        bounding_box = left, top, width, height
                        bounding_box = (el * scale_ratio for el in bounding_box)
                        bounding_box = (round(el) for el in bounding_box)
                        rest = -1, -1, -1, -1  # Confidence, 3D Coordinates
                        line = frame_no, track_id, *bounding_box, *rest
                        line = ','.join(str(el) for el in line)
                        stream.write(f'{line}\n')
                        count += 1
                    else:
                        print(f'  * skipping "{element.tag}.{child.tag}" in annotations')
            else:
                print(f'  * skipping "{element.tag}" in annotations')

    return count


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

    for subset, tracks in directories.items():
        for track in tracks:
            print('Processing', track)
            dir_path = source / subset / track
            video_path = dir_path / 'video.mp4'
            ann_path = dir_path / 'annotations.xml'
            output_dir = destination / subset / track
            images_dir = output_dir / 'images'
            ann_dir = output_dir / 'gt'
            if video_path.exists():
                ratio = extract_images(video_path, images_dir, args.max_width, args.skip_frames, args.quality)
                if ann_path.exists():
                    count = parse_annotations(ann_path, ann_dir, ratio)
                    print('* Extracted', count, 'bounding boxes.')
                else:
                    print('* Missing annotations.')
