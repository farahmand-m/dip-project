import argparse
import configparser
import os
import pathlib

import cv2
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--track', required=True, help='e.g., "IPM-Birds-01"')
    parser.add_argument('--data-dir', default='data/processed/train')
    parser.add_argument('--save-to', type=pathlib.Path)
    parser.add_argument('--show', action='store_true')
    args = parser.parse_args()

    dir_path = pathlib.Path(args.data_dir) / args.track
    assert dir_path.exists(), '%s does not exist.' % dir_path

    config = configparser.ConfigParser()
    config.read(dir_path / 'info.ini')
    sequence = config['Sequence']

    images = [dir_path / 'images' / filepath for filepath in sorted(os.listdir(dir_path / 'images'))]
    columns = ('frame', 'id', 'left', 'top', 'width', 'height', 'confidence', 'X', 'Y', 'Z')
    annotations = pd.read_csv(dir_path / 'gt' / 'gt.txt', names=columns, header=None)

    output_path = str(args.save_to or dir_path / 'annotated.mp4')
    encoding = cv2.VideoWriter_fourcc(*'mp4v')
    frame_rate = round(float(sequence['FrameRate']))
    frame_size = (int(sequence['ImWidth']), int(sequence['ImHeight']))
    video = cv2.VideoWriter(output_path, encoding, frame_rate, frame_size)

    for filepath in tqdm(images):
        filepath = str(filepath)
        image_dir, filename = os.path.split(filepath)
        image_name, extension = os.path.splitext(filename)
        frame_no = int(image_name)
        frame = cv2.imread(filepath)
        related = annotations.query(f'frame == {frame_no}')
        for index, row in related.iterrows():
            x, y, w, h = row.left, row.top, row.width, row.height
            color = np.random.default_rng(seed=row.id).random(3) * 255
            frame = cv2.rectangle(frame, (x, y), (x + w, y + h), color, 1)
            cv2.putText(frame, f'#{row.id}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        cv2.putText(frame, f'{frame_no}', (12, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2)
        if args.show:
            cv2.imshow('Current Frame', frame)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        video.write(frame)

    video.release()

    print('Saved to', output_path)
