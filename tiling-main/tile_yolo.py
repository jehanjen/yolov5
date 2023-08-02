import argparse
import os
from shutil import copyfile
import glob
import random
from PIL import Image
import numpy as np
import pandas as pd
from shapely.geometry import Polygon

def tiler(imnames, newpath, falsepath, slice_size, ext):
    for imname in imnames:
        im = Image.open(imname)
        imr = np.array(im, dtype=np.uint8)
        height = imr.shape[0]
        width = imr.shape[1]
        labname = imname.replace(ext, '.txt')
        labels = pd.read_csv(labname, sep=' ', names=['class', 'x1', 'y1', 'w', 'h'])

        # we need to rescale coordinates from 0-1 to real image height and width
        labels[['x1', 'w']] = labels[['x1', 'w']] * width
        labels[['y1', 'h']] = labels[['y1', 'h']] * height

        boxes = []

        # convert bounding boxes to shapely polygons. We need to invert Y and find polygon vertices from center points
        for row in labels.iterrows():
            x1 = row[1]['x1'] - row[1]['w']/2
            y1 = (height - row[1]['y1']) - row[1]['h']/2
            x2 = row[1]['x1'] + row[1]['w']/2
            y2 = (height - row[1]['y1']) + row[1]['h']/2

            boxes.append((int(row[1]['class']), Polygon([(x1, y1), (x2, y1), (x2, y2), (x1, y2)])))

        counter = 0
        print('Image:', imname)
        # create tiles and find intersection with bounding boxes for each tile
        for i in range((height // slice_size)):
            for j in range((width // slice_size)):
                x1 = j*slice_size
                y1 = height - (i*slice_size)
                x2 = ((j+1)*slice_size) - 1
                y2 = (height - (i+1)*slice_size) + 1

                pol = Polygon([(x1, y1), (x2, y1), (x2, y2), (x1, y2)])
                imsaved = False
                slice_labels = []

                for box in boxes:
                    if pol.intersects(box[1]):
                        inter = pol.intersection(box[1])

                        if not imsaved:
                            sliced = imr[i*slice_size:(i+1)*slice_size, j*slice_size:(j+1)*slice_size]
                            sliced_im = Image.fromarray(sliced)
                            filename = imname.split(os.sep)[-1]
                            slice_path = os.path.join(newpath, f'{filename.replace(ext, f"_{i}_{j}{ext}")}')
                            slice_labels_path = os.path.join(newpath, f'{filename.replace(ext, f"_{i}_{j}.txt")}')
                            sliced_im.save(slice_path)
                            imsaved = True

                        # get smallest rectangular polygon (with sides parallel to the coordinate axes) that contains the intersection
                        new_box = inter.envelope

                        # get central point for the new bounding box
                        centre = new_box.centroid

                        # get coordinates of the new bounding box
                        x = [new_box.bounds[0], new_box.bounds[2]]
                        y = [new_box.bounds[1], new_box.bounds[3]]

                        # calculate normalized coordinates relative to the tile
                        x_norm = [(coord - (j*slice_size)) / slice_size for coord in x]
                        y_norm = [(coord - ((height - (i+1)*slice_size) + 1)) / slice_size for coord in y]

                        # calculate width and height of the bounding box
                        w_norm = max(x_norm) - min(x_norm)
                        h_norm = max(y_norm) - min(y_norm)

                        # convert normalized coordinates and dimensions to image size
                        x_abs = [coord * slice_size for coord in x_norm]
                        y_abs = [coord * slice_size for coord in y_norm]
                        w_abs = w_norm * slice_size
                        h_abs = h_norm * slice_size

                        # add the label to the slice_labels list
                        slice_labels.append(f"{box[0]} {(centre.x / slice_size):.6f} {(centre.y / slice_size):.6f} {w_abs:.6f} {h_abs:.6f}")

                # save the slice labels to a text file
                if slice_labels:
                    with open(slice_labels_path, 'w') as f:
                        f.write('\n'.join(slice_labels))

                counter += 1

   

    # Copy obj.names file to the target folder
               

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tile YOLO images and labels")
    parser.add_argument("-source", "--source", type=str, default="./yolosample/ts", help="path to the source folder")
    parser.add_argument("-target", "--target", type=str, default="./yolosliced_tiles/ts", help="path to the target folder")
    parser.add_argument("-falsefolder", "--falsefolder", type=str, default="./yolosliced_tiles", help="path to the false positive folder")
    parser.add_argument("-size", "--size", type=int, default=512, help="size of the tiles")
    parser.add_argument("-ext", "--ext", type=str, default=".jpg", help="file extension of the images")
    args = parser.parse_args()

    imnames = glob.glob(os.path.join(args.source, f"*{args.ext}"))
    random.shuffle(imnames)

    tiler(imnames, args.target, args.falsefolder, args.size, args.ext)
    
    
    ###tile_yoloS