import xml.etree.ElementTree as Xet
import pandas as pd
import shutil

def xml_to_csv(file_name, output_file, label=True):
    """
    convert xml file to csv file
    """
    rows = []
    # print("processing: ", file_name)
    # Parsing the XML file
    xmlparse = Xet.parse(file_name)
    root = xmlparse.getroot()
    for child in root: # childs is a space
        space = []
        # print(child.tag, child.attrib)
        space.append(child.attrib['id'])
        if label:
            try: 
                space.append(child.attrib['occupied'])
            except:
                space.append(-1) #-1 if there is not label
        
        for elem in child: # elem are parking, rotatedRect, contour
            if elem.tag == 'rotatedRect':
                center = elem[0]
                space.append(center.attrib['x'])
                space.append(center.attrib['y'])
                size = elem[1]
                space.append(size.attrib['w'])
                space.append(size.attrib['h'])
                angle = elem[2]
                space.append(angle.attrib['d'])
                    
            elif elem.tag == 'contour':
                point1 = elem[0]
                space.append(point1.attrib['x'])
                space.append(point1.attrib['y'])
                point2 = elem[1]
                space.append(point2.attrib['x'])
                space.append(point2.attrib['y'])
                point3 = elem[2]
                space.append(point3.attrib['x'])
                space.append(point3.attrib['y'])
                point4 = elem[3]
                space.append(point4.attrib['x'])
                space.append(point4.attrib['y'])

        rows.append(space)
    if label:
        cols = ['id', 'occupied', 'center_x', 'center_y', 'size_w', 'size_h', 'angle', 
                'point1_x', 'point1_y', 'point2_x', 'point2_y', 'point3_x', 'point3_y', 'point4_x', 'point4_y']
    else:
        cols = ['id', 'center_x', 'center_y', 'size_w', 'size_h', 'angle', 
        'point1_x', 'point1_y', 'point2_x', 'point2_y', 'point3_x', 'point3_y', 'point4_x', 'point4_y']
 
    df = pd.DataFrame(rows, columns=cols)
    df['filename'] = file_name
    df.to_csv(output_file, index=False)

import argparse
parser = argparse.ArgumentParser(description='Convert xml to csv')
parser.add_argument('--ndata', help='number of images to use', default= 1000, type=int)
parser.add_argument('--camera_number', help='camera number', default=3, type=int)
parser.add_argument('--weather', help='weather', default= 'sunny', type=str)
args = parser.parse_args()


def main(ndata, camera_number, weather):
    """
    convert the xml file to csv file for pklot dataset, to make it easier to process in c++
    example of xml:
        <space id="36" occupied="0">
        <rotatedRect>
        <center x="746" y="285" />
        <size w="43" h="37" />
        <angle d="-88" />
        </rotatedRect>
        <contour>
        <point x="728" y="305" />
        <point x="731" y="265" />
        <point x="766" y="264" />
        <point x="765" y="307" />
        </contour>
    </space>

    since the dataset is huge, we also consider a limited number of images

    """
    if camera_number == 1: camera_label = "PUCPR"
    elif camera_number == 2: camera_label = "UFPR04"
    elif camera_number == 3: camera_label = "UFPR05"

    import glob
    import os

    # keep ndata for each camer
    paths = glob.glob("PKLot/PKLot/"+camera_label+"/"+weather+"/*/*.xml")[:ndata] 

    # create a folder with reduced data
    if not os.path.exists("PKLot_reduced/camera"+str(camera_number)+"/"+weather):
        os.makedirs("PKLot_reduced/camera"+str(camera_number)+"/"+weather)
    
    i = 0
    for p in paths:
        if i%100 == 0: print("iteration ", i)
        filename = p.split("/")[-1].split(".")[0]
        #this is the file containing also the right label
        xml_to_csv(p, "PKLot_reduced/camera"+str(camera_number)+"/"+weather+"/"+filename+".csv")
        #save photo in the reduced folder
        shutil.copy(p[:-3]+"jpg", "PKLot_reduced/camera"+str(camera_number)+"/"+weather+"/"+filename+".jpg")
        i+=1
    # we also want to save the lots position for each single camera, as it is done in the cnr dataset
    xml_to_csv(paths[0], "PKLot_reduced/camera"+str(camera_number)+".csv", label=False)

if __name__ == "__main__":
    args = parser.parse_args()
    main(args.ndata, args.camera_number, args.weather)
