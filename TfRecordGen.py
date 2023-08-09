import xml.etree.ElementTree as ET
import os
import tensorflow as tf

def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def float_list_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def bytes_list_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[item for item in value]))

def int64_list_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def parseXmlAnnotation(xmlPath):
    annotations = []

    tree = ET.parse(xmlPath)
    root = tree.getroot()

    objects = []
    fileName = root.find('filename').text

    sizeInfo = root.find('size')
    width = int(sizeInfo.find('width').text)
    height = int(sizeInfo.find('height').text)

    # Percorrem-se todos os objetos presentes na imagem.
    for object in root.findall('object'):
        name = object.find('name').text
        bbox_elem = object.find('bndbox')
        xmin = int(bbox_elem.find('xmin').text)
        ymin = int(bbox_elem.find('ymin').text)
        xmax = int(bbox_elem.find('xmax').text)
        ymax = int(bbox_elem.find('ymax').text)

        objects.append({
            'name': name,
            'bbox': (xmin, ymin, xmax, ymax)
        })

    annotations.append({
        'filename': fileName,
        'width': width,
        'height': height,
        'objects': objects
    })

    return annotations

def createTfRecord(annotation, imagesFolder):
    fileName = annotation['filename']
    imagePath = os.path.join(imagesFolder, fileName)

    with tf.io.gfile.GFile(imagePath, 'rb') as fid:
        encodedImageData = fid.read()

    width = annotation['width']
    height = annotation['height']

    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classesAsText = []
    classesAsIndex = []

    for obj in annotation['objects']:
        xmins.append(obj['bbox'][0] / width)
        xmaxs.append(obj['bbox'][2] / width)
        ymins.append(obj['bbox'][1] / height)
        ymaxs.append(obj['bbox'][3] / height)
        classesAsText.append(obj['name'].encode('utf8'))
        classesAsIndex.append(0 if obj['name'] == 'CocaCola' else 1)

    tfExample = tf.train.Example(features = tf.train.Features(feature = {
        'image/height': int64_feature(height),
        'image/width': int64_feature(width),
        'image/filename': bytes_feature(fileName.encode('utf8')),
        'image/source_id': bytes_feature(fileName.encode('utf8')),
        'image/encoded': bytes_feature(encodedImageData),
        'image/format': bytes_feature(b'jpg'),
        'image/object/bbox/xmin': float_list_feature(xmins),
        'image/object/bbox/xmax': float_list_feature(xmaxs),
        'image/object/bbox/ymin': float_list_feature(ymins),
        'image/object/bbox/ymax': float_list_feature(ymaxs),
        'image/object/class/text': bytes_list_feature(classesAsText),
        'image/object/class/label': int64_list_feature(classesAsIndex),
    }))

    return tfExample

annotationsAndImagesFolder = r'C:\Users\TECRA\Desktop\Uni\3ano\ESTAGIO\AutoML\train'
outputTfrecord = r'C:\Users\TECRA\Desktop\Uni\3ano\ESTAGIO\AutoML\tfrec'

xmlFiles = [f for f in os.listdir(annotationsAndImagesFolder) if f.endswith('.xml')]

aux = 0
for xmlFile in xmlFiles:
    tfRecordFile = os.path.join(outputTfrecord, 'tfrecord' + str(aux) + '.tfrecord')
    xmlPath = os.path.join(annotationsAndImagesFolder, xmlFile)
    annotations = parseXmlAnnotation(xmlPath)

    print(f"Annotations for {xmlFile}:")
    for annotation in annotations:
        for obj in annotation['objects']:
            print(f"Object: {obj['name']}, Bounding Box: {obj['bbox']}")
        print()

    # Crie e escreva os exemplos TFRecord
    with tf.io.TFRecordWriter(tfRecordFile) as writer:
        for annotation in annotations:
            tfRecord = createTfRecord(annotation, annotationsAndImagesFolder)
            writer.write(tfRecord.SerializeToString())

    aux = aux + 1