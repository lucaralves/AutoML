import tensorflow as tf
import autokeras as ak
import os

# Função para fazer o parse de um exemplo TFRecord
def parse_tfrecord_fn(example):
    feature_description = {
        'image/encoded': tf.io.FixedLenFeature([], tf.string),
        'image/object/bbox/xmin': tf.io.VarLenFeature(tf.float32),
        'image/object/bbox/xmax': tf.io.VarLenFeature(tf.float32),
        'image/object/bbox/ymin': tf.io.VarLenFeature(tf.float32),
        'image/object/bbox/ymax': tf.io.VarLenFeature(tf.float32),
        'image/object/class/text': tf.io.VarLenFeature(tf.string),
        'image/object/class/label': tf.io.VarLenFeature(tf.int64),
    }
    example = tf.io.parse_single_example(example, feature_description)
    return example

# Caminho para a pasta com os arquivos TFRecord
train_tfrecord_folder = r'C:\Users\TECRA\Desktop\Uni\3ano\ESTAGIO\AutoML\tfrec_train'
test_tfrecord_folder = r'C:\Users\TECRA\Desktop\Uni\3ano\ESTAGIO\AutoML\tfrec_test'

# Lista para armazenar os caminhos dos arquivos TFRecord
train_tfrecord_paths = []
test_tfrecord_paths = []

# Percorrer todos os arquivos na pasta e adicionar os arquivos TFRecord à lista de treino.
for file in os.listdir(train_tfrecord_folder):
    if file.endswith('.tfrecord'):
        tfrecord_path = os.path.join(train_tfrecord_folder, file)
        train_tfrecord_paths.append(tfrecord_path)

# Percorrer todos os arquivos na pasta e adicionar os arquivos TFRecord à lista de teste.
for file in os.listdir(train_tfrecord_folder):
    if file.endswith('.tfrecord'):
        tfrecord_path = os.path.join(train_tfrecord_folder, file)
        test_tfrecord_paths.append(tfrecord_path)

# Carregar os arquivos TFRecord usando tf.data.TFRecordDataset
train_dataset = tf.data.TFRecordDataset(train_tfrecord_paths)
# Fazer o parse dos exemplos TFRecord
train_dataset = train_dataset.map(parse_tfrecord_fn)

# Carregar os arquivos TFRecord usando tf.data.TFRecordDataset
test_dataset = tf.data.TFRecordDataset(test_tfrecord_paths)
# Fazer o parse dos exemplos TFRecord
test_dataset = test_dataset.map(parse_tfrecord_fn)

print("EOF")