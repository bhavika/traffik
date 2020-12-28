#!/usr/bin/python3
# Copyright 2020 Institute of Advanced Research in Artificial Intelligence (IARAI) GmbH.
# IARAI licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License. You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import sys, os
from PIL import Image
import h5py
import click


@click.group(name="utils")
def cli():
    pass


@cli.command(
    "compare_h5",
    help="usage: compare_h5 -l <left hand side file> -r <right hand side file>",
)
@click.option("-l", help="Left hand side file.")
@click.option("-r", help="Left hand side file.")
def compare_h5(l, r):
    lhs_data = load_h5_file(l)
    rhs_data = load_h5_file(r)

    if lhs_data.shape == rhs_data.shape:
        print((lhs_data == rhs_data).all())
    else:
        print("False")


@cli.command()
@click.option("--input_file", "-i")
@click.option("--channel", "-c", multiple=True)
@click.option("--output_file", "-o")
def extract_channel(input_file, channel, output_file):
    data = load_h5_file(input_file)
    data_shape = data.shape
    print("input file shape: {0}".format(data_shape))
    print("writing channel(s) {0}".format(str(channel)))
    out_array = np.take(data, list(channel), axis=-1)
    write_data_to_h5(out_array, output_file)
    print("File writen.")


@cli.command()
@click.option("--input_file", "-i")
@click.option("--channel", "-c", multiple=True)
@click.option("--output_file", "-o")
def extract_time_range(input_file, channel, output_file):
    data = load_h5_file(input_file)
    data_shape = data.shape
    print("input file shape: {0}".format(data_shape))
    print("writing channel(s) {0}".format(str(channel)))
    out_array = np.take(data, list(channel), axis=0)
    write_data_to_h5(out_array, output_file)
    print("File writen.")


@cli.command()
@click.option("--input_file", "-i")
def h5shape(input_file):
    data = load_h5_file(input_file)
    print_shape(data)


@cli.command()
@click.option("--input_file", "-i")
@click.option("--channel", "-c", multiple=True)
@click.option("--output_file", "-o")
def splice_test_tensor(input_file, channel, output_file):
    data = load_h5_file(input_file)
    data_shape = data.shape
    print("input file shape: {0}".format(data_shape))
    print("writing channel(s) {0}".format(str(channel)))
    out_array = np.take(data, list(channel), axis=0)
    write_data_to_h5(out_array[0], output_file)
    print("File writen.")


@cli.command(
    help="usage: tc2i.py -i <input file with h5 array> -c <channel(s)> -s <output file name stub> "
)
@click.option("--input_file", "-i")
@click.option("--channel", "-c", multiple=True, )
@click.option("--output_dir", "-s")
@click.option("--factor", "-f", default=1)
def tc2i(input_file, channel, output_dir, factor):
    channel = int(channel[0])
    factor = int(factor)
    data = load_h5_file(input_file)
    for i in range(data.shape[0]):
        d = np.clip(data[i, :, :, channel] * factor, 0, 255).astype("uint8")
        img = Image.fromarray(d)
        name = output_dir + "_" + "{:03d}".format(i) + ".png"
        img.save(name)
        print("image {0} written".format(name))


def load_h5_file(file_path):
    """
    Given a file path to an h5 file assumed to house a tensor,
    load that tensor into memory and return a pointer.
    """
    # load
    fr = h5py.File(file_path, "r")
    a_group_key = list(fr.keys())[0]
    data = list(fr[a_group_key])
    # transform to appropriate numpy array
    data = data[0:]
    data = np.stack(data, axis=0)
    return data


def print_shape(data):
    """
    print data shape
    """
    print(data.shape)


def write_data_to_h5(data, filename):
    """
    write data in gzipped h5 format
    """
    f = h5py.File(filename, "w", libver="latest")
    dset = f.create_dataset(
        "array", shape=(data.shape), data=data, compression="gzip", compression_opts=9
    )
    f.close()


def create_directory_structure(root, structure_path_list):
    """
    This command will create in the file system location root a subdirectory path
    determined in structure_path_list. For example
    creat_directory_path(".",["this","is","working") will create the directory path
    /this/is/working
    in the current directory. A touch of touch.
    """
    path = os.path.join(root, *structure_path_list)
    try:
        os.makedirs(path)
    except OSError:
        print("failed to create directory structure")
        sys.exit(2)


def list_filenames(directory):
    return os.listdir(directory)


if __name__ == "__main__":
    cli()
