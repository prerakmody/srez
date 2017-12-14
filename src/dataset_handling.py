"""
0. Download all the files from this URL : https://www.dropbox.com/sh/8oqt9vytwxb3s4r/AACA-mNX0tHOAfPLuOCmNfe7a/Img/img_align_celeba_png.7z?dl=0
1. Once all the files are download, simply double-click on the 0th .7z file, and it will automatically unizip images from other .7z files
2. The CelebA dataset is a really large repo of 200k images.
3. Due to memory constraints, we shall split the directory into multiple directories with N = 50000 images 
"""

import os
import argparse
import shutil

def move_files(abs_dirname, N):
    """Move files into subdirectories."""

    files = [os.path.join(abs_dirname, f) for f in os.listdir(abs_dirname)]

    i = 0
    curr_subdir = None

    for f in files:
        # create new subdir if necessary
        if i % N == 0:
            tmp_abs_dirname = os.path.dirname(abs_dirname)
            subdir_name = os.path.join(tmp_abs_dirname, '{0:03d}'.format(i / N + 1))
            os.mkdir(tmp_abs_dirname)
            curr_subdir = subdir_name

        # move file to current dir
        f_base = os.path.basename(f)
        shutil.move(f, os.path.join(subdir_name, f_base))
        i += 1

if __name__ == "__main__":
    cwd = os.getcwd()
    abs_dirname = os.path.abspath(os.path.join(cwd, '../../__data/CelebA/img_align_celeba_png/img_align_celeba_png'))
    print ('Data Path:', abs_dirname)
    print ('Total Files:', len(os.listdir(abs_dirname)))

    N = 500000  # the number of files in seach subfolder folder

    move_files(abs_dirname, N)