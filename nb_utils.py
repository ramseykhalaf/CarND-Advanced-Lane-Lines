import urllib.request
import matplotlib.pyplot as plt
import os


def download(url, file):
    """
    Download file from <url>
    :param url: URL to file
    :param file: Local file path
    """
    if not os.path.isfile(file):
        print('Downloading ' + file + '...')
        urllib.request.urlretrieve(url, file)
        print('Download Finished')
    else:
        print("File exists '{}' - skipping".format(file))


def draw_side_by_side(*images):
    cols = len(images)
    plt.figure(figsize=(20,20))
    for i in range(0,cols):
        plt.subplot(1, cols, i+1)
        plt.imshow(images[i])
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
