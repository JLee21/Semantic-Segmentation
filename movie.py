from moviepy.editor import ImageSequenceClip
import os
from glob import glob

def create_movie(path_input, movie_name):
    """
    :param path_input: abs folder path to computed images
    :param path_output:
    :param movie_name: name of the movie
    :param :
    """

    images = glob(os.path.join(path_input, '*.png'))

    clip = ImageSequenceClip(images, fps=5)

    if not movie_name.endswith('.mp4'):
        movie_name = '.'.join([movie_name, 'mp4'])
    clip.write_videofile(os.path.join('runs', movie_name), audio=False)

if __name__ == '__main__':
    path_input = os.path.join('runs', '1510606915.1779485')
    create_movie(path_input, 'movie')
