# -*- coding: utf-8 -*-

import requests
from bs4 import BeautifulSoup
from pathlib import Path

import os
import sys

def create_folder(path):
    Path(path).mkdir(parents=True, exist_ok=True)

def to_file(filename, data):
    '''
    Save the image data to file
    '''
    with open(filename, 'wb') as f:
        f.write(data)

def dl_image(folder_path, url):
    '''
    Download image from url and save
    '''
    print('Scraping: {}'.format(url))
    img_data = requests.get(url).content
    filename = url.split('/')[-1]
    to_file(os.path.join(folder_path, filename), img_data)

def get_img_pages(url):
    '''
    Get the list of all the pages contraining images
    '''
    html_content = requests.get(url).content
    soup = BeautifulSoup(html_content, 'html.parser')
    imgs = soup.find_all('a', {'class': 'img'})
    return imgs

def get_image_from_link(img_page_url, save_folder):
    '''
    Get an image from its page link:
        - Scrap the link and extract the image link
        - Download the image
    '''
    try:
        html_content = requests.get(img_page_url).content
        soup = BeautifulSoup(html_content, 'html.parser')
        a = soup.find('a', {'class': 'btn'})
        dl_image(save_folder, a['href'])
    except Exception as e:
        print("Could not request link: {}".format(img_page_url))

def get_images(imgs, save_folder):
    for img in imgs:
        href = img['href']
        print('Scraping url: {}'.format(href))
        get_image_from_link(href, save_folder)
        print()

def get_nb_elm(url):
    html_content = requests.get(url).content
    soup = BeautifulSoup(html_content, 'html.parser')
    a = soup.find('span', {'class': 'total'})
    nb_str = a.strong.contents
    nb_str = ''.join(nb_str[0].split(','))
    return int(nb_str)

def get_images_multi_links():
    pass

def main(argv):
    input_url, save_folder = argv[1], argv[2]
    create_folder(save_folder)

    nb_elm = get_nb_elm(input_url)
    url = input_url + '?start='

    for i in range(0, nb_elm, 100):
        print("Page: {}".format(i))
        url_req = url + '{}'.format(i)
        imgs = get_img_pages(url_req)
        get_images(imgs, save_folder)

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python3 reshape_img.py <input_url> <output_folder>")
        os._exit(1)

    main(sys.argv)