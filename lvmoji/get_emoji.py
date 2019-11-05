"""Scraper to download the emojis from Emojipedia

Attributes:
    DOWNLOAD_DIR (str): Description
    EmojiLink (TYPE): Description
    EMOJIPEDIA_URL (str): Description
    STYLES (list): Description
"""
import os
from collections import namedtuple
import time
from typing import List

import requests
from tqdm import tqdm
from bs4 import BeautifulSoup


EMOJIPEDIA_URL = "https://emojipedia.org/"
DOWNLOAD_DIR = 'emoji_set/'
STYLES = ['apple', 'google', 'facebook', 'microsoft', 'mozilla', 'whatsapp']

EmojiLink = namedtuple('EmojiLink', ['title', 'url', 'style'])


def get_contents_page_html(style: str) -> str:
    """Get the page listing all the emojis for a certain style (eg "apple", etc)

    :param style: the "style" of the emoji ("apple", "microsoft", etc)
    """
    response = requests.get(f'{EMOJIPEDIA_URL}/{style}')
    return response.content


def extract_emoji_links(html: str, style: str) -> List[EmojiLink]:
    """Extract resource urls for the emojis from the contents page

    :param html: The html of the contents page
    :param style: the "style" of the emoji page ("apple", "microsoft", etc)

    """
    bs = BeautifulSoup(html, 'html.parser')
    grid = bs.find(class_='emoji-grid')

    links = []
    for img in grid.find_all('img'):
        title = img.get('title')
        url = img.get('data-src')
        if url is not None and title is not None:
            links.append(EmojiLink(title, url, style))
    return links


def write_emoji_to_disk(data: bytes, title: str, style: str)-> None:
    """Write emoji bytes to disk

    :param data: the data representing the emoji
    :param title: the title of the emoji
    :param style: the "style" of the emoji ("apple", "microsoft", etc)
    """
    directory = f'{DOWNLOAD_DIR}/{style}'
    if not os.path.exists(directory):
        os.makedirs(directory)
    name = title.lower().replace(' ', '_')
    with open(f'{directory}/{name}', 'wb') as file:
        file.write(data)


def download_emojis(emoji_link_list: List[EmojiLink]) -> None:
    """Download the emojis in the emoji link list and write them to disk.

    :param emoji_link_list: a list of emoji links
    """
    if not os.path.exists(DOWNLOAD_DIR):
        os.mkdir(DOWNLOAD_DIR)
    for emoji_link in tqdm(emoji_link_list):
        resp = requests.get(emoji_link.url)
        if resp.status_code != 200:
            print(f'FAIL: {emoji_link.title}|{emoji_link.url} - {resp.content}')
            continue
        write_emoji_to_disk(resp.content, emoji_link.title, emoji_link.style)
        time.sleep(0.05)


def download_all_emoji_styles():
    """Download all the emoji styles and save them to disk."""
    for style in STYLES:
        html = get_contents_page_html(style)
        emoji_links = extract_emoji_links(html, style)
        download_emojis(emoji_links)


if __name__ == '__main__':
    download_all_emoji_styles()
