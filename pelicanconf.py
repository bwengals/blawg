#!/usr/bin/env python
# -*- coding: utf-8 -*- #
from __future__ import unicode_literals

AUTHOR = 'Bill Engels'
SITENAME = 'Bill Engels'
SITEURL = ''
THEME = "theme"

ABOUT_PAGE = '/pages/about.html'

PATH = 'content'

TIMEZONE = 'America/Los_Angeles'

DEFAULT_LANG = 'en'

# Feed generation is usually not desired when developing
FEED_ALL_ATOM = None
CATEGORY_FEED_ATOM = None
TRANSLATION_FEED_ATOM = None
AUTHOR_FEED_ATOM = None
AUTHOR_FEED_RSS = None

#ARTICLE_URL = 'blog/{date:%Y}/{date:%m}/{date:%d}/{slug}/'
#ARTICLE_SAVE_AS = 'blog/{date:%Y}/{date:%m}/{date:%d}/{slug}/index.html'


MARKUP = ['md']
IGNORE_FILES = ['.ipynb_checkpoints']

PLUGIN_PATHS = ['./pelican-plugins/']
PLUGINS = ["render_math", "summary"]
SUMMARY_MAX_LENGTH = 20

# Blogroll
LINKS = (('Github', 'https://github.com/bwengals'),)
#         ('Python.org', 'http://python.org/'),
#         ('Jinja2', 'http://jinja.pocoo.org/'),
#         ('You can modify those links in your config file', '#'),)

# Social widget
#SOCIAL = (('You can add links in your config file', '#'),
#          ('Another social link', '#'),)

DEFAULT_PAGINATION = False

# Uncomment following line if you want document-relative URLs when developing
RELATIVE_URLS = True
STATIC_PATHS = ['images']

MARKDOWN = {
    'extension_configs': {
        'markdown.extensions.codehilite': {'guess_lang': False,
                                           'css_class': 'highlight'},
        'markdown.extensions.extra': {},
    },
}
