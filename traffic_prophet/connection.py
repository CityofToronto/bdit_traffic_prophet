"""Base classes for connecting to Postgres database."""

import psycopg2
import configparser

from . import cfg


class Connection:
    # TO DO: this currently has NO FORMAL TEST SUITE, mainly because we'll
    # be replacing it with whatever Kedro cooked up at some point.

    def __init__(self, tablename):
        config = configparser.ConfigParser()
        config.read(cfg.postgres['cfgfile'].as_posix())
        self.credentials = dict(config[cfg.postgres['pg_name']])
        self.tablename = tablename

    def connect(self):
        return psycopg2.connect(**self.credentials)
